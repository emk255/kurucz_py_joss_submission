"""Core synthesis loop."""

from __future__ import annotations

import math
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from numba import jit, prange

from ..config import SynthesisConfig
from ..io import atmosphere, export
from ..io.lines import atomic, compiler as line_compiler, fort19 as fort19_io
from ..io import spectrv as spectrv_io
from ..physics import (
    bfudge,
    continuum,
    populations,
    tables,
    helium_profiles,
    line_opacity,
)
from ..physics.hydrogen_wings import (
    compute_hydrogen_continuum,
)
from ..physics.profiles import hydrogen_line_profile, voigt_profile
from .radiative import solve_lte_spectrum
from .buffers import SynthResult, allocate_buffers
from synthe_py.tools import pops_exact

MAX_PROFILE_STEPS = 1_000_000
H_PLANCK = 6.62607015e-27  # erg * s
C_LIGHT_CM = 2.99792458e10  # cm / s
C_LIGHT_NM = 2.99792458e17  # nm/s (for frequency calculation)
C_LIGHT_KM = 299792.458  # km/s
K_BOLTZ = 1.380649e-16  # erg / K
NM_TO_CM = 1e-7
MIN_NPZ_CONVERSION_VERSION = 3

# Hydrogen level energies (cm^-1) from Fortran atlas7v/synthe.
_EHYD_CM = np.array(
    [
        0.0,
        82259.105,
        97492.302,
        102823.893,
        105291.651,
        106632.160,
        107440.444,
        107965.051,
    ],
    dtype=np.float64,
)
_HYD_RYD_CM = 109677.576  # cm^-1
_HYD_EINF_CM = 109678.764  # cm^-1 (Fortran EHYD limit for n>=9)

# CGF conversion constants from rgfall.for line 267
CGF_CONSTANT = 0.026538 / 1.77245  # Factor for converting GF to CONGF


# Shared Voigt profile — single canonical JIT-compiled implementation
from synthe_py.physics.voigt_jit import voigt_profile_jit as _voigt_profile_jit


@jit(nopython=True, cache=True)
def _accumulate_metal_profile_kernel(
    buffer: np.ndarray,
    continuum_row: np.ndarray,
    wavelength_grid: np.ndarray,
    center_index: int,
    line_wavelength: float,
    kappa0: float,
    damping: float,
    doppler_width: float,
    cutoff: float,
    wcon: float,  # Use -1.0 as sentinel for None
    wtail: float,  # Use -1.0 as sentinel for None
    h0tab: np.ndarray,
    h1tab: np.ndarray,
    h2tab: np.ndarray,
) -> None:
    """JIT-compiled kernel for accumulating metal profile wings.

    EXACTLY matches Fortran synthe.for XLINOP (labels 320-350):

    Flow:
      320: N10DOP = int(10 * DOPPLE * RESOLU)
           Near-wing loop: compute PROFILE(1..N10DOP) via Voigt/table
             If PROFILE(NSTEP) < KAPMIN → GO TO 323 (NSTEP = k, skip far wings)
           If loop completes normally → fall through to far wings
           Far wings: X = PROFILE(N10DOP)*N10DOP^2, MAXSTEP = sqrt(X/KAPMIN)+1
             PROFILE(N10DOP+1..MAXSTEP) = X/NSTEP^2
             NSTEP = MAXSTEP
      323: Boundary check, then unconditional wing accumulation
           Red wing: DO 324 ISTEP=MINRED,MIN(LENGTH-NBUFF,NSTEP)
             BUFFER(NBUFF+ISTEP) += PROFILE(ISTEP)   ← no per-step cutoff
           Blue wing: DO 326 ISTEP=MINBLUE,MIN(NBUFF-1,NSTEP)
             BUFFER(NBUFF-ISTEP) += PROFILE(ISTEP)   ← no per-step cutoff

    Key Fortran behaviors matched:
    - Early KAPMIN cutoff: NSTEP=k, PROFILE(k) stored (<KAPMIN), no far wings
    - Normal completion: far wings computed, NSTEP=MAXSTEP
    - Wing accumulation is UNCONDITIONAL (no per-step cutoff)
    - MAXBLUE = MIN(NBUFF-1, NSTEP) → Python: min(center_index, nstep_final)
    - MINRED = MAX(1, 1-NBUFF) → Python: max(1, -center_index)
    """
    if doppler_width <= 0.0 or kappa0 <= 0.0:
        return

    n_points = buffer.size
    adamp = max(damping, 1e-12)

    # Clamp center_index for continuum access (Fortran: MIN(MAX(NBUFF,1),LENGTH))
    clamped_center = max(0, min(center_index, n_points - 1))
    kapmin = cutoff * continuum_row[clamped_center]

    # Compute RESOLU from wavelength grid
    if clamped_center < n_points - 1:
        ratio = wavelength_grid[clamped_center + 1] / wavelength_grid[clamped_center]
        resolu = 1.0 / (ratio - 1.0)
    else:
        if clamped_center > 0:
            ratio = (
                wavelength_grid[clamped_center] / wavelength_grid[clamped_center - 1]
            )
            resolu = 1.0 / (ratio - 1.0)
        else:
            resolu = 300000.0

    # DOPPLE is dimensionless (Fortran: DOPPLE = thermal_velocity / c)
    dopple = doppler_width / line_wavelength if line_wavelength > 0 else 1e-6

    # ========== NEAR WING PROFILE (Fortran labels 320-1321) ==========
    # N10DOP = int(10 * DOPPLE * RESOLU)
    n10dop = int(10.0 * dopple * resolu)
    n10dop = min(n10dop, MAX_PROFILE_STEPS)

    profile = np.zeros(MAX_PROFILE_STEPS + 1, dtype=np.float64)
    vsteps = 200.0

    # Track whether near-wing loop hit KAPMIN cutoff early.
    # In Fortran, early cutoff → GO TO 323 with NSTEP=k, skipping far wings.
    early_cutoff = False
    nstep_final = 0  # Fortran's NSTEP at label 323

    if adamp < 0.2:
        # Fortran: H0TAB/H1TAB table lookup
        tabstep = vsteps / (dopple * resolu) if (dopple * resolu) > 0 else vsteps
        tabi = 0.5  # 0-based indexing (Fortran uses 1.5 for 1-based arrays)
        for nstep in range(1, n10dop + 1):
            tabi = tabi + tabstep
            itab = min(max(int(tabi), 0), len(h0tab) - 1)
            profile[nstep] = kappa0 * (h0tab[itab] + adamp * h1tab[itab])
            if profile[nstep] < kapmin:
                # Fortran: GO TO 323 with NSTEP = nstep (profile[nstep] stored but < KAPMIN)
                nstep_final = nstep
                early_cutoff = True
                break
    else:
        # Fortran: Full Voigt function
        dvoigt = 1.0 / dopple / resolu if (dopple * resolu) > 0 else 1e-6
        for nstep in range(1, n10dop + 1):
            x_val = float(nstep) * dvoigt
            profile[nstep] = kappa0 * _voigt_profile_jit(
                x_val, adamp, h0tab, h1tab, h2tab
            )
            if profile[nstep] < kapmin:
                nstep_final = nstep
                early_cutoff = True
                break

    # ========== FAR WINGS (Fortran lines 580-587) ==========
    # Only reached if near-wing loop completed WITHOUT early cutoff.
    if not early_cutoff:
        if n10dop > 0 and profile[n10dop] > 0:
            x_far = profile[n10dop] * float(n10dop) ** 2
        else:
            x_far = 0.0

        if x_far > 0 and kapmin > 0:
            maxstep = int(np.sqrt(x_far / kapmin) + 1.0)
            maxstep = min(maxstep, MAX_PROFILE_STEPS)
        else:
            maxstep = n10dop

        n1 = n10dop + 1
        for nstep in range(n1, maxstep + 1):
            profile[nstep] = x_far / float(nstep) ** 2 if nstep > 0 else 0.0

        # Fortran: NSTEP = MAXSTEP (line 587)
        nstep_final = maxstep

    # ========== Label 323: Boundary check ==========
    # Fortran: IF(NBUFF+NSTEP.LT.1.OR.NBUFF-NSTEP.GT.LENGTH)GO TO 350
    # In 0-indexed: center_index + nstep_final < 0 or center_index - nstep_final >= n_points
    if center_index + nstep_final < 0 or center_index - nstep_final >= n_points:
        return

    use_wcon = wcon > 0.0
    use_wtail = wtail > 0.0

    # ========== RED WING (Fortran lines 589-614) ==========
    # Fortran: IF(NBUFF.GE.LENGTH)GO TO 325  → skip red wing
    # 0-indexed: center_index >= n_points - 1
    if center_index < n_points - 1:
        # Fortran: MAXRED = MIN0(LENGTH-NBUFF, NSTEP)
        # 0-indexed: min(n_points - 1 - center_index, nstep_final)
        if center_index >= 0:
            maxred = min(n_points - 1 - center_index, nstep_final)
        else:
            maxred = min(n_points - 1, nstep_final)

        # Fortran: MINRED = MAX0(1, 1-NBUFF) → 0-indexed: max(1, -center_index)
        minred = max(1, -center_index)

        for istep in range(minred, maxred + 1):
            idx = center_index + istep
            if idx < 0 or idx >= n_points:
                continue

            # WCON/WTAIL handling (for fort.19 lines only; wcon=-1 for regular fort.12)
            if use_wcon:
                wave = wavelength_grid[idx]
                if wave <= wcon:
                    continue

            # Profile value - Fortran: BUFFER(NBUFF+ISTEP) += PROFILE(ISTEP)
            # Unconditional accumulation (no per-step cutoff!)
            value = profile[istep]

            # Apply tapering if needed (fort.19 only)
            if use_wtail:
                wave = wavelength_grid[idx]
                base = wcon if use_wcon else line_wavelength
                if wave < wtail:
                    value = value * (wave - base) / max(wtail - base, 1e-12)

            buffer[idx] += value

    # ========== Fortran: IF(NBUFF.LE.1)GO TO 350 → skip blue wing ==========
    # 0-indexed: center_index <= 0
    if center_index <= 0:
        return

    # ========== BLUE WING (Fortran lines 617-639) ==========
    # Fortran: MAXBLUE = MIN0(NBUFF-1, NSTEP) → 0-indexed: min(center_index, nstep_final)
    maxblue = min(center_index, nstep_final)
    # Fortran: MINBLUE = MAX0(1, NBUFF-LENGTH) → 0-indexed: max(1, center_index + 1 - n_points)
    minblue = max(1, center_index + 1 - n_points)

    for istep in range(minblue, maxblue + 1):
        idx = center_index - istep
        if idx < 0 or idx >= n_points:
            continue

        # WCON/WTAIL handling
        if use_wcon:
            wave = wavelength_grid[idx]
            if wave <= wcon:
                break  # Blue wing terminates at WCON

        # Profile value - unconditional accumulation (no per-step cutoff!)
        value = profile[istep]

        # Apply tapering if needed
        if use_wtail:
            wave = wavelength_grid[idx]
            base = wcon if use_wcon else line_wavelength
            if wave < wtail:
                value = value * (wave - base) / max(wtail - base, 1e-12)

        buffer[idx] += value


_ATOMIC_MASS = {
    "H": 1.008,
    "HE": 4.002602,
    "LI": 6.94,
    "BE": 9.0121831,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998403163,
    "NE": 20.1797,
    "NA": 22.98976928,
    "MG": 24.305,
    "AL": 26.9815385,
    "SI": 28.085,
    "P": 30.973761998,
    "S": 32.06,
    "CL": 35.45,
    "AR": 39.948,
    "K": 39.0983,
    "CA": 40.078,
    "SC": 44.955908,
    "TI": 47.867,
    "V": 50.9415,
    "CR": 51.9961,
    "MN": 54.938044,
    "FE": 55.845,
    "CO": 58.933194,
    "NI": 58.6934,
    "CU": 63.546,
    "ZN": 65.38,
}

_ELEMENT_Z = {
    "H": 1,
    "HE": 2,
    "LI": 3,
    "BE": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "NE": 10,
    "NA": 11,
    "MG": 12,
    "AL": 13,
    "SI": 14,
    "P": 15,
    "S": 16,
    "CL": 17,
    "AR": 18,
    "K": 19,
    "CA": 20,
    "SC": 21,
    "TI": 22,
    "V": 23,
    "CR": 24,
    "MN": 25,
    "FE": 26,
    "CO": 27,
    "NI": 28,
    "CU": 29,
    "ZN": 30,
    "GA": 31,
    "GE": 32,
    "AS": 33,
    "SE": 34,
    "BR": 35,
    "KR": 36,
    "RB": 37,
    "SR": 38,
    "Y": 39,
    "ZR": 40,
    "NB": 41,
    "MO": 42,
    "TC": 43,
    "RU": 44,
    "RH": 45,
    "PD": 46,
    "AG": 47,
    "CD": 48,
    "IN": 49,
    "SN": 50,
    "SB": 51,
    "TE": 52,
    "I": 53,
    "XE": 54,
    "CS": 55,
    "BA": 56,
    "LA": 57,
    "CE": 58,
    "PR": 59,
    "ND": 60,
    "PM": 61,
    "SM": 62,
    "EU": 63,
    "GD": 64,
    "TB": 65,
    "DY": 66,
    "HO": 67,
    "ER": 68,
    "TM": 69,
    "YB": 70,
    "LU": 71,
    "HF": 72,
    "TA": 73,
    "W": 74,
    "RE": 75,
    "OS": 76,
    "IR": 77,
    "PT": 78,
    "AU": 79,
    "HG": 80,
    "TL": 81,
    "PB": 82,
    "BI": 83,
    "PO": 84,
    "AT": 85,
    "RN": 86,
    "FR": 87,
    "RA": 88,
    "AC": 89,
    "TH": 90,
    "PA": 91,
    "U": 92,
    "NP": 93,
    "PU": 94,
    "AM": 95,
    "CM": 96,
    "BK": 97,
    "CF": 98,
    "ES": 99,
}
_ELEMENT_SYMBOL_BY_Z = {value: key for key, value in _ELEMENT_Z.items()}

_EMPTY_FLOAT64: np.ndarray = np.empty(0, dtype=np.float64)


def _load_atmosphere(cfg: SynthesisConfig) -> atmosphere.AtmosphereModel:
    model_path = cfg.atmosphere.model_path

    def _npz_conversion_version(npz_path: Path) -> int:
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                raw = data.get("meta_npz_conversion_version", None)
                if raw is None:
                    return 0
                arr = np.asarray(raw).ravel()
                if arr.size == 0:
                    return 0
                return int(arr[0])
        except Exception:
            return 0

    def _refresh_stale_npz(npz_path: Path) -> None:
        if os.getenv("PY_DISABLE_AUTO_NPZ_REFRESH", "0") == "1":
            return
        if model_path.suffix.lower() != ".atm" or not model_path.exists():
            return
        current_version = _npz_conversion_version(npz_path)
        if current_version >= MIN_NPZ_CONVERSION_VERSION:
            return

        repo_root = Path(__file__).resolve().parents[2]
        atlas_tables = repo_root / "synthe_py" / "data" / "atlas_tables.npz"
        convert_script = repo_root / "synthe_py" / "tools" / "convert_atm_to_npz.py"
        if not convert_script.exists() or not atlas_tables.exists():
            logging.warning(
                "NPZ %s is stale (version=%s) but converter/tables are unavailable; proceeding with cached data.",
                npz_path,
                current_version,
            )
            return

        logging.info(
            "Refreshing stale NPZ cache %s (version=%s < %s).",
            npz_path,
            current_version,
            MIN_NPZ_CONVERSION_VERSION,
        )
        subprocess.run(
            [
                sys.executable,
                str(convert_script),
                str(model_path),
                str(npz_path),
                "--atlas-tables",
                str(atlas_tables),
            ],
            cwd=repo_root,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # If explicit NPZ path is provided, use it directly
    if cfg.atmosphere.npz_path is not None:
        npz_path = cfg.atmosphere.npz_path
        if not npz_path.exists():
            raise FileNotFoundError(f"Specified NPZ file does not exist: {npz_path}")
        _refresh_stale_npz(npz_path)
        logging.info(f"Loading atmosphere from specified NPZ file: {npz_path}")
        return atmosphere.load_cached(npz_path)

    # If model path is already an .npz file, use it directly
    if model_path.suffix == ".npz":
        return atmosphere.load_cached(model_path)

    # If given an .atm file, prefer a sibling .npz with the exact same stem.
    # Fortran uses the per-model .atm input (fort.10) for each Teff/logg, so the
    # Python cache lookup should not collapse distinct models to a shared base name.
    sibling_npz = model_path.with_suffix(".npz")
    if sibling_npz.exists():
        _refresh_stale_npz(sibling_npz)
        logging.info(f"Loading cached atmosphere alongside .atm: {sibling_npz}")
        return atmosphere.load_cached(sibling_npz)

    # Otherwise, fall back to shared cached atmospheres based on the base model name.
    # Extract model name from path (e.g., at12_aaaaa_t05770g4.44.atm -> at12_aaaaa)
    model_name = model_path.stem
    # Remove temperature/gravity suffix if present (e.g., at12_aaaaa_t05770g4.44 -> at12_aaaaa)
    if "_t" in model_name:
        model_name = model_name.split("_t")[0]

    # Look for cached NPZ file (prefer the fixed_interleaved version)
    cached_paths = [
        Path(f"synthe_py/data/{model_name}_atmosphere_fixed_interleaved.npz"),
        Path(f"synthe_py/data/{model_name}_atmosphere.npz"),
    ]

    for cached_path in cached_paths:
        if cached_path.exists():
            logging.info(f"Loading cached atmosphere: {cached_path}")
            return atmosphere.load_cached(cached_path)

    raise FileNotFoundError(
        f"Could not find cached .npz atmosphere file for {model_path}. "
        f"Tried: {[str(p) for p in cached_paths]}. "
        f"Please convert the atmosphere file to NPZ format first, or use --npz to specify the path."
    )


def _build_wavelength_grid(cfg: SynthesisConfig) -> np.ndarray:
    start = cfg.wavelength_grid.start
    end = cfg.wavelength_grid.end
    resolution = cfg.wavelength_grid.resolution
    if resolution <= 0.0:
        raise ValueError("Resolution must be positive for geometric wavelength grid")

    ratio = 1.0 + 1.0 / resolution
    rlog = math.log(ratio)
    ix_start = math.log(start) / rlog
    ix_floor = math.floor(ix_start)
    if math.exp(ix_floor * rlog) < start:
        ix_floor += 1
    wbegin = math.exp(ix_floor * rlog)

    wavelengths: List[float] = []
    wl = wbegin
    while wl <= end * (1.0 + 1e-9):
        wavelengths.append(wl)
        wl *= ratio

    return np.array(wavelengths, dtype=np.float64)


def _nearest_grid_indices(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Find nearest grid indices for given values.

    IMPORTANT: This now returns -1 for values below grid and grid.size for values
    above grid, to allow the wing kernel to handle margin lines correctly.
    Lines outside the grid can still contribute their wings TO the grid (DELLIM behavior).

    CRITICAL FIX (Dec 2025): Match Fortran rgfall.for EXACTLY.
    Fortran uses logarithmic rounding for exponential grids:
        IXWL = DLOG(WLVAC) / RATIOLG + 0.5D0
        NBUFF = IXWL - IXWLBEG + 1

    Previous Python code used linear nearest-neighbor (abs distance to left/right),
    which gives different results for ~30% of boundary cases on exponential grids.
    """
    # Derive grid parameters from the grid itself
    # For exponential grid: grid[i] = grid[0] * ratio^i
    if len(grid) < 2:
        return np.zeros(len(values), dtype=np.int64)

    ratio = grid[1] / grid[0]
    ratiolg = np.log(ratio)
    ix_start = int(
        np.log(grid[0]) / ratiolg + 0.5
    )  # Match Fortran's IXWLBEG calculation

    # Use Fortran's logarithmic rounding: IXWL = LOG(WL)/RATIOLG + 0.5
    # Then index = IXWL - IXWLBEG (0-based, since Fortran NBUFF is 1-based)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_values = np.log(values)
        ixwl = (log_values / ratiolg + 0.5).astype(np.int64)
        indices = ixwl - ix_start

    # Mark lines below grid as -1 (will be handled by wing kernel)
    below_grid = values < grid[0]
    # Mark lines above grid as grid.size (will be handled by wing kernel)
    above_grid = values > grid[-1]

    # Set special values for out-of-grid lines
    indices[below_grid] = -1
    indices[above_grid] = grid.size

    return indices


def _match_catalog_to_fort19(
    catalog_wavelength: np.ndarray, meta_wavelength: np.ndarray, tolerance: float = 1e-3
) -> Dict[int, int]:
    """Associate catalog entries with fort.19 wing metadata."""

    mapping: Dict[int, int] = {}
    if catalog_wavelength.size == 0 or meta_wavelength.size == 0:
        return mapping

    order = np.argsort(catalog_wavelength)
    sorted_wavelength = catalog_wavelength[order]
    indices = np.searchsorted(sorted_wavelength, meta_wavelength)
    for meta_idx, pos in enumerate(indices):
        best_idx: Optional[int] = None
        best_delta = tolerance
        for candidate in (pos, pos - 1):
            if 0 <= candidate < sorted_wavelength.size:
                delta = abs(sorted_wavelength[candidate] - meta_wavelength[meta_idx])
                if delta < best_delta:
                    best_delta = delta
                    best_idx = candidate
        if best_idx is not None:
            catalog_idx = int(order[best_idx])
            if catalog_idx not in mapping:
                mapping[catalog_idx] = meta_idx
    return mapping


def _atomic_mass_lookup(element_symbol: str) -> Optional[float]:
    key = element_symbol.strip().upper().replace(" ", "")
    return _ATOMIC_MASS.get(key)


def _layer_value(arr: Optional[np.ndarray], idx: int) -> float:
    if arr is None or arr.size <= idx:
        return 0.0
    return float(arr[idx])


def _element_atomic_number(symbol: str) -> Optional[int]:
    key = symbol.strip().upper().replace(" ", "")
    return _ELEMENT_Z.get(key)


@jit(nopython=True, cache=True)
def _vacuum_to_air_jit(w_nm: float) -> float:
    """Convert vacuum wavelength (nm) to air (nm) using SYNTHE's formula."""
    waven = 1.0e7 / w_nm
    denom = (
        1.0000834213
        + 2_406_030.0 / (1.30e10 - waven * waven)
        + 15_997.0 / (3.89e9 - waven * waven)
    )
    return w_nm / denom


def _vacuum_to_air(w_nm: float) -> float:
    """Convert vacuum wavelength (nm) to air (nm) using SYNTHE's formula."""
    return _vacuum_to_air_jit(w_nm)


@jit(nopython=True, cache=True)
def _compute_continuum_limits_jit(
    ncon: int,
    nelion: int,
    nelionx: int,
    emerge_val: float,
    emerge_h_val: float,
    contx: np.ndarray,  # metal_tables.contx
    ifvac: int,
) -> tuple[float, float]:
    """Compute WCON/WTAIL (nm) following the XLINOP continuum merge rules (Numba-compatible).

    Returns (wcon_nm, wtail_nm) where -1.0 indicates None/not set.
    """
    # Use -1.0 as sentinel for None
    if ncon <= 0 or nelionx <= 0:
        return -1.0, -1.0
    if nelionx > contx.shape[1] or ncon > contx.shape[0]:
        return -1.0, -1.0

    cont_val = contx[ncon - 1, nelionx - 1]
    if cont_val <= 0.0:
        return -1.0, -1.0

    emerge_line = emerge_h_val if nelion == 1 else emerge_val
    denom = cont_val - emerge_line
    if abs(denom) <= 1e-8:
        return -1.0, -1.0

    wcon_nm = 1.0e7 / denom

    denom_tail = cont_val - emerge_line - 500.0
    wtail_nm = -1.0
    if abs(denom_tail) > 1e-8:
        wtail_nm = 1.0e7 / denom_tail
        if wtail_nm < 0.0:
            wtail_nm = 2.0 * wcon_nm
        wtail_nm = min(2.0 * wcon_nm, wtail_nm)

    if ifvac == 0:
        wcon_nm = _vacuum_to_air_jit(wcon_nm)
        if wtail_nm > 0.0:
            wtail_nm = _vacuum_to_air_jit(wtail_nm)

    if wtail_nm > 0.0 and wtail_nm <= wcon_nm:
        wtail_nm = -1.0
    return wcon_nm, wtail_nm


def _compute_continuum_limits(
    ncon: int,
    nelion: int,
    nelionx: int,
    emerge_val: float,
    emerge_h_val: float,
    metal_tables: tables.MetalWingTables,
    ifvac: int,
) -> tuple[Optional[float], Optional[float]]:
    """Compute WCON/WTAIL (nm) following the XLINOP continuum merge rules."""
    wcon_nm, wtail_nm = _compute_continuum_limits_jit(
        ncon, nelion, nelionx, emerge_val, emerge_h_val, metal_tables.contx, ifvac
    )
    wcon = wcon_nm if wcon_nm > 0.0 else None
    wtail = wtail_nm if wtail_nm > 0.0 else None
    return wcon, wtail


def _compute_merged_continuum_limits(
    line_wavelength: float,
    nlast: int,
    emerge_val: float,
    emerge_h_val: float,
    ion_index: int,
    ifvac: int,
) -> tuple[Optional[float], Optional[float]]:
    """Compute WMERGE/WTAIL (nm) for TYPE>3 merged-continuum lines."""
    if line_wavelength <= 0.0 or nlast <= 0:
        return None, None
    ryd = 109677.576 if ion_index == 1 else 109737.312
    denom_shift = 1.0e7 / line_wavelength - ryd / float(nlast * nlast)
    if abs(denom_shift) <= 1e-12:
        return None, None
    wshift = 1.0e7 / denom_shift
    emerge_line = emerge_h_val if ion_index == 1 else emerge_val
    denom_merge = 1.0e7 / line_wavelength - emerge_line
    if abs(denom_merge) <= 1e-12:
        return None, None
    wmerge = 1.0e7 / denom_merge
    if wmerge < 0.0:
        wmerge = wshift + wshift
    wmerge = max(wmerge, wshift)
    wmerge = min(wshift + wshift, wmerge)
    denom_tail = 1.0e7 / wmerge - 500.0
    if abs(denom_tail) <= 1e-12:
        return wmerge, None
    wtail = 1.0e7 / denom_tail
    if wtail < 0.0:
        wtail = wmerge + wmerge
    wtail = min(wmerge + wmerge, wtail)
    if ifvac == 0:
        wmerge = _vacuum_to_air(wmerge)
        wtail = _vacuum_to_air(wtail)
    return wmerge, wtail


def _accumulate_line_profile(
    buffer: np.ndarray,
    continuum_row: np.ndarray,
    wavelength_grid: np.ndarray,
    center_index: int,
    line_wavelength: float,
    kappa0: float,
    damping: float,
    doppler_width: float,
    cutoff: float,
) -> None:
    if doppler_width <= 0.0:
        return

    n_points = buffer.size
    damping = max(damping, 1e-12)

    center_value = kappa0 * voigt_profile(0.0, damping)
    if center_value >= continuum_row[center_index] * cutoff:
        buffer[center_index] += center_value

    red_active = True
    blue_active = True
    offset = 1
    while offset <= MAX_PROFILE_STEPS and (red_active or blue_active):
        if red_active:
            idx = center_index + offset
            if idx >= n_points:
                red_active = False
            else:
                x_red = (wavelength_grid[idx] - line_wavelength) / doppler_width
                value_red = kappa0 * voigt_profile(x_red, damping)
                if value_red < continuum_row[idx] * cutoff:
                    red_active = False
                else:
                    buffer[idx] += value_red
        if blue_active:
            idx = center_index - offset
            if idx < 0:
                blue_active = False
            else:
                x_blue = (wavelength_grid[idx] - line_wavelength) / doppler_width
                value_blue = kappa0 * voigt_profile(x_blue, damping)
                if value_blue < continuum_row[idx] * cutoff:
                    blue_active = False
                else:
                    buffer[idx] += value_blue
        offset += 1


def _accumulate_metal_profile(
    buffer: np.ndarray,
    continuum_row: np.ndarray,
    wavelength_grid: np.ndarray,
    center_index: int,
    line_wavelength: float,
    kappa0: float,
    damping: float,
    doppler_width: float,
    cutoff: float,
    wcon: Optional[float] = None,
    wtail: Optional[float] = None,
) -> None:
    """
    Accumulate metal wings matching Fortran's two-phase approach:
    1. Near wings (within 10 Doppler widths): Full Voigt profile
    2. Far wings: 1/x² approximation

    Matches synthe.for lines 296-333.
    """
    if doppler_width <= 0.0 or kappa0 <= 0.0:
        return

    # Get Voigt tables
    voigt_tables = tables.voigt_tables()
    h0tab = voigt_tables.h0tab
    h1tab = voigt_tables.h1tab
    h2tab = voigt_tables.h2tab

    # Convert None to sentinel values
    wcon_val = wcon if wcon is not None else -1.0
    wtail_val = wtail if wtail is not None else -1.0

    _accumulate_metal_profile_kernel(
        buffer,
        continuum_row,
        wavelength_grid,
        center_index,
        line_wavelength,
        kappa0,
        damping,
        doppler_width,
        cutoff,
        wcon_val,
        wtail_val,
        h0tab,
        h1tab,
        h2tab,
    )
    return


def _accumulate_merged_continuum(
    buffer: np.ndarray,
    continuum_row: np.ndarray,
    wavelength_grid: np.ndarray,
    center_index: int,
    line_wavelength: float,
    kappa: float,
    cutoff: float,
    merge_wavelength: float,
    tail_wavelength: float,
) -> None:
    """Approximate the XLINOP merged-continuum ramp (TYPE=81)."""

    if kappa <= 0.0:
        return

    n_points = buffer.size
    idx_start = max(center_index, 0)
    idx_merge = np.searchsorted(wavelength_grid, merge_wavelength, side="left")
    idx_tail = np.searchsorted(wavelength_grid, tail_wavelength, side="right")
    idx_tail = min(idx_tail, n_points)
    if idx_tail <= idx_start:
        return

    denom = max(idx_tail - max(idx_merge, idx_start), 1)

    for idx in range(idx_start, idx_tail):
        wave = wavelength_grid[idx]
        if wave < line_wavelength:
            continue
        value = kappa
        if idx >= idx_merge:
            value *= max(idx_tail - idx, 0) / denom
        if value < continuum_row[idx] * cutoff:
            break
        buffer[idx] += value


def _accumulate_autoionizing_profile(
    buffer: np.ndarray,
    continuum_row: np.ndarray,
    wavelength_grid: np.ndarray,
    center_index: int,
    line_wavelength: float,
    kappa0: float,
    gamma_rad: float,
    gamma_stark: float,
    gamma_vdw: float,
    cutoff: float,
) -> bool:
    n_points = buffer.size
    center_cutoff = continuum_row[center_index] * cutoff
    if kappa0 < center_cutoff or kappa0 <= 0.0:
        return False

    freq_center = C_LIGHT_CM / (line_wavelength * NM_TO_CM)
    gamma = max(abs(gamma_rad), 1e-30)
    ashore = gamma_stark
    bshore = gamma_vdw
    if abs(bshore) < 1e-30:
        bshore = 1e-30

    buffer[center_index] += kappa0

    red_active = True
    blue_active = True
    offset = 1
    # Hydrogen wings can extend much farther than metal wings (Balmer series),
    # so allow expansion to the grid edge rather than a fixed MAX_PROFILE_STEPS cap.
    max_steps = max(center_index, n_points - center_index - 1)

    while offset <= max_steps and (red_active or blue_active):
        if red_active:
            idx = center_index + offset
            if idx >= n_points:
                red_active = False
            else:
                freq = C_LIGHT_CM / (wavelength_grid[idx] * NM_TO_CM)
                eps = 2.0 * (freq - freq_center) / gamma
                value = kappa0 * (ashore * eps + bshore) / (eps * eps + 1.0) / bshore
                if value <= 0.0 or value < continuum_row[idx] * cutoff:
                    red_active = False
                else:
                    buffer[idx] += value

        if blue_active:
            idx = center_index - offset
            if idx < 0:
                blue_active = False
            else:
                freq = C_LIGHT_CM / (wavelength_grid[idx] * NM_TO_CM)
                eps = 2.0 * (freq - freq_center) / gamma
                value = kappa0 * (ashore * eps + bshore) / (eps * eps + 1.0) / bshore
                if value <= 0.0 or value < continuum_row[idx] * cutoff:
                    blue_active = False
                else:
                    buffer[idx] += value

        offset += 1

    return True


def _apply_fort19_profile(
    wing_type: fort19_io.Fort19WingType,
    line_type_code: int,
    tmp_buffer: np.ndarray,
    continuum_row: np.ndarray,
    wavelength_grid: np.ndarray,
    center_index: int,
    line_wavelength: float,
    kappa0: float,
    cutoff: float,
    metal_wings_row: np.ndarray,
    metal_sources_row: np.ndarray,
    bnu_row: np.ndarray,
    wcon: Optional[float],
    wtail: Optional[float],
    he_solver: Optional[helium_profiles.HeliumWingSolver],
    use_numba_helium: bool,
    depth_idx: int,
    depth_state: populations.DepthState,
    n_lower: int,
    n_upper: int,
    gamma_rad: float,
    gamma_stark: float,
    gamma_vdw: float,
    doppler_width: float,
    line_index: int = -1,
) -> bool:
    """Handle special fort.19 wing prescriptions. Returns True if consumed."""
    if wing_type == fort19_io.Fort19WingType.CORONAL:
        # Fortran TYPE=2 (coronal) goes to label 500 -> 900 (skip line).
        return True

    if line_type_code < -2:
        # Match synthe.for XLINOP control flow:
        # IF(TYPE.LT.-2)GO TO 200
        # i.e., treat these as normal-line Voigt (not special He profile branch).
        tmp_buffer.fill(0.0)
        doppler = max(doppler_width, 1e-12)
        kappa_eff = kappa0
        if line_type_code == -4:
            # 3He branch adjustment in Fortran helium section.
            kappa_eff /= 1.155
            doppler *= 1.155
        damping_normal = (
            gamma_rad
            + gamma_stark * depth_state.electron_density
            + gamma_vdw * depth_state.txnxn
        ) / max(doppler / line_wavelength, 1e-40)
        adamp = max(damping_normal, 1e-12)
        n_points = tmp_buffer.size
        clamped_center = max(0, min(center_index, n_points - 1))
        base = wcon if wcon is not None else line_wavelength
        voigt_tables = tables.voigt_tables()
        h0tab = voigt_tables.h0tab
        h1tab = voigt_tables.h1tab
        h2tab = voigt_tables.h2tab

        # Red wing (Fortran 211 loop style): test cutoff before accumulation.
        if line_wavelength <= wavelength_grid[n_points - 1]:
            for idx in range(clamped_center, n_points):
                wave = wavelength_grid[idx]
                if wcon is not None and wave <= wcon:
                    continue
                x_val = abs(wave - line_wavelength) / doppler
                value = kappa_eff * _voigt_profile_jit(x_val, adamp, h0tab, h1tab, h2tab)
                if wtail is not None and wave < wtail:
                    value = value * (wave - base) / max(wtail - base, 1e-12)
                if value < continuum_row[idx] * cutoff:
                    break
                tmp_buffer[idx] += value

        # Blue wing (Fortran 214 loop style): accumulate then test cutoff.
        if clamped_center > 0 and line_wavelength >= wavelength_grid[0]:
            for idx in range(clamped_center - 1, -1, -1):
                wave = wavelength_grid[idx]
                if wcon is not None and wave <= wcon:
                    break
                x_val = abs(wave - line_wavelength) / doppler
                value = kappa_eff * _voigt_profile_jit(x_val, adamp, h0tab, h1tab, h2tab)
                if wtail is not None and wave < wtail:
                    value = value * (wave - base) / max(wtail - base, 1e-12)
                tmp_buffer[idx] += value
                if value < continuum_row[idx] * cutoff:
                    break

        metal_wings_row += tmp_buffer
        metal_sources_row += tmp_buffer * bnu_row
        return True

    if he_solver is not None and line_type_code in (-3, -4, -6):
        tmp_buffer.fill(0.0)
        doppler = max(doppler_width, 1e-12)
        kappa_eff = kappa0
        if line_type_code == -4:
            kappa_eff /= 1.155
            doppler *= 1.155
        if use_numba_helium and hasattr(he_solver, "evaluate_numba"):
            center_value = kappa_eff * he_solver.evaluate_numba(
                line_type=line_type_code,
                depth_idx=depth_idx,
                delta_nm=0.0,
                line_wavelength=line_wavelength,
                doppler_width=doppler,
                gamma_rad=gamma_rad,
                gamma_stark=gamma_stark,
            )
        else:
            center_value = kappa_eff * he_solver.evaluate(
                line_type=line_type_code,
                depth_idx=depth_idx,
                delta_nm=0.0,
                line_wavelength=line_wavelength,
                doppler_width=doppler,
                gamma_rad=gamma_rad,
                gamma_stark=gamma_stark,
            )
        center_cutoff = continuum_row[center_index] * cutoff
        center_pass = bool(center_value > 0.0 and center_value >= center_cutoff)
        if not center_pass:
            # Fortran XLINOP routes TYPE < -2 through label 200 (normal-line Voigt)
            # in this source, so when the specialized helium profile is below cutoff
            # we must not silently drop the line.
            tmp_buffer.fill(0.0)
            damping_fallback = (
                gamma_rad
                + gamma_stark * depth_state.electron_density
                + gamma_vdw * depth_state.txnxn
            ) / max(doppler / line_wavelength, 1e-40)
            if 0 <= center_index < tmp_buffer.size:
                tmp_buffer[center_index] = kappa_eff * voigt_profile(
                    0.0, max(damping_fallback, 1e-12)
                )
            _accumulate_metal_profile(
                buffer=tmp_buffer,
                continuum_row=continuum_row,
                wavelength_grid=wavelength_grid,
                center_index=center_index,
                line_wavelength=line_wavelength,
                kappa0=kappa_eff,
                damping=max(damping_fallback, 1e-12),
                doppler_width=doppler,
                cutoff=cutoff,
                wcon=wcon,
                wtail=wtail,
            )
            metal_wings_row += tmp_buffer
            metal_sources_row += tmp_buffer * bnu_row
            return True
        tmp_buffer[center_index] = center_value
        n_points = wavelength_grid.size
        red_active = True
        blue_active = True
        offset = 1
        last_red_idx = center_index
        last_blue_idx = center_index
        while offset <= MAX_PROFILE_STEPS and (red_active or blue_active):
            if red_active:
                idx = center_index + offset
                if idx >= n_points:
                    red_active = False
                else:
                    delta = wavelength_grid[idx] - line_wavelength
                    if use_numba_helium and hasattr(he_solver, "evaluate_numba"):
                        value = kappa_eff * he_solver.evaluate_numba(
                            line_type=line_type_code,
                            depth_idx=depth_idx,
                            delta_nm=delta,
                            line_wavelength=line_wavelength,
                            doppler_width=doppler,
                            gamma_rad=gamma_rad,
                            gamma_stark=gamma_stark,
                        )
                    else:
                        value = kappa_eff * he_solver.evaluate(
                            line_type=line_type_code,
                            depth_idx=depth_idx,
                            delta_nm=delta,
                            line_wavelength=line_wavelength,
                            doppler_width=doppler,
                            gamma_rad=gamma_rad,
                            gamma_stark=gamma_stark,
                        )
                    if value <= 0.0 or value < continuum_row[idx] * cutoff:
                        red_active = False
                    else:
                        tmp_buffer[idx] += value
                        last_red_idx = idx
            if blue_active:
                idx = center_index - offset
                if idx < 0:
                    blue_active = False
                else:
                    delta = wavelength_grid[idx] - line_wavelength
                    if use_numba_helium and hasattr(he_solver, "evaluate_numba"):
                        value = kappa_eff * he_solver.evaluate_numba(
                            line_type=line_type_code,
                            depth_idx=depth_idx,
                            delta_nm=delta,
                            line_wavelength=line_wavelength,
                            doppler_width=doppler,
                            gamma_rad=gamma_rad,
                            gamma_stark=gamma_stark,
                        )
                    else:
                        value = kappa_eff * he_solver.evaluate(
                            line_type=line_type_code,
                            depth_idx=depth_idx,
                            delta_nm=delta,
                            line_wavelength=line_wavelength,
                            doppler_width=doppler,
                            gamma_rad=gamma_rad,
                            gamma_stark=gamma_stark,
                        )
                    if value <= 0.0 or value < continuum_row[idx] * cutoff:
                        blue_active = False
                    else:
                        tmp_buffer[idx] += value
                        last_blue_idx = idx
            offset += 1
        metal_wings_row += tmp_buffer
        metal_sources_row += tmp_buffer * bnu_row
        return True

    if wing_type == fort19_io.Fort19WingType.CONTINUUM:
        merge_w = max(line_wavelength, wcon) if wcon is not None else line_wavelength
        tail_w = wtail if wtail is not None else merge_w * 1.1
        if tail_w <= merge_w:
            tail_w = merge_w * 1.05
        _accumulate_merged_continuum(
            buffer=tmp_buffer,
            continuum_row=continuum_row,
            wavelength_grid=wavelength_grid,
            center_index=center_index,
            line_wavelength=line_wavelength,
            kappa=kappa0,
            cutoff=cutoff,
            merge_wavelength=merge_w,
            tail_wavelength=tail_w,
        )
        tmp_buffer[center_index] = 0.0
        metal_wings_row += tmp_buffer
        metal_sources_row += tmp_buffer * bnu_row
        return True

    if wing_type == fort19_io.Fort19WingType.AUTOIONIZING:
        tmp_buffer.fill(0.0)
        if not _accumulate_autoionizing_profile(
            buffer=tmp_buffer,
            continuum_row=continuum_row,
            wavelength_grid=wavelength_grid,
            center_index=center_index,
            line_wavelength=line_wavelength,
            kappa0=kappa0,
            gamma_rad=gamma_rad,
            gamma_stark=gamma_stark,
            gamma_vdw=gamma_vdw,
            cutoff=cutoff,
        ):
            return True
        metal_wings_row += tmp_buffer
        metal_sources_row += tmp_buffer * bnu_row
        return True

    if he_solver is not None and wing_type in {
        fort19_io.Fort19WingType.HELIUM_3_II,
        fort19_io.Fort19WingType.HELIUM_3,
        fort19_io.Fort19WingType.HELIUM_4,
    }:
        tmp_buffer.fill(0.0)
        doppler = max(doppler_width, 1e-12)
        kappa_eff = kappa0
        if wing_type == fort19_io.Fort19WingType.HELIUM_3:
            kappa_eff /= 1.155
            doppler *= 1.155
        center_value = kappa_eff * he_solver.evaluate(
            line_type=line_type_code,
            depth_idx=depth_idx,
            delta_nm=0.0,
            line_wavelength=line_wavelength,
            doppler_width=doppler,
            gamma_rad=gamma_rad,
            gamma_stark=gamma_stark,
        )
        if center_value < continuum_row[center_index] * cutoff or center_value <= 0.0:
            return True
        tmp_buffer[center_index] = center_value
        n_points = wavelength_grid.size
        red_active = True
        blue_active = True
        offset = 1
        while offset <= MAX_PROFILE_STEPS and (red_active or blue_active):
            if red_active:
                idx = center_index + offset
                if idx >= n_points:
                    red_active = False
                else:
                    delta = wavelength_grid[idx] - line_wavelength
                    value = kappa_eff * he_solver.evaluate(
                        line_type=line_type_code,
                        depth_idx=depth_idx,
                        delta_nm=delta,
                        line_wavelength=line_wavelength,
                        doppler_width=doppler,
                        gamma_rad=gamma_rad,
                        gamma_stark=gamma_stark,
                    )
                    if value <= 0.0 or value < continuum_row[idx] * cutoff:
                        red_active = False
                    else:
                        tmp_buffer[idx] += value
            if blue_active:
                idx = center_index - offset
                if idx < 0:
                    blue_active = False
                else:
                    delta = wavelength_grid[idx] - line_wavelength
                    value = kappa_eff * he_solver.evaluate(
                        line_type=line_type_code,
                        depth_idx=depth_idx,
                        delta_nm=delta,
                        line_wavelength=line_wavelength,
                        doppler_width=doppler,
                        gamma_rad=gamma_rad,
                        gamma_stark=gamma_stark,
                    )
                    if value <= 0.0 or value < continuum_row[idx] * cutoff:
                        blue_active = False
                    else:
                        tmp_buffer[idx] += value
            offset += 1
        tmp_buffer[center_index] = 0.0
        metal_wings_row += tmp_buffer
        metal_sources_row += tmp_buffer * bnu_row
        return True

    if wing_type in {
        fort19_io.Fort19WingType.HYDROGEN,
        fort19_io.Fort19WingType.DEUTERIUM,
    }:
        tmp_buffer.fill(0.0)
        _accumulate_hydrogen_profile(
            buffer=tmp_buffer,
            continuum_row=continuum_row,
            stim_row=None,
            wavelength_grid=wavelength_grid,
            center_index=center_index,
            line_wavelength=line_wavelength,
            kappa0=kappa0,
            depth_state=depth_state,
            n_lower=max(n_lower, 1),
            n_upper=max(n_upper, n_lower + 1),
            wcon=line_wavelength,
            wtail=line_wavelength,
            wlminus1=line_wavelength,
            wlminus2=line_wavelength,
            wlplus1=line_wavelength,
            wlplus2=line_wavelength,
            redcut=line_wavelength,
            bluecut=line_wavelength,
            cutoff=cutoff,
        )
        tmp_buffer[center_index] = 0.0
        metal_wings_row += tmp_buffer
        metal_sources_row += tmp_buffer * bnu_row
        return True

    return False


def _add_fort19_asynth(
    asynth: np.ndarray,
    stim: np.ndarray,
    wavelength: np.ndarray,
    continuum: np.ndarray,
    contx: np.ndarray,
    emerge: np.ndarray,
    emerge_h: np.ndarray,
    catalog: atomic.LineCatalog,
    fort19_data: fort19_io.Fort19Data,
    catalog_to_fort19: Dict[int, int],
    pops: populations.Populations,
    atm: atmosphere.AtmosphereModel,
    cutoff: float,
) -> None:
    """Add fort.19 special profiles into ASYNTH (Fortran XLINOP N19 behavior)."""
    if fort19_data is None or len(fort19_data.wavelength_vacuum) == 0:
        return
    if atm.population_per_ion is None:
        return

    # Build inverse map: fort19 index -> catalog index
    fort19_to_catalog = {v: k for k, v in catalog_to_fort19.items()}
    fort19_indices = np.arange(len(fort19_data.wavelength_vacuum), dtype=np.int32)
    metal_tables = tables.metal_wing_tables()

    # Precompute fort19 center indices on the current wavelength grid
    fort19_centers = _nearest_grid_indices(wavelength, fort19_data.wavelength_vacuum)

    for depth_idx in range(atm.layers):
        tmp_buffer = np.zeros_like(wavelength, dtype=np.float64)
        for fidx in fort19_indices:
            wing_val = fort19_data.wing_type[fidx]
            if isinstance(wing_val, fort19_io.Fort19WingType):
                wing_type = wing_val
            else:
                wing_type = fort19_io.Fort19WingType.from_code(int(wing_val))

            # Handle fort.19 line families that are not present in fort.12:
            # - NORMAL lines with NBLO/NBUP metadata (e.g. Mg I 457.6574)
            # - AUTOIONIZING and CONTINUUM records
            if wing_type not in {
                fort19_io.Fort19WingType.NORMAL,
                fort19_io.Fort19WingType.AUTOIONIZING,
                fort19_io.Fort19WingType.CONTINUUM,
            }:
                continue

            cat_idx = fort19_to_catalog.get(int(fidx))
            if cat_idx is None:
                continue
            record = catalog.records[cat_idx]
            element_idx = _element_atomic_number(str(record.element))
            if element_idx is None:
                continue
            element_idx -= 1
            ion_stage = int(record.ion_stage)
            if ion_stage <= 0:
                continue
            pop_val = atm.population_per_ion[depth_idx, ion_stage - 1, element_idx]
            if pop_val <= 0.0:
                continue
            boltz = pops.layers[depth_idx].boltzmann_factor[cat_idx]
            line_wavelength = float(fort19_data.wavelength_vacuum[fidx])
            center_index = int(fort19_centers[fidx])
            # rgfall.for filters lines outside WLBEG/WLEND before writing fort.12/fort.19,
            # but merged-continuum records can still contribute when the line center is
            # below the grid start (Fortran clamps NBUFF1 to 1). Allow center_index < 0
            # for CONTINUUM lines, but skip if the center is above the grid.
            if center_index >= wavelength.size:
                continue

            if wing_type == fort19_io.Fort19WingType.NORMAL:
                rho = (
                    float(atm.mass_density[depth_idx])
                    if atm.mass_density is not None
                    else 0.0
                )
                if rho <= 0.0:
                    continue
                if cat_idx >= pops.layers[depth_idx].doppler_width.size:
                    continue
                doppler_width = float(pops.layers[depth_idx].doppler_width[cat_idx])
                if line_wavelength <= 0.0 or doppler_width <= 0.0:
                    continue
                dopple = doppler_width / line_wavelength
                if dopple <= 0.0:
                    continue
                xnfdop = pop_val / (rho * dopple)
                cgf = float(fort19_data.oscillator_strength[fidx])
                kappa0_pre = cgf * xnfdop

                clamped_center = max(0, min(center_index, wavelength.size - 1))
                kapmin = float(continuum[depth_idx, clamped_center]) * cutoff
                if kappa0_pre < kapmin:
                    continue

                kappa0 = kappa0_pre * boltz
                if kappa0 < kapmin:
                    continue

                depth_state = pops.layers[depth_idx]
                gamma_total = (
                    float(fort19_data.gamma_rad[fidx])
                    + float(fort19_data.gamma_stark[fidx])
                    * float(depth_state.electron_density)
                    + float(fort19_data.gamma_vdw[fidx]) * float(depth_state.txnxn)
                )
                adamp = gamma_total / dopple if dopple > 0.0 else 0.0
                if adamp < 0.2:
                    kapcen = kappa0 * (1.0 - 1.128 * adamp)
                else:
                    kapcen = kappa0 * voigt_profile(0.0, adamp)
                if kapcen >= kapmin:
                    tmp_buffer[clamped_center] += kapcen

                ncon = int(fort19_data.continuum_index[fidx])
                nelionx = int(fort19_data.element_index[fidx])
                nelion_f = int(fort19_data.ion_index[fidx])
                wcon = None
                wtail = None
                if ncon > 0 and nelionx > 0:
                    wcon, wtail = _compute_continuum_limits(
                        ncon=ncon,
                        nelion=nelion_f,
                        nelionx=nelionx,
                        emerge_val=float(emerge[depth_idx]),
                        emerge_h_val=float(emerge_h[depth_idx]),
                        metal_tables=metal_tables,
                        ifvac=1,
                    )

                _accumulate_metal_profile(
                    buffer=tmp_buffer,
                    continuum_row=continuum[depth_idx],
                    wavelength_grid=wavelength,
                    center_index=center_index,
                    line_wavelength=line_wavelength,
                    kappa0=kappa0,
                    damping=adamp,
                    doppler_width=doppler_width,
                    cutoff=cutoff,
                    wcon=wcon,
                    wtail=wtail,
                )
            elif wing_type == fort19_io.Fort19WingType.CONTINUUM:
                rho = (
                    float(atm.mass_density[depth_idx])
                    if atm.mass_density is not None
                    else 0.0
                )
                if rho <= 0.0:
                    continue
                # Fortran synthe.for merged-continuum block uses XNFPEL (per mass),
                # so convert number density to per-mass by dividing by rho.
                kappa = (
                    float(fort19_data.oscillator_strength[fidx])
                    * (pop_val / rho)
                    * boltz
                )
                nlast = int(fort19_data.line_type[fidx])
                wcon, wtail = _compute_merged_continuum_limits(
                    line_wavelength,
                    nlast,
                    float(emerge[depth_idx]),
                    float(emerge_h[depth_idx]),
                    int(fort19_data.ion_index[fidx]),
                    1,
                )
                _accumulate_merged_continuum(
                    buffer=tmp_buffer,
                    continuum_row=continuum[depth_idx],
                    wavelength_grid=wavelength,
                    center_index=center_index,
                    line_wavelength=line_wavelength,
                    kappa=kappa,
                    cutoff=cutoff,
                    merge_wavelength=wcon if wcon is not None else line_wavelength,
                    tail_wavelength=wtail if wtail is not None else line_wavelength,
                )
            elif wing_type == fort19_io.Fort19WingType.AUTOIONIZING:
                # Fortran XLINOP label 700: KAPPA0 = BSHORE * GF * XNFPEL * exp(-ELO*HCKT)
                rho = (
                    float(atm.mass_density[depth_idx])
                    if atm.mass_density is not None
                    else 0.0
                )
                if rho <= 0.0:
                    continue
                kappa0 = (
                    float(fort19_data.gamma_vdw[fidx])
                    * float(fort19_data.oscillator_strength[fidx])
                    * (pop_val / rho)
                    * boltz
                )
                _accumulate_autoionizing_profile(
                    buffer=tmp_buffer,
                    continuum_row=continuum[depth_idx],
                    wavelength_grid=wavelength,
                    center_index=center_index,
                    line_wavelength=line_wavelength,
                    kappa0=kappa0,
                    gamma_rad=float(fort19_data.gamma_rad[fidx]),
                    gamma_stark=float(fort19_data.gamma_stark[fidx]),
                    gamma_vdw=float(fort19_data.gamma_vdw[fidx]),
                    cutoff=cutoff,
                )

        if np.any(tmp_buffer > 0.0):
            asynth[depth_idx] += tmp_buffer * stim[depth_idx]


def _accumulate_hydrogen_profile(
    buffer: np.ndarray,
    continuum_row: np.ndarray,
    stim_row: Optional[np.ndarray],
    wavelength_grid: np.ndarray,
    center_index: int,
    line_wavelength: float,
    kappa0: float,
    depth_state: populations.DepthState,
    n_lower: int,
    n_upper: int,
    wcon: float,
    wtail: float,
    wlminus1: float,
    wlminus2: float,
    wlplus1: float,
    wlplus2: float,
    redcut: float,
    bluecut: float,
    cutoff: float,
) -> int:
    if depth_state.hydrogen is None:
        return 0

    n_points = buffer.size
    profile_eval_calls = 0
    # Fortran synthe.for: if NBUP == NBLO+1 (alpha) or NBUP == NBLO+2 (beta),
    # use the simpler wing accumulation path (labels 620/630) without WCON/WTAIL
    # or +/-2 line comparisons.
    simple_wings = n_upper <= n_lower + 2
    use_stim = stim_row is not None
    use_taper = (not simple_wings) and (wtail > wcon)
    upper_minus2 = max(n_upper - 2, n_lower + 1)
    upper_plus2 = n_upper + 2
    profile_fn = hydrogen_line_profile

    red_active = True
    blue_active = True
    offset = 1
    max_steps = max(center_index, n_points - center_index - 1)

    if 0 <= center_index < n_points:
        # Fortran skips hydrogen line contributions below WCON; if the line
        # center is below WCON, do not add the core at this wavelength.
        wave_center = wavelength_grid[center_index]
        if not simple_wings and wave_center < wcon:
            pass
        else:
            # Use the actual bin-center offset, not zero, because the nearest
            # wavelength bin generally does not land exactly on the line center.
            delta_center_nm = wave_center - line_wavelength
            profile_eval_calls += 1
            profile_center = kappa0 * profile_fn(n_lower, n_upper, depth_state, delta_center_nm)
            stim_center = stim_row[center_index] if use_stim else 1.0
            value_center = profile_center * stim_center
            if use_taper and wave_center < wtail:
                value_center *= (wave_center - wcon) / (wtail - wcon)
            if value_center >= continuum_row[center_index] * cutoff:
                buffer[center_index] += value_center
    else:
        # Line center is outside the grid: skip center, but still compute wings.
        if center_index >= n_points:
            red_active = False
            offset = max(1, center_index - (n_points - 1))
        else:
            blue_active = False
            offset = max(1, -center_index)

    while offset <= max_steps and (red_active or blue_active):
        if red_active:
            idx = center_index + offset
            if idx >= n_points:
                red_active = False
            else:
                wave = wavelength_grid[idx]
                if not simple_wings:
                    if wave > wlminus1:
                        red_active = False
                    elif wave < wcon:
                        # Fortran: IF(WAVE.LT.WCON) GO TO 611 (skip this step, continue)
                        pass
                    else:
                        delta_nm = wave - line_wavelength
                        stim_val = stim_row[idx] if use_stim else 1.0
                        profile_eval_calls += 1
                        value = kappa0 * profile_fn(n_lower, n_upper, depth_state, delta_nm) * stim_val
                        if use_taper and wave < wtail:
                            value *= (wave - wcon) / (wtail - wcon)
                        if wave > redcut:
                            delta_minus2 = wave - wlminus2
                            profile_eval_calls += 1
                            value_minus2 = (
                                kappa0
                                * profile_fn(n_lower, upper_minus2, depth_state, delta_minus2)
                                * stim_val
                            )
                            if use_taper and wave < wtail:
                                value_minus2 *= (wave - wcon) / (wtail - wcon)
                            if value_minus2 >= value:
                                red_active = False
                                value = 0.0
                        if value <= 0.0 or value < continuum_row[idx] * cutoff:
                            red_active = False
                        else:
                            buffer[idx] += value
                else:
                    delta_nm = wave - line_wavelength
                    stim_val = stim_row[idx] if use_stim else 1.0
                    profile_eval_calls += 1
                    value = (
                        kappa0
                        * profile_fn(n_lower, n_upper, depth_state, delta_nm)
                        * stim_val
                    )
                    if value <= 0.0 or value < continuum_row[idx] * cutoff:
                        red_active = False
                    else:
                        buffer[idx] += value
        if blue_active:
            idx = center_index - offset
            if idx < 0:
                blue_active = False
            else:
                wave = wavelength_grid[idx]
                if not simple_wings and (wave < wcon or wave < wlplus1):
                    blue_active = False
                else:
                    delta_nm = wave - line_wavelength
                    stim_val = stim_row[idx] if use_stim else 1.0
                    profile_eval_calls += 1
                    value = (
                        kappa0
                        * profile_fn(n_lower, n_upper, depth_state, delta_nm)
                        * stim_val
                    )
                    if not simple_wings:
                        if use_taper and wave < wtail:
                            value *= (wave - wcon) / (wtail - wcon)
                        if wave < bluecut:
                            delta_plus2 = wave - wlplus2
                            profile_eval_calls += 1
                            value_plus2 = (
                                kappa0
                                * profile_fn(n_lower, upper_plus2, depth_state, delta_plus2)
                                * stim_val
                            )
                            if use_taper and wave < wtail:
                                value_plus2 *= (wave - wcon) / (wtail - wcon)
                            if value_plus2 >= value:
                                blue_active = False
                                value = 0.0
                    if value <= 0.0 or value < continuum_row[idx] * cutoff:
                        blue_active = False
                    else:
                        buffer[idx] += value
        offset += 1
    return profile_eval_calls


def _compute_hydrogen_line_opacity(
    catalog: atomic.LineCatalog,
    pops: populations.Populations,
    atmosphere_model: atmosphere.AtmosphereModel,
    wavelength_grid: np.ndarray,
    continuum: np.ndarray,
    stim: np.ndarray,
    cutoff: float,
    microturb_kms: float = 0.0,
) -> np.ndarray:
    """Compute hydrogen line opacity using the HPROF4-style profile."""
    n_depths = atmosphere_model.layers
    n_wavelengths = wavelength_grid.size
    ahline = np.zeros((n_depths, n_wavelengths), dtype=np.float64)

    if (
        atmosphere_model.population_per_ion is None
        or atmosphere_model.doppler_per_ion is None
        or atmosphere_model.mass_density is None
    ):
        logging.getLogger(__name__).warning(
            "Missing population_per_ion/doppler_per_ion/mass_density; skipping hydrogen line opacity."
        )
        return ahline

    h_atomic_number = _element_atomic_number("H")
    if h_atomic_number is None:
        return ahline
    h_index = h_atomic_number - 1
    if h_index >= atmosphere_model.population_per_ion.shape[2]:
        return ahline

    pop_densities = atmosphere_model.population_per_ion[:, :, h_index]
    dop_velocity = atmosphere_model.doppler_per_ion[:, :, h_index]
    mass_density = atmosphere_model.mass_density
    layers = pops.layers
    use_micro = microturb_kms > 0.0
    micro_dop = (microturb_kms / C_LIGHT_KM) if use_micro else 0.0
    index_wavelength = (
        catalog.index_wavelength
        if hasattr(catalog, "index_wavelength")
        else catalog.wavelength
    )
    center_indices = _nearest_grid_indices(wavelength_grid, index_wavelength)

    conth = tables.metal_wing_tables().conth
    electron_density = np.maximum(atmosphere_model.electron_density, 1e-40)
    inglis = 1600.0 / np.power(electron_density, 2.0 / 15.0)
    # Fortran synthe.for: NMERGE = INGLIS - 1.5, EMERGEH = 109677.576 / NMERGE^2
    nmerge = np.maximum(inglis - 1.5, 1.0)
    emerge_h = _HYD_RYD_CM / np.maximum(nmerge * nmerge, 1e-12)

    def _ehyd_cm(n: int) -> float:
        if n <= 0:
            return 0.0
        idx = n - 1
        if 0 <= idx < _EHYD_CM.size:
            return float(_EHYD_CM[idx])
        # Fortran synthe.for: EHYD(N)=109678.764 - 109677.576/N**2 for N>=9.
        return _HYD_EINF_CM - _HYD_RYD_CM / float(n * n)

    line_types = catalog.line_types if catalog.line_types is not None else None
    n_lower_arr = catalog.n_lower if catalog.n_lower is not None else None
    n_upper_arr = catalog.n_upper if catalog.n_upper is not None else None

    for line_idx, record in enumerate(catalog.records):
        line_type = (
            int(line_types[line_idx]) if line_types is not None else record.line_type
        )
        if line_type not in {-1, -2}:
            continue
        if record.ion_stage != 1:
            continue

        # Use Fortran-style NBUFF mapping so H lines outside the grid can still
        # contribute wings (synthe.for label 623 for WL>WLEND).
        line_wavelength = float(record.wavelength)
        center_idx = int(center_indices[line_idx])

        gf_linear = float(catalog.gf[line_idx])
        freq_hz = C_LIGHT_NM / line_wavelength
        cgf = None
        if record.metadata:
            cgf = record.metadata.get("cgf")
        if cgf is None or cgf <= 0.0:
            cgf = CGF_CONSTANT * gf_linear / freq_hz

        n_lower = (
            int(n_lower_arr[line_idx])
            if n_lower_arr is not None
            else max(record.n_lower, 1)
        )
        n_upper = (
            int(n_upper_arr[line_idx])
            if n_upper_arr is not None
            else max(record.n_upper, n_lower + 1)
        )
        ncon_idx = max(1, min(n_lower, conth.size)) - 1
        conth_val = float(conth[ncon_idx])
        n_lower_eff = max(n_lower, 1)
        n_upper_eff = max(n_upper, n_lower + 1)
        ehyd_lower = _ehyd_cm(n_lower)
        wlminus1 = (
            1.0e7 / (_ehyd_cm(n_upper - 1) - ehyd_lower)
            if n_upper - 1 > n_lower
            else line_wavelength
        )
        wlminus2 = (
            1.0e7 / (_ehyd_cm(n_upper - 2) - ehyd_lower)
            if n_upper - 2 > n_lower
            else line_wavelength
        )
        wlplus1 = 1.0e7 / (_ehyd_cm(n_upper + 1) - ehyd_lower)
        wlplus2 = 1.0e7 / (_ehyd_cm(n_upper + 2) - ehyd_lower)
        redcut = 1.0e7 / (
            conth[0] - _HYD_RYD_CM / (float(n_upper) - 0.8) ** 2 - ehyd_lower
        )
        bluecut = 1.0e7 / (
            conth[0] - _HYD_RYD_CM / (float(n_upper) + 0.8) ** 2 - ehyd_lower
        )
        clamped_center = max(0, min(center_idx, n_wavelengths - 1))
        continuum_center_col = continuum[:, clamped_center]
        wshift = 1.0e7 / (conth_val - _HYD_RYD_CM / 81.0**2)

        for depth_idx in range(n_depths):
            pop_val = pop_densities[depth_idx, 0]
            dop_val = dop_velocity[depth_idx, 0]
            if use_micro:
                dop_val = math.sqrt(dop_val * dop_val + micro_dop * micro_dop)
            rho = float(mass_density[depth_idx])
            if pop_val <= 0.0 or dop_val <= 0.0 or rho <= 0.0:
                continue

            xnfdop = pop_val / (rho * dop_val)
            depth_state = layers[depth_idx]
            boltz = depth_state.boltzmann_factor[line_idx]
            kappa0_pre = cgf * xnfdop

            kapmin = continuum_center_col[depth_idx] * cutoff
            if kappa0_pre < kapmin:
                continue

            kappa0 = kappa0_pre * boltz
            if kappa0 < kapmin:
                continue
            # 1e7/(cm^-1) yields nm directly (match Fortran WL units).
            wmerge = 1.0e7 / (conth_val - emerge_h[depth_idx])
            if wmerge < 0.0:
                wmerge = wshift + wshift
            wcon = max(wshift, wmerge)
            wtail = 1.0e7 / (1.0e7 / wcon - 500.0) if wcon > 0.0 else wcon + wcon
            wcon = min(wshift + wshift, wcon)
            if wtail < 0.0:
                wtail = wcon + wcon
            wtail = min(wcon + wcon, wtail)

            profile_calls = _accumulate_hydrogen_profile(
                buffer=ahline[depth_idx],
                continuum_row=continuum[depth_idx],
                stim_row=stim[depth_idx],
                wavelength_grid=wavelength_grid,
                center_index=center_idx,
                line_wavelength=line_wavelength,
                kappa0=kappa0,
                depth_state=depth_state,
                n_lower=n_lower_eff,
                n_upper=n_upper_eff,
                wcon=wcon,
                wtail=wtail,
                wlminus1=wlminus1,
                wlminus2=wlminus2,
                wlplus1=wlplus1,
                wlplus2=wlplus2,
                redcut=redcut,
                bluecut=bluecut,
                cutoff=cutoff,
            )

    return ahline


# Numba-compatible atomic mass lookup (element_idx -> atomic_mass)
@jit(nopython=True)
def _get_atomic_mass_jit(element_idx: int, atomic_masses: np.ndarray) -> float:
    """Get atomic mass for element index. Returns 0.0 if invalid."""
    if element_idx >= 0 and element_idx < atomic_masses.size:
        return atomic_masses[element_idx]
    return 0.0


@jit(
    nopython=True, parallel=True, cache=True
)  # Safe: parallelize across depth (independent output rows), preserving per-depth line order.
def _process_metal_wings_kernel(
    metal_wings: np.ndarray,  # Output: n_depths × n_wavelengths
    metal_sources: np.ndarray,  # Output: n_depths × n_wavelengths
    wavelength_grid: np.ndarray,  # n_wavelengths
    line_indices: np.ndarray,  # n_lines (center_index for each line)
    line_wavelengths: np.ndarray,  # n_lines
    line_cgf: np.ndarray,  # n_lines (precomputed CONGF = (0.026538/1.77245) * GF / (C/λ))
    line_gamma_rad: np.ndarray,  # n_lines (LINEAR from catalog, not log10)
    line_gamma_stark: np.ndarray,  # n_lines (LINEAR from catalog, not log10)
    line_gamma_vdw: np.ndarray,  # n_lines (LINEAR from catalog, not log10)
    line_element_idx: np.ndarray,  # n_lines (element index, -1 if invalid)
    line_nelion_eff: np.ndarray,  # n_lines (effective ion stage, metadata-resolved)
    line_ncon: np.ndarray,  # n_lines (metadata-resolved, 0 if none)
    line_nelionx: np.ndarray,  # n_lines (metadata-resolved, 0 if none)
    line_alpha: np.ndarray,  # n_lines (metadata-resolved, 0 if none)
    line_start_idx: np.ndarray,  # n_lines (precomputed window start)
    line_end_idx: np.ndarray,  # n_lines (precomputed window end; exclusive)
    line_center_local: np.ndarray,  # n_lines (precomputed center within window)
    pop_densities_all: np.ndarray,  # n_elements × n_depths × max_ion_stage
    dop_velocity_all: np.ndarray,  # n_elements × n_depths
    continuum: np.ndarray,  # n_depths × n_wavelengths
    bnu: np.ndarray,  # n_depths × n_wavelengths
    electron_density: np.ndarray,  # n_depths
    temperature: np.ndarray,  # n_depths
    mass_density: np.ndarray,  # n_depths
    emerge: np.ndarray,  # n_depths
    emerge_h: np.ndarray,  # n_depths
    xnf_h: np.ndarray,  # n_depths
    xnf_he1: np.ndarray,  # n_depths
    xnf_h2: np.ndarray,  # n_depths
    txnxn: np.ndarray,  # n_depths
    boltzmann_factor: np.ndarray,  # n_depths × n_lines
    contx: np.ndarray,  # metal_tables.contx
    atomic_masses: np.ndarray,  # n_elements
    ifvac: int,
    cutoff: float,
    h0tab: np.ndarray,
    h1tab: np.ndarray,
    h2tab: np.ndarray,
) -> None:
    """Depth-parallel kernel for processing metal line wings.

    Parallelizes across depth (each thread writes to independent rows).
    For each depth, line accumulation order is identical to the sequential implementation
    (line_idx increasing), preserving bitwise reproducibility.
    """
    n_lines = line_indices.size
    n_depths = metal_wings.shape[0]
    n_wavelengths = wavelength_grid.size
    n_elements = pop_densities_all.shape[0]
    max_ion_stage = pop_densities_all.shape[2]

    max_window = 2 * MAX_PROFILE_STEPS + 2

    for depth_idx in prange(n_depths):
        rho = mass_density[depth_idx]
        if rho <= 0.0:
            continue

        t_j = temperature[depth_idx]
        xne = electron_density[depth_idx]
        emerge_j = emerge[depth_idx]
        emerge_h_j = emerge_h[depth_idx]
        txnxn_base = txnxn[depth_idx]
        xnf_h_j = xnf_h[depth_idx]
        xnf_he1_j = xnf_he1[depth_idx]
        xnf_h2_j = xnf_h2[depth_idx]

        tmp_buffer_full = np.zeros(max_window, dtype=np.float64)

        for line_idx in range(n_lines):
            center_index = line_indices[line_idx]
            if center_index < 0 or center_index >= n_wavelengths:
                continue

            element_idx = line_element_idx[line_idx]
            if element_idx < 0 or element_idx >= n_elements:
                continue

            line_wavelength = line_wavelengths[line_idx]

            # Use pre-resolved per-line metadata arrays to avoid per-iteration branching.
            ncon = line_ncon[line_idx]
            nelionx = line_nelionx[line_idx]
            nelion = line_nelion_eff[line_idx]
            alpha = line_alpha[line_idx]

            if nelion <= 0 or nelion > max_ion_stage:
                continue

            pop_val = pop_densities_all[element_idx, depth_idx, nelion - 1]
            dop_val = dop_velocity_all[element_idx, depth_idx]
            if pop_val <= 0.0 or dop_val <= 0.0:
                continue

            xnfdop = pop_val / (rho * dop_val)
            doppler_width = dop_val * line_wavelength
            if doppler_width <= 0.0:
                continue

            boltz = boltzmann_factor[depth_idx, line_idx]
            cgf = line_cgf[line_idx]

            kappa_min = continuum[depth_idx, center_index] * cutoff
            kappa0_pre = cgf * xnfdop
            if kappa0_pre < kappa_min:
                continue

            kappa0 = kappa0_pre * boltz
            if kappa0 < kappa_min:
                continue

            gamma_rad = line_gamma_rad[line_idx]
            gamma_stark = line_gamma_stark[line_idx]
            gamma_vdw = line_gamma_vdw[line_idx]

            txnxn_line = txnxn_base
            if abs(alpha) > 1e-8:
                atomic_mass = _get_atomic_mass_jit(element_idx, atomic_masses)
                if atomic_mass > 0.0:
                    v2 = 0.5 * (1.0 - alpha)
                    h_factor = (t_j / 10000.0) ** v2
                    # Fortran synthe.for line 467-468: 1/4 and 1/2 are INTEGER
                    # division → both evaluate to 0, leaving only 1.008/ATMASS.
                    he_factor = 0.628 * (2.0991e-4 * t_j * (1.008 / atomic_mass)) ** v2
                    h2_factor = 1.08 * (2.0991e-4 * t_j * (1.008 / atomic_mass)) ** v2
                    txnxn_line = (
                        xnf_h_j * h_factor
                        + xnf_he1_j * he_factor
                        + xnf_h2_j * h2_factor
                    )

            wcon, wtail = _compute_continuum_limits_jit(
                ncon,
                nelion,
                nelionx,
                emerge_j,
                emerge_h_j,
                contx,
                ifvac,
            )

            # Fortran synthe.for line 473: ADAMP = (GAMRF+GAMSF*XNE+GAMWF*TXNXN)/DOPPLE
            # GAMRF etc. are pre-normalized by 4πν in rgfall.for.
            dopple = doppler_width / line_wavelength if line_wavelength > 0 else dop_val
            gamma_total = gamma_rad + gamma_stark * xne + gamma_vdw * txnxn_line
            damping_value = gamma_total / max(dopple, 1e-40)

            # Use precomputed window bounds to reduce integer overhead.
            start_idx = line_start_idx[line_idx]
            end_idx = line_end_idx[line_idx]
            window_len = end_idx - start_idx
            if window_len <= 0 or window_len > max_window:
                continue

            tmp_buffer = tmp_buffer_full[:window_len]
            tmp_buffer.fill(0.0)
            center_local = line_center_local[line_idx]

            _accumulate_metal_profile_kernel(
                tmp_buffer,
                continuum[depth_idx, start_idx:end_idx],
                wavelength_grid[start_idx:end_idx],
                center_local,
                line_wavelength,
                kappa0,
                max(damping_value, 1e-12),
                doppler_width,
                cutoff,
                wcon,
                wtail,
                h0tab,
                h1tab,
                h2tab,
            )

            if 0 <= center_local < window_len:
                tmp_buffer[center_local] = 0.0

            metal_wings[depth_idx, start_idx:end_idx] += tmp_buffer
            metal_sources[depth_idx, start_idx:end_idx] += (
                tmp_buffer * bnu[depth_idx, start_idx:end_idx]
            )


def run_synthesis(cfg: SynthesisConfig) -> SynthResult:
    """Execute the high-level synthesis pipeline."""

    logger = logging.getLogger(__name__)
    logger.info("Starting synthesis pipeline")
    logger.info(
        f"Wavelength range: {cfg.wavelength_grid.start:.2f} - {cfg.wavelength_grid.end:.2f} nm"
    )
    t_pipeline = time.perf_counter()
    _timings: Dict[str, float] = {}

    logger.info("Loading atmosphere model...")
    t_stage = time.perf_counter()
    atm = _load_atmosphere(cfg)
    logger.info(f"Loaded atmosphere: {atm.layers} layers")
    # Fortran uses fort.10 populations/doppler directly during synthesis.
    # Match that behavior: do not recompute populations in the synthesis stage.
    _timings["atmosphere load"] = time.perf_counter() - t_stage
    logger.info("Timing: atmosphere load in %.3fs", _timings["atmosphere load"])

    # Stage dump directory (set via SYNTHE_PY_STAGE_DUMPS env var)
    _stage_dump_dir = os.environ.get("SYNTHE_PY_STAGE_DUMPS")
    if _stage_dump_dir:
        _stage_dump_path = Path(_stage_dump_dir)
        _stage_dump_path.mkdir(parents=True, exist_ok=True)
        logger.info("Stage dumps enabled: writing to %s", _stage_dump_path)
    else:
        _stage_dump_path = None

    # --- Stage 1 dump: Atmosphere / populations ---
    if _stage_dump_path is not None:
        _s1 = {
            "temperature": atm.temperature,
            "electron_density": atm.electron_density,
            "depth": atm.depth,
        }
        if atm.mass_density is not None:
            _s1["mass_density"] = atm.mass_density
        if atm.gas_pressure is not None:
            _s1["gas_pressure"] = atm.gas_pressure
        if atm.population_per_ion is not None:
            _s1["population_per_ion"] = atm.population_per_ion
        if atm.doppler_per_ion is not None:
            _s1["doppler_per_ion"] = atm.doppler_per_ion
        if atm.xnf_h is not None:
            _s1["xnf_h"] = np.asarray(atm.xnf_h)
        if atm.xnf_he1 is not None:
            _s1["xnf_he1"] = np.asarray(atm.xnf_he1)
        if atm.xnf_h2 is not None:
            _s1["xnf_h2"] = np.asarray(atm.xnf_h2)
        np.savez(_stage_dump_path / "stage_1_populations.npz", **_s1)
        logger.info("Stage 1 dump (populations) saved")

    asynth_npz: Optional[np.lib.npyio.NpzFile] = None
    fort19_data: Optional[fort19_io.Fort19Data] = None
    # Build or load wavelength grid
    # Always build wavelength grid from configuration (no fort.29 dependency)
    logger.info("Building wavelength grid from configuration...")
    t_stage = time.perf_counter()
    wavelength_full = _build_wavelength_grid(cfg)
    original_wavelength_size = wavelength_full.size
    grid_origin = float(wavelength_full[0]) if wavelength_full.size > 0 else None

    # Apply subsampling
    if cfg.wavelength_subsample > 1:
        original_size = wavelength_full.size
        wavelength = wavelength_full[:: cfg.wavelength_subsample]
        logger.info(
            f"Subsampled wavelength grid: {original_size} -> {wavelength.size} points (every {cfg.wavelength_subsample} points)"
        )
    else:
        wavelength = wavelength_full

    # Wavelength mask is no longer needed (no fort.29 filtering)
    # All wavelength filtering is done directly on the array

    logger.info(f"Final wavelength grid: {wavelength.size} points")
    _timings["wavelength grid"] = time.perf_counter() - t_stage
    logger.info("Timing: wavelength grid in %.3fs", _timings["wavelength grid"])
    if cfg.wavelength_subsample > 1:
        logger.info(f"  Subsample active: every {cfg.wavelength_subsample} points")

    catalog_path = Path(cfg.line_data.atomic_catalog)
    logger.info(
        "Using self-contained Python line compiler metadata (no tfort runtime inputs)"
    )
    logger.info("Allocating buffers...")
    buffers = allocate_buffers(wavelength, atm.layers)

    logger.info("Loading line catalog...")
    t_stage = time.perf_counter()
    compiled_lines = line_compiler.compile_atomic_catalog(
        catalog_path=catalog_path,
        wlbeg=cfg.wavelength_grid.start,
        wlend=cfg.wavelength_grid.end,
        resolution=cfg.wavelength_grid.resolution,
        line_filter=cfg.line_filter,
        cache_directory=cfg.line_data.cache_directory,
    )
    catalog = compiled_lines.catalog
    fort19_data = compiled_lines.fort19_data
    logger.info(
        "Compiled line metadata from catalog (contract=%s, lines=%d, fort19=%d)",
        line_compiler.LINE_COMPILER_CONTRACT.nbuff_indexing,
        len(catalog.records),
        len(fort19_data.wavelength_vacuum),
    )
    has_lines = len(catalog.records) > 0
    logger.info(f"Catalog: {len(catalog.records)} lines")
    _timings["line catalog"] = time.perf_counter() - t_stage
    logger.info("Timing: line catalog in %.3fs", _timings["line catalog"])
    catalog_wavelength = catalog.wavelength
    catalog_to_fort19: Dict[int, int] = {}
    if fort19_data is not None:
        catalog_to_fort19 = _match_catalog_to_fort19(
            catalog_wavelength, fort19_data.wavelength_vacuum
        )
    line_indices = _nearest_grid_indices(wavelength, catalog_wavelength)
    logger.info("Computing depth-dependent populations...")
    t_stage = time.perf_counter()
    pops = populations.compute_depth_state(
        atm,
        catalog.wavelength,
        catalog.excitation_energy,
        cfg.wavelength_grid.velocity_microturb,
    )
    _timings["populations"] = time.perf_counter() - t_stage
    logger.info("Timing: populations in %.3fs", _timings["populations"])

    logger.info("Computing frequency-dependent quantities...")
    t_stage = time.perf_counter()
    freq = C_LIGHT_CM / (wavelength * NM_TO_CM)
    freq_grid = freq[np.newaxis, :]
    temp_grid = atm.temperature[:, np.newaxis]
    # CRITICAL FIX: Match Fortran exactly - no clamping of temperature or STIM
    # Fortran (atlas7v.for line 186-187): EHVKT(J)=EXP(-FREQ*HKT(J)), STIM(J)=1.-EHVKT(J)
    # Fortran does NOT clamp temperature or STIM - use values directly
    hkt_vec = H_PLANCK / (K_BOLTZ * atm.temperature)
    hkt_grid = hkt_vec[:, np.newaxis]
    with np.errstate(over="ignore"):
        ehvkt = np.exp(-freq_grid * hkt_grid)
    stim = 1.0 - ehvkt
    freq15 = freq_grid / 1.0e15
    bnu = 1.47439e-02 * freq15**3 * ehvkt / stim
    line_source = bnu.copy()
    _timings["frequency quantities"] = time.perf_counter() - t_stage
    logger.info(
        "Timing: frequency quantities in %.3fs", _timings["frequency quantities"]
    )

    logger.info("Computing continuum absorption/scattering...")
    t_stage = time.perf_counter()
    cont_abs, cont_scat, _, _ = continuum.build_depth_continuum(atm, wavelength)
    # KAPMIN should use total continuum opacity (ABTOT = ACONT + SIGMAC),
    # consistent with the continuum seen by the transport solver.
    cont_kapmin = cont_abs + cont_scat
    if not np.any(cont_kapmin):
        # If ABLOG is unavailable in the atmosphere file, approximate it from the
        # depth-0 continuum coefficients (matches fort.10 structure).
        if atm.continuum_abs_coeff is not None and atm.continuum_wledge is not None:
            ablog = np.asarray(atm.continuum_abs_coeff[0], dtype=np.float64).T
            cont_tables = tables.build_continuum_tables(
                tuple(float(x) for x in atm.continuum_wledge.tolist()),
                tuple(float(x) for x in ablog.ravel().tolist()),
            )
            log_cont = continuum.interpolate_continuum(cont_tables, wavelength)
            cont = continuum.finalize_continuum(log_cont)
            cont_kapmin = np.tile(cont, (atm.layers, 1))
        else:
            cont_kapmin = cont_abs + cont_scat
    _timings["continuum"] = time.perf_counter() - t_stage
    logger.info("Timing: continuum in %.3fs", _timings["continuum"])

    # --- Stage 2 dump: Continuum opacity ---
    if _stage_dump_path is not None:
        np.savez(
            _stage_dump_path / "stage_2_continuum.npz",
            wavelength=wavelength,
            cont_abs=cont_abs,
            cont_scat=cont_scat,
            cont_kapmin=cont_kapmin,
        )
        logger.info("Stage 2 dump (continuum) saved")

    cont_kapmin_full = None
    wavelength_full = None
    logger.info("Computing hydrogen continuum...")
    t_stage = time.perf_counter()
    ahyd_cont, shyd_cont = compute_hydrogen_continuum(
        atm,
        freq,
        bnu,
        ehvkt,
        stim,
        hkt_vec,
    )
    buffers.hydrogen_continuum[:] = ahyd_cont
    buffers.hydrogen_source[:] = shyd_cont
    if atm.cont_absorption is None:
        cont_abs += ahyd_cont
    # CRITICAL FIX: Fortran synthe.for uses ABTOT (ACONT + SIGMAC) for CONTINUUM
    # Line 195: READ(10)QABLOG (this is CONTINALL = LOG10(ABTOT))
    # Line 212-218: CONTINUUM = interpolation of ABLOG = ABTOT
    # Python must match: continuum_row = ACONT + SIGMAC = ABTOT
    buffers.continuum[:] = cont_abs + cont_scat  # Use ABTOT instead of ACONT only!
    _timings["hydrogen continuum"] = time.perf_counter() - t_stage
    logger.info("Timing: hydrogen continuum in %.3fs", _timings["hydrogen continuum"])

    spectrv_params = None

    logger.info("Computing line opacity from line catalog...")
    t_line_opacity = time.perf_counter()
    fscat_vec: np.ndarray = np.zeros(atm.layers, dtype=np.float64)

    # Compute line opacity from first principles (no Fortran file dependency)
    if has_lines:
        logger.info(
            "Computing TRANSP and ASYNTH from line catalog using Saha-Boltzmann populations"
        )

        # Compute populations for all depths
        t_stage = time.perf_counter()
        pops = populations.compute_depth_state(
            atm,
            catalog.wavelength,
            catalog.excitation_energy,
            cfg.wavelength_grid.velocity_microturb,
        )
        logger.info(
            "Timing: populations (line opacity) in %.3fs", time.perf_counter() - t_stage
        )

        # Compute TRANSP (line opacity at line center)
        # Fortran synthe.for line 266: KAPMIN=CONTINUUM(...)*CUTOFF
        # CONTINUUM comes from ABLOG in fort.10 and is not depth-specific.
        t_transp = time.perf_counter()
        transp, valid_mask, line_indices = line_opacity.compute_transp(
            catalog=catalog,
            populations=pops,
            atmosphere=atm,
            cutoff=cfg.cutoff,
            continuum_absorption=cont_kapmin,
            wavelength_grid=wavelength,
            continuum_absorption_full=cont_kapmin_full,
            wavelength_grid_full=wavelength_full,
            microturb_kms=cfg.wavelength_grid.velocity_microturb,
        )
        logger.info("Timing: TRANSP in %.3fs", time.perf_counter() - t_transp)

        # --- Stage 3 dump: TRANSP (line centers) ---
        if _stage_dump_path is not None:
            np.savez(
                _stage_dump_path / "stage_3_transp.npz",
                transp=transp,
                valid_mask=valid_mask,
                line_indices=line_indices,
            )
            logger.info("Stage 3 dump (TRANSP) saved")

        logger.info(f"Computed TRANSP for {np.sum(valid_mask)} line-depth pairs")

        # Compute ASYNTH from TRANSP (with wing contributions)
        # Pass continuum absorption for cutoff calculation (matches Fortran: KAPMIN = CONTINUUM * CUTOFF)
        # CRITICAL FIX: Pass metal_tables for WCON/WTAIL computation (matches Fortran lines 676-681, 703, 706, 722, 726)
        metal_tables = None
        if hasattr(atm, "metal_tables") and atm.metal_tables is not None:
            metal_tables = atm.metal_tables
        t_asynth = time.perf_counter()
        asynth = line_opacity.compute_asynth_from_transp(
            transp=transp,
            catalog=catalog,
            atmosphere=atm,
            wavelength_grid=wavelength,
            valid_mask=valid_mask,
            populations=pops,
            cutoff=cfg.cutoff,
            continuum_absorption=cont_kapmin,
            metal_tables=metal_tables,
            grid_origin=grid_origin,
        )
        logger.info("Timing: ASYNTH in %.3fs", time.perf_counter() - t_asynth)

        # Add fort.19 special profiles into ASYNTH (autoionizing + merged continuum).
        if fort19_data is not None and len(catalog_to_fort19) > 0:
            t_f19 = time.perf_counter()
            electron_density = np.maximum(atm.electron_density, 1e-40)
            inglis = 1600.0 / np.power(electron_density, 2.0 / 15.0)
            nmerge = np.maximum(inglis - 1.5, 1.0)
            emerge = 109737.312 / np.maximum(nmerge**2, 1e-12)
            emerge_h = 109677.576 / np.maximum(nmerge**2, 1e-12)
            metal_tables = tables.metal_wing_tables()
            _add_fort19_asynth(
                asynth=asynth,
                stim=stim,
                wavelength=wavelength,
                continuum=cont_kapmin,
                contx=metal_tables.contx,
                emerge=emerge,
                emerge_h=emerge_h,
                catalog=catalog,
                fort19_data=fort19_data,
                catalog_to_fort19=catalog_to_fort19,
                pops=pops,
                atm=atm,
                cutoff=cfg.cutoff,
            )
            logger.info("Timing: fort.19 add in %.3fs", time.perf_counter() - t_f19)

        logger.info(
            f"Computed ASYNTH: shape {asynth.shape}, range [{np.min(asynth):.2e}, {np.max(asynth):.2e}]"
        )

        # --- Stage 4 dump: ASYNTH (full line opacity with wings) ---
        if _stage_dump_path is not None:
            np.savez(
                _stage_dump_path / "stage_4_asynth.npz",
                asynth=asynth,
                stim=stim,
                bnu=bnu,
                wavelength=wavelength,
            )
            logger.info("Stage 4 dump (ASYNTH) saved")

        # Apply scattering factor
        rhox = atm.depth
        rhox_scale = cfg.rhoxj_scale
        if rhox_scale > 0.0:
            fscat = np.exp(-rhox / rhox_scale)
        else:
            fscat = np.zeros_like(rhox)
        fscat_vec = fscat

        # ASYNTH = line opacity including stimulated emission
        # ALINE = ASYNTH * (1 - FSCAT)  (absorption)
        # SIGMAL = ASYNTH * FSCAT  (scattering)
        absorption = asynth * (1.0 - fscat[:, None])
        scattering = asynth * fscat[:, None]

        # Mark that we're using ASYNTH (computed from catalog)
        buffers._using_asynth = True

        # Check for NaN and Inf values (can occur from division by zero or overflow)
        nan_mask = np.isnan(absorption)
        inf_mask = np.isinf(absorption)
        if np.any(nan_mask):
            n_nan = np.sum(nan_mask)
            print(f"  WARNING: {n_nan:,} NaN values found in absorption, setting to 0")
            absorption[nan_mask] = 0.0
        if np.any(inf_mask):
            n_inf = np.sum(inf_mask)
            print(
                f"  WARNING: {n_inf:,} Inf values found in absorption, clamping to MAX_OPACITY"
            )
            absorption[inf_mask] = MAX_OPACITY

        buffers.line_opacity[:] = absorption
        buffers.line_scattering[:] = scattering

        # Metal wings will be computed below (if enabled)
        metal_wings = np.zeros_like(buffers.line_opacity)
        metal_sources = np.zeros_like(buffers.line_opacity)
        helium_wings = np.zeros_like(buffers.line_opacity)
        helium_sources = np.zeros_like(buffers.line_opacity)
    else:
        # Continuum-only synthesis (no lines)
        logger.info("No atomic lines in catalog - using continuum-only synthesis")
        buffers.line_opacity[:] = 0.0
        buffers.line_scattering[:] = 0.0
        metal_wings = np.zeros_like(buffers.line_opacity)
        metal_sources = np.zeros_like(buffers.line_opacity)
        helium_wings = np.zeros_like(buffers.line_opacity)
        helium_sources = np.zeros_like(buffers.line_opacity)

    abs_core_base = buffers.line_opacity.copy()
    with np.errstate(divide="ignore"):
        alinec_total = abs_core_base / np.maximum(1.0 - fscat_vec[:, None], 1e-12)

    # Compute wings for hydrogen and metal lines
    # Wings are always computed from the line catalog (no Fortran file dependency)
    use_wings = has_lines  # Compute wings when we have lines

    if use_wings and not cfg.skip_hydrogen_wings:
        logger.info("Computing hydrogen line wings...")
        ahline = _compute_hydrogen_line_opacity(
            catalog=catalog,
            pops=pops,
            atmosphere_model=atm,
            wavelength_grid=wavelength,
            continuum=buffers.continuum,
            stim=stim,
            cutoff=cfg.cutoff,
            microturb_kms=cfg.wavelength_grid.velocity_microturb,
        )
        shline = np.zeros_like(ahline)
    else:
        if asynth_npz is not None:
            logger.info(
                "Skipping hydrogen wings (using fort.29 ASYNTH, which doesn't include wings)"
            )
        else:
            logger.info("Skipping hydrogen wings (--skip-hydrogen-wings)")
        ahline = np.zeros_like(buffers.line_opacity)
        shline = np.zeros_like(buffers.line_opacity)

    # Ensure hydrogen lines are represented in the line opacity used by radiative transfer.
    # Fortran XLINOP adds hydrogen profiles directly into ASYNTH before SPECTRV applies FSCAT.
    # When using ASYNTH, merge AHLINE into the ASYNTH absorption instead of adding separately.
    skip_ahline = os.getenv("PY_SKIP_AHLINE") == "1"
    using_asynth = bool(getattr(buffers, "_using_asynth", False))
    if has_lines and np.any(ahline > 0) and not skip_ahline and using_asynth:
        buffers.line_opacity += ahline * (1.0 - fscat_vec[:, None])
        abs_core_base = buffers.line_opacity.copy()
        with np.errstate(divide="ignore"):
            alinec_total = abs_core_base / np.maximum(1.0 - fscat_vec[:, None], 1e-12)
        ahline_for_total = np.zeros_like(ahline)
    elif has_lines and np.any(ahline > 0) and not skip_ahline and not using_asynth:
        buffers.line_opacity += ahline
        ahline_for_total = np.zeros_like(ahline)
    else:
        ahline_for_total = ahline

    # Metal wings (non-hydrogen) computed with XLINOP-style tapering.
    if use_wings:
        logger.info("Computing metal line wings...")
        metal_tables = tables.metal_wing_tables()
        # Fortran fort.9 metadata is disabled - use catalog values directly
        metadata = None
        catalog_to_meta: Dict[int, int] = {}
        logger.info("Using line properties from catalog (fort.9 disabled)")

        he_solver: Optional[helium_profiles.HeliumWingSolver] = None
        if cfg.enable_helium_wings:
            he_solver = helium_profiles.HeliumWingSolver(
                temperature=atm.temperature,
                electron_density=atm.electron_density,
                xnfph=atm.xnfph,
                xnf_he2=atm.xnf_he2,
            )
        use_numba_helium = (
            he_solver is not None
            and os.getenv("PY_NUMBA_HELIUM", "1") != "0"
        )
        if use_numba_helium and hasattr(he_solver, "_prepare_numba_cache"):
            logger.info("Preparing Numba helium wing tables...")
            start_time = time.time()
            he_solver._prepare_numba_cache()
            logger.info("Numba helium tables ready in %.2fs", time.time() - start_time)

        # Precompute helium line indices for a fast helium-only pass (Fortran-style inline logic)
        helium_line_ids = None
        if he_solver is not None:
            helium_line_ids_list = []
            for line_idx, record in enumerate(catalog.records):
                line_type_code = int(getattr(record, "line_type", 0) or 0)
                if fort19_data is not None:
                    line19_idx = catalog_to_fort19.get(line_idx)
                    if line19_idx is not None:
                        line_type_code = int(fort19_data.line_type[line19_idx])
                if line_type_code in (-3, -4, -6):
                    helium_line_ids_list.append(line_idx)
            helium_line_ids = np.asarray(helium_line_ids_list, dtype=np.int32)
            logger.info("Helium wing lines: %d", helium_line_ids.size)

        electron_density = np.maximum(atm.electron_density, 1e-40)
        inglis = 1600.0 / np.power(electron_density, 2.0 / 15.0)
        nmerge = np.maximum(inglis - 1.5, 1.0)
        emerge = 109737.312 / np.maximum(nmerge**2, 1e-12)
        emerge_h = 109677.576 / np.maximum(nmerge**2, 1e-12)

        xnf_h_arr = (
            np.asarray(atm.xnf_h, dtype=np.float64) if atm.xnf_h is not None else None
        )
        xnf_he1_arr = (
            np.asarray(atm.xnf_he1, dtype=np.float64)
            if atm.xnf_he1 is not None
            else None
        )
        xnf_h2_arr = (
            np.asarray(atm.xnf_h2, dtype=np.float64) if atm.xnf_h2 is not None else None
        )
        # Populations are now always computed from Saha-Boltzmann (no fort.10 dependency)
        from ..physics.populations_saha import (
            compute_population_densities,
            compute_doppler_velocity,
        )

        # Cache population computations per element
        population_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # Pre-compute population cache for all unique elements (shared across depths)
        logger.info("Pre-computing population densities for all elements...")
        unique_elements = set()
        for record in catalog.records:
            element_symbol = str(record.element).strip().upper()
            if element_symbol not in {"H", "H I", "HI"} or record.ion_stage != 1:
                unique_elements.add(record.element)

        logger.info(f"Found {len(unique_elements)} unique elements to process")
        for element in unique_elements:
            if element not in population_cache:
                logger.debug(f"Computing populations for element: {element}")
                pop_densities = compute_population_densities(
                    atm, element, max_ion_stage=6
                )
                dop_velocity = compute_doppler_velocity(atm, element)
                population_cache[element] = (pop_densities, dop_velocity)
        logger.info(f"Pre-computed populations for {len(population_cache)} elements")

        # Determine number of workers for parallelization
        n_workers_metal = cfg.n_workers
        if n_workers_metal is None:
            import multiprocessing

            n_workers_metal = max(1, multiprocessing.cpu_count())
            logger.info(
                f"Auto-detected {multiprocessing.cpu_count()} CPUs, using {n_workers_metal} workers"
            )
        else:
            logger.info(f"Using {n_workers_metal} workers (from config)")

        # Use Numba parallel for metal wings when we have enough layers
        use_numba_parallel = atm.layers >= 10
        use_parallel = use_numba_parallel
        if he_solver is not None:
            # Helium wings are computed in Python; keep metal wings parallel and
            # run helium wings in a separate sequential pass for exact Fortran behavior.
            if use_numba_parallel:
                logger.info(
                    "Helium wings will be computed sequentially; using Numba parallel kernel for metal wings."
                )

        step1_wing_hoist = os.getenv("PY_OPT_STEP1_WING_HOIST", "1") != "0"
        record_elements_upper = [str(r.element).strip().upper() for r in catalog.records]
        record_ion_stage = np.asarray([int(r.ion_stage) for r in catalog.records], dtype=np.int16)
        record_line_type = np.asarray(
            [int(getattr(r, "line_type", 0) or 0) for r in catalog.records],
            dtype=np.int16,
        )

        if use_numba_parallel:
            logger.info(
                f"Using Numba parallel processing for {len(line_indices)} lines across {atm.layers} depth layers"
            )
        else:
            logger.info(
                f"Using sequential processing ({atm.layers} layers, {n_workers_metal} workers)"
            )

        # Kernel is now defined at module level (compiles once)
        # Process lines in batches for progress logging

        def process_depth(
            depth_idx: int,
            include_metals: bool = True,
            include_helium: bool = True,
        ) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
            """Process metal line wings for a single depth layer."""
            depth_start_perf = time.perf_counter()
            state = pops.layers[depth_idx]
            continuum_row = buffers.continuum[depth_idx]
            helium_only = include_helium and (not include_metals)
            tmp_buffer = np.zeros_like(wavelength, dtype=np.float64)
            if include_metals:
                local_wings = np.zeros_like(wavelength, dtype=np.float64)
                local_sources = np.zeros_like(wavelength, dtype=np.float64)
            else:
                # Helium-only pass does not read metal outputs; avoid 2 large allocations/depth.
                local_wings = _EMPTY_FLOAT64
                local_sources = _EMPTY_FLOAT64
            local_helium_wings = np.zeros_like(wavelength, dtype=np.float64)
            local_helium_sources = np.zeros_like(wavelength, dtype=np.float64)
            helium_pass_line_iter = 0
            helium_pass_is_helium = 0
            helium_pass_nonhelium_skips = 0
            helium_pass_with_fort19 = 0
            helium_profile_calls = 0
            helium_profile_consumed = 0
            helium_profile_time_ms = 0.0
            helium_fallback_calls = 0
            helium_fallback_time_ms = 0.0
            helium_tmp_fill_time_ms = 0.0
            helium_add_time_ms = 0.0

            lines_processed = 0
            lines_skipped = 0

            if include_metals and include_helium:
                line_iter = range(len(line_indices))
            elif helium_only:
                if helium_line_ids is None or helium_line_ids.size == 0:
                    return (
                        depth_idx,
                        local_wings,
                        local_sources,
                        local_helium_wings,
                        local_helium_sources,
                        lines_processed,
                        lines_skipped,
                    )
                line_iter = helium_line_ids
            else:
                line_iter = range(len(line_indices))
            if helium_only:
                helium_pass_line_iter = int(len(line_iter))

            for line_idx in line_iter:
                center_index = line_indices[int(line_idx)]
                # CRITICAL FIX: Match Fortran synthe.for line 301 EXACTLY
                # IF(NBUFF.LT.1.OR.NBUFF.GT.LENGTH)GO TO 320
                # Fortran SKIPS lines whose centers are outside the grid
                center_outside = center_index < 0 or center_index >= wavelength.size
                if center_outside:
                    lines_skipped += 1
                    continue
                # Get metadata index if fort.9 is available
                meta_idx = (
                    catalog_to_meta.get(line_idx) if metadata is not None else None
                )
                # If fort.9 metadata exists but line not found, still allow it (use catalog values)
                # This allows mixing of lines with and without fort.9 metadata
                record = catalog.records[line_idx]
                element_symbol = record_elements_upper[line_idx]
                line_wavelength = float(catalog.wavelength[line_idx])
                if element_symbol in {"H", "H I", "HI"} and record_ion_stage[line_idx] == 1:
                    lines_skipped += 1
                    continue

                line19_idx = None
                line_type_code = int(record_line_type[line_idx])
                wing_type = fort19_io.Fort19WingType.NORMAL
                if fort19_data is not None:
                    line19_idx = catalog_to_fort19.get(line_idx)
                    if line19_idx is not None:
                        line_type_code = int(fort19_data.line_type[line19_idx])
                        wing_val = fort19_data.wing_type[line19_idx]
                        if isinstance(wing_val, fort19_io.Fort19WingType):
                            wing_type = wing_val
                        else:
                            wing_type = fort19_io.Fort19WingType.from_code(
                                int(wing_val)
                            )
                is_helium = he_solver is not None and line_type_code in (-3, -4, -6)
                # In helium-only TYPE<-2 path, _apply_fort19_profile zeroes tmp_buffer
                # itself; skip duplicate pre-clear in caller.
                precleared_tmp = not (helium_only and line_type_code < -2)
                if precleared_tmp:
                    if helium_only:
                        _fill_t0 = time.perf_counter()
                        tmp_buffer.fill(0.0)
                        helium_tmp_fill_time_ms += (
                            time.perf_counter() - _fill_t0
                        ) * 1000.0
                    else:
                        tmp_buffer.fill(0.0)
                if helium_only:
                    if line19_idx is not None:
                        helium_pass_with_fort19 += 1
                    if is_helium:
                        helium_pass_is_helium += 1
                if is_helium and not include_helium:
                    lines_skipped += 1
                    continue
                if (not is_helium) and not include_metals:
                    if helium_only:
                        helium_pass_nonhelium_skips += 1
                    lines_skipped += 1
                    continue
                wings_target = local_helium_wings if is_helium else local_wings
                sources_target = local_helium_sources if is_helium else local_sources

                # ========== CORONAL LINE SKIP (Fortran line 793) ==========
                # TYPE = 2 (Coronal lines): Skip entirely - GO TO 900
                if line_type_code == 2:
                    lines_skipped += 1
                    continue

                # Get metadata from fort.9 if available, otherwise use catalog/default values
                ncon = (
                    metadata.ncon[meta_idx]
                    if (metadata is not None and meta_idx is not None)
                    else 0
                )
                nelionx = (
                    metadata.nelionx[meta_idx]
                    if (metadata is not None and meta_idx is not None)
                    else 0
                )
                nelion = (
                    metadata.nelion[meta_idx]
                    if (metadata is not None and meta_idx is not None)
                    else record.ion_stage
                )
                if fort19_data is not None and line19_idx is not None:
                    ncon = int(fort19_data.continuum_index[line19_idx])
                    nelionx = int(fort19_data.element_index[line19_idx])
                alpha = (
                    metadata.extra1[meta_idx]
                    if (metadata is not None and meta_idx is not None)
                    else 0.0
                )
                txnxn_line = state.txnxn

                # Compute TXNXN with alpha correction if needed
                if np.isfinite(alpha) and abs(alpha) > 1e-8:
                    atomic_mass = _atomic_mass_lookup(record.element)
                    if atomic_mass is not None and atomic_mass > 0.0:
                        t_j = atm.temperature[depth_idx]
                        v2 = 0.5 * (1.0 - alpha)
                        h_factor = (t_j / 10000.0) ** v2
                        # Fortran synthe.for line 467-468: 1/4 and 1/2 are INTEGER
                        # division → both evaluate to 0, leaving only 1.008/ATMASS.
                        he_factor = (
                            0.628 * (2.0991e-4 * t_j * (1.008 / atomic_mass)) ** v2
                        )
                        h2_factor = (
                            1.08 * (2.0991e-4 * t_j * (1.008 / atomic_mass)) ** v2
                        )
                        xnfh_val = _layer_value(xnf_h_arr, depth_idx)
                        xnfhe_val = _layer_value(xnf_he1_arr, depth_idx)
                        xnfh2_val = _layer_value(xnf_h2_arr, depth_idx)
                        txnxn_line = (
                            xnfh_val * h_factor
                            + xnfhe_val * he_factor
                            + xnfh2_val * h2_factor
                        )

                element = record.element
                # Normalize element symbol for cache lookup (same normalization as cache key)
                element_key = str(element).strip()

                use_atm_pop = (
                    atm.population_per_ion is not None
                    and atm.doppler_per_ion is not None
                )
                element_idx = _element_atomic_number(element_key)
                if (
                    use_atm_pop
                    and element_idx is not None
                    and element_idx - 1 < atm.population_per_ion.shape[2]
                ):
                    pop_densities = atm.population_per_ion[:, :, element_idx - 1]
                    dop_velocity = atm.doppler_per_ion[:, :, element_idx - 1]
                else:
                    # Get populations from cache (pre-computed Saha) if no atmosphere values.
                    if element_key not in population_cache:
                        lines_skipped += 1
                        continue
                    pop_densities, dop_velocity = population_cache[element_key]

                # Get population and Doppler for this ion stage
                if nelion > pop_densities.shape[1]:
                    lines_skipped += 1
                    continue  # Ion stage out of range

                pop_val = pop_densities[depth_idx, nelion - 1]
                if dop_velocity.ndim == 2:
                    dop_val = dop_velocity[depth_idx, nelion - 1]
                else:
                    dop_val = dop_velocity[depth_idx]

                if pop_val <= 0.0 or dop_val <= 0.0:
                    lines_skipped += 1
                    continue  # Invalid population or Doppler

                # XNFDOP = XNFPEL / DOPPLE
                # From Fortran synthe.for line 240: QXNFDOP = QXNFPEL / (QRHO * QDOPPLE)
                # Where:
                #   - QXNFPEL from fort.10 is population per unit mass (cm³/g)
                #   - QRHO is mass density (g/cm³)
                #   - QDOPPLE is Doppler velocity (dimensionless, in units of c)
                #
                # CRITICAL FIX: pop_val from populations_saha.compute_population_densities
                # returns population density (cm⁻³), NOT per unit mass!
                # We need to convert to per unit mass: pop_per_mass = pop_density / rho
                # Then: xnfdop = pop_per_mass / dop_val = pop_density / (rho * dop_val)
                #
                # This matches Fortran: QXNFDOP = QXNFPEL / QRHO / QDOPPLE
                rho = (
                    atm.mass_density[depth_idx]
                    if hasattr(atm, "mass_density") and atm.mass_density is not None
                    else 1.0
                )
                if rho > 0.0:
                    xnfdop = pop_val / (rho * dop_val)
                    # Compute Doppler width
                    doppler_width = dop_val * line_wavelength
                    doppler_override = doppler_width

                    # Get Boltzmann factor
                    boltz = state.boltzmann_factor[line_idx]

                    # CRITICAL FIX: Convert GF to CONGF by dividing by frequency
                    # From rgfall.for line 267: CGF = 0.026538/1.77245 * GF / FRELIN
                    # Where FRELIN = 2.99792458D17 / WLVAC (frequency in Hz)
                    freq_hz = C_LIGHT_NM / line_wavelength  # Frequency in Hz
                    gf_linear = catalog.gf[line_idx]  # Linear gf
                    cgf = CGF_CONSTANT * gf_linear / freq_hz  # CONGF conversion

                    # ========== DOUBLE KAPMIN CHECK (Fortran lines 266-272) ==========
                    # Clamp center_index for continuum access
                    clamped_idx = max(0, min(center_index, wavelength.size - 1))
                    kappa_min = continuum_row[clamped_idx] * cfg.cutoff

                    # First: KAPPA0 = CONGF * XNFDOP (BEFORE Boltzmann)
                    kappa0_pre = cgf * xnfdop

                    # First check (Fortran line 267)
                    if kappa0_pre < kappa_min or doppler_width <= 0.0:
                        lines_skipped += 1
                        continue

                    # Apply Boltzmann factor
                    kappa0 = kappa0_pre * boltz

                    # Second check (Fortran line 272): post-Boltzmann cutoff
                    # RE-ENABLED: This matches Fortran behavior
                    if kappa0 < kappa_min:
                        lines_skipped += 1
                        continue

                    population_lower = (
                        pop_val  # Store for potential use in special wing types
                    )
                else:
                    # Invalid mass density - skip this line/depth
                    lines_skipped += 1
                    continue

                # Compute continuum limits
                wcon, wtail = _compute_continuum_limits(
                    ncon=ncon,
                    nelion=nelion,
                    nelionx=nelionx,
                    emerge_val=emerge[depth_idx],
                    emerge_h_val=emerge_h[depth_idx],
                    metal_tables=metal_tables,
                    ifvac=1,
                )

                # ALWAYS use catalog gamma values (LINEAR, not normalized)
                # Catalog stores 10^GR, 10^GS, 10^GW from gfallvac.latest
                # NO fort.9 metadata dependency for gamma values!
                gamma_rad = catalog.gamma_rad[line_idx]
                gamma_stark = catalog.gamma_stark[line_idx]
                gamma_vdw = catalog.gamma_vdw[line_idx]

                line_doppler = doppler_width

                # Handle special wing types
                if (
                    wing_type == fort19_io.Fort19WingType.AUTOIONIZING
                    and fort19_data is not None
                    and not center_outside
                ):
                    population = (
                        population_lower
                        if (population_lower is not None and population_lower > 0.0)
                        else None
                    )
                    if population is None:
                        population = boltz
                    gf_value = catalog.gf[line_idx]
                    kappa_auto = gamma_vdw * gf_value * population * boltz
                    if kappa_auto < kappa_min:
                        lines_skipped += 1
                        continue
                    n_lower_val = (
                        int(metadata.nblo[meta_idx])
                        if (metadata is not None and meta_idx is not None)
                        else (
                            int(fort19_data.n_lower[line19_idx])
                            if (fort19_data is not None and line19_idx is not None)
                            else 1
                        )
                    )
                    n_upper_val = (
                        int(metadata.nbup[meta_idx])
                        if (metadata is not None and meta_idx is not None)
                        else (
                            int(fort19_data.n_upper[line19_idx])
                            if (fort19_data is not None and line19_idx is not None)
                            else 2
                        )
                    )
                    _profile_t0 = time.perf_counter()
                    _auto_consumed = _apply_fort19_profile(
                        wing_type=wing_type,
                        line_type_code=line_type_code,
                        tmp_buffer=tmp_buffer,
                        continuum_row=continuum_row,
                        wavelength_grid=wavelength,
                        center_index=center_index,
                        line_wavelength=line_wavelength,
                        kappa0=kappa_auto,
                        cutoff=cfg.cutoff,
                        metal_wings_row=wings_target,
                        metal_sources_row=sources_target,
                        bnu_row=bnu[depth_idx],
                        wcon=wcon,
                        wtail=wtail,
                        he_solver=he_solver,
                        use_numba_helium=use_numba_helium,
                        depth_idx=depth_idx,
                        depth_state=state,
                        n_lower=n_lower_val,
                        n_upper=n_upper_val,
                        gamma_rad=gamma_rad,
                        gamma_stark=gamma_stark,
                        gamma_vdw=gamma_vdw,
                        doppler_width=line_doppler,
                        line_index=int(line_idx),
                    )
                    if include_helium and not include_metals:
                        helium_profile_calls += 1
                        helium_profile_time_ms += (
                            time.perf_counter() - _profile_t0
                        ) * 1000.0
                        if _auto_consumed:
                            helium_profile_consumed += 1
                    if _auto_consumed:
                        lines_processed += 1
                        continue

                # Compute damping value (ADAMP in Fortran synthe.for line 473)
                # Fortran: ADAMP = (GAMRF + GAMSF*XNE + GAMWF*TXNXN) / DOPPLE(NELION)
                # GAMRF etc. are pre-normalized by 4πν in rgfall.for.
                dopple = (
                    doppler_width / line_wavelength if line_wavelength > 0 else dop_val
                )
                gamma_total = (
                    gamma_rad
                    + gamma_stark * state.electron_density
                    + gamma_vdw * txnxn_line
                )
                damping_value = gamma_total / max(dopple, 1e-40)

                # Apply fort.19 profile if available
                profile_consumed = False
                if not center_outside:
                    _profile_t0 = time.perf_counter()
                    profile_consumed = _apply_fort19_profile(
                        wing_type=wing_type,
                        line_type_code=line_type_code,
                        tmp_buffer=tmp_buffer,
                        continuum_row=continuum_row,
                        wavelength_grid=wavelength,
                        center_index=center_index,
                        line_wavelength=line_wavelength,
                        kappa0=kappa0,
                        cutoff=cfg.cutoff,
                        metal_wings_row=wings_target,
                        metal_sources_row=sources_target,
                        bnu_row=bnu[depth_idx],
                        wcon=wcon,
                        wtail=wtail,
                        he_solver=he_solver,
                        use_numba_helium=use_numba_helium,
                        depth_idx=depth_idx,
                        depth_state=state,
                        n_lower=(
                            int(metadata.nblo[meta_idx])
                            if (metadata is not None and meta_idx is not None)
                            else 1
                        ),
                        n_upper=(
                            int(metadata.nbup[meta_idx])
                            if (metadata is not None and meta_idx is not None)
                            else 2
                        ),
                        gamma_rad=gamma_rad,
                        gamma_stark=gamma_stark,
                        gamma_vdw=gamma_vdw,
                        doppler_width=line_doppler,
                        line_index=int(line_idx),
                    )
                    if helium_only:
                        helium_profile_calls += 1
                        helium_profile_time_ms += (
                            time.perf_counter() - _profile_t0
                        ) * 1000.0
                        if profile_consumed:
                            helium_profile_consumed += 1
                if profile_consumed:
                    lines_processed += 1
                    continue

                # Accumulate metal profile
                if not precleared_tmp:
                    if helium_only:
                        _fill_t0 = time.perf_counter()
                        tmp_buffer.fill(0.0)
                        helium_tmp_fill_time_ms += (
                            time.perf_counter() - _fill_t0
                        ) * 1000.0
                    else:
                        tmp_buffer.fill(0.0)
                    precleared_tmp = True
                _fallback_t0 = time.perf_counter()
                _accumulate_metal_profile(
                    buffer=tmp_buffer,
                    continuum_row=continuum_row,
                    wavelength_grid=wavelength,
                    center_index=center_index,
                    line_wavelength=line_wavelength,
                    kappa0=kappa0,
                    damping=max(damping_value, 1e-12),
                    doppler_width=line_doppler,
                    cutoff=cfg.cutoff,
                    wcon=wcon,
                    wtail=wtail,
                )
                if helium_only:
                    helium_fallback_calls += 1
                    helium_fallback_time_ms += (
                        time.perf_counter() - _fallback_t0
                    ) * 1000.0

                # Reset center (already handled by _accumulate_metal_profile, but ensure it's zero)
                # Only reset if center_index is within grid
                if 0 <= center_index < wavelength.size:
                    tmp_buffer[center_index] = 0.0

                # Accumulate into local buffers
                if helium_only:
                    _add_t0 = time.perf_counter()
                    local_wings += tmp_buffer
                    local_sources += tmp_buffer * bnu[depth_idx]
                    helium_add_time_ms += (time.perf_counter() - _add_t0) * 1000.0
                else:
                    local_wings += tmp_buffer
                    local_sources += tmp_buffer * bnu[depth_idx]
                lines_processed += 1

            # Return results (no locks needed - each depth writes to different indices)
            return (
                depth_idx,
                local_wings,
                local_sources,
                local_helium_wings,
                local_helium_sources,
                lines_processed,
                lines_skipped,
            )

        if use_numba_parallel:
            # Numba parallel processing (no pickling overhead)
            start_time = time.time()
            logger.info("Pre-processing data structures for Numba kernel...")

            # Build element-to-index mapping
            unique_elements_list = sorted(population_cache.keys())
            element_to_idx: Dict[str, int] = {
                elem: idx for idx, elem in enumerate(unique_elements_list)
            }
            n_elements = len(unique_elements_list)

            # Find max ion stage across all elements
            max_ion_stage = 0
            for pop_densities, _ in population_cache.values():
                if pop_densities.shape[1] > max_ion_stage:
                    max_ion_stage = pop_densities.shape[1]

            # Build population arrays: n_elements × n_depths × max_ion_stage
            pop_densities_all = np.zeros(
                (n_elements, atm.layers, max_ion_stage), dtype=np.float64
            )
            dop_velocity_all = np.zeros((n_elements, atm.layers), dtype=np.float64)
            for elem, (pop_densities, dop_velocity) in population_cache.items():
                elem_idx = element_to_idx[elem]
                n_depths_elem, n_ion_stages = pop_densities.shape
                pop_densities_all[elem_idx, :n_depths_elem, :n_ion_stages] = (
                    pop_densities
                )
                dop_velocity_all[elem_idx, :n_depths_elem] = dop_velocity

            # Build atomic masses array
            atomic_masses = np.zeros(n_elements, dtype=np.float64)
            for elem, elem_idx in element_to_idx.items():
                atomic_mass = _atomic_mass_lookup(elem)
                if atomic_mass is not None:
                    atomic_masses[elem_idx] = atomic_mass

            # Pre-process line data into arrays
            n_lines = len(line_indices)
            line_wavelengths = np.asarray(catalog.wavelength, dtype=np.float64)
            line_gf = np.asarray(catalog.gf, dtype=np.float64)
            line_gamma_rad = np.asarray(catalog.gamma_rad, dtype=np.float64)
            line_gamma_stark = np.asarray(catalog.gamma_stark, dtype=np.float64)
            line_gamma_vdw = np.asarray(catalog.gamma_vdw, dtype=np.float64)
            line_nelion = np.zeros(n_lines, dtype=np.int32)
            line_element_idx = np.full(n_lines, -1, dtype=np.int32)

            # Process catalog records to build element indices
            for line_idx in range(n_lines):
                if line_idx < len(catalog.records):
                    record = catalog.records[line_idx]
                    element_symbol = str(record.element).strip()
                    line_nelion[line_idx] = record.ion_stage
                    # Skip hydrogen lines
                    if (
                        element_symbol.upper() not in {"H", "H I", "HI"}
                        or record.ion_stage != 1
                    ):
                        if element_symbol in element_to_idx:
                            line_element_idx[line_idx] = element_to_idx[element_symbol]

            # Pre-resolve metadata-dependent per-line arrays (avoid branching inside kernel).
            # Defaults correspond to "no metadata".
            line_nelion_eff = np.asarray(line_nelion, dtype=np.int32)
            line_ncon = np.zeros(n_lines, dtype=np.int32)
            line_nelionx = np.zeros(n_lines, dtype=np.int32)
            line_alpha = np.zeros(n_lines, dtype=np.float64)

            has_metadata = metadata is not None
            if has_metadata and hasattr(metadata, "ncon"):
                n_meta = len(metadata.ncon)
                meta_ncon = np.asarray(metadata.ncon, dtype=np.int32)
                meta_nelionx = (
                    np.asarray(metadata.nelionx, dtype=np.int32)
                    if hasattr(metadata, "nelionx")
                    else None
                )
                meta_nelion = (
                    np.asarray(metadata.nelion, dtype=np.int32)
                    if hasattr(metadata, "nelion")
                    else None
                )
                meta_alpha = (
                    np.asarray(metadata.extra1, dtype=np.float64)
                    if hasattr(metadata, "extra1")
                    else None
                )

                # Build per-line meta index map (catalog line idx -> meta idx)
                line_meta_idx = np.full(n_lines, -1, dtype=np.int32)
                for li, mi in catalog_to_meta.items():
                    if li < n_lines and mi < n_meta:
                        line_meta_idx[li] = mi

                for li in range(n_lines):
                    mi = line_meta_idx[li]
                    if mi < 0:
                        continue
                    line_ncon[li] = meta_ncon[mi]
                    if meta_nelionx is not None and mi < meta_nelionx.size:
                        line_nelionx[li] = meta_nelionx[mi]
                    if (
                        meta_nelion is not None
                        and mi < meta_nelion.size
                        and meta_nelion[mi] > 0
                    ):
                        line_nelion_eff[li] = meta_nelion[mi]
                    if meta_alpha is not None and mi < meta_alpha.size:
                        line_alpha[li] = meta_alpha[mi]

            # Even when fort.9 metadata is unavailable, fort.19 lines still carry
            # NCON/NELIONX and must use them for XLINOP continuum taper limits.
            if fort19_data is not None and catalog_to_fort19:
                for li, f19i in catalog_to_fort19.items():
                    if 0 <= li < n_lines and 0 <= f19i < fort19_data.continuum_index.size:
                        line_ncon[li] = int(fort19_data.continuum_index[f19i])
                        line_nelionx[li] = int(fort19_data.element_index[f19i])

            # Precompute per-line window bounds (avoid recomputing indices in the hot loop).
            n_wl = wavelength.size
            max_window = 2 * MAX_PROFILE_STEPS + 2
            line_start_idx = np.zeros(n_lines, dtype=np.int32)
            line_end_idx = np.zeros(n_lines, dtype=np.int32)
            line_center_local = np.zeros(n_lines, dtype=np.int32)
            line_indices_arr = np.asarray(line_indices, dtype=np.int64)
            for li in range(n_lines):
                ci = int(line_indices_arr[li])
                if ci < 0 or ci >= n_wl:
                    # Keep zero window; kernel will skip via center_index range check.
                    continue
                start = ci - MAX_PROFILE_STEPS
                if start < 0:
                    start = 0
                end = ci + MAX_PROFILE_STEPS + 1
                if end > n_wl:
                    end = n_wl
                window_len = end - start
                if window_len <= 0:
                    continue
                if window_len > max_window:
                    window_len = max_window
                    start = ci - MAX_PROFILE_STEPS
                    if start < 0:
                        start = 0
                    end = start + window_len
                line_start_idx[li] = start
                line_end_idx[li] = end
                line_center_local[li] = ci - start

            # Pre-process depth-specific arrays
            electron_density_arr = np.zeros(atm.layers, dtype=np.float64)
            temperature_arr = np.zeros(atm.layers, dtype=np.float64)
            mass_density_arr = np.zeros(atm.layers, dtype=np.float64)
            emerge_arr = np.asarray(emerge, dtype=np.float64)
            emerge_h_arr = np.asarray(emerge_h, dtype=np.float64)
            xnf_h_arr_flat = np.zeros(atm.layers, dtype=np.float64)
            xnf_he1_arr_flat = np.zeros(atm.layers, dtype=np.float64)
            xnf_h2_arr_flat = np.zeros(atm.layers, dtype=np.float64)
            txnxn_arr = np.zeros(atm.layers, dtype=np.float64)
            # Store as (depth, line) to support depth-parallel kernel access.
            boltzmann_factor_arr = np.zeros((atm.layers, n_lines), dtype=np.float64)

            for depth_idx, state in pops.layers.items():
                electron_density_arr[depth_idx] = state.electron_density
                temperature_arr[depth_idx] = atm.temperature[depth_idx]
                mass_density_arr[depth_idx] = (
                    atm.mass_density[depth_idx]
                    if hasattr(atm, "mass_density") and atm.mass_density is not None
                    else 1.0
                )
                xnf_h_arr_flat[depth_idx] = _layer_value(xnf_h_arr, depth_idx)
                xnf_he1_arr_flat[depth_idx] = _layer_value(xnf_he1_arr, depth_idx)
                xnf_h2_arr_flat[depth_idx] = _layer_value(xnf_h2_arr, depth_idx)
                txnxn_arr[depth_idx] = state.txnxn
                boltzmann_factor_arr[depth_idx, :] = state.boltzmann_factor

            # Get Voigt tables
            voigt_tables = tables.voigt_tables()
            h0tab = voigt_tables.h0tab
            h1tab = voigt_tables.h1tab
            h2tab = voigt_tables.h2tab

            # Get metal tables contx array
            contx = metal_tables.contx
            ifvac_val = 1

            logger.info(
                f"Calling Numba kernel for {n_lines:,} lines × {atm.layers} depths..."
            )
            logger.info(
                "NOTE: First-time compilation may take 5-10 minutes. "
                "Subsequent runs will be much faster."
            )

            # Process in batches for progress logging
            # Use larger batches to minimize overhead (10 batches total)
            batch_size = max(1, n_lines // 10)
            n_batches = (n_lines + batch_size - 1) // batch_size

            kernel_start_time = time.time()

            # First call will trigger compilation - log when it starts executing
            logger.info(
                f"Processing {n_batches} batches of ~{batch_size:,} lines each..."
            )

            # Precompute depth-independent CONGF per line:
            # CONGF = (0.026538/1.77245) * GF / (C/λ) = const * GF * λ / C
            line_cgf = (0.026538 / 1.77245) * line_gf * line_wavelengths / C_LIGHT_NM
            # (1) Optimization: for non-INFO logging, avoid repeated kernel calls.
            # Batching is purely for progress logging; it adds overhead.
            use_progress_batches = logger.isEnabledFor(logging.INFO)
            if not use_progress_batches:
                _process_metal_wings_kernel(
                    metal_wings,
                    metal_sources,
                    wavelength,
                    line_indices_arr,
                    line_wavelengths,
                    line_cgf,
                    line_gamma_rad,
                    line_gamma_stark,
                    line_gamma_vdw,
                    line_element_idx,
                    line_nelion_eff,
                    line_ncon,
                    line_nelionx,
                    line_alpha,
                    line_start_idx,
                    line_end_idx,
                    line_center_local,
                    pop_densities_all,
                    dop_velocity_all,
                    buffers.continuum,
                    bnu,
                    electron_density_arr,
                    temperature_arr,
                    mass_density_arr,
                    emerge_arr,
                    emerge_h_arr,
                    xnf_h_arr_flat,
                    xnf_he1_arr_flat,
                    xnf_h2_arr_flat,
                    txnxn_arr,
                    boltzmann_factor_arr,
                    contx,
                    atomic_masses,
                    ifvac_val,
                    cfg.cutoff,
                    h0tab,
                    h1tab,
                    h2tab,
                )
            else:
                for batch_idx in range(n_batches):
                    batch_start = batch_idx * batch_size
                    batch_end = min(batch_start + batch_size, n_lines)

                # Log first batch (which includes compilation time)
                if batch_idx == 0:
                    batch_start_time = time.time()
                    logger.info(
                        f"Processing batch 1/{n_batches} (lines 0-{batch_end:,}) - "
                        "compiling kernel (this may take a few minutes)..."
                    )
                else:
                    batch_start_time = time.time()
                    elapsed_so_far = batch_start_time - kernel_start_time
                    progress_pct = 100.0 * batch_idx / n_batches
                    rate = (
                        batch_idx * batch_size / elapsed_so_far
                        if elapsed_so_far > 0
                        else 0
                    )
                    remaining_lines = n_lines - batch_start
                    eta = remaining_lines / rate if rate > 0 else 0
                    logger.info(
                        f"Processing batch {batch_idx + 1}/{n_batches} "
                        f"({batch_start:,}-{batch_end:,} lines, {progress_pct:.1f}%) - "
                        f"{rate:.0f} lines/s, ~{eta:.1f}s remaining"
                    )

                    # Create batch slices
                    batch_line_indices = line_indices_arr[batch_start:batch_end]
                    batch_line_wavelengths = line_wavelengths[batch_start:batch_end]
                    batch_line_cgf = line_cgf[batch_start:batch_end]
                    batch_line_gamma_rad = line_gamma_rad[batch_start:batch_end]
                    batch_line_gamma_stark = line_gamma_stark[batch_start:batch_end]
                    batch_line_gamma_vdw = line_gamma_vdw[batch_start:batch_end]
                    batch_line_element_idx = line_element_idx[batch_start:batch_end]
                    batch_line_nelion_eff = line_nelion_eff[batch_start:batch_end]
                    batch_line_ncon = line_ncon[batch_start:batch_end]
                    batch_line_nelionx = line_nelionx[batch_start:batch_end]
                    batch_line_alpha = line_alpha[batch_start:batch_end]
                    batch_line_start_idx = line_start_idx[batch_start:batch_end]
                    batch_line_end_idx = line_end_idx[batch_start:batch_end]
                    batch_line_center_local = line_center_local[batch_start:batch_end]
                    batch_boltzmann_factor = boltzmann_factor_arr[
                        :, batch_start:batch_end
                    ]

                    # Call kernel for this batch
                    _process_metal_wings_kernel(
                        metal_wings,
                        metal_sources,
                        wavelength,
                        batch_line_indices,
                        batch_line_wavelengths,
                        batch_line_cgf,
                        batch_line_gamma_rad,
                        batch_line_gamma_stark,
                        batch_line_gamma_vdw,
                        batch_line_element_idx,
                        batch_line_nelion_eff,
                        batch_line_ncon,
                        batch_line_nelionx,
                        batch_line_alpha,
                        batch_line_start_idx,
                        batch_line_end_idx,
                        batch_line_center_local,
                        pop_densities_all,
                        dop_velocity_all,
                        buffers.continuum,
                        bnu,
                        electron_density_arr,
                        temperature_arr,
                        mass_density_arr,
                        emerge_arr,
                        emerge_h_arr,
                        xnf_h_arr_flat,
                        xnf_he1_arr_flat,
                        xnf_h2_arr_flat,
                        txnxn_arr,
                        batch_boltzmann_factor,
                        contx,
                        atomic_masses,
                        ifvac_val,
                        cfg.cutoff,
                        h0tab,
                        h1tab,
                        h2tab,
                    )

                # Log batch completion
                batch_elapsed = time.time() - batch_start_time
                if batch_idx == 0:
                    # First batch includes compilation time
                    logger.info(
                        f"✓ Batch 1 completed in {batch_elapsed:.2f}s "
                        f"(includes kernel compilation time)"
                    )
                else:
                    elapsed_total = time.time() - kernel_start_time
                    progress_pct = 100.0 * batch_end / n_lines
                    rate = batch_end / elapsed_total if elapsed_total > 0 else 0
                    remaining_lines = n_lines - batch_end
                    eta = remaining_lines / rate if rate > 0 else 0
                    logger.info(
                        f"✓ Batch {batch_idx + 1}/{n_batches} completed in {batch_elapsed:.2f}s - "
                        f"{batch_end:,}/{n_lines:,} lines ({progress_pct:.1f}%) - "
                        f"{rate:.0f} lines/s, ~{eta:.1f}s remaining"
                    )

            elapsed_time = time.time() - start_time
            total_kernel_time = time.time() - kernel_start_time
            logger.info(
                f"Completed Numba parallel processing: {n_lines:,} lines × {atm.layers} depths in {elapsed_time:.2f}s"
            )
            logger.info(
                f"Kernel execution time: {total_kernel_time:.2f}s "
                f"({n_lines * atm.layers / total_kernel_time:.0f} line-depth pairs/s)"
            )

        else:
            # Sequential processing
            start_time = time.time()
            logger.info(
                f"Starting sequential processing of {atm.layers} depth layers..."
            )

            total_lines_processed = 0
            total_lines_skipped = 0

            for depth_idx, state in pops.layers.items():
                if depth_idx % 10 == 0:
                    logger.info(f"Processing depth layer {depth_idx+1}/{atm.layers}")

                try:
                    (
                        _,
                        local_wings,
                        local_sources,
                        local_helium_wings,
                        local_helium_sources,
                        lines_proc,
                        lines_skip,
                    ) = process_depth(depth_idx)
                    # Accumulate results
                    metal_wings[depth_idx] += local_wings
                    metal_sources[depth_idx] += local_sources
                    helium_wings[depth_idx] += local_helium_wings
                    helium_sources[depth_idx] += local_helium_sources
                    total_lines_processed += lines_proc
                    total_lines_skipped += lines_skip
                except Exception as e:
                    logger.error(
                        f"Error processing depth {depth_idx}: {e}", exc_info=True
                    )
                    raise

            elapsed_time = time.time() - start_time
            logger.info(
                f"Completed sequential processing: {atm.layers} depths in {elapsed_time:.2f}s "
                f"({atm.layers/elapsed_time:.2f} depths/s)"
            )
            logger.info(
                f"Lines processed: {total_lines_processed:,} total, "
                f"{total_lines_processed/atm.layers:.0f} per depth on average"
            )
            if total_lines_skipped > 0:
                logger.info(f"Lines skipped: {total_lines_skipped:,} total")

    if he_solver is not None and use_wings and (use_numba_parallel or use_parallel):
        if helium_line_ids is None or helium_line_ids.size == 0:
            logger.info("No helium wing lines found; skipping helium wings.")
        else:
            logger.info("Computing helium wings (helium-only pass, inline)...")
            start_time = time.time()
            helium_lines_processed = 0
            helium_lines_skipped = 0
            last_log_time = start_time

            helium_progress = os.getenv("PY_HELIUM_PROGRESS", "0") == "1"
            for depth_idx in range(atm.layers):
                (
                    _depth_idx,
                    _local_wings,
                    _local_sources,
                    local_helium_wings,
                    local_helium_sources,
                    lines_proc,
                    lines_skip,
                ) = process_depth(depth_idx, include_metals=False, include_helium=True)
                helium_lines_processed += lines_proc
                helium_lines_skipped += lines_skip
                helium_wings[depth_idx] += local_helium_wings
                helium_sources[depth_idx] += local_helium_sources

                if helium_progress and time.time() - last_log_time > 5:
                    logger.info(
                        f"Helium wings progress: {depth_idx + 1}/{atm.layers} "
                        f"({100.0 * (depth_idx + 1) / atm.layers:.1f}%)"
                    )
                    last_log_time = time.time()

            elapsed_time = time.time() - start_time
            logger.info(
                f"Completed helium wings: {atm.layers} depths in {elapsed_time:.2f}s "
                f"({atm.layers/elapsed_time:.2f} depths/s)"
            )
            if helium_lines_skipped > 0:
                logger.info(f"Helium lines skipped: {helium_lines_skipped:,} total")

    if use_wings:
        logger.info("Metal wings computation complete")
    else:
        # No lines - wings are already zero
        logger.info("No lines - skipping metal wings")

    # --- line source reconstruction -------------------------------------------------
    # Reuse AHLINE computed above; this avoids a second expensive pass with
    # identical inputs and preserves behavior.
    if use_wings and not cfg.skip_hydrogen_wings:
        logger.info(
            "Computing hydrogen wings for source function... (reusing precomputed AHLINE)"
        )
        shline = np.zeros_like(ahline)
    else:
        if cfg.skip_hydrogen_wings:
            logger.info("Skipping hydrogen wings (--skip-hydrogen-wings)")
        else:
            logger.info("No lines - skipping hydrogen wings")
        ahline = np.zeros_like(buffers.line_opacity)
        shline = np.zeros_like(buffers.line_opacity)

    logger.info("Computing line source functions...")
    spectrv_params = spectrv_params or spectrv_io.SpectrvParams(
        rhoxj=0.0, ph1=0.0, pc1=0.0, psi1=0.0, prddop=0.0, prdpow=0.0
    )

    bfudge_values, slinec = bfudge.compute_bfudge_and_slinec(
        atm,
        spectrv_params,
        bnu,
        stim,
        ehvkt,
    )
    buffers.bfudge[:] = bfudge_values
    buffers.slinec[:] = slinec

    combined_wings = metal_wings + helium_wings
    combined_sources = metal_sources + helium_sources

    # Reconstruct SXLINE (metal/helium wing source function state)
    with np.errstate(divide="ignore", invalid="ignore"):
        sxline = np.divide(
            combined_sources,
            np.maximum(combined_wings, 1e-40),
            out=np.zeros_like(combined_sources),
            where=combined_wings > 1e-40,
        )

    abs_core = abs_core_base
    total_line_absorption = abs_core + ahline_for_total + combined_wings

    # Include metal wings in the line opacity used by radiative transfer.
    # ASYNTH from compute_transp does not include metal_wings; add them here.
    using_asynth = asynth_npz is not None or has_lines
    if not using_asynth:
        buffers.line_opacity[:] = total_line_absorption
    else:
        # Using ASYNTH mode: buffers.line_opacity was already set to absorption = asynth * (1 - fscat)
        # For range-filtered runs, add off-grid wing contributions to match Fortran's
        # full-grid accumulation prior to ASYNTH.
        if np.any(helium_wings > 0):
            # Apply the same ASYNTH split used by Fortran:
            # absorption += ASYNTH_he * (1-FSCAT), scattering += ASYNTH_he * FSCAT.
            # `helium_wings` here is pre-STIM opacity from the wing pass; convert to
            # ASYNTH-equivalent opacity before applying the FSCAT split.
            helium_asynth = helium_wings * stim
            buffers.line_opacity += helium_asynth * (1.0 - fscat_vec[:, None])
            alinec_total = alinec_total + helium_asynth
    buffers.line_scattering[:] = alinec_total * fscat_vec[:, None]

    # CRITICAL: When using ASYNTH (whether from fort.29 or computed from catalog),
    # Fortran ALWAYS uses SLINE = BNU*STIM/(BFUDGE-EHVKT) = slinec
    # (spectrv.for line 314: SLINE(J)=BNU(J)*STIM(J)/(BFUDGE(J)-EHVKT(J)))
    # The weighted average (commented out in Fortran lines 315-316) is NEVER used with ASYNTH
    #
    # Detection: If ASYNTH was computed from catalog (has_lines=True means we computed ASYNTH),
    # OR if ASYNTH was loaded from fort.29 (asynth_npz is not None), we're using ASYNTH.
    # The weighted average is only used when using fort.9 ALINEC (not ASYNTH)
    #
    # Note: has_lines=True indicates we computed ASYNTH from catalog (see line 1426-1456)
    # So if has_lines=True, we're using ASYNTH (regardless of asynth_npz)
    using_asynth = asynth_npz is not None or has_lines
    if using_asynth:
        # Use SLINEC in ASYNTH mode to match the source-function scaling used in
        # the opacity/source pipeline and avoid spurious line emission in cool models.
        line_source = slinec.copy()
    else:
        # Using fort.9 ALINEC or computed lines: compute weighted source function
        # Match Fortran atlas7v.for line 4497-4498:
        # SLINE = (AHLINE*SHLINE + ALINES*BNU + AXLINE*SXLINE) / ALINE
        # Where SXLINE=0 (initialized at line 4462, never set in XLINOP)
        # However, in our Python implementation, we compute metal_wings and metal_sources,
        # so we should include metal_wings * sxline in the numerator.
        # If sxline is zero (metal_sources not computed), metal wings contribute with Planck source.
        # Reference: atlas7v.for lines 4462, 4497-4498
        # CRITICAL FIX: Include metal wings in numerator, using sxline if available, else bnu
        numerator = ahline * shline + abs_core * slinec
        # Add metal wings contribution: if sxline is available and non-zero, use it; else use Planck
        metal_source = np.where(
            (metal_wings > 1e-40) & (np.abs(sxline) > 1e-40), sxline, bnu
        )
        metal_contribution = metal_wings * metal_source
        numerator = numerator + metal_contribution


        with np.errstate(divide="ignore", invalid="ignore"):
            line_source = np.divide(
                numerator,
                np.maximum(total_line_absorption, 1e-40),
                out=np.zeros_like(total_line_absorption),
            )

        # When line opacity is zero (or very small), line_source should default to Planck
        line_opacity_mask = total_line_absorption < 1e-30
        if np.any(line_opacity_mask):
            line_source[line_opacity_mask] = bnu[line_opacity_mask]

    _timings["line opacity stage"] = time.perf_counter() - t_line_opacity
    logger.info("Timing: line opacity stage in %.3fs", _timings["line opacity stage"])
    logger.info("Solving radiative transfer equation...")

    # Diagnostic: check line opacity magnitude
    if wavelength.size > 0 and cont_abs.shape[1] == wavelength.size:
        idx_check = wavelength.size // 2  # Check middle wavelength
        logger.info(f"Diagnostic (wavelength {float(wavelength[idx_check]):.2f} nm):")
        logger.info(
            f"  Continuum absorption (surface): {float(cont_abs[0, idx_check]):.6E}"
        )
        logger.info(
            f"  Line opacity (surface): {float(buffers.line_opacity[0, idx_check]):.6E}"
        )
        logger.info(
            f"  Line/Continuum ratio: {float(buffers.line_opacity[0, idx_check]) / max(float(cont_abs[0, idx_check]), 1e-40):.6f}"
        )
        logger.info(
            f"  Total opacity (surface): {float(cont_abs[0, idx_check] + buffers.line_opacity[0, idx_check]):.6E}"
        )

    # Determine number of workers for parallel processing
    n_workers = cfg.n_workers
    if n_workers is None:
        # Auto-detect: use parallel processing for large wavelength grids
        if wavelength.size > 10000:
            import multiprocessing

            n_workers = max(1, multiprocessing.cpu_count())
        else:
            n_workers = 1  # Sequential for small grids
    # Add right before line 1462 (before solve_lte_spectrum call)
    logger.info("=" * 70)
    logger.info("DIAGNOSTIC: Before solve_lte_spectrum")
    logger.info("=" * 70)
    logger.info(f"Line opacity shape: {buffers.line_opacity.shape}")
    logger.info(
        f"Line opacity non-zero count: {np.count_nonzero(buffers.line_opacity)}"
    )
    logger.info(f"Line opacity max: {np.max(buffers.line_opacity):.2e}")
    if np.any(buffers.line_opacity > 0):
        non_zero_indices = np.where(buffers.line_opacity > 0)
        logger.info(
            f"Sample non-zero line opacity: {float(buffers.line_opacity[non_zero_indices[0][0], non_zero_indices[1][0]]):.2e} at depth {int(non_zero_indices[0][0])}, wavelength idx {int(non_zero_indices[1][0])}"
        )
    logger.info(f"Continuum absorption shape: {cont_abs.shape}")
    logger.info(f"Continuum absorption max: {np.max(cont_abs):.2e}")
    logger.info(
        f"Line source shape: {line_source.shape if line_source is not None else 'None'}"
    )
    if line_source is not None:
        logger.info(f"Line source max: {np.max(line_source):.2e}")
        logger.info(f"Line source min: {np.min(line_source):.2e}")
    logger.info("=" * 70)
    # Right before solve_lte_spectrum call
    logger.info(f"Line opacity stats before solve_lte_spectrum:")
    logger.info(f"  Shape: {buffers.line_opacity.shape}")
    logger.info(f"  Non-zero count: {np.count_nonzero(buffers.line_opacity)}")
    logger.info(f"  Max: {np.max(buffers.line_opacity):.2e}")
    if len(line_indices) > 0:
        first_line_idx = line_indices[0]
        logger.info(
            f"  At first line wavelength (idx {first_line_idx}): {float(buffers.line_opacity[0, first_line_idx]):.2e}"
        )
        logger.info(
            f"  At first line wavelength (all depths): min={float(np.min(buffers.line_opacity[:, first_line_idx])):.2e}, max={float(np.max(buffers.line_opacity[:, first_line_idx])):.2e}"
        )
    # CRITICAL CHECK: Verify line opacity is not all zeros
    if np.all(buffers.line_opacity == 0.0):
        logger.error(
            "ERROR: buffers.line_opacity is ALL ZEROS! This will cause flux == continuum!"
        )
    else:
        logger.info(
            f"  Line opacity is NOT all zeros - should produce different flux and continuum"
        )

    t_rt = time.perf_counter()
    flux_total, flux_cont = solve_lte_spectrum(
        wavelength,
        atm.temperature,
        atm.depth,
        cont_abs,
        cont_scat,
        buffers.line_opacity,
        buffers.line_scattering,
        line_source=line_source,
        n_workers=n_workers,
    )
    _timings["radiative transfer"] = time.perf_counter() - t_rt
    logger.info("Timing: radiative transfer in %.3fs", _timings["radiative transfer"])

    # --- Stage 5 dump: RT output ---
    if _stage_dump_path is not None:
        np.savez(
            _stage_dump_path / "stage_5_rt.npz",
            wavelength=wavelength,
            flux_total_hz=flux_total,
            flux_cont_hz=flux_cont,
            flux_total=flux_total,
            flux_cont=flux_cont,
            cont_abs=cont_abs,
            cont_scat=cont_scat,
            line_opacity=buffers.line_opacity,
            line_scattering=buffers.line_scattering,
        )
        logger.info("Stage 5 dump (RT output) saved")

    # Diagnostic: check flux before conversion
    if wavelength.size > 0:
        idx_check = wavelength.size // 2
        logger.info(
            f"Flux BEFORE conversion (wavelength {float(wavelength[idx_check]):.2f} nm):"
        )
        logger.info(f"  Flux total: {float(flux_total[idx_check]):.6E}")
        logger.info(f"  Flux continuum: {float(flux_cont[idx_check]):.6E}")
        logger.info(
            f"  Line opacity (surface): {float(buffers.line_opacity[0, idx_check]):.6E}"
        )
        logger.info(f"  Line source (surface): {float(line_source[0, idx_check]):.6E}")
        logger.info(
            f"  Continuum absorption (surface): {float(cont_abs[0, idx_check]):.6E}"
        )
        logger.info(
            f"  Line/Continuum ratio: {float(buffers.line_opacity[0, idx_check]) / max(float(cont_abs[0, idx_check]), 1e-40):.6f}"
        )
        flux_ratio = float(flux_total[idx_check]) / max(
            float(flux_cont[idx_check]), 1e-40
        )
        logger.info(f"  Flux ratio (total/cont): {flux_ratio:.6f}")
        # CRITICAL CHECK: Are flux_total and flux_cont identical?
        if np.allclose(flux_total, flux_cont, rtol=1e-10):
            logger.error(
                "ERROR: flux_total and flux_cont are IDENTICAL! This means line opacity is not being used!"
            )
            logger.error(f"  This will cause all output flux == continuum!")
        else:
            n_different = np.sum(~np.isclose(flux_total, flux_cont, rtol=1e-10))
            logger.info(
                f"  Flux values differ at {n_different}/{len(flux_total)} wavelengths"
            )

    # Convert from per Hz to per nm: F_λ = F_ν * c / λ^2
    # Fortran uses: FREQTOWAVE = 2.99792458D17 / WAVE^2 (where WAVE is in nm)
    # 2.99792458D17 = speed of light in nm/s = 2.99792458e10 cm/s * 1e7 nm/cm
    C_LIGHT_NM_PER_S = 2.99792458e17  # nm/s
    conversion = C_LIGHT_NM_PER_S / np.maximum(wavelength**2, 1e-40)

    # CRITICAL FIX: Make copies before conversion to avoid modifying original arrays
    # The multiplication creates new arrays, but we need to ensure they're independent
    flux_total = (flux_total * conversion).copy()
    flux_cont = (flux_cont * conversion).copy()

    if wavelength.size > 0:
        idx_check = wavelength.size // 2
        logger.info(
            f"Conversion factor (wavelength {float(wavelength[idx_check]):.2f} nm): {float(conversion[idx_check]):.6E}"
        )
        logger.info(f"Flux AFTER conversion:")
        logger.info(f"  Flux total: {float(flux_total[idx_check]):.6E}")
        logger.info(f"  Flux continuum: {float(flux_cont[idx_check]):.6E}")
        # Note: Ground truth comparison removed to avoid hardcoded configuration-specific paths

    # Optional post-conversion RT dump to make units explicit for diagnostics.
    if _stage_dump_path is not None:
        np.savez(
            _stage_dump_path / "stage_5_rt_converted.npz",
            wavelength=wavelength,
            flux_total=flux_total,
            flux_cont=flux_cont,
        )

    logger.info("Converting flux units...")
    _timings["total pipeline"] = time.perf_counter() - t_pipeline
    result = SynthResult(
        wavelength=buffers.wavelength.copy(),
        intensity=flux_total,
        continuum=flux_cont,
        timings=_timings,
    )
    logger.info(f"Writing spectrum to {cfg.output.spec_path}")
    export.write_spec_file(result, cfg.output.spec_path)

    diagnostics_path = cfg.output.diagnostics_path
    if diagnostics_path is not None:
        logger.info(f"Writing diagnostics to {diagnostics_path}")
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            diagnostics_path,
            wavelength=wavelength,
            continuum_absorption=cont_abs,
            continuum_scattering=cont_scat,
            hydrogen_continuum=buffers.hydrogen_continuum,
            hydrogen_source=buffers.hydrogen_source,
            line_opacity=buffers.line_opacity,
            line_scattering=buffers.line_scattering,
            line_source=line_source,
            bfudge=buffers.bfudge,
            slinec=buffers.slinec,
            flux_total=flux_total,
            flux_continuum=flux_cont,
        )
    logger.info("Synthesis complete!")
    logger.info("Timing: total pipeline in %.3fs", time.perf_counter() - t_pipeline)
    return result
