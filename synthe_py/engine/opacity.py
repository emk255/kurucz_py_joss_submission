"""Core synthesis loop."""

from __future__ import annotations

import json
import math
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from numba import jit, prange

from ..config import SynthesisConfig
from ..io import atmosphere, export
from ..io.lines import atomic, compiler as line_compiler, fort19 as fort19_io, tfort
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

_fort19_unhandled_types: Set[fort19_io.Fort19WingType] = set()

# Fortran synthe.for XLINOP debug targets (WRITE 199 when J in 1,41,80 and WAVE near these)
_XLINOP_DEBUG_TARGET_WAVES = (457.656906, 636.360933)
_XLINOP_DEBUG_TARGET_TOL = 5.0e-4
_XLINOP_DEBUG_DEPTHS = (0, 40, 79)  # 0-based (Fortran J=1,41,80)
_EMPTY_FLOAT64 = np.empty(0, dtype=np.float64)


def _agent_load_fortran_spec(stem: str) -> Optional[np.ndarray]:
    repo_root = Path(__file__).resolve().parents[2]
    spec_path = repo_root / "results/validation_100/fortran_specs" / f"{stem}.spec"
    if not spec_path.exists():
        return None
    try:
        return np.loadtxt(spec_path, usecols=(0, 1, 2), dtype=np.float64)
    except Exception:
        # Some Fortran .spec rows can omit whitespace before negative flux values
        # (e.g., "300.03840812-0.775425E-15 ..."), which breaks loadtxt tokenization.
        # Fallback: extract float-like tokens via regex per line.
        float_pattern = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?")
        rows: List[Tuple[float, float, float]] = []
        try:
            with open(spec_path, "r", encoding="utf-8", errors="ignore") as fp:
                for raw_line in fp:
                    line = raw_line.strip()
                    if not line:
                        continue
                    tokens = float_pattern.findall(line)
                    if len(tokens) < 3:
                        continue
                    try:
                        wl = float(tokens[0])
                        flux = float(tokens[1])
                        cont = float(tokens[2])
                    except ValueError:
                        continue
                    rows.append((wl, flux, cont))
            if not rows:
                return None
            return np.asarray(rows, dtype=np.float64)
        except Exception:
            return None


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


def _recompute_population_per_ion(atm: atmosphere.AtmosphereModel) -> None:
    logger = logging.getLogger(__name__)
    if (
        atm.xabund is None
        or atm.xnatm is None
        or atm.tkev is None
        or atm.tk is None
        or atm.hkt is None
        or atm.hckt is None
        or atm.tlog is None
    ):
        logger.warning(
            "Skipping population_per_ion recompute (missing xabund/xnatm or thermo arrays)."
        )
        return

    try:
        from synthe_py.tools import pops_exact
    except Exception as exc:  # pragma: no cover - debug only
        logger.warning("POPS exact not available: %s", exc)
        return

    logger.info("Recomputing population_per_ion using POPS exact (Fortran matching).")
    pops_exact.load_fortran_data()

    # Reset POPS iteration state to match fresh xnfpelsyn behavior.
    if hasattr(pops_exact, "_ITEMP"):
        pops_exact._ITEMP = 0
    if hasattr(pops_exact, "_ITEMP1"):
        pops_exact._ITEMP1 = 0

    n_layers = atm.layers
    population = np.zeros((n_layers, 6, 139), dtype=np.float64)
    xne = np.asarray(atm.electron_density, dtype=np.float64).copy()
    xnatm = np.asarray(atm.xnatm, dtype=np.float64).copy()

    def get_element_code(elem_num: int) -> float:
        code_map = {
            1: 1.01,
            2: 2.02,
            3: 3.03,
            4: 4.03,
            5: 5.03,
            6: 6.05,
            7: 7.05,
            8: 8.05,
            9: 9.05,
            10: 10.05,
            11: 11.05,
            12: 12.05,
            13: 13.05,
            14: 14.05,
            15: 15.05,
            16: 16.05,
            17: 17.04,
            18: 18.04,
            19: 19.04,
            20: 20.09,
            21: 21.09,
            22: 22.09,
            23: 23.09,
            24: 24.09,
            25: 25.09,
            26: 26.09,
            27: 27.09,
            28: 28.09,
        }
        return code_map.get(elem_num, float(elem_num) + 0.02)

    for elem_num in range(1, 100):
        number = np.zeros((n_layers, 10), dtype=np.float64)
        code = get_element_code(elem_num)
        pops_exact.pops_exact(
            code,
            11,
            number,
            np.asarray(atm.temperature, dtype=np.float64),
            np.asarray(atm.tkev, dtype=np.float64),
            np.asarray(atm.tk, dtype=np.float64),
            np.asarray(atm.hkt, dtype=np.float64),
            np.asarray(atm.hckt, dtype=np.float64),
            np.asarray(atm.tlog, dtype=np.float64),
            np.asarray(atm.gas_pressure, dtype=np.float64),
            xne,
            xnatm,
            np.asarray(atm.xabund, dtype=np.float64),
        )
        population[:, :6, elem_num - 1] = number[:, :6]

    atm.population_per_ion = population


def _load_line_data(
    cfg: SynthesisConfig,
    wl_min: float,
    wl_max: float,
) -> atomic.LineCatalog:
    """Load line catalog from atomic catalog file."""
    _logger = logging.getLogger(__name__)
    catalog: Optional[atomic.LineCatalog] = None
    is_fort19_catalog = False
    is_tfort12_catalog = False
    is_tfort14_catalog = False
    if cfg.line_data.atomic_catalog is not None:
        try:
            _logger.info(f"Loading atomic catalog from: {cfg.line_data.atomic_catalog}")
            catalog_path = Path(cfg.line_data.atomic_catalog)
            suffix = catalog_path.suffix.lower()
            if suffix == ".12" and "fort.12" in catalog_path.name:
                tfort12_records = tfort.parse_tfort12(catalog_path)
                tfort93_path = catalog_path.with_suffix(".93")
                if not tfort93_path.exists():
                    raise RuntimeError(
                        "tfort.12 requires companion tfort.93 for WLBEG/RATIO (Fortran exact behavior)."
                    )
                t93 = tfort.parse_tfort93(tfort93_path)
                ratio = t93.ratio
                rlog = t93.ratiolg if t93.ratiolg != 0.0 else math.log(ratio)
                ixwlbeg = math.floor(math.log(t93.wlbeg) / rlog)
                if math.exp(ixwlbeg * rlog) < t93.wlbeg:
                    ixwlbeg += 1
                wbegin = math.exp(ixwlbeg * rlog)
                cgf_constant = 0.026538 / 1.77245
                records: List[atomic.LineRecord] = []
                for rec in tfort12_records:
                    wl = wbegin * (ratio ** (rec.nbuff - 1))
                    if wl <= 0.0:
                        continue
                    freq_hz = C_LIGHT_NM / wl
                    gf_linear = rec.cgf * freq_hz / cgf_constant
                    log_gf = math.log10(gf_linear) if gf_linear > 0.0 else -99.0
                    nelion = rec.nelion
                    if nelion <= 0:
                        continue
                    elem_z = (nelion - 1) // 6 + 1
                    ion_stage = nelion - 6 * (elem_z - 1)
                    element = _ELEMENT_SYMBOL_BY_Z.get(elem_z)
                    if element is None or ion_stage <= 0:
                        continue
                    # tfort.12 gamma values are PRE-NORMALIZED by rgfall.for
                    # (lines 271-273):
                    #   GAMMAR = GAMMAR / 12.5664 / FRELIN
                    # where 12.5664 = 4π and FRELIN = c/λ (Hz).
                    # So GAMRF = γ_linear / (4πν).
                    #
                    # The ADAMP formula must use: ADAMP = gamma_total / DOPPLE
                    # (matching Fortran synthe.for line 473) — NOT gamma / (4π*ν*DOPPLE).
                    # Both line_opacity.py and opacity.py now use gamma_total / dopple.
                    records.append(
                        atomic.LineRecord(
                            wavelength=wl,
                            index_wavelength=wl,
                            element=element,
                            ion_stage=ion_stage,
                            log_gf=log_gf,
                            excitation_energy=rec.elo_cm,
                            gamma_rad=rec.gamma_rad,
                            gamma_stark=rec.gamma_stark,
                            gamma_vdw=rec.gamma_vdw,
                            metadata={"cgf": float(rec.cgf)},
                            line_type=0,
                            n_lower=0,
                            n_upper=0,
                        )
                    )
                catalog = atomic.LineCatalog.from_records(records)
                is_tfort12_catalog = True
                _logger.info("Loaded %d lines from tfort.12", len(catalog.records))
                is_tfort12_catalog = True
            elif suffix == ".14" and "fort.14" in catalog_path.name:
                tfort14_records = tfort.parse_tfort14(catalog_path)
                records: List[atomic.LineRecord] = []
                for rec in tfort14_records:
                    line_type = 0
                    if rec.element_symbol == "H" and rec.ion_stage == 1:
                        line_type = -1
                    elif rec.element_symbol == "He" and rec.ion_stage == 1:
                        line_type = -3
                    elif rec.element_symbol == "He" and rec.ion_stage == 2:
                        line_type = -6
                    gf_linear = rec.gf if rec.gf > 0.0 else 0.0
                    log_gf = math.log10(gf_linear) if gf_linear > 0.0 else rec.log_gf
                    records.append(
                        atomic.LineRecord(
                            wavelength=rec.wavelength_vac,
                            index_wavelength=rec.wavelength_vac,
                            element=rec.element_symbol,
                            ion_stage=rec.ion_stage,
                            log_gf=log_gf,
                            excitation_energy=rec.excitation_energy_cm,
                            gamma_rad=rec.gamma_rad,
                            gamma_stark=rec.gamma_stark,
                            gamma_vdw=rec.gamma_vdw,
                            metadata={},
                            line_type=line_type,
                            n_lower=rec.n_lower or 0,
                            n_upper=rec.n_upper or 0,
                        )
                    )
                catalog = atomic.LineCatalog.from_records(records)
                _logger.info("Loaded %d lines from tfort.14", len(catalog.records))
                is_tfort14_catalog = True
            elif suffix in {".19", ".npz"} and "fort.19" in catalog_path.name:
                fort19_data = fort19_io.load(catalog_path)
                catalog = _catalog_from_fort19(fort19_data)
                is_fort19_catalog = True
                _logger.info(
                    "Loaded %d lines from fort.19/tfort.19", len(catalog.records)
                )
            else:
                catalog = atomic.load_catalog(catalog_path)
                _logger.info(f"Loaded {len(catalog.records)} lines from catalog")
        except (FileNotFoundError, ValueError) as e:
            _logger.error(
                f"Could not load atomic line catalog {cfg.line_data.atomic_catalog}: {e}"
            )
            raise RuntimeError(
                f"Failed to load atomic catalog from {cfg.line_data.atomic_catalog}. "
                f"Please check that the file exists and is a valid atomic line catalog."
            ) from e

    if catalog is None:
        raise RuntimeError(
            "No atomic catalog provided. "
            "An atomic line catalog is required for line synthesis. "
            "Please provide --atomic-catalog or use the 'atomic' positional argument."
        )

    if is_fort19_catalog or is_tfort12_catalog or is_tfort14_catalog:
        filtered_catalog = catalog
        _logger.info(
            "Line filtering skipped for fort.* catalog: using %d lines",
            len(filtered_catalog.records),
        )
    elif cfg.line_filter:
        filtered_catalog = atomic.filter_by_range(catalog, wl_min, wl_max)
        _logger.info(
            f"After filtering to wavelength range [{wl_min:.2f}, {wl_max:.2f}] nm: "
            f"{len(filtered_catalog.records)} lines"
        )
    else:
        filtered_catalog = catalog
        _logger.info(
            "Line filtering disabled: using full catalog of %d lines",
            len(filtered_catalog.records),
        )
    # For narrow windows (e.g. detailed debugging over a small wavelength span),
    # log the strongest few lines to make sure the expected transitions are present.
    try:
        if len(filtered_catalog.records) > 0 and (wl_max - wl_min) <= 5.0:
            wl_arr = filtered_catalog.wavelength
            loggf_arr = filtered_catalog.log_gf
            order = np.argsort(loggf_arr)[::-1]
            n_show = int(min(10, order.size))
            _logger.info(
                "Top %d lines in [%.3f, %.3f] nm by log(gf):", n_show, wl_min, wl_max
            )
            for rank, idx in enumerate(order[:n_show], start=1):
                rec = filtered_catalog.records[int(idx)]
                _logger.info(
                    "  %2d: wl=%.4f nm element=%s ion=%d loggf=%.3f E_exc=%.3f eV",
                    rank,
                    float(rec.wavelength),
                    rec.element,
                    int(rec.ion_stage),
                    float(rec.log_gf),
                    float(rec.excitation_energy),
                )
    except Exception as exc:  # pragma: no cover - debug logging only
        _logger.debug("Failed to log detailed line list diagnostics: %s", exc)
    if len(filtered_catalog.records) == 0:
        _logger.warning(
            f"No lines found in wavelength range [{wl_min:.2f}, {wl_max:.2f}] nm. "
            f"Original catalog had {len(catalog.records)} lines. "
            f"This will result in continuum-only synthesis."
        )
    return filtered_catalog


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


def _match_catalog_to_fort9(
    catalog_wavelength: np.ndarray, meta_wavelength: np.ndarray, tolerance: float = 1e-3
) -> Dict[int, int]:
    """Build a mapping from catalog index to fort.9 metadata index."""

    mapping: Dict[int, int] = {}
    if catalog_wavelength.size == 0 or meta_wavelength.size == 0:
        return mapping

    indices = np.searchsorted(catalog_wavelength, meta_wavelength)
    for meta_idx, pos in enumerate(indices):
        best_idx: Optional[int] = None
        best_delta = tolerance
        for candidate in (pos, pos - 1):
            if 0 <= candidate < catalog_wavelength.size:
                delta = abs(catalog_wavelength[candidate] - meta_wavelength[meta_idx])
                if delta < best_delta:
                    best_delta = delta
                    best_idx = candidate
        if best_idx is not None and best_idx not in mapping:
            mapping[best_idx] = meta_idx
    return mapping


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


def _catalog_from_fort19(fort19_data: fort19_io.Fort19Data) -> atomic.LineCatalog:
    """Build a LineCatalog from a fort.19/tfort.19 file."""
    records: List[atomic.LineRecord] = []
    for idx in range(fort19_data.wavelength_vacuum.size):
        wl = float(fort19_data.wavelength_vacuum[idx])
        nelion = int(fort19_data.ion_index[idx])
        if nelion <= 0:
            continue
        elem_z = (nelion - 1) // 6 + 1
        ion_stage = nelion - 6 * (elem_z - 1)
        element = _ELEMENT_SYMBOL_BY_Z.get(elem_z)
        if element is None or ion_stage <= 0:
            continue
        gf = float(fort19_data.oscillator_strength[idx])
        log_gf = math.log10(gf) if gf > 0.0 else -99.0
        rec = atomic.LineRecord(
            wavelength=wl,
            index_wavelength=wl,
            element=element,
            ion_stage=ion_stage,
            log_gf=log_gf,
            excitation_energy=float(fort19_data.energy_lower[idx]),
            gamma_rad=float(fort19_data.gamma_rad[idx]),
            gamma_stark=float(fort19_data.gamma_stark[idx]),
            gamma_vdw=float(fort19_data.gamma_vdw[idx]),
            metadata={"cgf": gf},
            line_type=int(fort19_data.line_type[idx]),
            n_lower=int(fort19_data.n_lower[idx]),
            n_upper=int(fort19_data.n_upper[idx]),
        )
        records.append(rec)
    return atomic.LineCatalog.from_records(records)


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

    if wing_type not in _fort19_unhandled_types:
        _fort19_unhandled_types.add(wing_type)
        logging.getLogger(__name__).debug(
            "fort.19 line_type %s not yet implemented; falling back to standard wings",
            wing_type,
        )
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

    debug_line = os.getenv("PY_DEBUG_HLINE")
    debug_wave = os.getenv("PY_DEBUG_HLINE_WAVE")
    debug_sum_wave = os.getenv("PY_DEBUG_HLINE_SUM_WAVE")
    debug_sum_depth = os.getenv("PY_DEBUG_HLINE_SUM_DEPTH")
    debug_sum_top = os.getenv("PY_DEBUG_HLINE_SUM_TOP")
    debug_line_val = None
    debug_wave_idx = None
    sum_wave_val = None
    sum_wave_idx = None
    sum_depth_idx = None
    sum_top_n = 10
    if debug_line and debug_wave:
        try:
            debug_line_val = float(debug_line)
            debug_wave_idx = int(np.argmin(np.abs(wavelength_grid - float(debug_wave))))
        except ValueError:
            debug_line_val = None
            debug_wave_idx = None
    if debug_sum_wave:
        try:
            sum_wave_val = float(debug_sum_wave)
            sum_wave_idx = int(np.argmin(np.abs(wavelength_grid - sum_wave_val)))
        except ValueError:
            sum_wave_val = None
            sum_wave_idx = None
    if debug_sum_depth:
        try:
            depth_val = int(debug_sum_depth)
            if depth_val > 0:
                sum_depth_idx = depth_val - 1
        except ValueError:
            sum_depth_idx = None
    if debug_sum_top:
        try:
            sum_top_n = max(1, int(debug_sum_top))
        except ValueError:
            sum_top_n = 10
    sum_entries = []

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

            debug_before = None
            sum_before = None
            if (
                debug_line_val is not None
                and abs(line_wavelength - debug_line_val) < 1e-3
                and depth_idx in (0, 19, 39, 49, 59, 79)
            ):
                if debug_wave_idx is not None:
                    debug_before = ahline[depth_idx, debug_wave_idx]
                print(
                    f"PY_DEBUG_HLINE_KAPPA: line={line_wavelength:.6f} nm depth={depth_idx + 1} "
                    f"center_idx={center_idx} kappa0_pre={kappa0_pre:.6e} "
                    f"kappa0={kappa0:.6e} kapmin={kapmin:.6e}"
                )
                if debug_wave_idx is not None:
                    dbg_wave = float(wavelength_grid[debug_wave_idx])
                    dbg_stim = stim[depth_idx, debug_wave_idx]
                    dbg_delta = dbg_wave - line_wavelength
                    dbg_profile = hydrogen_line_profile(
                        n_lower_eff,
                        n_upper_eff,
                        depth_state,
                        dbg_delta,
                    )
                    dbg_value = kappa0 * dbg_profile * dbg_stim
                    dbg_thresh = continuum[depth_idx, debug_wave_idx] * cutoff
                    dbg_simple = n_upper <= n_lower + 2
                    dbg_side = "red" if dbg_wave >= line_wavelength else "blue"
                    dbg_gate = "open"
                    if not dbg_simple:
                        if dbg_side == "red" and dbg_wave > wlminus1:
                            dbg_gate = "blocked_wlminus1"
                        elif dbg_side == "blue" and (
                            dbg_wave < wcon or dbg_wave < wlplus1
                        ):
                            dbg_gate = "blocked_wcon_wlplus1"
                    dbg_pass = (
                        dbg_gate == "open" and dbg_value > 0.0 and dbg_value >= dbg_thresh
                    )
                    print(
                        f"PY_DEBUG_HLINE_TARGET: line={line_wavelength:.6f} nm depth={depth_idx + 1} "
                        f"wave={dbg_wave:.6f} side={dbg_side} simple={dbg_simple} gate={dbg_gate} "
                        f"delta={dbg_delta:.6e} profile={dbg_profile:.6e} value={dbg_value:.6e} "
                        f"threshold={dbg_thresh:.6e} pass={dbg_pass}"
                    )
            if (
                sum_wave_idx is not None
                and sum_depth_idx is not None
                and depth_idx == sum_depth_idx
            ):
                sum_before = ahline[depth_idx, sum_wave_idx]
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
            if debug_before is not None and debug_wave_idx is not None:
                debug_after = ahline[depth_idx, debug_wave_idx]
                debug_wave_val = float(wavelength_grid[debug_wave_idx])
                print(
                    f"PY_DEBUG_HLINE: line={line_wavelength:.6f} nm depth={depth_idx + 1} "
                    f"center_idx={center_idx} wcon={wcon:.6f} wtail={wtail:.6f} "
                    f"wlplus1={wlplus1:.6f} wlminus1={wlminus1:.6f} "
                    f"wave={debug_wave_val:.6f} before={debug_before:.6e} after={debug_after:.6e}"
                )
            if sum_before is not None and sum_wave_idx is not None:
                sum_after = ahline[depth_idx, sum_wave_idx]
                contrib = sum_after - sum_before
                if contrib > 0.0:
                    sum_entries.append(
                        (contrib, float(line_wavelength), int(n_lower), int(n_upper))
                    )

    if sum_wave_idx is not None and sum_depth_idx is not None and sum_entries:
        sum_entries.sort(reverse=True, key=lambda item: item[0])
        target_wave = float(wavelength_grid[sum_wave_idx])
        print(
            f"PY_DEBUG_HLINE_SUM: wave={target_wave:.6f} depth={sum_depth_idx + 1} "
            f"lines={len(sum_entries)}"
        )
        total = 0.0
        for contrib, wl_line, n_lower, n_upper in sum_entries[:sum_top_n]:
            total += contrib
            print(f"  line={wl_line:.6f} n={n_lower}->{n_upper} contrib={contrib:.6e}")
        print(f"  top_sum={total:.6e}")

    return ahline


def _process_metal_wings_depth_standalone(
    args_tuple: Tuple,
) -> Tuple[
    int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, Optional[int], list
]:
    """Standalone function to process metal line wings for a single depth (for multiprocessing).

    Returns (depth_idx, local_wings, local_sources, helium_wings, helium_sources,
    lines_processed, lines_skipped, debug_idx, debug_hits)
    """
    (
        depth_idx,
        state_dict,  # Serialized DepthState data
        continuum_row,
        wavelength,
        line_indices,
        catalog_wavelength,
        catalog_gf,
        catalog_gamma_rad,
        catalog_gamma_stark,
        catalog_gamma_vdw,
        catalog_records,
        metadata_dict,  # Serialized metadata or None
        fort19_data_dict,  # Serialized fort19 data or None
        catalog_to_meta,
        catalog_to_fort19,
        electron_density_val,
        emerge_val,
        emerge_h_val,
        xnf_h_val,
        xnf_he1_val,
        xnf_h2_val,
        temperature_val,
        mass_density_val,
        txnxn_val,
        boltzmann_factor,
        metal_tables_dict,
        population_cache_dict,
        ifvac,
        cutoff,
        bnu_row,
        skip_helium,
        skip_metals,
        debug_wave_val,
        debug_depth_val,
        debug_top_n,
        debug_line_val,
        debug_line_eps,
        debug_line_filter,
    ) = args_tuple

    # Use values directly (no need to reconstruct DepthState object)
    # We only need: txnxn_val, electron_density_val, boltzmann_factor

    # Reconstruct metadata if available
    metadata = None
    if metadata_dict is not None:
        # Reconstruct from dict (simplified - may need full reconstruction)
        metadata = metadata_dict

    # Reconstruct fort19 data if available
    fort19_data = None
    if fort19_data_dict is not None:
        fort19_data = fort19_data_dict

    # Reconstruct metal tables
    metal_tables = metal_tables_dict

    # Reconstruct population cache
    population_cache = population_cache_dict

    tmp_buffer = np.zeros_like(wavelength, dtype=np.float64)
    local_wings = np.zeros_like(wavelength, dtype=np.float64)
    local_sources = np.zeros_like(wavelength, dtype=np.float64)
    local_helium_wings = np.zeros_like(wavelength, dtype=np.float64)
    local_helium_sources = np.zeros_like(wavelength, dtype=np.float64)

    debug_idx = None
    debug_hits = []
    if debug_wave_val is not None and debug_depth_val is not None:
        if debug_depth_val == depth_idx + 1:
            debug_idx = int(np.argmin(np.abs(wavelength - debug_wave_val)))

    lines_processed = 0
    lines_skipped = 0

    # Pre-compute arrays to reduce Python overhead in loop
    # Convert catalog arrays to numpy for faster access
    n_lines = len(line_indices)
    valid_line_mask = np.zeros(n_lines, dtype=bool)
    line_wavelengths_array = np.asarray(catalog_wavelength, dtype=np.float64)
    line_gf_array = np.asarray(catalog_gf, dtype=np.float64)
    line_gamma_rad_array = np.asarray(catalog_gamma_rad, dtype=np.float64)
    line_gamma_stark_array = np.asarray(catalog_gamma_stark, dtype=np.float64)
    line_gamma_vdw_array = np.asarray(catalog_gamma_vdw, dtype=np.float64)
    line_indices_array = np.asarray(line_indices, dtype=np.int64)

    # Pre-compute metadata arrays if available
    meta_ncon = None
    meta_nelionx = None
    meta_nelion = None
    meta_alpha = None
    meta_gamma_rad = None
    meta_gamma_stark = None
    meta_gamma_vdw = None
    if metadata is not None:
        meta_ncon = (
            np.asarray(metadata.ncon, dtype=np.int32)
            if hasattr(metadata, "ncon")
            else None
        )
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
        meta_gamma_rad = (
            np.asarray(metadata.gamma_rad, dtype=np.float64)
            if hasattr(metadata, "gamma_rad")
            else None
        )
        meta_gamma_stark = (
            np.asarray(metadata.gamma_stark, dtype=np.float64)
            if hasattr(metadata, "gamma_stark")
            else None
        )
        meta_gamma_vdw = (
            np.asarray(metadata.gamma_vdw, dtype=np.float64)
            if hasattr(metadata, "gamma_vdw")
            else None
        )

    # Pre-compute catalog_to_meta mapping as array
    meta_idx_array = np.full(n_lines, -1, dtype=np.int32)
    if metadata is not None and catalog_to_meta:
        for line_idx, meta_idx in catalog_to_meta.items():
            if line_idx < n_lines:
                meta_idx_array[line_idx] = meta_idx

    # Pre-compute element keys and filter valid lines
    element_keys_array = []
    nelion_array = np.zeros(n_lines, dtype=np.int32)
    for line_idx in range(n_lines):
        if line_idx < len(catalog_records):
            record = catalog_records[line_idx]
            element_key = str(record.element).strip()
            element_keys_array.append(element_key)
            nelion_array[line_idx] = record.ion_stage
        else:
            element_keys_array.append("")

    # Get Voigt tables once (for JIT kernel)
    voigt_tables = tables.voigt_tables()
    h0tab = voigt_tables.h0tab
    h1tab = voigt_tables.h1tab
    h2tab = voigt_tables.h2tab

    for line_idx, center_index in enumerate(line_indices):
        if (
            debug_line_filter
            and debug_line_val is not None
            and abs(line_wavelengths_array[line_idx] - debug_line_val) > debug_line_eps
        ):
            continue
        # CRITICAL FIX: Match Fortran synthe.for line 301 EXACTLY
        # IF(NBUFF.LT.1.OR.NBUFF.GT.LENGTH)GO TO 320
        # Fortran SKIPS lines whose centers are outside the grid - NO wing computation!
        # Previous Python code computed wings for margin lines, causing ~60x too deep
        # absorption at grid boundaries (first few wavelength points).
        center_outside = center_index < 0 or center_index >= wavelength.size
        if center_outside:
            lines_skipped += 1
            continue

        # Use pre-computed arrays instead of dictionary lookups
        meta_idx_val = meta_idx_array[line_idx] if meta_idx_array[line_idx] >= 0 else -1
        record = catalog_records[line_idx]

        # Skip hydrogen lines (faster check using pre-computed nelion)
        if nelion_array[line_idx] == 1:
            element_key = element_keys_array[line_idx]
            if element_key.upper() in {"H", "H I", "HI"}:
                lines_skipped += 1
                continue

        tmp_buffer.fill(0.0)
        line19_idx = None
        line_type_code = 0
        wing_type = fort19_io.Fort19WingType.NORMAL
        if fort19_data is not None:
            line19_idx = catalog_to_fort19.get(line_idx)
            if line19_idx is not None:
                line_type_code = int(fort19_data.line_type[line19_idx])
                wing_val = fort19_data.wing_type[line19_idx]
                if isinstance(wing_val, fort19_io.Fort19WingType):
                    wing_type = wing_val
                else:
                    wing_type = fort19_io.Fort19WingType.from_code(int(wing_val))

        record_line_type = int(getattr(record, "line_type", 0) or 0)
        is_helium_line = line_type_code in (-3, -4, -6) or record_line_type in (
            -3,
            -4,
            -6,
        )
        if skip_helium and is_helium_line:
            lines_skipped += 1
            continue
        if skip_metals and not is_helium_line:
            lines_skipped += 1
            continue
        # ========== CORONAL LINE SKIP (Fortran line 793) ==========
        # TYPE = 2 (Coronal lines): Skip entirely - GO TO 900
        if line_type_code == 2:
            lines_skipped += 1
            continue

        # Get metadata values using pre-computed arrays
        if meta_idx_val >= 0:
            ncon = meta_ncon[meta_idx_val] if meta_ncon is not None else 0
            nelionx = meta_nelionx[meta_idx_val] if meta_nelionx is not None else 0
            nelion = (
                meta_nelion[meta_idx_val]
                if meta_nelion is not None
                else nelion_array[line_idx]
            )
            alpha = meta_alpha[meta_idx_val] if meta_alpha is not None else 0.0
        else:
            ncon = 0
            nelionx = 0
            nelion = nelion_array[line_idx]
            alpha = 0.0
        # When fort.9 metadata is disabled, still honor fort.19 continuum indices
        # for special-profile lines (Fortran XLINOP uses NCON/NELIONX from fort.19).
        if fort19_data is not None and line19_idx is not None:
            ncon = int(fort19_data.continuum_index[line19_idx])
            nelionx = int(fort19_data.element_index[line19_idx])
        txnxn_line = txnxn_val

        # Compute TXNXN with alpha correction if needed
        if np.isfinite(alpha) and abs(alpha) > 1e-8:
            atomic_mass = _atomic_mass_lookup(record.element)
            if atomic_mass is not None and atomic_mass > 0.0:
                t_j = temperature_val
                v2 = 0.5 * (1.0 - alpha)
                h_factor = (t_j / 10000.0) ** v2
                # Fortran synthe.for line 467-468: 1/4 and 1/2 are INTEGER
                # division → both evaluate to 0, leaving only 1.008/ATMASS.
                he_factor = 0.628 * (2.0991e-4 * t_j * (1.008 / atomic_mass)) ** v2
                h2_factor = 1.08 * (2.0991e-4 * t_j * (1.008 / atomic_mass)) ** v2
                txnxn_line = (
                    xnf_h_val * h_factor
                    + xnf_he1_val * he_factor
                    + xnf_h2_val * h2_factor
                )

        # Use pre-computed arrays
        line_wavelength = line_wavelengths_array[line_idx]
        element_key = element_keys_array[line_idx]

        # Get populations from cache
        if element_key not in population_cache:
            lines_skipped += 1
            continue
        pop_densities, dop_velocity = population_cache[element_key]

        if nelion > pop_densities.shape[1]:
            lines_skipped += 1
            continue

        pop_val = pop_densities[depth_idx, nelion - 1]
        if dop_velocity.ndim == 2:
            dop_val = (
                dop_velocity[depth_idx, nelion - 1]
                if nelion <= dop_velocity.shape[1]
                else dop_velocity[depth_idx, 0]
            )
        else:
            dop_val = dop_velocity[depth_idx]

        if pop_val <= 0.0 or dop_val <= 0.0:
            lines_skipped += 1
            continue

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
        rho = mass_density_val if mass_density_val > 0.0 else 1.0
        if rho > 0.0:
            xnfdop = pop_val / (rho * dop_val)
            doppler_width = dop_val * line_wavelength
            boltz = boltzmann_factor[line_idx]

            # CRITICAL FIX: Convert GF to CONGF by dividing by frequency
            # From rgfall.for line 267: CGF = 0.026538/1.77245 * GF / FRELIN
            # Where FRELIN = 2.99792458D17 / WLVAC (frequency in Hz)
            freq_hz = C_LIGHT_NM / line_wavelength  # Frequency in Hz
            gf_linear = line_gf_array[line_idx]  # Linear gf (use pre-computed array)
            cgf = CGF_CONSTANT * gf_linear / freq_hz  # CONGF conversion

            # ========== DOUBLE KAPMIN CHECK (Fortran lines 266-272) ==========
            # Clamp center_index for continuum access
            clamped_idx = max(0, min(center_index, wavelength.size - 1))
            kappa_min = continuum_row[clamped_idx] * cutoff

            # First: KAPPA0 = CONGF * XNFDOP (BEFORE Boltzmann)
            kappa0_pre = cgf * xnfdop

            # First check (Fortran line 267): IF(KAPPA0.LT.KAPMIN)GO TO 350
            if kappa0_pre < kappa_min:
                lines_skipped += 1
                continue

            # Apply Boltzmann factor (Fortran line 270)
            kappa0 = kappa0_pre * boltz

            # Second check (Fortran line 272): IF(KAPPA0.LT.KAPMIN)GO TO 350
            # RE-ENABLED: This matches Fortran behavior and prevents weak line accumulation
            if kappa0 < kappa_min:
                lines_skipped += 1
                continue

        # Post-Boltzmann cutoff already applied at line 1575
        # Only check doppler_width validity here
        if doppler_width <= 0.0:
            lines_skipped += 1
            continue

        population_lower = pop_val

        # Compute continuum limits
        wcon, wtail = _compute_continuum_limits(
            ncon=ncon,
            nelion=nelion,
            nelionx=nelionx,
            emerge_val=emerge_val,
            emerge_h_val=emerge_h_val,
            metal_tables=metal_tables,
            ifvac=ifvac,
        )

        # ALWAYS use catalog gamma values (LINEAR, not normalized)
        # NO fort.9 metadata dependency for gamma values!
        gamma_rad = line_gamma_rad_array[line_idx]
        gamma_stark = line_gamma_stark_array[line_idx]
        gamma_vdw = line_gamma_vdw_array[line_idx]
        line_doppler = doppler_width

        # ========== TYPE=1 AUTOIONIZING / STARK (Fortran synthe.for label 700) ==========
        # KAPPA0 = BSHORE*G*XNFPEL*exp(-ELO*HCKT); KAPPA(ν) = KAPPA0*(ASHORE*ε+BSHORE)/(ε²+1)/BSHORE, ε=2(ν-ν0)/GAMMAR
        if line_type_code == 1:
            if rho <= 0.0:
                lines_skipped += 1
                continue
            boltz = boltzmann_factor[line_idx]
            gf_linear = line_gf_array[line_idx]
            clamped_idx = max(0, min(center_index, wavelength.size - 1))
            kappa_min_auto = continuum_row[clamped_idx] * cutoff
            # Fortran 1078: KAPPA0=BSHORE*G*XNFPEL(NELION); 1080: KAPPA0=KAPPA0*FASTEX(ELO*HCKT(J))
            # BSHORE=GAMMAW, G=GF. XNFPEL is population (per unit mass in Fortran).
            xnfpel = pop_val / rho
            kappa0_auto = gamma_vdw * gf_linear * xnfpel * boltz
            if kappa0_auto < kappa_min_auto:
                lines_skipped += 1
                continue
            bshore_safe = max(gamma_vdw, 1e-30)
            gamrad_safe = max(gamma_rad, 1e-30)
            frelin_hz = 2.99792458e17 / line_wavelength
            for i in range(wavelength.size):
                freq_hz_i = 2.99792458e17 / wavelength[i]
                epsil = 2.0 * (freq_hz_i - frelin_hz) / gamrad_safe
                kappa_auto = (
                    kappa0_auto
                    * (gamma_stark * epsil + gamma_vdw)
                    / (epsil * epsil + 1.0)
                    / bshore_safe
                )
                if kappa_auto >= continuum_row[i] * cutoff:
                    tmp_buffer[i] += kappa_auto
            lines_processed += 1
            if _xlinop_debug_path is not None and depth_idx in _XLINOP_DEBUG_DEPTHS:
                for idx in range(wavelength.size):
                    if not any(
                        abs(wavelength[idx] - tw) <= _XLINOP_DEBUG_TARGET_TOL
                        for tw in _XLINOP_DEBUG_TARGET_WAVES
                    ):
                        continue
                    kapmin_bin = float(continuum_row[idx] * cutoff)
                    if tmp_buffer[idx] > 0.0:
                        with open(_xlinop_debug_path, "a") as _f:
                            _f.write(
                                f"DEBUG XLINOP TARGET TYPE1: J={depth_idx + 1:3d} ILINE={line_idx:7d} TYPE=   1 "
                                f"WL={line_wavelength:10.4f} WAVE={wavelength[idx]:10.6f} "
                                f"KAPPA={tmp_buffer[idx]:12.4e} KAPMIN={kapmin_bin:12.4e}\n"
                            )
            continue

        # Handle special wing types (simplified - he_solver not available in standalone)
        if (
            wing_type == fort19_io.Fort19WingType.AUTOIONIZING
            and fort19_data is not None
        ):
            population = (
                population_lower
                if (population_lower is not None and population_lower > 0.0)
                else None
            )
            if population is None:
                population = boltz
            gf_value = catalog_gf[line_idx]
            kappa_auto = gamma_vdw * gf_value * population * boltz
            if kappa_auto < kappa_min:
                lines_skipped += 1
                continue
            # Skip fort19 profile application in standalone (he_solver not available)
            # This is a limitation but should be rare
            lines_skipped += 1
            continue

        # Compute damping value (ADAMP in Fortran synthe.for line 473)
        # Fortran: ADAMP = (GAMRF + GAMSF*XNE + GAMWF*TXNXN) / DOPPLE(NELION)
        # Where GAMRF etc. are NORMALIZED gamma values (divided by 4πν in rgfall.for)
        # and DOPPLE is dimensionless (v_th/c).
        #
        # For tfort.12 catalogs, gamma values are PRE-NORMALIZED by rgfall.for.
        # The Fortran-exact formula is simply: ADAMP = gamma_total / DOPPLE.
        dopple = doppler_width / line_wavelength if line_wavelength > 0 else dop_val
        gamma_total = (
            gamma_rad + gamma_stark * electron_density_val + gamma_vdw * txnxn_line
        )
        damping_value = gamma_total / max(dopple, 1e-40)

        # Apply fort.19 profile if available (simplified - he_solver not available)
        if fort19_data is not None and line19_idx is not None:
            # Skip fort19 profiles in standalone mode (he_solver limitation)
            pass

        # Use JIT kernel directly (bypass wrapper)
        wcon_val = wcon if wcon is not None else -1.0
        wtail_val = wtail if wtail is not None else -1.0
        _accumulate_metal_profile_kernel(
            tmp_buffer,
            continuum_row,
            wavelength,
            center_index,
            line_wavelength,
            kappa0,
            max(damping_value, 1e-12),
            line_doppler,
            cutoff,
            wcon_val,
            wtail_val,
            h0tab,
            h1tab,
            h2tab,
        )

        # Reset center (but only if center_index is within grid)
        # For lines outside grid, wings are still added but no center to reset
        if 0 <= center_index < wavelength.size:
            tmp_buffer[center_index] = 0.0
        local_wings += tmp_buffer
        local_sources += tmp_buffer * bnu_row
        if debug_idx is not None and 0 <= debug_idx < tmp_buffer.size:
            contrib = float(tmp_buffer[debug_idx])
            if contrib > 0.0:
                delta_nm = float(wavelength[debug_idx] - line_wavelength)
                local_kapmin = float(continuum_row[debug_idx] * cutoff)
                profile_val = contrib / float(kappa0) if kappa0 > 0.0 else 0.0
                resolu_val = None
                v_val = None
                if 0 <= center_index < wavelength.size and line_wavelength > 0.0:
                    if center_index < wavelength.size - 1:
                        ratio = wavelength[center_index + 1] / wavelength[center_index]
                    elif center_index > 0:
                        ratio = wavelength[center_index] / wavelength[center_index - 1]
                    else:
                        ratio = None
                    if ratio is not None:
                        resolu_val = 1.0 / (ratio - 1.0)
                        dopple_val = line_doppler / line_wavelength
                        if dopple_val > 0.0:
                            v_val = abs(debug_idx - center_index) / (
                                dopple_val * resolu_val
                            )
                debug_hits.append(
                    (
                        contrib,
                        float(line_wavelength),
                        int(line_idx),
                        str(record.element).strip(),
                        int(record.ion_stage),
                        float(kappa0),
                        float(damping_value),
                        float(line_doppler),
                        int(line_type_code),
                        delta_nm,
                        local_kapmin,
                        profile_val,
                        None if wcon_val < 0.0 else float(wcon_val),
                        None if wtail_val < 0.0 else float(wtail_val),
                        bool(center_outside),
                        int(center_index),
                        int(debug_idx),
                        resolu_val,
                        v_val,
                    )
                )
        lines_processed += 1

    return (
        depth_idx,
        local_wings,
        local_sources,
        local_helium_wings,
        local_helium_sources,
        lines_processed,
        lines_skipped,
        debug_idx,
        debug_hits,
    )


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

        print("\n" + "=" * 70)
        print("DEBUG: Mass density (RHO) values at different depths")
        print("=" * 70)
        sample_depths = [
            0,
            atm.layers // 4,
            atm.layers // 2,
            3 * atm.layers // 4,
            atm.layers - 1,
        ]
        for depth_idx in sample_depths:
            if depth_idx < atm.layers:
                rho_val = float(atm.mass_density[depth_idx])
                temp_val = float(atm.temperature[depth_idx])
                press_val = (
                    float(atm.gas_pressure[depth_idx])
                    if hasattr(atm, "gas_pressure")
                    else 0.0
                )
                xne_val = (
                    float(atm.electron_density[depth_idx])
                    if hasattr(atm, "electron_density")
                    else 0.0
                )
                print(
                    f"  Depth {depth_idx}: RHO={rho_val:.6e} g/cm³, T={temp_val:.2f} K, P={press_val:.6e}, XNE={xne_val:.6e}"
                )
                # Compare with Fortran: XNATOM = P/TK - XNE, where TK = k_B * T
                if press_val > 0:
                    tk_val = K_BOLTZ * temp_val
                    xnatom_approx = press_val / tk_val - xne_val
                    print(
                        f"    XNATOM (approx from P/TK-XNE): {xnatom_approx:.6e} cm⁻³"
                    )
        print("=" * 70 + "\n")
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

    # Fortran-generated metadata inputs are only required for explicit tfort.* runs.
    catalog_path = Path(cfg.line_data.atomic_catalog)
    suffix = catalog_path.suffix.lower()
    is_tfort_catalog = suffix in {".12", ".14"} and "fort.1" in catalog_path.name
    if is_tfort_catalog:
        if not cfg.line_data.allow_tfort_runtime:
            raise RuntimeError(
                "tfort.* runtime input is disabled by default. "
                "Use gfallvac.latest for self-contained runtime or pass --allow-tfort-runtime for compatibility mode."
            )
        logger.info("Fortran metadata inputs enabled: tfort.19/tfort.93")
    else:
        logger.info(
            "Using self-contained Python line compiler metadata (no tfort runtime inputs)"
        )
    logger.info("Allocating buffers...")
    buffers = allocate_buffers(wavelength, atm.layers)

    logger.info("Loading line catalog...")
    t_stage = time.perf_counter()
    if is_tfort_catalog:
        catalog = _load_line_data(cfg, wavelength.min(), wavelength.max())
        tfort19_path = catalog_path.with_suffix(".19")
        tfort19_npz = catalog_path.parent / "tfort19.npz"
        if tfort19_npz.exists():
            logger.info("Loading tfort.19 metadata from: %s", tfort19_npz)
            fort19_data = fort19_io.load(tfort19_npz)
        elif tfort19_path.exists():
            logger.info("Loading tfort.19 metadata from: %s", tfort19_path)
            fort19_data = fort19_io.load(tfort19_path)
        else:
            raise RuntimeError(
                "tfort.12/14 requires companion tfort.19 for exact Fortran behavior."
            )
        if suffix == ".12" and fort19_data is not None:
            fort19_catalog = _catalog_from_fort19(fort19_data)
            extra_records = [
                rec
                for rec in fort19_catalog.records
                if rec.line_type < 0 or rec.line_type == 1 or rec.line_type > 3
            ]
            if extra_records:
                catalog = atomic.LineCatalog.from_records(
                    list(catalog.records) + extra_records
                )
                logger.info(
                    "Augmented tfort.12 catalog with %d H/He/auto/merged lines from tfort.19",
                    len(extra_records),
                )
    else:
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
    if is_tfort_catalog:
        tfort93_path = catalog_path.with_suffix(".93")
        if not tfort93_path.exists():
            raise RuntimeError(
                "tfort.12/14 requires companion tfort.93 for exact Fortran KAPMIN grid."
            )
        t93 = tfort.parse_tfort93(tfort93_path)
        wavelength_full = []
        wl = t93.wlbeg
        while wl <= t93.wlend * (1.0 + 1e-9):
            wavelength_full.append(wl)
            wl *= t93.ratio
        wavelength_full = np.array(wavelength_full, dtype=np.float64)
        t_full = time.perf_counter()
        cont_abs_full, cont_scat_full, _, _ = continuum.build_depth_continuum(
            atm, wavelength_full
        )
        cont_kapmin_full = cont_abs_full + cont_scat_full
        if not np.any(cont_kapmin_full):
            if atm.continuum_abs_coeff is not None and atm.continuum_wledge is not None:
                ablog = np.asarray(atm.continuum_abs_coeff[0], dtype=np.float64).T
                cont_tables = tables.build_continuum_tables(
                    tuple(float(x) for x in atm.continuum_wledge.tolist()),
                    tuple(float(x) for x in ablog.ravel().tolist()),
                )
                log_cont = continuum.interpolate_continuum(cont_tables, wavelength_full)
                cont = continuum.finalize_continuum(log_cont)
                cont_kapmin_full = np.tile(cont, (atm.layers, 1))
            else:
                cont_kapmin_full = None
        logger.info(
            "Timing: full-grid KAPMIN in %.3fs (%d points)",
            time.perf_counter() - t_full,
            wavelength_full.size if wavelength_full is not None else 0,
        )
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

    # CRITICAL DEBUG: Check has_lines value
    if cfg.debug:
        print("\n" + "=" * 70)
        print("CRITICAL DEBUG: Before line opacity computation")
        print("=" * 70)
        print(f"has_lines: {has_lines}")
        print(f"catalog.records length: {len(catalog.records)}")
        print(f"line_indices length: {len(line_indices)}")
        if len(line_indices) > 0:
            print(f"First line index: {line_indices[0]}")
        print("=" * 70 + "\n")

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

        # CRITICAL DEBUG: Check if asynth is all zeros
        if cfg.debug:
            print("\n" + "=" * 70)
            print("CRITICAL DEBUG: After computing ASYNTH")
            print("=" * 70)
            print(f"asynth shape: {asynth.shape}")
            print(f"asynth non-zero count: {np.count_nonzero(asynth)}")
            print(f"asynth max: {np.max(asynth):.2e}")
            print(f"asynth min: {np.min(asynth):.2e}")
            if np.all(asynth == 0.0):
                print("ERROR: asynth is ALL ZEROS!")
            else:
                print("asynth is NOT all zeros")
            print("=" * 70 + "\n")

        # Apply scattering factor
        rhox = atm.depth
        rhox_scale = cfg.rhoxj_scale
        if rhox_scale <= 0.0 and spectrv_params is not None:
            rhox_scale = spectrv_params.rhoxj
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

        # CRITICAL DEBUG: Check fscat values
        if cfg.debug:
            print("\n" + "=" * 70)
            print("CRITICAL DEBUG: After computing absorption from ASYNTH")
            print("=" * 70)
            print(f"fscat shape: {fscat.shape}")
            print(f"fscat min: {np.min(fscat):.2e}, max: {np.max(fscat):.2e}")
            print(f"fscat non-zero count: {np.count_nonzero(fscat)}")
            print(
                f"(1.0 - fscat) min: {np.min(1.0 - fscat):.2e}, max: {np.max(1.0 - fscat):.2e}"
            )
            print(f"absorption shape: {absorption.shape}")
            print(f"absorption non-zero count: {np.count_nonzero(absorption)}")
            print(f"absorption max: {np.max(absorption):.2e}")
            if np.all(absorption == 0.0):
                print("ERROR: absorption is ALL ZEROS!")
                if np.all(asynth == 0.0):
                    print("  Reason: asynth is all zeros")
                elif np.all(fscat == 1.0):
                    print("  Reason: fscat is all 1.0, so (1.0 - fscat) = 0")
                else:
                    print(
                        "  Reason: Unknown - asynth and fscat both have non-zero values"
                    )
            else:
                print("absorption is NOT all zeros")
            print("=" * 70 + "\n")

        buffers.line_opacity[:] = absorption
        # Mark that we're using ASYNTH (computed from catalog)
        buffers._using_asynth = True
        debug_metal_wave = os.getenv("PY_DEBUG_METAL_WING_WAVE")
        debug_metal_depth = os.getenv("PY_DEBUG_METAL_WING_DEPTH")
        if debug_metal_wave and debug_metal_depth:
            try:
                debug_wave_val = float(debug_metal_wave)
                depth_val = int(debug_metal_depth)
                if depth_val > 0:
                    depth_idx = depth_val - 1
                    wl_idx = int(np.argmin(np.abs(wavelength - debug_wave_val)))
                    print(
                        "PY_DEBUG_LINE_OPACITY_INIT: "
                        f"wave={wavelength[wl_idx]:.6f} depth={depth_val} "
                        f"line_opacity={buffers.line_opacity[depth_idx, wl_idx]:.6e}"
                    )
            except ValueError:
                pass
        # After line 971: buffers.line_opacity[:] = absorption
        logger.info(
            f"Line opacity after ASYNTH: shape {buffers.line_opacity.shape}, "
            f"non-zero count: {np.count_nonzero(buffers.line_opacity)}, "
            f"max: {np.max(buffers.line_opacity):.2e}, "
            f"min (non-zero): {float(np.min(buffers.line_opacity[buffers.line_opacity > 0])) if np.any(buffers.line_opacity > 0) else 0:.2e}"
        )
        # CRITICAL DEBUG: Check if absorption is all zeros and analyze large values
        if cfg.debug:
            print("\n" + "=" * 70)
            print(
                "CRITICAL DEBUG: After setting buffers.line_opacity = absorption (from TRANSP/ASYNTH)"
            )
            print("=" * 70)
            print(f"absorption shape: {absorption.shape}")
            print(f"absorption non-zero count: {np.count_nonzero(absorption)}")
            print(f"absorption max: {np.max(absorption):.2e}")
            absorption_nonzero = (
                absorption[absorption > 0]
                if np.any(absorption > 0)
                else np.array([0.0])
            )
            print(f"absorption min (non-zero): {float(np.min(absorption_nonzero)):.2e}")
            print(
                f"absorption mean (non-zero): {float(np.mean(absorption_nonzero)):.2e}"
            )
            print(
                f"absorption median (non-zero): {float(np.median(absorption_nonzero)):.2e}"
            )
            print(f"Values > 1e10: {np.sum(absorption > 1e10):,}")
            print(f"Values > 1e20: {np.sum(absorption > 1e20):,}")
            print(f"Values > 1e24: {np.sum(absorption > 1e24):,}")
        # CRITICAL FIX: Match Fortran exactly - no clamping of line opacity
        # Fortran does NOT clamp ASYNTH/ALINE values - it uses them directly
        # Remove clamping to match Fortran behavior exactly
        if cfg.debug and np.any(absorption > 1e10):
            n_overflow = np.sum(absorption > 1e10)
            print(
                f"  WARNING: {n_overflow:,} values exceed 1e10 (for diagnostic only, not clamping)"
            )

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

        if cfg.debug:
            print(
                f"buffers.line_opacity non-zero count: {np.count_nonzero(buffers.line_opacity)}"
            )
            print(f"buffers.line_opacity max: {np.max(buffers.line_opacity):.2e}")
            if np.all(absorption == 0.0):
                print("ERROR: absorption is ALL ZEROS!")
            else:
                print("absorption is NOT all zeros")
            print("=" * 70 + "\n")

        # Assign absorption to line_opacity (with overflow protection applied)
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

    # CRITICAL DEBUG: Check buffers.line_opacity before copying to abs_core_base
    if cfg.debug:
        print("\n" + "=" * 70)
        print("CRITICAL DEBUG: Before abs_core_base = buffers.line_opacity.copy()")
        print("=" * 70)
        print(f"buffers.line_opacity shape: {buffers.line_opacity.shape}")
        print(
            f"buffers.line_opacity non-zero count: {np.count_nonzero(buffers.line_opacity)}"
        )
        print(f"buffers.line_opacity max: {np.max(buffers.line_opacity):.2e}")
        print(
            f"buffers.line_opacity min (non-zero): {float(np.min(buffers.line_opacity[buffers.line_opacity > 0])) if np.any(buffers.line_opacity > 0) else 0:.2e}"
        )
        # Check first few wavelengths specifically
        print(f"\nFirst 10 wavelengths (first 5 depths):")
        for wl_idx in range(min(10, wavelength.size)):
            wl = wavelength[wl_idx]
            line_op_wl = buffers.line_opacity[:, wl_idx]
            non_zero_depths = np.sum(line_op_wl > 0)
            max_val = np.max(line_op_wl)
            print(
                f"  Wavelength {wl:.8f} nm (idx {wl_idx}): max={max_val:.2e}, non-zero depths={non_zero_depths}/{atm.layers}"
            )
            if max_val > 0:
                print(
                    f"    Surface value: {line_op_wl[0]:.2e}, Deep value: {line_op_wl[-1]:.2e}"
                )
        print("=" * 70 + "\n")

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

    # Optional debug: report hydrogen vs metal contributions at a target wavelength.
    debug_wave = os.getenv("PY_DEBUG_WAVE")
    if debug_wave:
        try:
            target_wave = float(debug_wave)
        except ValueError:
            target_wave = None
        if target_wave is not None and wavelength.size > 0:
            idx_target = int(np.argmin(np.abs(wavelength - target_wave)))
            for depth_idx in (0, 19, 39, 59, 79):
                if depth_idx < ahline.shape[0]:
                    print(
                        f"PY_DEBUG_COMPONENTS depth {depth_idx + 1}: "
                        f"AHLINE={float(ahline[depth_idx, idx_target]):.6e} "
                        f"ASYNTH_METAL={float(buffers.line_opacity[depth_idx, idx_target]):.6e}"
                    )

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

        # Optional XLINOP target debug (Fortran 9060: J, ILINE, TYPE, WL, WAVE, KAPPA, KAPMIN, NELION, NCON, ELO, GF, ADAMP)
        _xlinop_debug_path = None
        logger.info("XLINOP target debug enabled: %s", _xlinop_debug_path)
        if os.getenv("PY_DEBUG_XLINOP_TARGET", "") == "1":
            _stem = cfg.atmosphere.model_path.stem.replace(".npz", "").replace(".atm", "")
            _out_dir = Path("results/validation_100")
            _out_dir.mkdir(parents=True, exist_ok=True)
            _xlinop_debug_path = _out_dir / f"debug_xlinop_target_{_stem}.txt"
            with open(_xlinop_debug_path, "w") as _f:
                _f.write(
                    "# PY_DEBUG_XLINOP_TARGET: J ILINE TYPE WL WAVE KAPPA KAPMIN NELION NCON ELO GF ADAMP (matches Fortran 9060)\n"
                )

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

        # Prepare arguments for standalone processing (shared by metal/helium parallel paths)
        debug_wave_val = None
        debug_depth_val = None
        debug_top_n = 20
        debug_line_val = None
        debug_line_eps = 1e-3
        debug_line_filter = os.getenv("PY_DEBUG_METAL_WING_FILTER") == "1"
        debug_wave = os.getenv("PY_DEBUG_METAL_WING_WAVE")
        debug_depth = os.getenv("PY_DEBUG_METAL_WING_DEPTH")
        debug_top = os.getenv("PY_DEBUG_METAL_WING_TOP")
        debug_line = os.getenv("PY_DEBUG_METAL_WING_LINE")
        debug_line_tol = os.getenv("PY_DEBUG_METAL_WING_LINE_TOL")
        if debug_wave and debug_depth:
            try:
                debug_wave_val = float(debug_wave)
                debug_depth_val = int(debug_depth)
            except ValueError:
                debug_wave_val = None
                debug_depth_val = None
        if debug_top:
            try:
                debug_top_n = max(1, int(debug_top))
            except ValueError:
                debug_top_n = 20
        if debug_line:
            try:
                debug_line_val = float(debug_line)
            except ValueError:
                debug_line_val = None
        if debug_line_tol:
            try:
                debug_line_eps = float(debug_line_tol)
            except ValueError:
                debug_line_eps = 1e-3
        step1_wing_hoist = os.getenv("PY_OPT_STEP1_WING_HOIST", "1") != "0"
        record_elements_upper = [str(r.element).strip().upper() for r in catalog.records]
        record_ion_stage = np.asarray([int(r.ion_stage) for r in catalog.records], dtype=np.int16)
        record_line_type = np.asarray(
            [int(getattr(r, "line_type", 0) or 0) for r in catalog.records],
            dtype=np.int16,
        )

        def _build_process_args(skip_helium_flag: bool, skip_metals_flag: bool) -> list:
            args = []
            for depth_idx in pops.layers.keys():
                state = pops.layers[depth_idx]
                continuum_row = buffers.continuum[depth_idx]
                bnu_row = bnu[depth_idx]

                # Extract scalar values for this depth
                args_tuple = (
                    depth_idx,
                    None,  # state_dict - will reconstruct from values
                    continuum_row,
                    wavelength,
                    line_indices,
                    catalog.wavelength,
                    catalog.gf,
                    catalog.gamma_rad,
                    catalog.gamma_stark,
                    catalog.gamma_vdw,
                    catalog.records,
                    metadata,  # May not be picklable - will handle
                    fort19_data,  # May not be picklable - will handle
                    catalog_to_meta,
                    catalog_to_fort19,
                    state.electron_density,
                    emerge[depth_idx],
                    emerge_h[depth_idx],
                    _layer_value(xnf_h_arr, depth_idx),
                    _layer_value(xnf_he1_arr, depth_idx),
                    _layer_value(xnf_h2_arr, depth_idx),
                    atm.temperature[depth_idx],
                    (
                        atm.mass_density[depth_idx]
                        if hasattr(atm, "mass_density") and atm.mass_density is not None
                        else 1.0
                    ),
                    state.txnxn,
                    state.boltzmann_factor,
                    metal_tables,
                    population_cache,
                    1,
                    cfg.cutoff,
                    bnu_row,
                    skip_helium_flag,
                    skip_metals_flag,
                    debug_wave_val,
                    debug_depth_val,
                    debug_top_n,
                    debug_line_val,
                    debug_line_eps,
                    debug_line_filter,
                )
                args.append(args_tuple)
            return args

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
            debug_hits = []
            debug_idx = None
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
            if step1_wing_hoist:
                if debug_wave_val is not None and debug_depth_val is not None:
                    if debug_depth_val == depth_idx + 1:
                        debug_idx = int(np.argmin(np.abs(wavelength - debug_wave_val)))
            else:
                debug_wave = os.getenv("PY_DEBUG_METAL_WING_WAVE")
                debug_depth = os.getenv("PY_DEBUG_METAL_WING_DEPTH")
                if debug_wave and debug_depth:
                    try:
                        debug_wave_local = float(debug_wave)
                        debug_depth_local = int(debug_depth)
                        if debug_depth_local == depth_idx + 1:
                            debug_idx = int(np.argmin(np.abs(wavelength - debug_wave_local)))
                    except ValueError:
                        debug_idx = None

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

            xlinop_target_indices = []
            if (
                _xlinop_debug_path is not None
                and depth_idx in _XLINOP_DEBUG_DEPTHS
                and include_metals
            ):
                xlinop_target_indices = [
                    i
                    for i in range(wavelength.size)
                    if any(
                        abs(wavelength[i] - tw) <= _XLINOP_DEBUG_TARGET_TOL
                        for tw in _XLINOP_DEBUG_TARGET_WAVES
                    )
                ]

            for line_idx in line_iter:
                center_index = line_indices[int(line_idx)]
                if (
                    debug_line_filter
                    and debug_line_val is not None
                    and abs(catalog_wavelength[line_idx] - debug_line_val)
                    > debug_line_eps
                ):
                    continue
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
                        if (
                            debug_line_val is not None
                            and debug_depth_val is not None
                            and debug_depth_val == depth_idx + 1
                            and abs(line_wavelength - debug_line_val) <= debug_line_eps
                        ):
                            print(
                                "PY_DEBUG_METAL_WING_SKIP_PRE: "
                                f"wl={line_wavelength:.6f} depth={depth_idx + 1} "
                                f"kappa0_pre={kappa0_pre:.6e} kappa_min={kappa_min:.6e} "
                                f"center_outside={center_outside}"
                            )
                        lines_skipped += 1
                        continue

                    # Apply Boltzmann factor
                    kappa0 = kappa0_pre * boltz

                    # Second check (Fortran line 272): post-Boltzmann cutoff
                    # RE-ENABLED: This matches Fortran behavior
                    if kappa0 < kappa_min:
                        if (
                            debug_line_val is not None
                            and debug_depth_val is not None
                            and debug_depth_val == depth_idx + 1
                            and abs(line_wavelength - debug_line_val) <= debug_line_eps
                        ):
                            print(
                                "PY_DEBUG_METAL_WING_SKIP_POST: "
                                f"wl={line_wavelength:.6f} depth={depth_idx + 1} "
                                f"kappa0={kappa0:.6e} kappa_min={kappa_min:.6e} "
                                f"center_outside={center_outside}"
                            )
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
                    if debug_idx is not None and 0 <= debug_idx < tmp_buffer.size:
                        contrib = float(tmp_buffer[debug_idx])
                        if contrib > 0.0:
                            debug_hits.append(
                                (
                                    contrib,
                                    float(line_wavelength),
                                    int(line_idx),
                                    element_symbol,
                                    int(record.ion_stage),
                                    float(kappa0),
                                    float(damping_value),
                                    float(line_doppler),
                                    int(line_type_code),
                                )
                            )
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
                buf_before_xlinop = None
                if (
                    _xlinop_debug_path is not None
                    and depth_idx in _XLINOP_DEBUG_DEPTHS
                    and xlinop_target_indices
                    and include_metals
                    and not helium_only
                ):
                    buf_before_xlinop = np.copy(tmp_buffer)
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
                if (
                    _xlinop_debug_path is not None
                    and depth_idx in _XLINOP_DEBUG_DEPTHS
                    and xlinop_target_indices
                    and buf_before_xlinop is not None
                ):
                    elo = float(getattr(record, "excitation_energy", 0.0))
                    gf_val = float(catalog.gf[line_idx])
                    for idx in xlinop_target_indices:
                        delta = float(tmp_buffer[idx] - buf_before_xlinop[idx])
                        if delta > 0.0:
                            wave_bin = float(wavelength[idx])
                            kapmin_bin = float(continuum_row[idx] * cfg.cutoff)
                            with open(_xlinop_debug_path, "a") as _f:
                                _f.write(
                                    f"DEBUG XLINOP TARGET: J={depth_idx + 1:3d} ILINE={line_idx:7d} TYPE={line_type_code:4d} "
                                    f"WL={line_wavelength:10.4f} WAVE={wave_bin:10.6f} "
                                    f"KAPPA={delta:12.4e} KAPMIN={kapmin_bin:12.4e} "
                                    f"NELION={record.ion_stage:4d} NCON={0:4d} "
                                    f"ELO={elo:12.4e} GF={gf_val:12.4e} ADAMP={max(damping_value, 1e-12):12.4e}\n"
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
                if debug_idx is not None and 0 <= debug_idx < tmp_buffer.size:
                    contrib = float(tmp_buffer[debug_idx])
                    if contrib > 0.0:
                        delta_nm = float(wavelength[debug_idx] - line_wavelength)
                        local_kapmin = float(continuum_row[debug_idx] * cfg.cutoff)
                        profile_val = contrib / float(kappa0) if kappa0 > 0.0 else 0.0
                        resolu_val = None
                        v_val = None
                        if (
                            0 <= center_index < wavelength.size
                            and line_wavelength > 0.0
                        ):
                            if center_index < wavelength.size - 1:
                                ratio = (
                                    wavelength[center_index + 1]
                                    / wavelength[center_index]
                                )
                            elif center_index > 0:
                                ratio = (
                                    wavelength[center_index]
                                    / wavelength[center_index - 1]
                                )
                            else:
                                ratio = None
                            if ratio is not None:
                                resolu_val = 1.0 / (ratio - 1.0)
                                dopple_val = line_doppler / line_wavelength
                                if dopple_val > 0.0:
                                    v_val = abs(debug_idx - center_index) / (
                                        dopple_val * resolu_val
                                    )
                        debug_hits.append(
                            (
                                contrib,
                                float(line_wavelength),
                                int(line_idx),
                                element_symbol,
                                int(record.ion_stage),
                                float(kappa0),
                                float(damping_value),
                                float(line_doppler),
                                int(line_type_code),
                                delta_nm,
                                local_kapmin,
                                profile_val,
                                float(wcon) if wcon is not None else None,
                                float(wtail) if wtail is not None else None,
                                bool(center_outside),
                                int(center_index),
                                int(debug_idx),
                                resolu_val,
                                v_val,
                            )
                        )
                lines_processed += 1

            if debug_idx is not None and debug_hits:
                debug_hits.sort(reverse=True, key=lambda item: item[0])
                target_wave = float(wavelength[debug_idx])
                print(
                    f"PY_DEBUG_METAL_WING_SUM: wave={target_wave:.6f} depth={depth_idx + 1} "
                    f"hits={len(debug_hits)}"
                )
                for item in debug_hits[:debug_top_n]:
                    if len(item) == 9:
                        (
                            contrib,
                            wl_line,
                            line_id,
                            elem,
                            ion_stage,
                            kappa0_val,
                            adamp_val,
                            dop_val,
                            line_type,
                        ) = item
                        print(
                            "  hit "
                            f"contrib={contrib:.6e} wl={wl_line:.6f} "
                            f"line={line_id} elem={elem} ion={ion_stage} "
                            f"kappa0={kappa0_val:.6e} adamp={adamp_val:.6e} "
                            f"doppler={dop_val:.6e} type={line_type} "
                            "delta=NA kapmin=NA profile=NA wcon=NA wtail=NA "
                            "center_outside=NA center_idx=NA hit_idx=NA istep=NA "
                            "resolu=NA v=NA"
                        )
                        continue
                    (
                        contrib,
                        wl_line,
                        line_id,
                        elem,
                        ion_stage,
                        kappa0_val,
                        adamp_val,
                        dop_val,
                        line_type,
                        delta_nm,
                        local_kapmin,
                        profile_val,
                        wcon_val,
                        wtail_val,
                        center_outside_val,
                        center_index_val,
                        debug_idx_val,
                        resolu_val,
                        v_val,
                    ) = item
                    print(
                        "  hit "
                        f"contrib={contrib:.6e} wl={wl_line:.6f} "
                        f"line={line_id} elem={elem} ion={ion_stage} "
                        f"kappa0={kappa0_val:.6e} adamp={adamp_val:.6e} "
                        f"doppler={dop_val:.6e} type={line_type} "
                        f"delta={delta_nm:.6e} kapmin={local_kapmin:.6e} "
                        f"profile={profile_val:.6e} wcon={wcon_val} "
                        f"wtail={wtail_val} center_outside={center_outside_val} "
                        f"center_idx={center_index_val} hit_idx={debug_idx_val} "
                        f"istep={debug_idx_val - center_index_val} "
                        f"resolu={resolu_val} v={v_val}"
                    )
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

    debug_metal_wave = os.getenv("PY_DEBUG_METAL_WING_WAVE")
    debug_metal_depth = os.getenv("PY_DEBUG_METAL_WING_DEPTH")
    if debug_metal_wave and debug_metal_depth:
        try:
            debug_wave_val = float(debug_metal_wave)
            depth_val = int(debug_metal_depth)
            if depth_val > 0:
                depth_idx = depth_val - 1
                wl_idx = int(np.argmin(np.abs(wavelength - debug_wave_val)))
                fscat_val = (
                    float(fscat_vec[depth_idx]) if depth_idx < fscat_vec.size else 0.0
                )
                alinec_val = float(alinec_total[depth_idx, wl_idx])
                helium_val = float(helium_wings[depth_idx, wl_idx])
                print(
                    "PY_DEBUG_METAL_WING: "
                    f"wave={wavelength[wl_idx]:.6f} depth={depth_val} "
                    f"abs_core={abs_core_base[depth_idx, wl_idx]:.6e} "
                    f"metal_wings={metal_wings[depth_idx, wl_idx]:.6e} "
                    f"helium_wings={helium_val:.6e} "
                    f"line_opacity={buffers.line_opacity[depth_idx, wl_idx]:.6e} "
                    f"alinec_total={alinec_val:.6e} fscat={fscat_val:.6e}"
                )
        except ValueError:
            pass

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
        # CRITICAL DEBUG: Check values before overwriting buffers.line_opacity
        print("\n" + "=" * 70)
        if cfg.debug:
            print(
                "CRITICAL DEBUG: Before setting buffers.line_opacity = total_line_absorption (NOT using ASYNTH)"
            )
            print("=" * 70)
            print(f"Array shapes:")
            print(f"  abs_core_base: {abs_core_base.shape}")
            print(f"  ahline: {ahline.shape}")
            print(f"  metal_wings: {metal_wings.shape}")
            print(f"  wavelength: {wavelength.shape}")
            print(f"\nabs_core_base statistics:")
            print(
                f"  non-zero count: {np.count_nonzero(abs_core_base)} / {abs_core_base.size}"
            )
            print(f"  max: {np.max(abs_core_base):.2e}")
            if np.any(abs_core_base > 0):
                print(f"  min (non-zero): {np.min(abs_core_base[abs_core_base > 0]):.2e}")
            else:
                print("  min (non-zero): N/A")
            print(
                f"  percentage non-zero: {100*np.count_nonzero(abs_core_base)/abs_core_base.size:.2f}%"
            )
            print(f"\nahline statistics:")
            print(f"  non-zero count: {np.count_nonzero(ahline)} / {ahline.size}")
            print(f"  max: {np.max(ahline):.2e}")
            if np.any(ahline > 0):
                print(f"  min (non-zero): {np.min(ahline[ahline > 0]):.2e}")
            else:
                print("  min (non-zero): N/A")
            print(f"  percentage non-zero: {100*np.count_nonzero(ahline)/ahline.size:.2f}%")
            print(f"\nmetal_wings statistics:")
            print(f"  non-zero count: {np.count_nonzero(metal_wings)} / {metal_wings.size}")
            print(f"  max: {np.max(metal_wings):.2e}")
            if np.any(metal_wings > 0):
                print(f"  min (non-zero): {np.min(metal_wings[metal_wings > 0]):.2e}")
            else:
                print("  min (non-zero): N/A")
            print(
                f"  percentage non-zero: {100*np.count_nonzero(metal_wings)/metal_wings.size:.2f}%"
            )
            print(f"\ntotal_line_absorption statistics:")
            print(
                f"  non-zero count: {np.count_nonzero(total_line_absorption)} / {total_line_absorption.size}"
            )
            print(f"  max: {np.max(total_line_absorption):.2e}")
            if np.any(total_line_absorption > 0):
                print(
                    f"  min (non-zero): {np.min(total_line_absorption[total_line_absorption > 0]):.2e}"
                )
            else:
                print("  min (non-zero): N/A")
            print(
                f"  percentage non-zero: {100*np.count_nonzero(total_line_absorption)/total_line_absorption.size:.2f}%"
            )

            # Check per-wavelength statistics
            if abs_core.shape[1] == wavelength.size:
                abs_core_nonzero_wl = np.sum(np.any(abs_core > 0, axis=0))
                print(
                    f"  abs_core: {abs_core_nonzero_wl} wavelengths with at least one non-zero depth"
                )
            if ahline.shape[1] == wavelength.size:
                ahline_nonzero_wl = np.sum(np.any(ahline > 0, axis=0))
                print(
                    f"  ahline: {ahline_nonzero_wl} wavelengths with at least one non-zero depth"
                )
            if metal_wings.shape[1] == wavelength.size:
                metal_wings_nonzero_wl = np.sum(np.any(metal_wings > 0, axis=0))
                print(
                    f"  metal_wings: {metal_wings_nonzero_wl} wavelengths with at least one non-zero depth"
                )
            total_nonzero_wl = np.sum(np.any(total_line_absorption > 0, axis=0))
            print(
                f"  total_line_absorption: {total_nonzero_wl} wavelengths with at least one non-zero depth"
            )

            if np.all(total_line_absorption == 0.0):
                print("\nERROR: total_line_absorption is ALL ZEROS!")
                print("  This means: abs_core_base + ahline + metal_wings = 0")
                print("  Check why each component is zero.")
            else:
                print("\ntotal_line_absorption is NOT all zeros - has some non-zero values")
            print("=" * 70 + "\n")

        buffers.line_opacity[:] = total_line_absorption
    else:
        # Using ASYNTH mode: buffers.line_opacity was already set to absorption = asynth * (1 - fscat)
        # For range-filtered runs, add off-grid wing contributions to match Fortran's
        # full-grid accumulation prior to ASYNTH.
        if cfg.debug:
            print("\n" + "=" * 70)
            print(
                "CRITICAL: Using ASYNTH mode - NOT overwriting buffers.line_opacity with metal_wings!"
            )
            print("=" * 70)
            print("  Fortran spectrv.for line 300: ALINE(J) = ASYNTH(J) * (1 - FSCAT(J))")
            print(
                "  Python buffers.line_opacity was already set to absorption = asynth * (1 - fscat)"
            )
            print("  Keeping it as is (NOT adding metal_wings)")
            print("=" * 70 + "\n")
        if np.any(helium_wings > 0):
            # Apply the same ASYNTH split used by Fortran:
            # absorption += ASYNTH_he * (1-FSCAT), scattering += ASYNTH_he * FSCAT.
            # `helium_wings` here is pre-STIM opacity from the wing pass; convert to
            # ASYNTH-equivalent opacity before applying the FSCAT split.
            helium_asynth = helium_wings * stim
            buffers.line_opacity += helium_asynth * (1.0 - fscat_vec[:, None])
            alinec_total = alinec_total + helium_asynth
    buffers.line_scattering[:] = alinec_total * fscat_vec[:, None]

    if cfg.debug:
        # CRITICAL DEBUG: Wavelength-by-wavelength analysis of line opacity
        print("\n" + "=" * 70)
        print("WAVELENGTH-BY-WAVELENGTH LINE OPACITY ANALYSIS")
        print("=" * 70)
        zero_wl_count = 0
        nonzero_wl_count = 0
        zero_wl_indices = []
        nonzero_wl_indices = []

        for wl_idx in range(wavelength.size):
            wl = wavelength[wl_idx]
            line_op_wl = buffers.line_opacity[:, wl_idx]
            abs_core_wl = (
                abs_core[:, wl_idx]
                if abs_core.shape[1] == wavelength.size
                else abs_core[:, 0]
            )
            ahline_wl = (
                ahline[:, wl_idx] if ahline.shape[1] == wavelength.size else ahline[:, 0]
            )
            metal_wings_wl = (
                metal_wings[:, wl_idx]
                if metal_wings.shape[1] == wavelength.size
                else metal_wings[:, 0]
            )

            is_zero = np.all(line_op_wl == 0.0)
            max_val = np.max(line_op_wl)

            if is_zero:
                zero_wl_count += 1
                zero_wl_indices.append(wl_idx)
                # Show first 10 zero wavelengths with breakdown
                if zero_wl_count <= 10:
                    print(f"\nWavelength {wl:.8f} nm (idx {wl_idx}): ALL ZEROS")
                    print(f"  abs_core max: {np.max(abs_core_wl):.2e}")
                    print(f"  ahline max: {np.max(ahline_wl):.2e}")
                    print(f"  metal_wings max: {np.max(metal_wings_wl):.2e}")
                    print(f"  total_line_absorption max: {max_val:.2e}")
            else:
                nonzero_wl_count += 1
                nonzero_wl_indices.append(wl_idx)
                # Show first 10 non-zero wavelengths with breakdown
                if nonzero_wl_count <= 10:
                    print(
                        f"\nWavelength {wl:.8f} nm (idx {wl_idx}): NON-ZERO (max={max_val:.2e})"
                    )
                    print(
                        f"  abs_core max: {np.max(abs_core_wl):.2e} ({100*np.max(abs_core_wl)/max_val:.1f}%)"
                    )
                    print(
                        f"  ahline max: {np.max(ahline_wl):.2e} ({100*np.max(ahline_wl)/max_val:.1f}%)"
                    )
                    print(
                        f"  metal_wings max: {np.max(metal_wings_wl):.2e} ({100*np.max(metal_wings_wl)/max_val:.1f}%)"
                    )
                    print(f"  Surface value: {line_op_wl[0]:.2e}")
                    print(f"  Deep value: {line_op_wl[-1]:.2e}")
            if zero_wl_count == 10:
                print(
                    f"\n  ... (showing first 10 zero wavelengths, {wavelength.size - len(zero_wl_indices) - len(nonzero_wl_indices)} more to check)"
                )
            if nonzero_wl_count == 10:
                print(
                    f"\n  ... (showing first 10 non-zero wavelengths, {wavelength.size - len(zero_wl_indices) - len(nonzero_wl_indices)} more to check)"
                )

        print("\n" + "=" * 70)
        print(f"SUMMARY: Line Opacity by Wavelength")
        print("=" * 70)
        print(f"Total wavelengths: {wavelength.size}")
        print(
            f"Wavelengths with ZERO line opacity: {zero_wl_count} ({100*zero_wl_count/wavelength.size:.1f}%)"
        )
        print(
            f"Wavelengths with NON-ZERO line opacity: {nonzero_wl_count} ({100*nonzero_wl_count/wavelength.size:.1f}%)"
        )

        if zero_wl_count > 0:
            print(f"\nFirst 5 zero-wavelength indices: {zero_wl_indices[:5]}")
            print(
                f"First 5 zero wavelengths: {[wavelength[i] for i in zero_wl_indices[:5]]}"
            )

        if nonzero_wl_count > 0:
            print(f"\nFirst 5 non-zero-wavelength indices: {nonzero_wl_indices[:5]}")
            print(
                f"First 5 non-zero wavelengths: {[wavelength[i] for i in nonzero_wl_indices[:5]]}"
            )
            # Check if non-zero wavelengths are clustered
            if len(nonzero_wl_indices) > 1:
                gaps = [
                    nonzero_wl_indices[i + 1] - nonzero_wl_indices[i]
                    for i in range(min(10, len(nonzero_wl_indices) - 1))
                ]
                avg_gap = np.mean(gaps)
                print(
                    f"Average gap between non-zero wavelengths (first 10): {avg_gap:.1f} indices"
                )

        # Component breakdown
        print(f"\nComponent Breakdown (max values across all wavelengths):")
        print(f"  abs_core_base max: {np.max(abs_core):.2e}")
        print(f"  abs_core_base non-zero count: {np.count_nonzero(abs_core)}")
        print(f"  ahline max: {np.max(ahline):.2e}")
        print(f"  ahline non-zero count: {np.count_nonzero(ahline)}")
        print(f"  metal_wings max: {np.max(metal_wings):.2e}")
        print(f"  metal_wings non-zero count: {np.count_nonzero(metal_wings)}")
        print("=" * 70 + "\n")

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
    if cfg.debug:
        print(f"\n{'='*70}")
        print(f"DEBUG: Line source computation decision")
        print(f"{'='*70}")
        print(f"  asynth_npz is not None: {asynth_npz is not None}")
        print(f"  has_lines: {has_lines}")
        print(f"  using_asynth: {using_asynth}")
        print(f"  slinec shape: {slinec.shape}")
        print(f"  slinec min/max: {np.min(slinec):.6e} / {np.max(slinec):.6e}")
        print(f"  bnu min/max: {np.min(bnu):.6e} / {np.max(bnu):.6e}")
        print(f"  slinec == bnu (all close): {np.allclose(slinec, bnu, rtol=1e-6)}")
    if using_asynth:
        # Use SLINEC in ASYNTH mode to match the source-function scaling used in
        # the opacity/source pipeline and avoid spurious line emission in cool models.
        line_source = slinec.copy()
        if cfg.debug:
            print(f"  -> Using slinec directly (ASYNTH mode)")
    else:
        if cfg.debug:
            print(f"  -> Computing weighted average (fort.9 ALINEC mode)")
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
        # CRITICAL DEBUG: Check if metal contribution is being computed correctly
        if cfg.debug and np.any(metal_wings > 1e10):
            print(
                f"\nWARNING: Large metal_wings detected! Max: {np.max(metal_wings):.2e}"
            )
            print(f"  metal_source max: {np.max(metal_source):.2e}")
            print(f"  metal_contribution max: {np.max(metal_contribution):.2e}")
            print(
                f"  numerator before adding metal: max={np.max(numerator):.2e}, non-zero count={np.count_nonzero(numerator)}"
            )
        numerator = numerator + metal_contribution
        if cfg.debug and np.any(metal_wings > 1e10):
            print(
                f"  numerator after adding metal: max={np.max(numerator):.2e}, non-zero count={np.count_nonzero(numerator)}"
            )
            print(
                f"  Check: Is metal_contribution finite? {np.all(np.isfinite(metal_contribution))}"
            )
            print(f"  Check: Is numerator finite? {np.all(np.isfinite(numerator))}")


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

    debug_wave = os.getenv("PY_DEBUG_WAVE")
    if debug_wave:
        try:
            target_wave = float(debug_wave)
        except ValueError:
            target_wave = None
        if (
            target_wave is not None
            and wavelength.size > 0
            and cont_abs.shape[1] == wavelength.size
        ):
            idx_target = int(np.argmin(np.abs(wavelength - target_wave)))
            print(
                f"PY_DEBUG_WAVE: requested {target_wave:.6f} nm, nearest {float(wavelength[idx_target]):.6f} nm (idx={idx_target})"
            )
            debug_depths = os.getenv("PY_DEBUG_WAVE_DEPTHS")
            if debug_depths:
                depth_list = []
                for item in debug_depths.split(","):
                    item = item.strip()
                    if not item:
                        continue
                    try:
                        depth_val = int(item)
                    except ValueError:
                        continue
                    if depth_val > 0:
                        depth_list.append(depth_val - 1)
            else:
                depth_list = [0, 19, 39, 59, 79]
            for depth_idx in depth_list:
                if 0 <= depth_idx < cont_abs.shape[0]:
                    line_val = float(buffers.line_opacity[depth_idx, idx_target])
                    cont_val = float(cont_abs[depth_idx, idx_target])
                    line_scat_val = float(
                        buffers.line_scattering[depth_idx, idx_target]
                    )
                    cont_scat_val = float(cont_scat[depth_idx, idx_target])
                    asynth_val = (
                        float(asynth[depth_idx, idx_target])
                        if "asynth" in locals()
                        else float("nan")
                    )
                    fscat_val = (
                        float(fscat_vec[depth_idx])
                        if "fscat_vec" in locals()
                        else float("nan")
                    )
                    print(
                        f"PY_DEBUG_WAVE depth {depth_idx + 1}: ALINE={line_val:.6e} "
                        f"ACONT={cont_val:.6e} ALINE/ACONT={line_val / max(cont_val, 1e-40):.6e} "
                        f"SIGMAL={line_scat_val:.6e} SIGMAC={cont_scat_val:.6e}"
                    )
                    print(
                        f"  ASYNTH={asynth_val:.6e} FSCAT={fscat_val:.6e} "
                        f"(1-FSCAT)={1.0 - fscat_val:.6e}"
                    )
            if os.getenv("PY_DEBUG_ABTOT_FULL") == "1":
                dump_path = os.getenv("PY_DEBUG_ABTOT_PATH")
                if not dump_path:
                    dump_path = f"out/abtot_{float(wavelength[idx_target]):.6f}.npz"
                np.savez(
                    dump_path,
                    acont=cont_abs[:, idx_target],
                    aline=buffers.line_opacity[:, idx_target],
                    sigmac=cont_scat[:, idx_target],
                    sigmal=buffers.line_scattering[:, idx_target],
                )
                print(f"PY_DEBUG_ABTOT_FULL: wrote {dump_path}")

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

    # CRITICAL DEBUG: Print directly to stdout (bypasses logger)
    if cfg.debug:
        print("\n" + "=" * 70)
        print("CRITICAL DEBUG: Before solve_lte_spectrum")
        print("=" * 70)
        print(f"Line opacity shape: {buffers.line_opacity.shape}")
        print(f"Line opacity non-zero count: {np.count_nonzero(buffers.line_opacity)}")
        print(f"Line opacity max: {np.max(buffers.line_opacity):.2e}")
        if len(line_indices) > 0:
            first_line_idx = line_indices[0]
            print(
                f"At first line wavelength (idx {first_line_idx}): {float(buffers.line_opacity[0, first_line_idx]):.2e}"
            )
        if np.all(buffers.line_opacity == 0.0):
            print("ERROR: buffers.line_opacity is ALL ZEROS!")
        else:
            print("Line opacity is NOT all zeros")
        print("=" * 70 + "\n")

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
        debug=cfg.debug,
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

    # CRITICAL DEBUG: Check if flux values are identical
    if cfg.debug:
        print("\n" + "=" * 70)
        print("CRITICAL DEBUG: After solve_lte_spectrum")
        print("=" * 70)
        print(f"flux_total shape: {flux_total.shape}")
        print(f"flux_cont shape: {flux_cont.shape}")
        if len(flux_total) > 0:
            print(f"First 5 flux_total: {flux_total[:5]}")
            print(f"First 5 flux_cont: {flux_cont[:5]}")
            # Check if they're identical with more detailed diagnostics
            are_close = np.isclose(flux_total, flux_cont, rtol=1e-10)
            n_identical = np.sum(are_close)
            n_different = np.sum(~are_close)
            print(f"Are they identical? {n_identical == len(flux_total)}")
            print(
                f"  Identical at {n_identical}/{len(flux_total)} wavelengths ({100*n_identical/len(flux_total):.2f}%)"
            )
            print(
                f"  Different at {n_different}/{len(flux_total)} wavelengths ({100*n_different/len(flux_total):.2f}%)"
            )
            if n_different > 0:
                # Show statistics for different values
                diff_mask = ~are_close
                diff_ratios = flux_total[diff_mask] / np.maximum(
                    flux_cont[diff_mask], 1e-40
                )
                print(
                    f"  Ratio (total/cont) for different values: min={np.min(diff_ratios):.6f}, max={np.max(diff_ratios):.6f}, mean={np.mean(diff_ratios):.6f}"
                )
                # Show first few different wavelengths
                diff_indices = np.where(diff_mask)[0]
                print(f"  First 5 different wavelengths:")
                for idx in diff_indices[:5]:
                    wl = wavelength[idx] if wavelength.size > idx else 0.0
                    print(
                        f"    idx={idx}, λ={wl:.2f} nm: flux_total={flux_total[idx]:.6e}, flux_cont={flux_cont[idx]:.6e}, ratio={flux_total[idx]/max(flux_cont[idx],1e-40):.6f}"
                    )

            if n_identical == len(flux_total):
                print(
                    "ERROR: flux_total and flux_cont are IDENTICAL at ALL wavelengths!"
                )
            elif n_identical > len(flux_total) * 0.9:
                print(
                    f"WARNING: flux_total ≈ flux_cont at {100*n_identical/len(flux_total):.1f}% of wavelengths!"
                )
                print("  This suggests line opacity is not effectively reducing flux.")

        print("=" * 70 + "\n")

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
