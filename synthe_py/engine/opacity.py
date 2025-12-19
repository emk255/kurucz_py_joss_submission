"""Core synthesis loop."""

from __future__ import annotations

import math
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def prange(*args, **kwargs):
        return range(*args)


from ..config import SynthesisConfig
from ..io import atmosphere, export
from ..io.lines import atomic, fort19 as fort19_io, fort9 as fort9_io
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
    compute_hydrogen_wings,
)
from ..physics.profiles import hydrogen_line_profile, voigt_profile
from .radiative import solve_lte_spectrum
from .buffers import SynthResult, allocate_buffers

MAX_PROFILE_STEPS = 2000
H_PLANCK = 6.62607015e-27  # erg * s
C_LIGHT_CM = 2.99792458e10  # cm / s
C_LIGHT_NM = 2.99792458e17  # nm/s (for frequency calculation)
K_BOLTZ = 1.380649e-16  # erg / K
NM_TO_CM = 1e-7

# CGF conversion constants from rgfall.for line 267
CGF_CONSTANT = 0.026538 / 1.77245  # Factor for converting GF to CONGF


@jit(nopython=True, cache=True)
def _voigt_profile_jit(
    v: float, a: float, h0tab: np.ndarray, h1tab: np.ndarray, h2tab: np.ndarray
) -> float:
    """JIT-compiled Voigt profile function."""
    # CRITICAL FIX: Voigt function is symmetric in v, use abs(v) for table lookup
    # Bug was: negative v -> negative index -> clamped to 0 -> returned center value!
    iv = int(abs(v) * 200.0 + 0.5)
    iv = max(0, min(iv, h0tab.size - 1))

    if a < 0.2:
        if abs(v) > 10.0:
            return 0.5642 * a / (v * v)
        else:
            return (h2tab[iv] * a + h1tab[iv]) * a + h0tab[iv]
    elif a > 1.4 or (a + abs(v)) > 3.2:
        aa = a * a
        vv = v * v
        u = (aa + vv) * 1.4142
        voigt_val = a * 0.79788 / u
        if a <= 100.0:
            aau = aa / u
            vvu = vv / u
            uu = u * u
            voigt_val = (
                (((aau - 10.0 * vvu) * aau * 3.0 + 15.0 * vvu * vvu) + 3.0 * vv - aa)
                / uu
                + 1.0
            ) * voigt_val
        return voigt_val
    else:
        vv = v * v
        h0 = h0tab[iv]
        h1 = h1tab[iv] + h0 * 1.12838
        h2 = h2tab[iv] + h1 * 1.12838 - h0
        h3 = (1.0 - h2tab[iv]) * 0.37613 - h1 * 0.66667 * vv + h2 * 1.12838
        h4 = (3.0 * h3 - h1) * 0.37613 + h0 * 0.66667 * vv * vv
        poly_a = (((h4 * a + h3) * a + h2) * a + h1) * a + h0
        poly_b = ((-0.122727278 * a + 0.532770573) * a - 0.96284325) * a + 0.979895032
        return poly_a * poly_b


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

    EXACTLY matches Fortran synthe.for lines 255-348:
    - Line center: IF(ADAMP.LT..2) KAPCEN=KAPPA0*(1.-1.128*ADAMP) else VOIGT
    - Near wings: IF(ADAMP.LT..2) use H0TAB/H1TAB tables else full Voigt
    - Far wings: 1/x² approximation with MAXSTEP=SQRT(X/KAPMIN)+1
    - Boundary handling: Skip line CENTER if outside grid, still compute wings
    """
    if doppler_width <= 0.0 or kappa0 <= 0.0:
        return

    n_points = buffer.size
    adamp = max(damping, 1e-12)  # Use adamp to match Fortran naming

    # Clamp center_index for continuum access (Fortran line 266: MIN(MAX(NBUFF,1),LENGTH))
    clamped_center = max(0, min(center_index, n_points - 1))
    kapmin = cutoff * continuum_row[clamped_center]

    # Check if line center is OUTSIDE the grid (Fortran line 301)
    center_outside_grid = center_index < 0 or center_index >= n_points

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

    # ========== LINE CENTER ==========
    # NOTE: Line centers (ALINEC) are handled separately in fort.9/fort.19 processing
    # This kernel computes WINGS ONLY, matching Fortran's BUFFER accumulation
    # The calling code sets tmp_buffer[center_index] = 0 after this kernel to exclude centers
    # (Fortran BUFFER gets line center at line 309, but we handle it separately via line_opacity)

    # ========== NEAR WING PROFILE (Fortran lines 311-326) ==========
    # N10DOP = 10 * (DOPPLE * RESOLU) - number of steps within 10 Doppler widths
    # CRITICAL: DO NOT force n10dop >= 1! Fortran uses integer truncation.
    # If DOPPLE*RESOLU < 0.1, N10DOP = 0 and NO wings are computed.
    n10dop = int(10.0 * dopple * resolu)
    n10dop = min(
        n10dop, MAX_PROFILE_STEPS
    )  # Don't use max(1, ...) - match Fortran exactly!

    # Pre-compute profile values - CRITICAL: Must be MAX_PROFILE_STEPS+1 like Fortran's PROFILE(MAXPROF)
    # Bug fix: Previously was n10dop+1, causing out-of-bounds access for far wings
    profile = np.zeros(MAX_PROFILE_STEPS + 1, dtype=np.float64)

    # VSTEPS = 200 (200 table steps per Doppler width)
    vsteps = 200.0

    if adamp < 0.2:
        # Fortran lines 313-319: Use H0TAB/H1TAB tables
        # TABSTEP = VSTEPS / (DOPPLE * RESOLU)
        tabstep = vsteps / (dopple * resolu) if (dopple * resolu) > 0 else vsteps
        tabi = 1.5

        for nstep in range(1, n10dop + 1):
            tabi = tabi + tabstep
            # Clamp table index to valid range [0, len(h0tab)-1]
            itab = min(max(int(tabi), 0), len(h0tab) - 1)
            # PROFILE(NSTEP) = KAPPA0 * (H0TAB(IFIX(TABI)) + ADAMP * H1TAB(IFIX(TABI)))
            profile[nstep] = kappa0 * (h0tab[itab] + adamp * h1tab[itab])

            if profile[nstep] < kapmin:
                n10dop = nstep - 1
                break
    else:
        # Fortran lines 321-325: Full Voigt function
        # DVOIGT = 1 / DOPPLE / RESOLU
        dvoigt = 1.0 / dopple / resolu if (dopple * resolu) > 0 else 1e-6

        for nstep in range(1, n10dop + 1):
            # x = NSTEP * DVOIGT (velocity in Doppler units)
            x_val = float(nstep) * dvoigt
            profile[nstep] = kappa0 * _voigt_profile_jit(
                x_val, adamp, h0tab, h1tab, h2tab
            )

            if profile[nstep] < kapmin:
                n10dop = nstep - 1
                break

    # ========== FAR WINGS (Fortran lines 328-334) ==========
    # X = PROFILE(N10DOP) * FLOAT(N10DOP)**2
    if n10dop > 0 and profile[n10dop] > 0:
        x_far = profile[n10dop] * float(n10dop) ** 2
    else:
        x_far = 0.0

    # MAXSTEP = SQRT(X/KAPMIN) + 1, limited to MAXPROF
    if x_far > 0 and kapmin > 0:
        maxstep = int(np.sqrt(x_far / kapmin) + 1.0)
        maxstep = min(maxstep, MAX_PROFILE_STEPS)
    else:
        maxstep = n10dop

    # Compute far wing profile values (1/x² approximation)
    # PROFILE(NSTEP) = X / FLOAT(NSTEP)**2
    n1 = n10dop + 1
    for nstep in range(n1, maxstep + 1):
        profile[nstep] = x_far / float(nstep) ** 2 if nstep > 0 else 0.0

    # Check if entire profile is outside grid (Fortran line 335)
    if center_index + maxstep < 0 or center_index - maxstep >= n_points:
        return

    use_wcon = wcon > 0.0
    use_wtail = wtail > 0.0

    # ========== RED WING (Fortran lines 337-341) ==========
    # MAXRED = MIN0(LENGTH-NBUFF, NSTEP)
    # MINRED = MAX0(1, 1-NBUFF)
    if center_index < n_points:
        maxred = (
            min(n_points - 1 - center_index, maxstep)
            if center_index >= 0
            else min(n_points - 1, maxstep)
        )
        minred = max(1, 1 - center_index)

        for istep in range(minred, maxred + 1):
            idx = center_index + istep
            if idx < 0 or idx >= n_points:
                continue

            wave = wavelength_grid[idx]

            # Check continuum limits (WCON/WTAIL)
            if use_wcon and wave <= wcon:
                continue

            # Get profile value
            if istep <= maxstep and profile[istep] > 0:
                value = profile[istep]
            else:
                value = x_far / float(istep) ** 2 if istep > 0 else 0.0

            # Apply tapering if needed
            if use_wtail:
                base = wcon if use_wcon else line_wavelength
                if wave < wtail:
                    value = value * (wave - base) / max(wtail - base, 1e-12)

            # Check cutoff and accumulate
            if value >= continuum_row[idx] * cutoff and value >= kapmin:
                buffer[idx] += value

    # ========== BLUE WING (Fortran lines 343-347) ==========
    # Skip blue wing if center is at or before start (Fortran line 342)
    if center_index <= 0:
        return

    # MAXBLUE = MIN0(NBUFF-1, NSTEP)
    # MINBLUE = MAX0(1, NBUFF-LENGTH)
    maxblue = min(center_index - 1, maxstep) if center_index > 0 else 0
    minblue = max(1, center_index - (n_points - 1))

    for istep in range(minblue, maxblue + 1):
        idx = center_index - istep
        if idx < 0 or idx >= n_points:
            continue

        wave = wavelength_grid[idx]

        # Check continuum limits (WCON/WTAIL) - note: different termination behavior
        if use_wcon and wave <= wcon:
            break  # Blue wing terminates at WCON

        # Get profile value
        if istep <= maxstep and profile[istep] > 0:
            value = profile[istep]
        else:
            value = x_far / float(istep) ** 2 if istep > 0 else 0.0

        # Apply tapering if needed
        if use_wtail:
            base = wcon if use_wcon else line_wavelength
            if wave < wtail:
                value = value * (wave - base) / max(wtail - base, 1e-12)

        # Check cutoff and accumulate
        if value >= continuum_row[idx] * cutoff and value >= kapmin:
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

_fort19_unhandled_types: Set[fort19_io.Fort19WingType] = set()


def _load_atmosphere(cfg: SynthesisConfig) -> atmosphere.AtmosphereModel:
    model_path = cfg.atmosphere.model_path

    # If explicit NPZ path is provided, use it directly
    if cfg.atmosphere.npz_path is not None:
        npz_path = cfg.atmosphere.npz_path
        if not npz_path.exists():
            raise FileNotFoundError(f"Specified NPZ file does not exist: {npz_path}")
        logging.info(f"Loading atmosphere from specified NPZ file: {npz_path}")
        return atmosphere.load_cached(npz_path)

    # If model path is already an .npz file, use it directly
    if model_path.suffix == ".npz":
        return atmosphere.load_cached(model_path)

    # If given an .atm file, try to find the corresponding cached NPZ file
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


def _load_line_data(
    cfg: SynthesisConfig,
    wl_min: float,
    wl_max: float,
    fort9_data: Optional[fort9_io.Fort9Data] = None,
) -> atomic.LineCatalog:
    """Load line catalog from atomic catalog file.

    fort.9 is optional and only used for metadata if provided.
    Line catalog is required - cannot proceed without it.
    """
    _logger = logging.getLogger(__name__)
    catalog: Optional[atomic.LineCatalog] = None
    if cfg.line_data.atomic_catalog is not None:
        try:
            _logger.info(f"Loading atomic catalog from: {cfg.line_data.atomic_catalog}")
            catalog = atomic.load_catalog(cfg.line_data.atomic_catalog)
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
        # Try to use fort.9 as fallback (for compatibility, but not recommended)
        if fort9_data is not None:
            _logger.warning(
                "No atomic catalog provided - using fort.9 to create catalog. "
                "This is deprecated - provide an atomic catalog file instead."
            )
            catalog = atomic.catalog_from_fort9(fort9_data)
        else:
            # No catalog and no fort.9 - raise error
            raise RuntimeError(
                "No atomic catalog provided and no fort.9 available. "
                "An atomic line catalog is required for line synthesis. "
                "Please provide --atomic-catalog or use the 'atomic' positional argument."
            )

    filtered_catalog = atomic.filter_by_range(catalog, wl_min, wl_max)
    _logger.info(
        f"After filtering to wavelength range [{wl_min:.2f}, {wl_max:.2f}] nm: "
        f"{len(filtered_catalog.records)} lines"
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


@jit(nopython=True)
def _vacuum_to_air_jit(w_angstrom: float) -> float:
    """Convert vacuum wavelength (Å) to air using the standard SYNTHE formula (Numba-compatible)."""
    waven = 1.0e7 / w_angstrom
    denom = (
        1.0000834213
        + 2_406_030.0 / (1.30e10 - waven * waven)
        + 15_997.0 / (3.89e9 - waven * waven)
    )
    return w_angstrom / denom


def _vacuum_to_air(w_angstrom: float) -> float:
    """Convert vacuum wavelength (Å) to air using the standard SYNTHE formula."""
    return _vacuum_to_air_jit(w_angstrom)


@jit(nopython=True)
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

    wcon_ang = 1.0e7 / denom

    denom_tail = cont_val - emerge_line - 500.0
    wtail_ang = -1.0
    if abs(denom_tail) > 1e-8:
        wtail_ang = 1.0e7 / denom_tail
        if wtail_ang < 0.0:
            wtail_ang = 2.0 * wcon_ang
        wtail_ang = min(2.0 * wcon_ang, wtail_ang)

    if ifvac == 0:
        wcon_ang = _vacuum_to_air_jit(wcon_ang)
        if wtail_ang > 0.0:
            wtail_ang = _vacuum_to_air_jit(wtail_ang)

    wcon_nm = wcon_ang * 0.1
    wtail_nm = wtail_ang * 0.1 if wtail_ang > 0.0 else -1.0
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

    # Use JIT kernel if Numba is available
    if NUMBA_AVAILABLE:
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

    # ========== FALLBACK (non-Numba) - MATCHES FORTRAN EXACTLY ==========
    n_points = buffer.size
    adamp = max(damping, 1e-12)

    # Clamp center_index for continuum access (Fortran line 266)
    clamped_center = max(0, min(center_index, n_points - 1))
    kapmin = cutoff * continuum_row[clamped_center]

    # Check if line center is OUTSIDE the grid
    center_outside_grid = center_index < 0 or center_index >= n_points

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

    dopple = doppler_width / line_wavelength if line_wavelength > 0 else 1e-6

    # ========== LINE CENTER ==========
    # NOTE: Line centers are handled separately - this kernel computes WINGS ONLY
    # The calling code sets tmp_buffer[center_index] = 0 after this function

    # ========== NEAR WING PROFILE (Fortran lines 311-326) ==========
    # CRITICAL: DO NOT force n10dop >= 1! Fortran uses integer truncation.
    # If DOPPLE*RESOLU < 0.1, N10DOP = 0 and NO wings are computed.
    n10dop = int(10.0 * dopple * resolu)
    n10dop = min(
        n10dop, MAX_PROFILE_STEPS
    )  # Don't use max(1, ...) - match Fortran exactly!

    profile_near = {}
    x_far_wing = 0.0
    vsteps = 200.0

    if adamp < 0.2:
        # Use H0TAB-like approximation with TABSTEP
        tabstep = vsteps / (dopple * resolu) if (dopple * resolu) > 0 else vsteps
        tabi = 1.5

        for nstep in range(1, n10dop + 1):
            tabi = tabi + tabstep
            # Approximate H0TAB: exp(-v²) where v = tabi/vsteps
            v = tabi / vsteps
            h0_approx = np.exp(-v * v)
            # H1TAB approximation (first-order correction)
            h1_approx = 2.0 / np.sqrt(np.pi) * (1.0 - 2.0 * v * v) * h0_approx
            value = kappa0 * (h0_approx + adamp * h1_approx)

            if value < kapmin:
                n10dop = nstep - 1
                break
            profile_near[nstep] = value
    else:
        dvoigt = 1.0 / dopple / resolu if (dopple * resolu) > 0 else 1e-6
        for nstep in range(1, n10dop + 1):
            x_val = float(nstep) * dvoigt
            value = kappa0 * voigt_profile(x_val, adamp)

            if value < kapmin:
                n10dop = nstep - 1
                break
            profile_near[nstep] = value

    # ========== FAR WINGS (Fortran lines 328-334) ==========
    if n10dop > 0 and n10dop in profile_near:
        x_far_wing = profile_near[n10dop] * float(n10dop) ** 2
    elif profile_near:
        last_step = max(profile_near.keys())
        x_far_wing = profile_near[last_step] * float(last_step) ** 2

    if x_far_wing > 0 and kapmin > 0:
        maxstep = int(np.sqrt(x_far_wing / kapmin) + 1.0)
        maxstep = min(maxstep, MAX_PROFILE_STEPS)
    else:
        maxstep = n10dop

    # Skip center contribution (core already handled by ALINEC)
    # Process red wing (matches Fortran lines 323-327)
    # For lines BEFORE grid (center_index < 0), start from offset that reaches grid
    # Matches Fortran line 339: MINRED=MAX0(1,1-NBUFF)
    min_red_offset = max(1, 1 - center_index) if center_index < 0 else 1

    red_active = True
    for offset in range(min_red_offset, maxstep + 1):
        if not red_active:
            break
        idx = center_index + offset
        if idx >= n_points:
            red_active = False
            break
        if idx < 0:
            continue  # Skip indices before grid

        wave = wavelength_grid[idx]

        # Check continuum limits (wcon/wtail)
        if wcon is not None and wave <= wcon:
            continue

        # Apply tapering if needed
        taper = 1.0
        if wtail is not None:
            base = wcon if wcon is not None else line_wavelength
            if wave < wtail:
                taper = (wave - base) / max(wtail - base, 1e-12)
        taper = max(min(taper, 1.0), 0.0)

        # Get profile value
        if offset <= n10dop and offset in profile_near:
            # Near wing: use precomputed Voigt profile
            value = profile_near[offset] * taper
        else:
            # Far wing: use 1/x² approximation (matches Fortran line 319)
            # PROFILE(NSTEP) = X / FLOAT(NSTEP)**2
            value = (x_far_wing / float(offset) ** 2) * taper

        # Check cutoff (matches Fortran logic)
        if value < continuum_row[idx] * cutoff or value < kapmin:
            red_active = False
        else:
            buffer[idx] += value

    # Process blue wing (matches Fortran lines 329-333)
    # For lines AFTER grid (center_index >= n_points), start from offset that reaches grid
    # Matches Fortran line 345: MINBLUE=MAX0(1,NBUFF-LENGTH)
    min_blue_offset = (
        max(1, center_index - (n_points - 1)) if center_index >= n_points else 1
    )

    blue_active = True
    for offset in range(min_blue_offset, maxstep + 1):
        if not blue_active:
            break
        idx = center_index - offset
        if idx < 0:
            blue_active = False
            break
        if idx >= n_points:
            continue  # Skip indices beyond grid

        wave = wavelength_grid[idx]

        # Check continuum limits (wcon/wtail)
        if wcon is not None and wave <= wcon:
            blue_active = False
            break

        # Apply tapering if needed
        taper = 1.0
        if wtail is not None:
            base = wcon if wcon is not None else line_wavelength
            if wave < wtail:
                taper = (wave - base) / max(wtail - base, 1e-12)
        taper = max(min(taper, 1.0), 0.0)

        # Get profile value
        if offset <= n10dop and offset in profile_near:
            # Near wing: use precomputed Voigt profile
            value = profile_near[offset] * taper
        else:
            # Far wing: use 1/x² approximation
            value = (x_far_wing / float(offset) ** 2) * taper

        # Check cutoff
        if value < continuum_row[idx] * cutoff or value < kapmin:
            blue_active = False
        else:
            buffer[idx] += value


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

    while offset <= MAX_PROFILE_STEPS and (red_active or blue_active):
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
    depth_idx: int,
    depth_state: populations.DepthState,
    n_lower: int,
    n_upper: int,
    gamma_rad: float,
    gamma_stark: float,
    gamma_vdw: float,
    doppler_width: float,
) -> bool:
    """Handle special fort.19 wing prescriptions. Returns True if consumed."""

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
            wavelength_grid=wavelength_grid,
            center_index=center_index,
            line_wavelength=line_wavelength,
            kappa0=kappa0,
            depth_state=depth_state,
            n_lower=max(n_lower, 1),
            n_upper=max(n_upper, n_lower + 1),
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


def _accumulate_hydrogen_profile(
    buffer: np.ndarray,
    continuum_row: np.ndarray,
    wavelength_grid: np.ndarray,
    center_index: int,
    line_wavelength: float,
    kappa0: float,
    depth_state: populations.DepthState,
    n_lower: int,
    n_upper: int,
    cutoff: float,
) -> None:
    if depth_state.hydrogen is None:
        return

    n_points = buffer.size

    profile_center = kappa0 * hydrogen_line_profile(n_lower, n_upper, depth_state, 0.0)
    if profile_center >= continuum_row[center_index] * cutoff:
        buffer[center_index] += profile_center

    red_active = True
    blue_active = True
    offset = 1

    while offset <= MAX_PROFILE_STEPS and (red_active or blue_active):
        if red_active:
            idx = center_index + offset
            if idx >= n_points:
                red_active = False
            else:
                delta_nm = wavelength_grid[idx] - line_wavelength
                value = kappa0 * hydrogen_line_profile(
                    n_lower, n_upper, depth_state, delta_nm
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
                delta_nm = wavelength_grid[idx] - line_wavelength
                value = kappa0 * hydrogen_line_profile(
                    n_lower, n_upper, depth_state, delta_nm
                )
                if value <= 0.0 or value < continuum_row[idx] * cutoff:
                    blue_active = False
                else:
                    buffer[idx] += value
        offset += 1


def _process_metal_wings_depth_standalone(
    args_tuple: Tuple,
) -> Tuple[int, np.ndarray, np.ndarray, int, int]:
    """Standalone function to process metal line wings for a single depth (for multiprocessing).

    Returns (depth_idx, local_wings, local_sources, lines_processed, lines_skipped)
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
    voigt_tables = None
    h0tab = None
    h1tab = None
    h2tab = None
    if NUMBA_AVAILABLE:
        voigt_tables = tables.voigt_tables()
        h0tab = voigt_tables.h0tab
        h1tab = voigt_tables.h1tab
        h2tab = voigt_tables.h2tab

    for line_idx, center_index in enumerate(line_indices):
        # CRITICAL FIX: Match Fortran synthe.for line 301 EXACTLY
        # IF(NBUFF.LT.1.OR.NBUFF.GT.LENGTH)GO TO 320
        # Fortran SKIPS lines whose centers are outside the grid - NO wing computation!
        # Previous Python code computed wings for margin lines, causing ~60x too deep
        # absorption at grid boundaries (first few wavelength points).
        if center_index < 0 or center_index >= wavelength.size:
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
        txnxn_line = txnxn_val

        # Compute TXNXN with alpha correction if needed
        if np.isfinite(alpha) and abs(alpha) > 1e-8:
            atomic_mass = _atomic_mass_lookup(record.element)
            if atomic_mass is not None and atomic_mass > 0.0:
                t_j = temperature_val
                v2 = 0.5 * (1.0 - alpha)
                h_factor = (t_j / 10000.0) ** v2
                he_factor = (
                    0.628 * (2.0991e-4 * t_j * (0.25 + 1.008 / atomic_mass)) ** v2
                )
                h2_factor = 1.08 * (2.0991e-4 * t_j * (0.5 + 1.008 / atomic_mass)) ** v2
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

            # DEBUG: Track large kappa0 values (potential source of huge metal_wings)
            # Only print for first depth layer (depth_idx == 0) and first 5 lines to avoid spam
            if (
                (kappa0 > 1e10 or xnfdop > 1e20 or pop_val > 1e20)
                and depth_idx == 0
                and lines_processed < 5
            ):
                print(
                    f"\nDEBUG: Large values detected in metal wings computation (line {line_idx}, depth {depth_idx}):"
                )
                print(
                    f"  Element: {record.element}, Ion: {record.ion_stage}, λ: {line_wavelength:.6f} nm"
                )
                rho_debug = mass_density_val if mass_density_val > 0 else 1.0
                print(f"  pop_val (population density, cm⁻³): {pop_val:.6e}")
                print(f"  rho (mass density, g/cm³): {rho_debug:.6e}")
                print(
                    f"  pop_val / rho (population per unit mass, cm³/g): {pop_val / rho_debug if rho_debug > 0 else 0:.6e}"
                )
                print(
                    f"  dop_val (doppler velocity, dimensionless in units of c): {dop_val:.6e}"
                )
                print(f"  xnfdop = pop_val / (rho * dop_val): {xnfdop:.6e}")
                print(f"  boltzmann_factor: {boltz:.6e}")
                print(f"  catalog.gf (linear): {float(catalog_gf[line_idx]):.6e}")
                print(f"  wavelength: {float(line_wavelength):.6f} nm")
                print(f"  frequency: {freq_hz:.6e} Hz")
                print(f"  CGF = {CGF_CONSTANT:.6e} * GF / FREQ = {cgf:.6e}")
                print(f"  kappa0 = CGF * xnfdop * boltz: {kappa0:.6e}")
                print(f"  doppler_width = dop_val * λ (nm): {doppler_width:.6e}")
                print(
                    f"  continuum_row[center]: {float(continuum_row[center_index]):.6e}"
                )
                print(f"  kappa_min: {kappa_min:.6e}")
                # Additional verification: check if pop_val might already be divided by rho
                pop_val_check = pop_val * rho if rho > 0 else pop_val
                print(
                    f"  Verification: pop_val * rho = {pop_val_check:.6e} (should be original population density)"
                )

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

        # Compute damping value (ADAMP in Fortran synthe.for line 299)
        # Fortran: ADAMP = (GAMRF + GAMSF*XNE + GAMWF*TXNXN) / DOPPLE(NELION)
        # Where GAMRF etc. are NORMALIZED gamma values (divided by 4π*freq in rgfall.for)
        # And DOPPLE is dimensionless (v_th/c)
        #
        # Catalog gamma is LINEAR (10^GR from gfallvac.latest)
        # Need to normalize: gamma_normalized = gamma_linear / (4π*freq)
        # Then: ADAMP = gamma_normalized / DOPPLE = gamma_linear / (4π*freq*dopple)
        dopple = doppler_width / line_wavelength if line_wavelength > 0 else dop_val
        freq_hz = C_LIGHT_NM / line_wavelength if line_wavelength > 0 else 1e15
        delta_nu_doppler = freq_hz * dopple  # Doppler width in Hz
        gamma_total = (
            gamma_rad + gamma_stark * electron_density_val + gamma_vdw * txnxn_line
        )
        damping_value = gamma_total / max(4.0 * 3.14159265359 * delta_nu_doppler, 1e-40)

        # Apply fort.19 profile if available (simplified - he_solver not available)
        if fort19_data is not None and line19_idx is not None:
            # Skip fort19 profiles in standalone mode (he_solver limitation)
            pass

        # Accumulate metal profile
        # Pass Voigt tables directly to kernel if available (avoid repeated table lookups)
        if not NUMBA_AVAILABLE or h0tab is None:
            raise RuntimeError(
                "Numba is required for metal profile accumulation. "
                "Please install numba: pip install numba"
            )

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

        # DEBUG: Check if metal profile accumulation produced large values
        # Only print for first depth layer (depth_idx == 0) and first 3 lines to avoid spam
        if np.any(tmp_buffer > 1e10) and depth_idx == 0 and lines_processed < 3:
            max_idx = np.argmax(tmp_buffer)
            max_val = tmp_buffer[max_idx]
            print(
                f"\nDEBUG: Large metal profile value after accumulation (line {line_idx}, depth {depth_idx}):"
            )
            print(f"  Max value: {max_val:.6e} at wavelength index {max_idx}")
            print(f"  kappa0: {kappa0:.6e}")
            print(f"  damping: {damping_value:.6e}")
            print(f"  doppler_width: {line_doppler:.6e} cm")
            print(f"  Number of non-zero points: {np.count_nonzero(tmp_buffer)}")
            print(f"  Sum of tmp_buffer: {np.sum(tmp_buffer):.6e}")

        # Reset center (but only if center_index is within grid)
        # For lines outside grid, wings are still added but no center to reset
        if 0 <= center_index < wavelength.size:
            tmp_buffer[center_index] = 0.0
        local_wings += tmp_buffer
        local_sources += tmp_buffer * bnu_row
        lines_processed += 1

    return depth_idx, local_wings, local_sources, lines_processed, lines_skipped


# Numba-compatible atomic mass lookup (element_idx -> atomic_mass)
@jit(nopython=True)
def _get_atomic_mass_jit(element_idx: int, atomic_masses: np.ndarray) -> float:
    """Get atomic mass for element index. Returns 0.0 if invalid."""
    if element_idx >= 0 and element_idx < atomic_masses.size:
        return atomic_masses[element_idx]
    return 0.0


@jit(
    nopython=True, parallel=False, cache=False
)  # CRITICAL: parallel=False to avoid race conditions
def _process_metal_wings_kernel(
    metal_wings: np.ndarray,  # Output: n_depths × n_wavelengths
    metal_sources: np.ndarray,  # Output: n_depths × n_wavelengths
    wavelength_grid: np.ndarray,  # n_wavelengths
    line_indices: np.ndarray,  # n_lines (center_index for each line)
    line_wavelengths: np.ndarray,  # n_lines
    line_gf: np.ndarray,  # n_lines
    line_gamma_rad: np.ndarray,  # n_lines (LINEAR from catalog, not log10)
    line_gamma_stark: np.ndarray,  # n_lines (LINEAR from catalog, not log10)
    line_gamma_vdw: np.ndarray,  # n_lines (LINEAR from catalog, not log10)
    line_element_idx: np.ndarray,  # n_lines (element index, -1 if invalid)
    line_nelion: np.ndarray,  # n_lines (ion stage from catalog)
    line_meta_idx: np.ndarray,  # n_lines (metadata index, -1 if none)
    line_meta_ncon: np.ndarray,  # n_meta (or empty)
    line_meta_nelionx: np.ndarray,  # n_meta
    line_meta_nelion: np.ndarray,  # n_meta
    line_meta_alpha: np.ndarray,  # n_meta
    line_meta_gamma_rad: np.ndarray,  # n_meta (log10, or empty)
    line_meta_gamma_stark: np.ndarray,  # n_meta (log10, or empty)
    line_meta_gamma_vdw: np.ndarray,  # n_meta (log10, or empty)
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
    boltzmann_factor: np.ndarray,  # n_lines × n_depths
    contx: np.ndarray,  # metal_tables.contx
    atomic_masses: np.ndarray,  # n_elements
    ifvac: int,
    cutoff: float,
    h0tab: np.ndarray,
    h1tab: np.ndarray,
    h2tab: np.ndarray,
) -> None:
    """Sequential kernel for processing metal line wings.

    Processes all lines sequentially to avoid race conditions when accumulating
    results into metal_wings and metal_sources (multiple lines can contribute
    to the same wavelength bin via their wings).
    """
    n_lines = line_indices.size
    n_depths = metal_wings.shape[0]
    n_wavelengths = wavelength_grid.size
    n_elements = pop_densities_all.shape[0]
    max_ion_stage = pop_densities_all.shape[2]
    has_meta = line_meta_ncon.size > 0

    # Process lines sequentially to avoid race conditions
    for line_idx in range(n_lines):
        center_index = line_indices[line_idx]

        # CRITICAL FIX: Match Fortran synthe.for line 301 EXACTLY
        # IF(NBUFF.LT.1.OR.NBUFF.GT.LENGTH)GO TO 320
        # Fortran SKIPS lines whose centers are outside the grid - NO wing computation!
        if center_index < 0 or center_index >= n_wavelengths:
            continue

        line_wavelength = line_wavelengths[line_idx]
        element_idx = line_element_idx[line_idx]

        # Skip invalid elements or hydrogen lines
        if element_idx < 0 or element_idx >= n_elements:
            continue

        # Get metadata values for continuum limits (ncon, nelion, nelionx)
        # But ALWAYS use catalog gamma values - no fort.9 dependency!
        meta_idx = line_meta_idx[line_idx]
        if has_meta and meta_idx >= 0 and meta_idx < line_meta_ncon.size:
            ncon = line_meta_ncon[meta_idx]
            nelionx = line_meta_nelionx[meta_idx]
            nelion = (
                line_meta_nelion[meta_idx]
                if line_meta_nelion.size > meta_idx and line_meta_nelion[meta_idx] > 0
                else line_nelion[line_idx]
            )
            alpha = (
                line_meta_alpha[meta_idx] if line_meta_alpha.size > meta_idx else 0.0
            )
        else:
            ncon = 0
            nelionx = 0
            nelion = line_nelion[line_idx]
            alpha = 0.0

        # ALWAYS use catalog gamma values (LINEAR, not normalized)
        # Catalog stores 10^GR, 10^GS, 10^GW from gfallvac.latest
        # NO fort.9 metadata dependency for gamma values!
        gamma_rad = line_gamma_rad[line_idx]
        gamma_stark = line_gamma_stark[line_idx]
        gamma_vdw = line_gamma_vdw[line_idx]

        # Allocate a small reusable buffer per line limited to the wing span (2*MAX_PROFILE_STEPS)
        max_window = 2 * MAX_PROFILE_STEPS + 2
        tmp_buffer_full = np.zeros(max_window, dtype=np.float64)

        # Get population data for this element
        if nelion <= 0 or nelion > max_ion_stage:
            continue

        # Process each depth for this line
        for depth_idx in range(n_depths):
            # Get population and Doppler velocity
            pop_val = pop_densities_all[element_idx, depth_idx, nelion - 1]
            dop_val = dop_velocity_all[element_idx, depth_idx]

            if pop_val <= 0.0 or dop_val <= 0.0:
                continue

            # Compute XNFDOP
            rho = mass_density[depth_idx]
            if rho <= 0.0:
                continue

            xnfdop = pop_val / (rho * dop_val)
            doppler_width = dop_val * line_wavelength

            if doppler_width <= 0.0:
                continue

            # Get Boltzmann factor
            boltz = boltzmann_factor[line_idx, depth_idx]

            # Convert GF to CONGF
            # CGF_CONSTANT = 0.026538 / 1.77245 (from rgfall.for line 267)
            freq_hz = 2.99792458e17 / line_wavelength  # C_LIGHT_NM
            gf_linear = line_gf[line_idx]
            cgf = (0.026538 / 1.77245) * gf_linear / freq_hz

            # ========== DOUBLE KAPMIN CHECK (Fortran lines 266-272) ==========
            # Clamp center_index for continuum access
            clamped_idx = max(0, min(center_index, n_wavelengths - 1))
            kappa_min = continuum[depth_idx, clamped_idx] * cutoff

            # First: KAPPA0 = CONGF * XNFDOP (BEFORE Boltzmann)
            kappa0_pre = cgf * xnfdop

            # First check (Fortran line 267)
            if kappa0_pre < kappa_min:
                continue

            # Apply Boltzmann factor
            kappa0 = kappa0_pre * boltz

            # Second check (Fortran line 272): post-Boltzmann cutoff
            # RE-ENABLED: This matches Fortran behavior and prevents weak line accumulation
            if kappa0 < kappa_min:
                continue

            # Compute TXNXN with alpha correction if needed
            txnxn_line = txnxn[depth_idx]
            if abs(alpha) > 1e-8:
                atomic_mass = _get_atomic_mass_jit(element_idx, atomic_masses)
                if atomic_mass > 0.0:
                    t_j = temperature[depth_idx]
                    v2 = 0.5 * (1.0 - alpha)
                    h_factor = (t_j / 10000.0) ** v2
                    he_factor = (
                        0.628 * (2.0991e-4 * t_j * (0.25 + 1.008 / atomic_mass)) ** v2
                    )
                    h2_factor = (
                        1.08 * (2.0991e-4 * t_j * (0.5 + 1.008 / atomic_mass)) ** v2
                    )
                    txnxn_line = (
                        xnf_h[depth_idx] * h_factor
                        + xnf_he1[depth_idx] * he_factor
                        + xnf_h2[depth_idx] * h2_factor
                    )

            # Compute continuum limits
            wcon, wtail = _compute_continuum_limits_jit(
                ncon,
                nelion,
                nelionx,
                emerge[depth_idx],
                emerge_h[depth_idx],
                contx,
                ifvac,
            )

            # Compute damping (ADAMP in Fortran synthe.for line 299)
            # Fortran: ADAMP = (GAMRF + GAMSF*XNE + GAMWF*TXNXN) / DOPPLE(NELION)
            # Where GAMRF etc. are NORMALIZED gamma values (divided by 4π*freq in rgfall.for)
            # And DOPPLE is dimensionless (v_th/c)
            #
            # Catalog gamma is LINEAR (10^GR from gfallvac.latest)
            # Need to normalize: gamma_normalized = gamma_linear / (4π*freq)
            # Then: ADAMP = gamma_normalized / DOPPLE = gamma_linear / (4π*freq*dopple)
            dopple = doppler_width / line_wavelength if line_wavelength > 0 else dop_val
            freq_hz = C_LIGHT_NM / line_wavelength if line_wavelength > 0 else 1e15
            delta_nu_doppler = freq_hz * dopple  # Doppler width in Hz
            gamma_total = (
                gamma_rad
                + gamma_stark * electron_density[depth_idx]
                + gamma_vdw * txnxn_line
            )
            damping_value = gamma_total / max(
                4.0 * 3.14159265359 * delta_nu_doppler, 1e-40
            )

            # Compute window bounds around the line center, limited by MAX_PROFILE_STEPS
            start_idx = center_index - MAX_PROFILE_STEPS if center_index >= 0 else 0
            start_idx = max(start_idx, 0)
            end_idx = center_index + MAX_PROFILE_STEPS + 1
            end_idx = min(end_idx, n_wavelengths)

            if end_idx <= start_idx:
                continue

            # Local buffer view sized to the window
            window_len = end_idx - start_idx
            if window_len > max_window:
                window_len = max_window
                start_idx = max(0, center_index - MAX_PROFILE_STEPS)
                end_idx = start_idx + window_len
            tmp_buffer = tmp_buffer_full[:window_len]
            tmp_buffer.fill(0.0)

            # Local slices for continuum and wavelength grid
            continuum_slice = continuum[depth_idx, start_idx:end_idx]
            wavelength_slice = wavelength_grid[start_idx:end_idx]
            center_local = center_index - start_idx

            # Accumulate metal profile into the local window
            _accumulate_metal_profile_kernel(
                tmp_buffer,
                continuum_slice,
                wavelength_slice,
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

            # Reset center within local window if in range
            if 0 <= center_local < window_len:
                tmp_buffer[center_local] = 0.0

            # Scatter-add local window into full outputs
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

    logger.info("Loading atmosphere model...")
    atm = _load_atmosphere(cfg)
    logger.info(f"Loaded atmosphere: {atm.layers} layers")

    # DEBUG: Print mass density (rho) values at different depths for comparison with Fortran
    if hasattr(atm, "mass_density") and atm.mass_density is not None:
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
    fort9_data: Optional[fort9_io.Fort9Data] = None
    fort19_data: Optional[fort19_io.Fort19Data] = None
    # Build or load wavelength grid
    # Always build wavelength grid from configuration (no fort.29 dependency)
    logger.info("Building wavelength grid from configuration...")
    wavelength_full = _build_wavelength_grid(cfg)
    original_wavelength_size = wavelength_full.size

    # Apply wavelength range filter first (before subsampling)
    wavelength_mask_full = None
    if cfg.wavelength_range_filter is not None:
        wl_min, wl_max = cfg.wavelength_range_filter
        wavelength_mask_full = (wavelength_full >= wl_min) & (wavelength_full <= wl_max)
        original_size = wavelength_full.size
        wavelength_full = wavelength_full[wavelength_mask_full]
        logger.info(
            f"Filtered wavelength grid: {original_size} -> {wavelength_full.size} points (range {wl_min:.2f}-{wl_max:.2f} nm)"
        )

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
    if cfg.wavelength_range_filter is not None:
        logger.info(
            f"  Range filter active: {cfg.wavelength_range_filter[0]:.2f}-{cfg.wavelength_range_filter[1]:.2f} nm"
        )
    if cfg.wavelength_subsample > 1:
        logger.info(f"  Subsample active: every {cfg.wavelength_subsample} points")

    # Fort.9 and fort.19 are now optional (for metadata only, not required for computation)
    # Populations and line opacity are computed from first principles
    fort9_data = None
    if cfg.line_data.fort9 is not None:
        logger.info("Loading fort.9 metadata (optional, for line metadata only)...")
        fort9_data = fort9_io.load(cfg.line_data.fort9)
        logger.info(f"fort.9: {fort9_data.n_lines} lines (metadata only)")
    else:
        logger.info(
            "No fort.9 provided - using line catalog only (all metadata from catalog)"
        )

    fort19_data = None
    if cfg.line_data.fort19 is not None:
        logger.info(
            "Loading fort.19 wing metadata (optional, for special wing profiles)..."
        )
        fort19_data = fort19_io.load(cfg.line_data.fort19)
    else:
        logger.info("No fort.19 provided - using standard wing profiles from catalog")
    logger.info("Allocating buffers...")
    buffers = allocate_buffers(wavelength, atm.layers)

    logger.info("Loading line catalog...")
    catalog = _load_line_data(cfg, wavelength.min(), wavelength.max(), fort9_data)
    has_lines = len(catalog.records) > 0
    logger.info(f"Catalog: {len(catalog.records)} lines")
    line_indices = _nearest_grid_indices(wavelength, catalog.wavelength)
    logger.info("Computing depth-dependent populations...")
    pops = populations.compute_depth_state(
        atm,
        catalog.wavelength,
        catalog.excitation_energy,
        cfg.wavelength_grid.velocity_microturb,
    )

    logger.info("Computing frequency-dependent quantities...")
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

    logger.info("Computing continuum absorption/scattering...")
    cont_abs, cont_scat, _, _ = continuum.build_depth_continuum(atm, wavelength)
    logger.info("Computing hydrogen continuum...")
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

    spectrv_params = None
    if cfg.line_data.spectrv_input is not None:
        spectrv_params = spectrv_io.load(cfg.line_data.spectrv_input)

    logger.info("Computing line opacity from line catalog...")
    fscat_vec: np.ndarray = np.zeros(atm.layers, dtype=np.float64)

    # CRITICAL DEBUG: Check has_lines value
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
        pops = populations.compute_depth_state(
            atm,
            catalog.wavelength,
            catalog.excitation_energy,
            cfg.wavelength_grid.velocity_microturb,
        )

        # Compute TRANSP (line opacity at line center)
        # CRITICAL: Pass continuum_absorption for dynamic KAPMIN = CONTINUUM * CUTOFF (matches Fortran)
        # CRITICAL FIX: Pass ABTOT (cont_abs + cont_scat) for cutoff check
        # Fortran synthe.for line 266: KAPMIN=CONTINUUM(...)*CUTOFF
        # where CONTINUUM = ABTOT = ACONT + SIGMAC (from xnfpelsyn.for lines 382-383)
        # Using only cont_abs would make KAPMIN ~900x too low for cool stars
        transp, valid_mask, line_indices = line_opacity.compute_transp(
            catalog=catalog,
            populations=pops,
            atmosphere=atm,
            cutoff=cfg.cutoff,
            continuum_absorption=cont_abs + cont_scat,  # Match Fortran: ABTOT
            wavelength_grid=wavelength,
        )

        logger.info(f"Computed TRANSP for {np.sum(valid_mask)} line-depth pairs")

        # Compute ASYNTH from TRANSP (with wing contributions)
        # Pass continuum absorption for cutoff calculation (matches Fortran: KAPMIN = CONTINUUM * CUTOFF)
        # CRITICAL FIX: Pass metal_tables for WCON/WTAIL computation (matches Fortran lines 676-681, 703, 706, 722, 726)
        metal_tables = None
        if hasattr(atm, "metal_tables") and atm.metal_tables is not None:
            metal_tables = atm.metal_tables
        asynth = line_opacity.compute_asynth_from_transp(
            transp=transp,
            catalog=catalog,
            atmosphere=atm,
            wavelength_grid=wavelength,
            valid_mask=valid_mask,
            populations=pops,
            cutoff=cfg.cutoff,
            continuum_absorption=cont_abs + cont_scat,  # Match Fortran: ABTOT
            metal_tables=metal_tables,
        )

        logger.info(
            f"Computed ASYNTH: shape {asynth.shape}, range [{np.min(asynth):.2e}, {np.max(asynth):.2e}]"
        )

        # CRITICAL DEBUG: Check if asynth is all zeros
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
                print("  Reason: Unknown - asynth and fscat both have non-zero values")
        else:
            print("absorption is NOT all zeros")
        print("=" * 70 + "\n")

        buffers.line_opacity[:] = absorption
        # Mark that we're using ASYNTH (computed from catalog)
        buffers._using_asynth = True
        # After line 971: buffers.line_opacity[:] = absorption
        logger.info(
            f"Line opacity after ASYNTH: shape {buffers.line_opacity.shape}, "
            f"non-zero count: {np.count_nonzero(buffers.line_opacity)}, "
            f"max: {np.max(buffers.line_opacity):.2e}, "
            f"min (non-zero): {float(np.min(buffers.line_opacity[buffers.line_opacity > 0])) if np.any(buffers.line_opacity > 0) else 0:.2e}"
        )
        # CRITICAL DEBUG: Check if absorption is all zeros and analyze large values
        print("\n" + "=" * 70)
        print(
            "CRITICAL DEBUG: After setting buffers.line_opacity = absorption (from TRANSP/ASYNTH)"
        )
        print("=" * 70)
        print(f"absorption shape: {absorption.shape}")
        print(f"absorption non-zero count: {np.count_nonzero(absorption)}")
        print(f"absorption max: {np.max(absorption):.2e}")
        absorption_nonzero = (
            absorption[absorption > 0] if np.any(absorption > 0) else np.array([0.0])
        )
        print(f"absorption min (non-zero): {float(np.min(absorption_nonzero)):.2e}")
        print(f"absorption mean (non-zero): {float(np.mean(absorption_nonzero)):.2e}")
        print(
            f"absorption median (non-zero): {float(np.median(absorption_nonzero)):.2e}"
        )
        print(f"Values > 1e10: {np.sum(absorption > 1e10):,}")
        print(f"Values > 1e20: {np.sum(absorption > 1e20):,}")
        print(f"Values > 1e24: {np.sum(absorption > 1e24):,}")
        # CRITICAL FIX: Match Fortran exactly - no clamping of line opacity
        # Fortran does NOT clamp ASYNTH/ALINE values - it uses them directly
        # Remove clamping to match Fortran behavior exactly
        if np.any(absorption > 1e10):
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
    else:
        # Continuum-only synthesis (no lines)
        logger.info("No atomic lines in catalog - using continuum-only synthesis")
        buffers.line_opacity[:] = 0.0
        buffers.line_scattering[:] = 0.0
        metal_wings = np.zeros_like(buffers.line_opacity)
        metal_sources = np.zeros_like(buffers.line_opacity)

    # CRITICAL DEBUG: Check buffers.line_opacity before copying to abs_core_base
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
        ahline, shline = compute_hydrogen_wings(
            atm,
            freq,
            bnu,
            ehvkt,
            stim,
            hkt_vec,
        )
    else:
        if asynth_npz is not None:
            logger.info(
                "Skipping hydrogen wings (using fort.29 ASYNTH, which doesn't include wings)"
            )
        else:
            logger.info("Skipping hydrogen wings (--skip-hydrogen-wings)")
        ahline = np.zeros_like(buffers.line_opacity)
        shline = np.zeros_like(buffers.line_opacity)

    # Metal wings (non-hydrogen) computed with XLINOP-style tapering.
    if use_wings:
        logger.info("Computing metal line wings...")
        metal_tables = tables.metal_wing_tables()
        # fort.9 metadata is optional - if not provided, use catalog values directly
        metadata = (
            fort9_io.decode_metadata(fort9_data) if fort9_data is not None else None
        )
        catalog_wavelength = catalog.wavelength
        catalog_to_meta: Dict[int, int] = {}
        if metadata is not None and fort9_data is not None:
            # Match catalog to fort.9 metadata if available
            catalog_to_meta = _match_catalog_to_fort9(
                catalog_wavelength, fort9_data.wavelength
            )
        else:
            # No fort.9 metadata - all line properties come from catalog
            logger.info("No fort.9 metadata - using line properties from catalog")

        catalog_to_fort19: Dict[int, int] = {}
        he_solver: Optional[helium_profiles.HeliumWingSolver] = None
        if fort19_data is not None and cfg.enable_helium_wings:
            catalog_to_fort19 = _match_catalog_to_fort19(
                catalog_wavelength, fort19_data.wavelength_vacuum
            )
            he_solver = helium_profiles.HeliumWingSolver(
                temperature=atm.temperature,
                electron_density=atm.electron_density,
                xnfph=atm.xnfph,
                xnf_he2=atm.xnf_he2,
            )

        electron_density = np.maximum(atm.electron_density, 1e-40)
        inglis = 1600.0 / np.power(electron_density, 2.0 / 15.0)
        emerge = 109737.312 / np.maximum(inglis**2, 1e-12)
        emerge_h = 109677.576 / np.maximum(inglis**2, 1e-12)

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

                # DEBUG: Print sample population values for verification
                if element in {"Fe", "V", "Al"} and pop_densities.size > 0:
                    print(f"\nDEBUG: Population densities for {element}:")
                    sample_depths = [0, atm.layers // 2, atm.layers - 1]
                    for depth_idx in sample_depths:
                        if (
                            depth_idx < atm.layers
                            and depth_idx < pop_densities.shape[0]
                        ):
                            for ion_stage in range(min(3, pop_densities.shape[1])):
                                pop_val = float(pop_densities[depth_idx, ion_stage])
                                rho_val = (
                                    float(atm.mass_density[depth_idx])
                                    if hasattr(atm, "mass_density")
                                    and atm.mass_density is not None
                                    else 0.0
                                )
                                dop_val = float(dop_velocity[depth_idx])
                                print(
                                    f"  Depth {depth_idx}, Ion {ion_stage+1}: pop={pop_val:.6e} cm⁻³, rho={rho_val:.6e} g/cm³, dop={dop_val:.6e}"
                                )
                                if rho_val > 0:
                                    pop_per_rho = pop_val / rho_val
                                    print(
                                        f"    pop/rho = {pop_per_rho:.6e} (should be ~1e24 for typical values)"
                                    )
        logger.info(f"Pre-computed populations for {len(population_cache)} elements")

        # Determine number of workers for parallelization
        n_workers_metal = cfg.n_workers
        if n_workers_metal is None:
            import multiprocessing

            n_workers_metal = min(multiprocessing.cpu_count(), 8)
            logger.info(
                f"Auto-detected {multiprocessing.cpu_count()} CPUs, using {n_workers_metal} workers"
            )
        else:
            logger.info(f"Using {n_workers_metal} workers (from config)")

        # Determine if we should use Numba parallel (preferred) or ProcessPoolExecutor fallback
        use_numba_parallel = NUMBA_AVAILABLE and atm.layers >= 10
        use_parallel = use_numba_parallel or (atm.layers >= 10 and n_workers_metal > 1)

        if use_numba_parallel:
            logger.info(
                f"Using Numba parallel processing for {len(line_indices)} lines across {atm.layers} depth layers"
            )
        elif use_parallel:
            logger.info(
                f"Using ProcessPoolExecutor parallel processing for {atm.layers} depth layers with {n_workers_metal} workers"
            )
        else:
            logger.info(
                f"Using sequential processing ({atm.layers} layers, {n_workers_metal} workers)"
            )

        # Kernel is now defined at module level (compiles once)
        # Process lines in batches for progress logging

        def process_depth(
            depth_idx: int,
        ) -> Tuple[int, np.ndarray, np.ndarray, int, int]:
            """Process metal line wings for a single depth layer."""
            state = pops.layers[depth_idx]
            continuum_row = buffers.continuum[depth_idx]
            tmp_buffer = np.zeros_like(wavelength, dtype=np.float64)
            local_wings = np.zeros_like(wavelength, dtype=np.float64)
            local_sources = np.zeros_like(wavelength, dtype=np.float64)

            lines_processed = 0
            lines_skipped = 0

            for line_idx, center_index in enumerate(line_indices):
                # CRITICAL FIX: Match Fortran synthe.for line 301 EXACTLY
                # IF(NBUFF.LT.1.OR.NBUFF.GT.LENGTH)GO TO 320
                # Fortran SKIPS lines whose centers are outside the grid
                if center_index < 0 or center_index >= wavelength.size:
                    lines_skipped += 1
                    continue
                # Get metadata index if fort.9 is available
                meta_idx = (
                    catalog_to_meta.get(line_idx) if metadata is not None else None
                )
                # If fort.9 metadata exists but line not found, still allow it (use catalog values)
                # This allows mixing of lines with and without fort.9 metadata
                record = catalog.records[line_idx]
                element_symbol = str(record.element).strip().upper()
                if element_symbol in {"H", "H I", "HI"} and record.ion_stage == 1:
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
                            wing_type = fort19_io.Fort19WingType.from_code(
                                int(wing_val)
                            )

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
                        he_factor = (
                            0.628
                            * (2.0991e-4 * t_j * (0.25 + 1.008 / atomic_mass)) ** v2
                        )
                        h2_factor = (
                            1.08 * (2.0991e-4 * t_j * (0.5 + 1.008 / atomic_mass)) ** v2
                        )
                        xnfh_val = _layer_value(xnf_h_arr, depth_idx)
                        xnfhe_val = _layer_value(xnf_he1_arr, depth_idx)
                        xnfh2_val = _layer_value(xnf_h2_arr, depth_idx)
                        txnxn_line = (
                            xnfh_val * h_factor
                            + xnfhe_val * he_factor
                            + xnfh2_val * h2_factor
                        )

                line_wavelength = catalog.wavelength[line_idx]
                element = record.element
                # Normalize element symbol for cache lookup (same normalization as cache key)
                element_key = str(element).strip()

                # Get populations from cache (pre-computed)
                if element_key not in population_cache:
                    lines_skipped += 1
                    continue
                pop_densities, dop_velocity = population_cache[element_key]

                # Get population and Doppler for this ion stage
                if nelion > pop_densities.shape[1]:
                    lines_skipped += 1
                    continue  # Ion stage out of range

                pop_val = pop_densities[depth_idx, nelion - 1]
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
                    ifvac=fort9_data.ifvac if fort9_data is not None else 1,
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
                    if _apply_fort19_profile(
                        wing_type=wing_type,
                        line_type_code=line_type_code,
                        tmp_buffer=tmp_buffer,
                        continuum_row=continuum_row,
                        wavelength_grid=wavelength,
                        center_index=center_index,
                        line_wavelength=line_wavelength,
                        kappa0=kappa_auto,
                        cutoff=cfg.cutoff,
                        metal_wings_row=local_wings,
                        metal_sources_row=local_sources,
                        bnu_row=bnu[depth_idx],
                        wcon=wcon,
                        wtail=wtail,
                        he_solver=he_solver,
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
                    ):
                        lines_processed += 1
                        continue

                # Compute damping value (ADAMP in Fortran synthe.for line 299)
                # Fortran: ADAMP = (GAMRF + GAMSF*XNE + GAMWF*TXNXN) / DOPPLE(NELION)
                # Where GAMRF etc. are NORMALIZED gamma values (divided by 4π*freq in rgfall.for)
                # And DOPPLE is dimensionless (v_th/c)
                #
                # Catalog gamma is LINEAR (10^GR from gfallvac.latest)
                # Need to normalize: gamma_normalized = gamma_linear / (4π*freq)
                # Then: ADAMP = gamma_normalized / DOPPLE = gamma_linear / (4π*freq*dopple)
                dopple = (
                    doppler_width / line_wavelength if line_wavelength > 0 else dop_val
                )
                freq_hz = C_LIGHT_NM / line_wavelength if line_wavelength > 0 else 1e15
                delta_nu_doppler = freq_hz * dopple  # Doppler width in Hz
                gamma_total = (
                    gamma_rad
                    + gamma_stark * state.electron_density
                    + gamma_vdw * txnxn_line
                )
                damping_value = gamma_total / max(
                    4.0 * 3.14159265359 * delta_nu_doppler, 1e-40
                )

                # Apply fort.19 profile if available
                if _apply_fort19_profile(
                    wing_type=wing_type,
                    line_type_code=line_type_code,
                    tmp_buffer=tmp_buffer,
                    continuum_row=continuum_row,
                    wavelength_grid=wavelength,
                    center_index=center_index,
                    line_wavelength=line_wavelength,
                    kappa0=kappa0,
                    cutoff=cfg.cutoff,
                    metal_wings_row=local_wings,
                    metal_sources_row=local_sources,
                    bnu_row=bnu[depth_idx],
                    wcon=wcon,
                    wtail=wtail,
                    he_solver=he_solver,
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
                ):
                    lines_processed += 1
                    continue

                # Accumulate metal profile
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

                # Reset center (already handled by _accumulate_metal_profile, but ensure it's zero)
                # Only reset if center_index is within grid
                if 0 <= center_index < wavelength.size:
                    tmp_buffer[center_index] = 0.0

                # Accumulate into local buffers
                local_wings += tmp_buffer
                local_sources += tmp_buffer * bnu[depth_idx]
                lines_processed += 1

            # Return results (no locks needed - each depth writes to different indices)
            return depth_idx, local_wings, local_sources, lines_processed, lines_skipped

        if use_numba_parallel:
            # Numba parallel processing (no pickling overhead)
            import time

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

            # Pre-process metadata arrays
            has_metadata = metadata is not None
            if has_metadata:
                n_meta = len(metadata.ncon) if hasattr(metadata, "ncon") else 0
                line_meta_idx = np.full(n_lines, -1, dtype=np.int32)
                for line_idx, meta_idx in catalog_to_meta.items():
                    if line_idx < n_lines and meta_idx < n_meta:
                        line_meta_idx[line_idx] = meta_idx

                line_meta_ncon = (
                    np.asarray(metadata.ncon, dtype=np.int32)
                    if hasattr(metadata, "ncon")
                    else np.array([], dtype=np.int32)
                )
                line_meta_nelionx = (
                    np.asarray(metadata.nelionx, dtype=np.int32)
                    if hasattr(metadata, "nelionx")
                    else np.array([], dtype=np.int32)
                )
                line_meta_nelion = (
                    np.asarray(metadata.nelion, dtype=np.int32)
                    if hasattr(metadata, "nelion")
                    else np.array([], dtype=np.int32)
                )
                line_meta_alpha = (
                    np.asarray(metadata.extra1, dtype=np.float64)
                    if hasattr(metadata, "extra1")
                    else np.array([], dtype=np.float64)
                )
                line_meta_gamma_rad = (
                    np.asarray(metadata.gamma_rad, dtype=np.float64)
                    if hasattr(metadata, "gamma_rad")
                    else np.array([], dtype=np.float64)
                )
                line_meta_gamma_stark = (
                    np.asarray(metadata.gamma_stark, dtype=np.float64)
                    if hasattr(metadata, "gamma_stark")
                    else np.array([], dtype=np.float64)
                )
                line_meta_gamma_vdw = (
                    np.asarray(metadata.gamma_vdw, dtype=np.float64)
                    if hasattr(metadata, "gamma_vdw")
                    else np.array([], dtype=np.float64)
                )
            else:
                line_meta_idx = np.full(n_lines, -1, dtype=np.int32)
                line_meta_ncon = np.array([], dtype=np.int32)
                line_meta_nelionx = np.array([], dtype=np.int32)
                line_meta_nelion = np.array([], dtype=np.int32)
                line_meta_alpha = np.array([], dtype=np.float64)
                line_meta_gamma_rad = np.array([], dtype=np.float64)
                line_meta_gamma_stark = np.array([], dtype=np.float64)
                line_meta_gamma_vdw = np.array([], dtype=np.float64)

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
            boltzmann_factor_arr = np.zeros((n_lines, atm.layers), dtype=np.float64)

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
                boltzmann_factor_arr[:, depth_idx] = state.boltzmann_factor

            # Get Voigt tables
            voigt_tables = tables.voigt_tables()
            h0tab = voigt_tables.h0tab
            h1tab = voigt_tables.h1tab
            h2tab = voigt_tables.h2tab

            # Get metal tables contx array
            contx = metal_tables.contx
            ifvac_val = fort9_data.ifvac if fort9_data is not None else 1

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

            line_indices_arr = np.asarray(line_indices, dtype=np.int64)

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
                batch_line_gf = line_gf[batch_start:batch_end]
                batch_line_gamma_rad = line_gamma_rad[batch_start:batch_end]
                batch_line_gamma_stark = line_gamma_stark[batch_start:batch_end]
                batch_line_gamma_vdw = line_gamma_vdw[batch_start:batch_end]
                batch_line_element_idx = line_element_idx[batch_start:batch_end]
                batch_line_nelion = line_nelion[batch_start:batch_end]
                batch_line_meta_idx = line_meta_idx[batch_start:batch_end]
                batch_boltzmann_factor = boltzmann_factor_arr[batch_start:batch_end, :]

                # Call kernel for this batch
                _process_metal_wings_kernel(
                    metal_wings,
                    metal_sources,
                    wavelength,
                    batch_line_indices,
                    batch_line_wavelengths,
                    batch_line_gf,
                    batch_line_gamma_rad,
                    batch_line_gamma_stark,
                    batch_line_gamma_vdw,
                    batch_line_element_idx,
                    batch_line_nelion,
                    batch_line_meta_idx,
                    line_meta_ncon,
                    line_meta_nelionx,
                    line_meta_nelion,
                    line_meta_alpha,
                    line_meta_gamma_rad,
                    line_meta_gamma_stark,
                    line_meta_gamma_vdw,
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

            # Handle fort.19 special wing types (fallback to Python for complex cases)
            # Note: This is a simplified version - full fort.19 support would require
            # more complex Numba kernels or keeping Python fallback
            if fort19_data is not None and len(catalog_to_fort19) > 0:
                logger.info(
                    "Processing fort.19 special wing types (Python fallback)..."
                )
                # For now, we skip fort.19 special types in Numba kernel
                # Full implementation would require additional kernels

        elif use_parallel:
            # Fallback to ProcessPoolExecutor if Numba not available
            import time

            start_time = time.time()
            logger.info(
                f"Starting ProcessPoolExecutor parallel processing of {atm.layers} depth layers with {n_workers_metal} workers..."
            )

            # Prepare arguments for standalone function (extract data needed for pickling)
            process_args = []
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
                    fort9_data.ifvac if fort9_data is not None else 1,
                    cfg.cutoff,
                    bnu_row,
                )
                process_args.append(args_tuple)

            with ProcessPoolExecutor(max_workers=n_workers_metal) as executor:
                futures = {
                    executor.submit(_process_metal_wings_depth_standalone, args): args[
                        0
                    ]
                    for args in process_args
                }

                completed = 0
                total_lines_processed = 0
                total_lines_skipped = 0
                last_log_time = start_time

                for future in as_completed(futures):
                    try:
                        (
                            depth_idx,
                            local_wings,
                            local_sources,
                            lines_proc,
                            lines_skip,
                        ) = future.result()
                        completed += 1
                        total_lines_processed += lines_proc
                        total_lines_skipped += lines_skip

                        # Accumulate results (no locks needed - different indices)
                        metal_wings[depth_idx] += local_wings
                        metal_sources[depth_idx] += local_sources

                        # Log immediately on first completion
                        if completed == 1:
                            elapsed = time.time() - start_time
                            logger.info(
                                f"✓ First depth completed in {elapsed:.2f}s "
                                f"(estimated {elapsed * atm.layers / n_workers_metal:.0f}s total)"
                            )

                        # Log progress more frequently
                        current_time = time.time()
                        if (
                            completed == 1  # Always log first
                            or completed % max(1, min(atm.layers // 20, 5))
                            == 0  # Every 5% or every 5 completions
                            or completed == atm.layers
                            or (current_time - last_log_time)
                            >= 2.0  # Or every 2 seconds
                        ):
                            elapsed = current_time - start_time
                            rate = completed / elapsed if elapsed > 0 else 0
                            remaining = (
                                (atm.layers - completed) / rate if rate > 0 else 0
                            )
                            logger.info(
                                f"Progress: {completed}/{atm.layers} depths completed "
                                f"({100.0*completed/atm.layers:.1f}%) - "
                                f"{rate:.2f} depths/s, ~{remaining:.1f}s remaining"
                            )
                            last_log_time = current_time
                    except Exception as e:
                        depth_idx = futures[future]
                        logger.error(
                            f"Error processing depth {depth_idx}: {e}", exc_info=True
                        )
                        raise

            elapsed_time = time.time() - start_time
            logger.info(
                f"Completed ProcessPoolExecutor parallel processing: {atm.layers} depths in {elapsed_time:.2f}s "
                f"({atm.layers/elapsed_time:.2f} depths/s)"
            )
            logger.info(
                f"Lines processed: {total_lines_processed:,} total, "
                f"{total_lines_processed/atm.layers:.0f} per depth on average"
            )
            if total_lines_skipped > 0:
                logger.info(f"Lines skipped: {total_lines_skipped:,} total")
        else:
            # Sequential processing
            import time

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
                    _, local_wings, local_sources, lines_proc, lines_skip = (
                        process_depth(depth_idx)
                    )
                    # Accumulate results
                    metal_wings[depth_idx] += local_wings
                    metal_sources[depth_idx] += local_sources
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

    if use_wings:
        logger.info("Metal wings computation complete")
    else:
        # No lines - wings are already zero
        logger.info("No lines - skipping metal wings")

    # --- line source reconstruction -------------------------------------------------
    # Compute hydrogen wings for source function
    if use_wings and not cfg.skip_hydrogen_wings:
        logger.info("Computing hydrogen wings for source function...")
        ahline, shline = compute_hydrogen_wings(
            atm,
            freq,
            bnu,
            ehvkt,
            stim,
            hkt_vec,
        )
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

    # Reconstruct SXLINE (metal-wing source function state)
    with np.errstate(divide="ignore", invalid="ignore"):
        sxline = np.divide(
            metal_sources,
            np.maximum(metal_wings, 1e-40),
            out=np.zeros_like(metal_sources),
            where=metal_wings > 1e-40,
        )

    abs_core = abs_core_base
    total_line_absorption = abs_core + ahline + metal_wings

    # CRITICAL FIX: Fortran spectrv.for line 300: ALINE(J) = ASYNTH(J) * (1 - FSCAT(J))
    # This does NOT include metal_wings! Metal_wings are handled separately in synthe.for.
    # When using ASYNTH mode, Python should NOT add metal_wings to ALINE!
    # Python already set buffers.line_opacity = absorption = asynth * (1 - fscat) earlier (line 1522)
    # So we should NOT overwrite it with total_line_absorption that includes metal_wings!
    using_asynth = asynth_npz is not None or has_lines
    if not using_asynth:
        # Only overwrite buffers.line_opacity if NOT using ASYNTH mode
        # CRITICAL DEBUG: Check values before overwriting buffers.line_opacity
        print("\n" + "=" * 70)
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
        # Do NOT overwrite with total_line_absorption that includes metal_wings!
        print("\n" + "=" * 70)
        print(
            "CRITICAL: Using ASYNTH mode - NOT overwriting buffers.line_opacity with metal_wings!"
        )
        print("=" * 70)
        print("  Fortran spectrv.for line 300: ALINE(J) = ASYNTH(J) * (1 - FSCAT(J))")
        print("  This does NOT include metal_wings!")
        print(
            "  Python buffers.line_opacity was already set to absorption = asynth * (1 - fscat)"
        )
        print("  Keeping it as is (NOT adding metal_wings)")
        print("=" * 70 + "\n")
    buffers.line_scattering[:] = alinec_total * fscat_vec[:, None]

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
        # Using ASYNTH (from fort.29 or computed): line source is slinec (bfudge-corrected Planck)
        line_source = slinec.copy()
        print(f"  -> Using slinec directly (ASYNTH mode)")
    else:
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
        if np.any(metal_wings > 1e10):
            print(
                f"\nWARNING: Large metal_wings detected! Max: {np.max(metal_wings):.2e}"
            )
            print(f"  metal_source max: {np.max(metal_source):.2e}")
            print(f"  metal_contribution max: {np.max(metal_contribution):.2e}")
            print(
                f"  numerator before adding metal: max={np.max(numerator):.2e}, non-zero count={np.count_nonzero(numerator)}"
            )
        numerator = numerator + metal_contribution
        if np.any(metal_wings > 1e10):
            print(
                f"  numerator after adding metal: max={np.max(numerator):.2e}, non-zero count={np.count_nonzero(numerator)}"
            )
            print(
                f"  Check: Is metal_contribution finite? {np.all(np.isfinite(metal_contribution))}"
            )
            print(f"  Check: Is numerator finite? {np.all(np.isfinite(numerator))}")

        # DEBUG: Detailed breakdown of line source computation
        print("\n" + "=" * 70)
        print("DEBUG: Line source computation breakdown")
        print("=" * 70)
        # Sample a few wavelengths to analyze
        sample_indices = [
            0,
            wavelength.size // 4,
            wavelength.size // 2,
            3 * wavelength.size // 4,
            wavelength.size - 1,
        ]
        for idx in sample_indices:
            if idx < wavelength.size:
                # These are 2D arrays (depth, wavelength), so index with [0, idx] for surface layer
                depth_idx = 0  # Surface layer for debugging
                print(
                    f"\nWavelength index {idx} (λ={float(wavelength[idx]):.6f} nm), depth {depth_idx}:"
                )
                print(f"  ahline: {float(ahline[depth_idx, idx]):.6e}")
                print(f"  abs_core: {float(abs_core[depth_idx, idx]):.6e}")
                print(f"  metal_wings: {float(metal_wings[depth_idx, idx]):.6e}")
                print(
                    f"  total_line_absorption: {float(total_line_absorption[depth_idx, idx]):.6e}"
                )
                tot_abs = float(total_line_absorption[depth_idx, idx])
                print(
                    f"  Ratio: ahline={float(ahline[depth_idx, idx])/max(tot_abs,1e-40):.2%}, "
                    f"abs_core={float(abs_core[depth_idx, idx])/max(tot_abs,1e-40):.2%}, "
                    f"metal_wings={float(metal_wings[depth_idx, idx])/max(tot_abs,1e-40):.2%}"
                )
                print(f"  shline: {float(shline[depth_idx, idx]):.6e}")
                print(f"  slinec: {float(slinec[depth_idx, idx]):.6e}")
                print(f"  sxline: {float(sxline[depth_idx, idx]):.6e}")
                print(f"  bnu: {float(bnu[depth_idx, idx]):.6e}")
                metal_src_val = float(metal_source[depth_idx, idx])
                print(f"  metal_source (sxline or bnu): {metal_src_val:.6e}")
                num_val = float(numerator[depth_idx, idx])
                print(
                    f"  numerator = ahline*shline + abs_core*slinec + metal_wings*metal_source: {num_val:.6e}"
                )
                print(
                    f"  numerator / total_line_absorption: {num_val/max(tot_abs,1e-40):.6e}"
                )
        print("=" * 70 + "\n")

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

            n_workers = min(multiprocessing.cpu_count(), 8)  # Cap at 8 workers
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

    # CRITICAL DEBUG: Check if flux values are identical
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
            print("ERROR: flux_total and flux_cont are IDENTICAL at ALL wavelengths!")
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

    logger.info("Converting flux units...")
    result = SynthResult(
        wavelength=buffers.wavelength.copy(),
        intensity=flux_total,
        continuum=flux_cont,
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
    return result
