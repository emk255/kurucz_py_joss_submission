"""
Line opacity computation (TRANSP) for removing fort.9/fort.29 dependency.

This module computes line opacity at line center (TRANSP) from first principles,
following the Fortran XLINOP subroutine implementation.

Key formula from synthe.for line 692:
    KAPCEN = KAPPA0 * VOIGT(0., ADAMP)

Where:
    KAPPA0 = CGF * XNFDOP(NELION) * BOLT
    CGF = (0.026538/1.77245) * GF / FREQ  (from rgfall.for line 267)
    XNFDOP = XNFPEL / (RHO * DOPPLE) (population per unit mass per Doppler width)
    BOLT = exp(-ELO * HCKT) (Boltzmann factor)
    ADAMP = (GAMMAR + GAMMAS*XNE + GAMMAW*TXNXN) / DOPPLE
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Optional, Tuple, Dict
import logging
import numpy as np

from .profiles import voigt_profile
from . import tables

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


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..io.atmosphere import AtmosphereModel
    from ..io.lines.atomic import LineCatalog
    from ..physics.populations import Populations

# Constants matching Fortran
C_LIGHT_CM = 2.99792458e10  # cm/s
C_LIGHT_KM = 299792.458  # km/s
C_LIGHT_NM = 2.99792458e17  # nm/s (for frequency calculation)
H_PLANCK = 6.62607015e-27  # erg * s
K_BOLTZ = 1.380649e-16  # erg / K

# CGF conversion constants from rgfall.for line 267
CGF_CONSTANT = 0.026538 / 1.77245  # Factor for converting GF to CONGF
# Fortran synthe.for PARAMETER: MAXPROF=1000000
MAX_PROFILE_STEPS = 1_000_000


# Shared Voigt profile — single canonical JIT-compiled implementation
from synthe_py.physics.voigt_jit import voigt_profile_jit as _voigt_profile_jit


@jit(
    nopython=True, parallel=False, cache=True
)  # CRITICAL: parallel=False to avoid race conditions
def _compute_asynth_wings_kernel(
    asynth: np.ndarray,
    wavelength_grid: np.ndarray,
    transp: np.ndarray,
    valid_mask: np.ndarray,
    line_wavelengths: np.ndarray,
    line_indices: np.ndarray,
    line_types: np.ndarray,
    stim_factors: np.ndarray,
    kappa0_values: np.ndarray,
    adamp_values: np.ndarray,
    doppler_widths: np.ndarray,
    gamma_rad_values: np.ndarray,
    gamma_stark_values: np.ndarray,
    gamma_vdw_values: np.ndarray,
    kapmin_ref_values: np.ndarray,
    continuum_absorption: np.ndarray,
    wcon_values: np.ndarray,
    wtail_values: np.ndarray,
    cutoff: float,
    max_profile_steps: int,
    h0tab: np.ndarray,
    h1tab: np.ndarray,
    h2tab: np.ndarray,
) -> None:
    """JIT-compiled kernel for computing ASYNTH wing contributions.

    CRITICAL FIX (Dec 2025): Match Fortran N10DOP logic exactly.
    Fortran synthe.for line 311: N10DOP = 10 * (DOPPLE * RESOLU)
    If N10DOP = 0 (which happens when DOPPLE*RESOLU < 0.1), NO wings are computed.
    This is critical for high-resolution spectra where Doppler widths are << grid spacing.

    NOTE: parallel=False is required because multiple lines can contribute to the same
    wavelength bin via their wings. With parallel=True, the += operations create race
    conditions that cause ~50% of contributions to be lost.
    """
    n_lines = transp.shape[0]
    n_depths = transp.shape[1]
    n_wavelengths = wavelength_grid.size

    use_cutoff = continuum_absorption.size > 0
    use_wcon = wcon_values.size > 0

    # Compute RESOLU from wavelength grid (matches Fortran)
    # RESOLU = 1 / (ratio - 1) where ratio = wavelength[i+1] / wavelength[i]
    resolu = 300000.0  # Default fallback
    if n_wavelengths > 1:
        ratio = wavelength_grid[1] / wavelength_grid[0]
        if ratio > 1.0:
            resolu = 1.0 / (ratio - 1.0)

    for line_idx in range(n_lines):  # Sequential loop to avoid race conditions
        line_wavelength = line_wavelengths[line_idx]
        center_idx = line_indices[line_idx]
        line_type = line_types[line_idx]

        # CRITICAL FIX: Skip fort.19 special lines (line_type != 0).
        # Hydrogen (type -1/-2) → _compute_hydrogen_line_opacity (HPROF4 profile)
        # Autoionizing (type 1) → _add_fort19_asynth (Lorentz profile)
        # Helium (type -3/-6) → helium wings path
        # Processing them here with Voigt/table profiles DOUBLE-COUNTS them
        # with the wrong profile shape.
        if line_type != 0:
            continue

        # Allow wing contributions from lines whose centers fall just outside the grid.
        # This is required to match full-range Fortran runs when we synthesize a subrange.
        if (
            center_idx < -max_profile_steps
            or center_idx > n_wavelengths - 1 + max_profile_steps
        ):
            continue

        for depth_idx in range(n_depths):
            if not valid_mask[line_idx, depth_idx]:
                continue

            transp_val = transp[line_idx, depth_idx]
            if transp_val <= 0.0:
                continue

            kappa0 = kappa0_values[line_idx, depth_idx]
            adamp = adamp_values[line_idx, depth_idx]
            doppler_width = doppler_widths[line_idx, depth_idx]
            stim_factor = stim_factors[line_idx, depth_idx]

            if line_type == 1:
                # Autoionizing line (Fortran synthe.for label 700)
                gamma_rad = gamma_rad_values[line_idx]
                ashore = gamma_stark_values[line_idx]
                bshore = gamma_vdw_values[line_idx]
                if gamma_rad <= 0.0 or bshore <= 0.0:
                    continue

                # Use per-position cutoff (Fortran checks after add)
                maxstep = max_profile_steps
                offset = 1
                red_active = True
                blue_active = True
                freq_line = C_LIGHT_NM / line_wavelength

                while offset <= maxstep and (red_active or blue_active):
                    # Red wing
                    if red_active:
                        idx = center_idx + offset
                        if idx < 0:
                            # Center below grid; wait for offset to bring idx in-range.
                            pass
                        elif idx >= n_wavelengths:
                            red_active = False
                        else:
                            freq = C_LIGHT_NM / wavelength_grid[idx]
                            epsil = 2.0 * (freq - freq_line) / gamma_rad
                            profile_val = (
                                kappa0
                                * (ashore * epsil + bshore)
                                / (epsil * epsil + 1.0)
                                / bshore
                            )
                            value_red = profile_val * stim_factor
                            asynth[depth_idx, idx] += value_red
                            if use_cutoff:
                                kapmin_at_idx = continuum_absorption[0, idx] * cutoff
                                if value_red < kapmin_at_idx:
                                    red_active = False

                    # Blue wing
                    if blue_active:
                        idx = center_idx - offset
                        if idx < 0:
                            blue_active = False
                        elif idx >= n_wavelengths:
                            # Center is above grid; wait for offset to bring idx in-range.
                            pass
                        else:
                            freq = C_LIGHT_NM / wavelength_grid[idx]
                            epsil = 2.0 * (freq - freq_line) / gamma_rad
                            profile_val = (
                                kappa0
                                * (ashore * epsil + bshore)
                                / (epsil * epsil + 1.0)
                                / bshore
                            )
                            value_blue = profile_val * stim_factor
                            asynth[depth_idx, idx] += value_blue
                            if use_cutoff:
                                kapmin_at_idx = continuum_absorption[0, idx] * cutoff
                                if value_blue < kapmin_at_idx:
                                    blue_active = False

                    offset += 1

                continue

            if doppler_width <= 0.0:
                continue

            # AUTOIONIZING LINES (TYPE=1): Lorentzian wings with ASHORE/BSHORE
            # Fortran synthe.for label 700:
            #   KAPPA = KAPPA0*(ASHORE*EPSIL+BSHORE)/(EPSIL**2+1)/BSHORE
            #   EPSIL = 2*(FREQ-FRELIN)/GAMMAR
            if line_type == 1:
                gamma_rad = gamma_rad_values[line_idx]
                bshore = gamma_vdw_values[line_idx]
                ashore = gamma_stark_values[line_idx]
                if gamma_rad <= 0.0 or bshore <= 0.0:
                    continue

                maxstep = center_idx
                if n_wavelengths - center_idx - 1 > maxstep:
                    maxstep = n_wavelengths - center_idx - 1

                offset = 1
                red_active = True
                blue_active = True
                while offset <= maxstep and (red_active or blue_active):
                    # Red wing: add then cutoff check
                    if red_active:
                        idx = center_idx + offset
                        if idx < 0:
                            # Center below grid; wait for offset to bring idx in-range.
                            pass
                        elif idx >= n_wavelengths:
                            red_active = False
                        else:
                            freq = C_LIGHT_NM / wavelength_grid[idx]
                            frelin = C_LIGHT_NM / line_wavelength
                            epsil = 2.0 * (freq - frelin) / gamma_rad
                            profile_val = (
                                kappa0
                                * (ashore * epsil + bshore)
                                / (epsil * epsil + 1.0)
                                / bshore
                            )
                            profile_val *= stim_factor
                            asynth[depth_idx, idx] += profile_val
                            if use_cutoff:
                                kapmin_at_idx = continuum_absorption[0, idx] * cutoff
                                if profile_val < kapmin_at_idx:
                                    red_active = False

                    # Blue wing: add then cutoff check
                    if blue_active:
                        idx = center_idx - offset
                        if idx < 0:
                            blue_active = False
                        else:
                            freq = C_LIGHT_NM / wavelength_grid[idx]
                            frelin = C_LIGHT_NM / line_wavelength
                            epsil = 2.0 * (freq - frelin) / gamma_rad
                            profile_val = (
                                kappa0
                                * (ashore * epsil + bshore)
                                / (epsil * epsil + 1.0)
                                / bshore
                            )
                            profile_val *= stim_factor
                            asynth[depth_idx, idx] += profile_val
                            if use_cutoff:
                                kapmin_at_idx = continuum_absorption[0, idx] * cutoff
                                if profile_val < kapmin_at_idx:
                                    blue_active = False

                    offset += 1
                continue

            # CRITICAL FIX: Compute N10DOP to match Fortran behavior exactly
            # Fortran synthe.for line 311: N10DOP = 10 * (DOPPLE * RESOLU)
            # DOPPLE is dimensionless: doppler_width / line_wavelength
            dopple = doppler_width / line_wavelength if line_wavelength > 0.0 else 1e-10
            n10dop = int(10.0 * dopple * resolu)

            # Keep at least one wing step when N10DOP rounds to zero.
            # This preserves weak far-wing contributions from very narrow lines that
            # still affect nearby grid points in Fortran parity runs.
            if n10dop == 0:
                n10dop = 1

            # Get WCON/WTAIL for this line/depth (if available)
            wcon = -1.0  # Use -1.0 as sentinel for "not set"
            wtail = -1.0
            if use_wcon:
                idx_wcon = line_idx * n_depths + depth_idx
                if idx_wcon < wcon_values.size:
                    wcon_val = wcon_values[idx_wcon]
                    if wcon_val > 0.0:
                        wcon = wcon_val
                        if idx_wcon < wtail_values.size:
                            wtail_val = wtail_values[idx_wcon]
                            if wtail_val > 0.0:
                                wtail = wtail_val

            # Wing contributions (center contributions are added separately)
            # All lines are now in-grid (we skip out-of-grid lines above to match Fortran)
            red_active = True
            blue_active = True
            offset = 1

            # CRITICAL FIX (Dec 2025): Use DYNAMIC continuum at EACH WING POSITION
            # Fortran synthe.for line 767: IF(KAPPA.LT.CONTINUUM(IBUFF)*CUTOFF)GO TO 212
            # The IBUFF changes with each wing iteration, so cutoff threshold varies!
            # Previous code incorrectly used kapmin_center (continuum at line center) everywhere.

            # For MAXSTEP estimation, use depth-specific KAPMIN at line center.
            kapmin_ref = kapmin_ref_values[line_idx, depth_idx] if use_cutoff else 0.0

            # CRITICAL FIX (Dec 2025): Match Fortran XLINOP behavior EXACTLY
            #
            # Fortran XLINOP (synthe.for lines 757-786) for gfallvac lines:
            # 1. Use FULL VOIGT at every wing step (not 1/x^2 approximation)
            # 2. Per-step cutoff check: IF(KAPPA.LT.CONTINUUM(IBUFF)*CUTOFF)
            # 3. Red wing: Check BEFORE adding (line 767-768)
            # 4. Blue wing: Add FIRST, then check (lines 784-785)
            #
            # Key differences from previous Python code:
            # - KAPMIN check uses continuum at LINE CENTER, not wing position
            # - Near-wing KAPMIN check exits BOTH wings, not just one
            # - If near-wing exits early, far-wing is SKIPPED entirely

            # Pre-compute PROFILE array (matching Fortran's PROFILE(NSTEP))
            # This stores kappa0 * voigt (no stim_factor - that's applied later)
            dvoigt = 1.0 / (dopple * resolu) if dopple > 0 else 1.0

            # Phase 1: Near-wing profile with KAPMIN check at line center
            nstep_cutoff = n10dop  # Max near-wing step before cutoff
            profile_at_n10dop = 0.0
            # Fortran XLINOP uses H0TAB/H1TAB for ADAMP < 0.2
            vsteps = 200.0
            tabstep = vsteps * dvoigt
            tabi = 0.5  # 0-based indexing (Fortran uses 1.5 for 1-based arrays)
            for nstep in range(1, n10dop + 1):
                if adamp < 0.2:
                    # Match Fortran's incremental TABI update to preserve rounding behavior.
                    tabi += tabstep
                    idx = int(tabi)
                    if idx < 0:
                        idx = 0
                    x_step = float(nstep) * dvoigt
                    if x_step > 10.0:
                        profile_val = kappa0 * (0.5642 * adamp / (x_step * x_step))
                    else:
                        if idx >= h0tab.size:
                            idx = h0tab.size - 1
                        profile_val = kappa0 * (h0tab[idx] + adamp * h1tab[idx])
                else:
                    x_step = float(nstep) * dvoigt
                    voigt_val = _voigt_profile_jit(x_step, adamp, h0tab, h1tab, h2tab)
                    profile_val = kappa0 * voigt_val  # No stim_factor here
                if nstep == n10dop:
                    profile_at_n10dop = profile_val
                # Check against KAPMIN at LINE CENTER (kapmin_ref)
                if use_cutoff and profile_val < kapmin_ref:
                    nstep_cutoff = nstep
                    break
            else:
                # Near-wing completed without cutoff - compute far-wing X
                nstep_cutoff = -1  # Flag: no early cutoff

            # Phase 2: Far-wing setup
            #
            # CRITICAL FIX (Dec 2025): Match Fortran XLINOP behavior exactly.
            # Fortran XLINOP does NOT pre-limit maxstep based on near-wing cutoff!
            # Instead, it uses MAXBLUE = NBUFF-1 (line center index - 1), allowing
            # wings to extend all the way to the grid start if the per-step cutoff
            # doesn't terminate them earlier.
            #
            # The per-step cutoff checks in the wing loop (lines 309-310, 343-344)
            # will naturally terminate wings when profile value falls below kapmin.
            # This allows wings to extend further than the old nstep_cutoff limit
            # when XLINOP's full Voigt profile has higher far-wing values.
            #
            # Previous code (WRONG):
            #   if nstep_cutoff == -1:
            #       maxstep = max_profile_steps
            #   else:
            #       maxstep = nstep_cutoff  # <-- This was too restrictive!
            #
            # If near-wing cutoff triggers, Fortran skips far wings entirely.
            if nstep_cutoff != -1:
                maxstep = nstep_cutoff
                use_far_wing = False
                x_far = 0.0
            else:
                # Fortran far-wing: X = PROFILE(N10DOP) * N10DOP**2
                # MAXSTEP = SQRT(X / KAPMIN) + 1, capped by MAXPROF
                use_far_wing = True
                if n10dop > 0 and profile_at_n10dop > 0.0 and kapmin_ref > 0.0:
                    x_far = profile_at_n10dop * float(n10dop) ** 2
                    maxstep = int(np.sqrt(x_far / kapmin_ref) + 1.0)
                else:
                    x_far = 0.0
                    maxstep = 0
                if maxstep > max_profile_steps:
                    maxstep = max_profile_steps

            # Phase 3: Apply profile to both red and blue wings
            tabi_offset = 0.5  # 0-based indexing (Fortran uses 1.5 for 1-based arrays)
            while offset <= maxstep and (red_active or blue_active):
                # Compute profile value for this offset (Fortran near-wing vs far-wing)
                if use_far_wing and offset > n10dop:
                    profile_val = x_far / float(offset) ** 2
                else:
                    if adamp < 0.2:
                        tabi_offset += tabstep
                        idx = int(tabi_offset)
                        if idx < 0:
                            idx = 0
                        x_offset = float(offset) * dvoigt
                        if x_offset > 10.0:
                            profile_val = kappa0 * (
                                0.5642 * adamp / (x_offset * x_offset)
                            )
                        else:
                            if idx >= h0tab.size:
                                idx = h0tab.size - 1
                            profile_val = kappa0 * (h0tab[idx] + adamp * h1tab[idx])
                    else:
                        x_offset = float(offset) * dvoigt
                        voigt_val = _voigt_profile_jit(
                            x_offset, adamp, h0tab, h1tab, h2tab
                        )
                        profile_val = kappa0 * voigt_val
                profile_val = profile_val * stim_factor

                # Process red wing
                # Fortran XLINOP (lines 767-768): Check BEFORE adding, exit if below cutoff
                if red_active:
                    idx = center_idx + offset
                    if idx < 0:
                        pass  # Below grid, will reach it as offset increases
                    elif idx >= n_wavelengths:
                        red_active = False
                    else:
                        wave = wavelength_grid[idx]
                        skip_red = wcon > 0.0 and wave < wcon
                        if not skip_red:
                            value_red = profile_val

                            # Taper between WCON and WTAIL
                            if wtail > 0.0 and wcon > 0.0 and wave < wtail:
                                taper = (wave - wcon) / max(wtail - wcon, 1e-10)
                                value_red = value_red * taper

                            # Fortran uses KAPMIN at line center to set MAXSTEP,
                            # no per-position cutoff in the wing loop.
                            asynth[depth_idx, idx] += value_red

                # Process blue wing
                # Fortran XLINOP (lines 784-785): Add FIRST, then check for exit
                if blue_active:
                    idx = center_idx - offset
                    if idx < 0:
                        blue_active = False
                    elif idx >= n_wavelengths:
                        # Center is above grid; wait for offset to bring idx in-range.
                        pass
                    else:
                        wave = wavelength_grid[idx]
                        skip_blue = wcon > 0.0 and wave < wcon
                        if not skip_blue:
                            value_blue = profile_val

                            # Taper between WCON and WTAIL
                            if wtail > 0.0 and wcon > 0.0 and wave < wtail:
                                taper = (wave - wcon) / max(wtail - wcon, 1e-10)
                                value_blue = value_blue * taper

                            # XLINOP behavior (line 784-785): Add FIRST, then check
                            asynth[depth_idx, idx] += value_blue

                            # Fortran uses KAPMIN at line center to set MAXSTEP,
                            # no per-position cutoff in the wing loop.

                offset += 1


def compute_transp(
    catalog: "LineCatalog",
    populations: "Populations",
    atmosphere: "AtmosphereModel",
    cutoff: float = 1e-3,
    continuum_absorption: Optional[np.ndarray] = None,
    wavelength_grid: Optional[np.ndarray] = None,
    continuum_absorption_full: Optional[np.ndarray] = None,
    wavelength_grid_full: Optional[np.ndarray] = None,
    microturb_kms: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute TRANSP (line opacity at line center) for all lines and depths.

    This is the core computation that replaces fort.9/fort.29 dependency.

    Parameters
    ----------
    catalog:
        Line catalog containing line properties (gf, wavelength, excitation, etc.)
    populations:
        Pre-computed populations and Doppler widths for all depths
    atmosphere:
        Atmosphere model with temperature, electron density, etc.
    cutoff:
        Opacity cutoff factor (lines below this are ignored)
    continuum_absorption:
        Continuum absorption array, shape (n_depths, n_wavelengths).
        Used for KAPMIN = CONTINUUM * CUTOFF check (matches Fortran exactly).
    wavelength_grid:
        Wavelength grid for mapping lines to grid indices.
        Required when continuum_absorption is provided.

    Returns
    -------
    transp:
        Array of shape (n_lines, n_depths) containing line opacity at line center
    valid_mask:
        Boolean array of shape (n_lines, n_depths) indicating which lines/depths are valid
    line_indices:
        Array of line indices that contribute (for wavelength grid mapping)

    Notes
    -----
    TRANSP computation follows synthe.for XLINOP:
    1. KAPPA0 = gf * (population / doppler_width) * exp(-E/kT)
    2. ADAMP = (gamma_rad + gamma_stark*XNE + gamma_vdw*TXNXN) / doppler_width
    3. KAPCEN = KAPPA0 * VOIGT(0, ADAMP)
    """
    n_lines = len(catalog.records)
    n_depths = atmosphere.layers

    logger.info(f"Computing TRANSP for {n_lines:,} lines across {n_depths} depths...")

    # Initialize output arrays
    transp = np.zeros((n_lines, n_depths), dtype=np.float64)
    valid_mask = np.zeros((n_lines, n_depths), dtype=bool)

    # Progress logging
    log_interval = max(1, n_lines // 20)  # Log every 5%

    # Pre-compute center indices for all lines
    # This is used for KAPMIN = CONTINUUM(center_idx) * CUTOFF (matches Fortran exactly)
    # Fortran has no fallback - KAPMIN always uses CONTINUUM * CUTOFF
    if continuum_absorption is None or wavelength_grid is None:
        raise ValueError(
            "continuum_absorption and wavelength_grid are required for compute_transp. "
            "Fortran always uses KAPMIN = CONTINUUM * CUTOFF with no fallback."
        )

    from ..engine.opacity import _nearest_grid_indices

    index_wavelength = (
        catalog.index_wavelength
        if hasattr(catalog, "index_wavelength")
        else catalog.wavelength
    )
    center_indices = _nearest_grid_indices(wavelength_grid, index_wavelength)
    center_indices_full = None
    if continuum_absorption_full is not None and wavelength_grid_full is not None:
        center_indices_full = _nearest_grid_indices(
            wavelength_grid_full, index_wavelength
        )
    n_wavelengths = len(wavelength_grid)
    logger.info(f"Using dynamic KAPMIN = CONTINUUM * CUTOFF (Fortran-matching)")

    # Population data comes from NPZ (computed by pops_exact in convert_atm_to_npz.py)

    # Compute TXNXN if not available
    xnf_h = atmosphere.xnf_h if atmosphere.xnf_h is not None else np.zeros(n_depths)
    xnf_he1 = (
        atmosphere.xnf_he1 if atmosphere.xnf_he1 is not None else np.zeros(n_depths)
    )
    xnf_h2 = atmosphere.xnf_h2 if atmosphere.xnf_h2 is not None else np.zeros(n_depths)

    # Cache population computations per element to avoid redundant calculations
    # Format: {element: (pop_densities, dop_velocity)}
    population_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # DEBUG: Track cutoff statistics for diagnosis
    debug_stats = {
        "total_line_depth_pairs": 0,
        "skipped_pre_boltz": 0,
        "skipped_post_boltz": 0,
        "skipped_zero_kappa": 0,
        "passed_cutoff": 0,
        "lines_near_300nm": 0,  # Lines between 299-301nm
        "passed_near_300nm": 0,
        "kapmin_at_300nm": [],
        "kappa0_at_300nm": [],
    }
    DEBUG_WL_MIN = 299.5  # nm
    DEBUG_WL_MAX = 300.5  # nm

    debug_transp_wl = os.getenv("PY_DEBUG_TRANSP_WL")
    debug_transp_depth = os.getenv("PY_DEBUG_TRANSP_DEPTH")
    debug_transp_tol = float(os.getenv("PY_DEBUG_TRANSP_TOL", "2e-4"))
    debug_transp_wl_val = None
    debug_transp_depth_idx = None
    if debug_transp_wl and debug_transp_depth:
        try:
            debug_transp_wl_val = float(debug_transp_wl)
            depth_val = int(debug_transp_depth)
            if depth_val > 0:
                debug_transp_depth_idx = depth_val - 1
        except ValueError:
            debug_transp_wl_val = None
            debug_transp_depth_idx = None

    include_h_lines = os.getenv("PY_INCLUDE_H_LINES") == "1"

    # Process each line
    for line_idx in range(n_lines):
        if line_idx % log_interval == 0:
            logger.info(
                f"TRANSP progress: {line_idx:,}/{n_lines:,} lines ({100.0*line_idx/n_lines:.1f}%)"
            )

        record = catalog.records[line_idx]
        element = record.element
        nelion = record.ion_stage

        # Hydrogen lines are handled with HPROF4-style profiles elsewhere unless overridden.
        if not include_h_lines and (
            record.line_type == -1
            or (str(element).strip().upper() in {"H", "H I", "HI"} and nelion == 1)
        ):
            continue

        # CRITICAL FIX: Use pre-computed QXNFPEL and QDOPPLE from NPZ (matching Fortran fort.10)
        # Fortran synthe.for line 220: READ(10)QXNFPEL,QDOPPLE
        # These are pre-computed by xnfpelsyn.for and stored in NPZ as population_per_ion and doppler_per_ion
        # QXNFPEL is per unit mass (cm³/g), QDOPPLE is Doppler velocity (dimensionless, in units of c)

        # Get populations for this element from NPZ (required)
        if element not in population_cache:
            from ..engine.opacity import _element_atomic_number

            atomic_number = _element_atomic_number(element)

            # NPZ must provide population_per_ion and doppler_per_ion
            # These are computed by pops_exact in convert_atm_to_npz.py
            if (
                atomic_number is None
                or not hasattr(atmosphere, "population_per_ion")
                or atmosphere.population_per_ion is None
                or not hasattr(atmosphere, "doppler_per_ion")
                or atmosphere.doppler_per_ion is None
            ):
                raise ValueError(
                    f"NPZ file must contain population_per_ion and doppler_per_ion. "
                    f"Regenerate NPZ with convert_atm_to_npz.py. Element: {element}"
                )

            elem_idx = atomic_number - 1
            # Array shape is (n_depths, n_ion_stages, n_elements)
            if elem_idx >= atmosphere.population_per_ion.shape[2]:
                raise ValueError(
                    f"Element {element} (Z={atomic_number}) not in NPZ population array. "
                    f"Array has {atmosphere.population_per_ion.shape[2]} elements."
                )

            # Indexing [:, :, elem_idx] gives (n_depths, n_ion_stages) for this element
            pop_densities = atmosphere.population_per_ion[:, :, elem_idx]
            dop_velocity = atmosphere.doppler_per_ion[:, :, elem_idx]
            population_cache[element] = (pop_densities, dop_velocity)
        else:
            pop_densities, dop_velocity = population_cache[element]

        # Process each depth
        for depth_idx in range(n_depths):
            state = populations.layers[depth_idx]

            # Get population and Doppler for this ion stage
            if nelion <= pop_densities.shape[1]:
                pop_val = pop_densities[depth_idx, nelion - 1]
                # CRITICAL FIX: QDOPPLE is per ion stage, use the Doppler for this specific ion stage
                # Fortran synthe.for line 236: QDOPPLE(NELION) - uses Doppler for this ion stage
                if dop_velocity.ndim > 1:
                    # dop_velocity is shape (n_depths, n_ion_stages) - use the Doppler for this ion stage
                    dop_val = (
                        dop_velocity[depth_idx, nelion - 1]
                        if nelion <= dop_velocity.shape[1]
                        else dop_velocity[depth_idx, 0]
                    )
                else:
                    # dop_velocity is shape (n_depths,) - use it directly
                    dop_val = dop_velocity[depth_idx]
                if microturb_kms > 0.0:
                    micro = microturb_kms / C_LIGHT_KM
                    dop_val = math.sqrt(dop_val * dop_val + micro * micro)

                if pop_val > 0.0 and dop_val > 0.0:
                    # CRITICAL FIX: Match Fortran synthe.for line 240 exactly
                    # Line 240: QXNFDOP(NELION)=QXNFPEL(NELION)/QRHO(J)/QDOPPLE(NELION)
                    # Where:
                    #   - QXNFPEL from fort.10 is population per unit mass (cm³/g)
                    #   - QRHO is mass density (g/cm³)
                    #   - QDOPPLE is Doppler velocity (dimensionless, in units of c)
                    #
                    # If using pre-computed QXNFPEL (population_per_ion):
                    #   pop_val is already number density (cm⁻³) = QXNFPEL * rho
                    #   So: xnfdop = (QXNFPEL * rho) / (rho * QDOPPLE) = QXNFPEL / QDOPPLE
                    #   But we need: xnfdop = QXNFPEL / (rho * QDOPPLE)
                    #   So: xnfdop = pop_val / (rho * dop_val) ✓
                    #
                    # If computing from scratch:
                    #   pop_val is number density (cm⁻³)
                    #   dop_val is Doppler velocity (dimensionless, v/c)
                    #
                    # Fortran formula from synthe.for line 240:
                    #   QXNFDOP(NELION)=QXNFPEL(NELION)/QRHO(J)/QDOPPLE(NELION)
                    # where QXNFPEL is population (cm⁻³), QRHO is density (g/cm³),
                    # and QDOPPLE is Doppler velocity (v/c, dimensionless)
                    #
                    # This gives mass absorption coefficient (cm²/g)
                    rho = (
                        atmosphere.mass_density[depth_idx]
                        if hasattr(atmosphere, "mass_density")
                        and atmosphere.mass_density is not None
                        else 1.0
                    )
                    if rho > 0.0:
                        xnfdop = pop_val / (rho * dop_val)
                        doppler_width = dop_val * record.wavelength
                    else:
                        # Invalid mass density - skip this line/depth
                        continue
                else:
                    # Invalid population or Doppler - skip this line/depth
                    continue
            else:
                # Ion stage out of range - skip
                continue

            # Get Boltzmann factor
            boltz = state.boltzmann_factor[line_idx]

            # Compute KAPPA0
            # From Fortran rgfall.for line 267: CGF = 0.026538/1.77245 * GF / FRELIN
            # Where FRELIN = 2.99792458D17 / WLVAC (frequency in Hz)
            # Then synthe.for line 262: KAPPA0 = CONGF * QXNFDOP(NELION)
            # Then line 268: KAPPA0 = KAPPA0 * FASTEX(ELO*HCKT(J))
            # So: KAPPA0 = CGF * XNFDOP * exp(-E/kT)
            # Where CGF = (0.026538/1.77245) * GF / FREQ
            # And XNFDOP = QXNFPEL / (QRHO * QDOPPLE)

            # CRITICAL FIX: Convert GF to CONGF by dividing by frequency
            # From rgfall.for line 266: FRELIN = 2.99792458D17/WLVAC
            # rgfall.for uses WLVAC (energy-derived, with shifts) for FRELIN/CGF.
            wavelength_nm = (
                index_wavelength[line_idx]
                if index_wavelength is not None
                else record.wavelength
            )
            freq_hz = C_LIGHT_NM / wavelength_nm  # Frequency in Hz
            gf_linear = catalog.gf[
                line_idx
            ]  # Linear gf (already converted from log_gf)
            cgf_meta = None
            if record.metadata:
                cgf_meta = record.metadata.get("cgf")
            if cgf_meta is not None and cgf_meta > 0.0:
                cgf = float(cgf_meta)
            else:
                cgf = CGF_CONSTANT * gf_linear / freq_hz  # CONGF conversion

            # Gamma values are already linear (s^-1) after catalog load.
            gamma_rad = catalog.gamma_rad[line_idx]
            gamma_stark = catalog.gamma_stark[line_idx]
            gamma_vdw = catalog.gamma_vdw[line_idx]

            # PRE-BOLTZMANN CUTOFF CHECK (matches Fortran synthe.for lines 262-267)
            # Fortran: KAPMIN = CONTINUUM(MIN(MAX(NBUFF,1),LENGTH)) * CUTOFF
            # Then: IF(KAPPA0.LT.KAPMIN)GO TO 350
            if record.line_type == 1:
                # Autoionizing line (Fortran synthe.for label 700)
                # KAPPA0 = BSHORE * G * XNFPEL(NELION)
                # XNFPEL in fort.10 is per mass, so convert number density to per mass.
                if rho <= 0.0:
                    continue
                kappa0_pre_boltz = gamma_vdw * gf_linear * (pop_val / rho)
            else:
                kappa0_pre_boltz = cgf * xnfdop

            # Compute KAPMIN dynamically (matching Fortran exactly)
            # Fortran: KAPMIN = CONTINUUM(MIN(MAX(NBUFF,1),LENGTH)) * CUTOFF
            # No fallback - Fortran always uses this formula
            center_idx = center_indices[line_idx]
            clamped_idx = max(0, min(center_idx, n_wavelengths - 1))
            if (
                center_indices_full is not None
                and continuum_absorption_full is not None
            ):
                full_idx = center_indices_full[line_idx]
                full_idx = max(0, min(full_idx, continuum_absorption_full.shape[1] - 1))
                kapmin = continuum_absorption_full[depth_idx, full_idx] * cutoff
            else:
                # Fortran uses CONTINUUM(NBUFF) derived from ABLOG (depth-specific in synthe.for),
                # but when a full-grid continuum isn't available (e.g. full-range runs),
                # use the local continuum array.
                kapmin = continuum_absorption[depth_idx, clamped_idx] * cutoff

            # DEBUG: Track statistics (only for first depth to avoid spam)
            is_near_300nm = DEBUG_WL_MIN <= record.wavelength <= DEBUG_WL_MAX
            if depth_idx == 0:
                debug_stats["total_line_depth_pairs"] += 1
                if is_near_300nm:
                    debug_stats["lines_near_300nm"] += 1

            if (
                debug_transp_wl_val is not None
                and debug_transp_depth_idx == depth_idx
                and abs(record.wavelength - debug_transp_wl_val) <= debug_transp_tol
            ):
                idx_wl = (
                    index_wavelength[line_idx]
                    if index_wavelength is not None
                    else record.wavelength
                )
                full_idx_dbg = None
                full_wave_dbg = None
                full_cont_dbg = None
                if (
                    center_indices_full is not None
                    and continuum_absorption_full is not None
                    and wavelength_grid_full is not None
                ):
                    full_idx_dbg = int(center_indices_full[line_idx])
                    full_idx_dbg = max(
                        0, min(full_idx_dbg, continuum_absorption_full.shape[1] - 1)
                    )
                    full_wave_dbg = float(wavelength_grid_full[full_idx_dbg])
                    full_cont_dbg = float(
                        continuum_absorption_full[depth_idx, full_idx_dbg]
                    )
                print(
                    "PY_DEBUG_TRANSP_PRE: "
                    f"line_wl={record.wavelength:.6f} depth={depth_idx + 1} "
                    f"index_wl={idx_wl:.6f} full_idx={full_idx_dbg} "
                    f"full_wave={full_wave_dbg} full_cont={full_cont_dbg} "
                    f"pop={pop_val:.6e} rho={rho:.6e} dop={dop_val:.6e} "
                    f"xnfdop={xnfdop:.6e} cgf={cgf:.6e} "
                    f"boltz={boltz:.6e} kappa0_pre={kappa0_pre_boltz:.6e} "
                    f"kapmin={kapmin:.6e}"
                )

            if kappa0_pre_boltz < kapmin:
                if depth_idx == 0:
                    debug_stats["skipped_pre_boltz"] += 1
                continue  # Skip weak lines (matches Fortran cutoff behavior)

            # Apply Boltzmann factor AFTER pre-Boltzmann cutoff (Fortran line 268-270)
            kappa0 = kappa0_pre_boltz * boltz

            # POST-BOLTZMANN CUTOFF CHECK (Fortran line 272)
            # RE-ENABLED: This matches Fortran behavior and prevents weak line accumulation
            # Note: The Ca I 551nm issue was actually due to molecular opacity, not this cutoff
            if (
                debug_transp_wl_val is not None
                and debug_transp_depth_idx == depth_idx
                and abs(record.wavelength - debug_transp_wl_val) <= debug_transp_tol
            ):
                print(
                    "PY_DEBUG_TRANSP_POST: "
                    f"line_wl={record.wavelength:.6f} depth={depth_idx + 1} "
                    f"kappa0={kappa0:.6e} kapmin={kapmin:.6e}"
                )

            if kappa0 < kapmin:
                if depth_idx == 0:
                    debug_stats["skipped_post_boltz"] += 1
                continue

            # DEBUG: Track large values in TRANSP computation
            if kappa0 > 1e10 or xnfdop > 1e20 or pop_val > 1e20:
                if (
                    line_idx < 10 and depth_idx == 0
                ):  # Only log first 10 lines at surface
                    rho_debug = (
                        atmosphere.mass_density[depth_idx]
                        if hasattr(atmosphere, "mass_density")
                        and atmosphere.mass_density is not None
                        else 1.0
                    )
                    print(
                        f"\nDEBUG: Large values in TRANSP computation (line {line_idx}, depth {depth_idx}):"
                    )
                    print(
                        f"  Element: {element}, Ion: {nelion}, λ: {record.wavelength:.6f} nm"
                    )
                    print(f"  pop_val (population density, cm⁻³): {pop_val:.6e}")
                    print(f"  rho (mass density, g/cm³): {rho_debug:.6e}")
                    print(
                        f"  pop_val / rho (population per unit mass, cm³/g): {pop_val / rho_debug if rho_debug > 0 else 0:.6e}"
                    )
                    print(
                        f"  dop_val (doppler velocity, dimensionless in units of c): {dop_val:.6e}"
                    )
                    print(f"  doppler_width = dop_val * λ (nm): {doppler_width:.6e}")
                    print(f"  xnfdop = pop_val / (rho * dop_val): {xnfdop:.6e}")
                    print(f"  boltzmann_factor: {boltz:.6e}")
                    print(f"  catalog.gf (linear): {float(catalog.gf[line_idx]):.6e}")
                    print(f"  wavelength: {float(wavelength_nm):.6f} nm")
                    print(f"  frequency: {freq_hz:.6e} Hz")
                    print(f"  CGF = {CGF_CONSTANT:.6e} * GF / FREQ = {cgf:.6e}")
                    print(f"  kappa0 = CGF * xnfdop * boltz: {kappa0:.6e}")
                    print(
                        f"  Units check: CGF (s), xnfdop (cm³/g / dimensionless = cm³/g), boltz (dimensionless)"
                    )
                    print(f"  Units: kappa0 should be in cm²/g (opacity per unit mass)")

            # Skip if opacity is too small
            if kappa0 <= 0.0:
                if depth_idx == 0:
                    debug_stats["skipped_zero_kappa"] += 1
                continue

            # DEBUG: Track passed lines
            if depth_idx == 0:
                debug_stats["passed_cutoff"] += 1
                if is_near_300nm:
                    debug_stats["passed_near_300nm"] += 1
                    debug_stats["kapmin_at_300nm"].append(kapmin)
                    debug_stats["kappa0_at_300nm"].append(kappa0)
                    # Print detailed info for first 5 lines near 300nm
                    if debug_stats["passed_near_300nm"] <= 5:
                        rho_dbg = (
                            atmosphere.mass_density[depth_idx]
                            if hasattr(atmosphere, "mass_density")
                            and atmosphere.mass_density is not None
                            else 1.0
                        )
                        print(
                            f"\nDEBUG 300nm LINE #{debug_stats['passed_near_300nm']}:"
                        )
                        print(
                            f"  λ={record.wavelength:.4f}nm, Element={element}, Ion={nelion}"
                        )
                        print(f"  GF={gf_linear:.4e}, CGF={cgf:.4e}")
                        print(f"  pop_val={pop_val:.4e}, dop_val={dop_val:.4e}")
                        print(f"  rho={rho_dbg:.4e}, xnfdop={xnfdop:.4e}")
                        print(f"  boltz={boltz:.4e}")
                        print(f"  KAPMIN={kapmin:.4e}, KAPPA0={kappa0:.4e}")

            # Compute damping parameter
            # ADAMP = (GAMMAR + GAMMAS*XNE + GAMMAW*TXNXN) / DOPPLE

            xne = state.electron_density
            txnxn = state.txnxn

            # Compute damping parameter ADAMP
            # The Voigt damping parameter a = gamma / (4 * pi * delta_nu_D)
            # where gamma is total damping rate (s^-1) and delta_nu_D is Doppler width (Hz)
            #
            # gamma = GAMMAR + GAMMAS*XNE + GAMMAW*TXNXN  (s^-1)
            # delta_nu_D = frequency * dopple = (c / wavelength) * dopple  (Hz)
            #
            # So: adamp = gamma / (4 * pi * (c / wavelength) * dopple)
            #          = gamma * wavelength / (4 * pi * c * dopple)
            if doppler_width > 0:
                # Convert Doppler width to Doppler velocity (DOPPLE)
                # doppler_width is in units of wavelength (nm)
                # DOPPLE = doppler_width / wavelength (dimensionless)
                dopple = doppler_width / record.wavelength

                if dopple > 0:
                    # Total damping rate (s^-1)
                    gamma_total = gamma_rad + gamma_stark * xne + gamma_vdw * txnxn

                    # Fortran synthe.for: ADAMP = (GAMMAR + GAMMAS*XNE + GAMMAW*TXNXN) / DOPPLE
                    # GAMMA* values are already normalized by (4*pi*freq) in rgfall.
                    adamp = gamma_total / dopple
                else:
                    adamp = 0.0
            else:
                adamp = 0.0

            # Compute line center opacity
            # For normal lines: Voigt center (Fortran line 286-290)
            # For autoionizing lines: KAPPA0 directly (Fortran label 700 writes KAPPA0)
            if adamp >= 0 and kappa0 > 0:
                if record.line_type == 1:
                    kapcen = kappa0
                else:
                    if adamp < 0.2:
                        # Small damping approximation (matches Fortran line 287)
                        kapcen = kappa0 * (1.0 - 1.128 * adamp)
                    else:
                        # Full Voigt profile (matches Fortran line 289)
                        voigt_center = voigt_profile(0.0, adamp)
                        kapcen = kappa0 * voigt_center

                # Store result
                transp[line_idx, depth_idx] = kapcen
                valid_mask[line_idx, depth_idx] = True

                # DEBUG: Track very large TRANSP values
                if kapcen > 1e10 and line_idx < 10 and depth_idx == 0:
                    print(
                        f"\nDEBUG: Large TRANSP value (line {line_idx}, depth {depth_idx}):"
                    )
                    print(f"  kapcen: {kapcen:.6e}")
                    print(f"  kappa0: {kappa0:.6e}")
                    print(f"  adamp: {adamp:.6e}")
                    print(
                        f"  voigt_center: {voigt_center if adamp >= 0.2 else (1.0 - 1.128 * adamp):.6e}"
                    )

    # DEBUG: Summary statistics for TRANSP
    if np.any(valid_mask):
        transp_valid = transp[valid_mask]
        print(f"\nDEBUG: TRANSP statistics:")
        print(f"  Valid values: {np.sum(valid_mask):,}")
        print(f"  TRANSP min: {np.min(transp_valid):.6e}")
        print(f"  TRANSP max: {np.max(transp_valid):.6e}")
        print(f"  TRANSP mean: {np.mean(transp_valid):.6e}")
        print(f"  TRANSP median: {np.median(transp_valid):.6e}")
        print(f"  Values > 1e10: {np.sum(transp_valid > 1e10):,}")
        print(f"  Values > 1e20: {np.sum(transp_valid > 1e20):,}")

    logger.info(
        f"TRANSP computation complete: {np.sum(valid_mask):,} valid line-depth pairs"
    )

    # DEBUG: Print cutoff statistics summary
    print("\n" + "=" * 70)
    print("DEBUG: TRANSP CUTOFF STATISTICS (depth=0 only)")
    print("=" * 70)
    print(f"  Total lines processed: {debug_stats['total_line_depth_pairs']:,}")
    print(f"  Skipped (pre-Boltzmann < KAPMIN): {debug_stats['skipped_pre_boltz']:,}")
    print(f"  Skipped (post-Boltzmann < KAPMIN): {debug_stats['skipped_post_boltz']:,}")
    print(f"  Skipped (kappa0 <= 0): {debug_stats['skipped_zero_kappa']:,}")
    print(f"  PASSED CUTOFF: {debug_stats['passed_cutoff']:,}")
    print(
        f"  Pass rate: {100.0 * debug_stats['passed_cutoff'] / max(debug_stats['total_line_depth_pairs'], 1):.2f}%"
    )
    print()
    print(f"  Lines near 300nm (299.5-300.5nm): {debug_stats['lines_near_300nm']:,}")
    print(f"  PASSED near 300nm: {debug_stats['passed_near_300nm']:,}")
    if debug_stats["kapmin_at_300nm"]:
        print(
            f"  KAPMIN at 300nm: min={min(debug_stats['kapmin_at_300nm']):.4e}, max={max(debug_stats['kapmin_at_300nm']):.4e}"
        )
        print(
            f"  KAPPA0 at 300nm (passed): min={min(debug_stats['kappa0_at_300nm']):.4e}, max={max(debug_stats['kappa0_at_300nm']):.4e}"
        )
    print("=" * 70 + "\n")

    # Return pre-computed center indices for wing and center accumulation.
    # These are computed using Fortran's logarithmic rounding on the wavelength grid.
    return transp, valid_mask, center_indices


def _compute_fortran_profile_steps(
    offset: int,
    kappa0: float,
    adamp: float,
    dopple: float,
    resolu: float,
    kapmin_ref: float,
    h0tab: np.ndarray,
    h1tab: np.ndarray,
    h2tab: np.ndarray,
    max_profile_steps: int,
) -> Tuple[Optional[float], int, int, Optional[float], bool]:
    """Compute per-offset profile using Fortran XLINOP steps (labels 320-323)."""
    offset_abs = abs(int(offset))
    if dopple <= 0.0:
        return None, 0, 0, None, False

    n10dop = int(10.0 * dopple * resolu)
    if n10dop <= 0:
        return None, n10dop, 0, None, False

    profile_at_offset = None
    profile_at_n10dop = None
    cutoff_hit = False

    dvoigt = 1.0 / (dopple * resolu)
    if adamp < 0.2:
        vsteps = 200.0
        tabstep = vsteps * dvoigt
        tabi = 0.5  # 0-based indexing (Fortran uses 1.5 for 1-based arrays)
        for nstep in range(1, n10dop + 1):
            tabi += tabstep
            idx_tab = int(tabi)
            if idx_tab < 0:
                idx_tab = 0
            x_step = float(nstep) * dvoigt
            if x_step > 10.0:
                profile = kappa0 * (0.5642 * adamp / (x_step * x_step))
            else:
                if idx_tab >= h0tab.size:
                    idx_tab = h0tab.size - 1
                profile = kappa0 * (h0tab[idx_tab] + adamp * h1tab[idx_tab])
            if nstep == offset_abs:
                profile_at_offset = profile
            if nstep == n10dop:
                profile_at_n10dop = profile
            if profile < kapmin_ref:
                cutoff_hit = True
                maxstep = nstep
                if offset_abs > maxstep:
                    return None, n10dop, maxstep, profile_at_n10dop, cutoff_hit
                return profile_at_offset, n10dop, maxstep, profile_at_n10dop, cutoff_hit
    else:
        for nstep in range(1, n10dop + 1):
            x_step = float(nstep) * dvoigt
            profile = kappa0 * _voigt_profile_jit(x_step, adamp, h0tab, h1tab, h2tab)
            if nstep == offset_abs:
                profile_at_offset = profile
            if nstep == n10dop:
                profile_at_n10dop = profile
            if profile < kapmin_ref:
                cutoff_hit = True
                maxstep = nstep
                if offset_abs > maxstep:
                    return None, n10dop, maxstep, profile_at_n10dop, cutoff_hit
                return profile_at_offset, n10dop, maxstep, profile_at_n10dop, cutoff_hit

    if profile_at_n10dop is None or kapmin_ref <= 0.0:
        return profile_at_offset, n10dop, 0, profile_at_n10dop, cutoff_hit

    x_far = profile_at_n10dop * float(n10dop) ** 2
    maxstep = int(np.sqrt(x_far / kapmin_ref) + 1.0)
    if maxstep > max_profile_steps:
        maxstep = max_profile_steps

    if offset_abs > n10dop:
        if offset_abs > maxstep or x_far <= 0.0:
            return None, n10dop, maxstep, profile_at_n10dop, cutoff_hit
        profile_at_offset = x_far / float(offset_abs) ** 2

    return profile_at_offset, n10dop, maxstep, profile_at_n10dop, cutoff_hit


def compute_asynth_from_transp(
    transp: np.ndarray,
    catalog: "LineCatalog",
    atmosphere: "AtmosphereModel",
    wavelength_grid: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    populations: Optional["Populations"] = None,
    cutoff: float = 1e-3,
    continuum_absorption: Optional[np.ndarray] = None,
    continuum_absorption_full: Optional[np.ndarray] = None,
    wavelength_grid_full: Optional[np.ndarray] = None,
    metal_tables: Optional["tables.MetalWingTables"] = None,
    grid_origin: Optional[float] = None,
) -> np.ndarray:
    """
    Compute ASYNTH from TRANSP using the stimulated emission correction.

    Formula from synthe.for line 368:
        ASYNTH(J) = TRANSP(J,I) * (1. - EXP(-FREQ*HKT(J)))

    CRITICAL: This function now includes wing contributions via Voigt profiles,
    matching Fortran's behavior where lines contribute to nearby wavelengths.

    Parameters
    ----------
    transp:
        Line opacity at line center, shape (n_lines, n_depths)
    catalog:
        Line catalog
    atmosphere:
        Atmosphere model
    wavelength_grid:
        Wavelength grid for output, shape (n_wavelengths,)
    valid_mask:
        Optional mask indicating valid lines/depths
    populations:
        Populations object (needed for computing damping and doppler widths)
    cutoff:
        Opacity cutoff factor for wing contributions (matches Fortran CUTOFF)
    continuum_absorption:
        Continuum absorption array, shape (n_depths, n_wavelengths).
        If None, cutoff check is skipped (wings extend to MAX_PROFILE_STEPS)

    Returns
    -------
    asynth:
        ASYNTH array, shape (n_depths, n_wavelengths)
    """
    n_wavelengths = wavelength_grid.size
    n_depths = atmosphere.layers

    # Initialize ASYNTH array
    asynth = np.zeros((n_depths, n_wavelengths), dtype=np.float64)

    # CRITICAL FIX: Match Fortran frequency calculation exactly
    # Fortran line 369: FREQ=2.99792458D17/WAVE (WAVE in nm, result in Hz)
    # C_LIGHT_NM = 2.99792458e17 nm/s = speed of light in nm/s
    # Frequency = C_LIGHT_NM / wavelength_nm (Hz)

    # Compute frequency grid
    freq_grid = C_LIGHT_NM / wavelength_grid  # Shape: (n_wavelengths,)

    # Compute HKT for each depth
    hkt = np.zeros(n_depths, dtype=np.float64)
    for depth_idx in range(n_depths):
        temp = atmosphere.temperature[depth_idx]
        if atmosphere.hckt is not None:
            # HKT = H_PLANCK / (K_BOLTZ * T) = hckt / T
            hkt[depth_idx] = H_PLANCK / (K_BOLTZ * max(temp, 1.0))
        else:
            hkt[depth_idx] = H_PLANCK / (K_BOLTZ * max(temp, 1.0))

    # Map lines to wavelength grid
    from ..engine.opacity import _nearest_grid_indices

    index_wavelength = (
        catalog.index_wavelength
        if hasattr(catalog, "index_wavelength")
        else catalog.wavelength
    )
    line_indices = _nearest_grid_indices(wavelength_grid, index_wavelength)

    # Raw (unclamped) indices for wing contributions so outside-center lines
    # still map to correct offset distances.
    def _nearest_grid_indices_raw(
        grid: np.ndarray, values: np.ndarray, origin_start: Optional[float] = None
    ) -> np.ndarray:
        if len(grid) < 2:
            return np.zeros(len(values), dtype=np.int64)
        ratio = grid[1] / grid[0]
        ratiolg = np.log(ratio)
        start_val = grid[0] if origin_start is None else origin_start
        ix_floor = int(np.floor(np.log(start_val) / ratiolg))
        wbegin = np.exp(ix_floor * ratiolg)
        if wbegin < start_val:
            ix_floor += 1
            wbegin = np.exp(ix_floor * ratiolg)
        with np.errstate(divide="ignore", invalid="ignore"):
            ix = np.rint(np.log(values / wbegin) / ratiolg).astype(np.int64)
        return ix

    line_indices_wing = _nearest_grid_indices_raw(
        wavelength_grid, index_wavelength, origin_start=grid_origin
    )
    if grid_origin is not None and wavelength_grid.size > 1:
        ratio = wavelength_grid[1] / wavelength_grid[0]
        ratiolg = np.log(ratio)
        ix_floor = int(np.floor(np.log(grid_origin) / ratiolg))
        wbegin = np.exp(ix_floor * ratiolg)
        if wbegin < grid_origin:
            ix_floor += 1
            wbegin = np.exp(ix_floor * ratiolg)
        grid_offset = int(np.rint(np.log(wavelength_grid[0] / wbegin) / ratiolg))
        line_indices_wing = line_indices_wing - grid_offset

    # Vectorized ASYNTH computation
    # Compute frequencies for all lines at once (matches Fortran line 369)
    line_freqs = C_LIGHT_NM / catalog.wavelength  # Shape: (n_lines,)

    # Fortran applies the stimulated emission factor after TRANSP is transposed
    # to each wavelength (synthe.for lines 439-443). Use grid frequency, not line centers.
    stim_grid = 1.0 - np.exp(-freq_grid[np.newaxis, :] * hkt[:, np.newaxis])

    # Kernel still expects stim_factors array; keep it as ones so wing profiles are unscaled here.
    stim_factors = np.ones((len(catalog.records), n_depths), dtype=np.float64)

    # Import needed functions
    from .profiles.voigt import voigt_profile
    from ..engine.opacity import MAX_PROFILE_STEPS, _element_atomic_number

    # Cache populations per element
    population_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # Get continuum absorption for cutoff calculation (matches Fortran: KAPMIN = CONTINUUM * CUTOFF)
    # If not provided, skip cutoff check and extend wings to MAX_PROFILE_STEPS
    use_cutoff = continuum_absorption is not None and continuum_absorption.shape == (
        n_depths,
        n_wavelengths,
    )

    # Add center contributions first (TRANSP only; stim applied after wing accumulation).
    # CRITICAL FIX: Skip fort.19 special lines (line_type != 0) here.
    # Hydrogen (type -1/-2) → handled by _compute_hydrogen_line_opacity
    # Autoionizing (type 1) → handled by _add_fort19_asynth
    # Helium (type -3/-6) → handled by helium wings path
    # Adding them here would DOUBLE-COUNT their center opacity.
    asynth_per_line = transp  # Shape: (n_lines, n_depths)
    for line_idx in range(len(catalog.records)):
        rec = catalog.records[line_idx]
        if int(getattr(rec, "line_type", 0) or 0) != 0:
            continue  # Skip fort.19 special lines
        center_idx = line_indices[line_idx]
        if center_idx >= 0 and center_idx < n_wavelengths:
            for depth_idx in range(n_depths):
                if valid_mask is None or valid_mask[line_idx, depth_idx]:
                    asynth[depth_idx, center_idx] += asynth_per_line[
                        line_idx, depth_idx
                    ]

    debug_wave = os.getenv("PY_DEBUG_ASYNTH_WAVE")
    debug_depth = os.getenv("PY_DEBUG_ASYNTH_DEPTH")
    if debug_wave and debug_depth:
        try:
            target_wave = float(debug_wave)
            target_depth = int(debug_depth) - 1
        except ValueError:
            target_wave = None
            target_depth = -1
        if (
            target_wave is not None
            and 0 <= target_depth < n_depths
            and n_wavelengths > 0
        ):
            idx_target = int(np.argmin(np.abs(wavelength_grid - target_wave)))
            center_mask = line_indices == idx_target
            center_transp = transp[center_mask, target_depth]
            sum_center = float(np.sum(center_transp)) if center_transp.size else 0.0
            min_center = float(np.min(center_transp)) if center_transp.size else 0.0
            neg_count = int(np.sum(center_transp < 0.0)) if center_transp.size else 0
            print(
                f"PY_DEBUG_ASYNTH: wave={float(wavelength_grid[idx_target]):.6f} "
                f"depth={target_depth + 1} center lines={center_transp.size} "
                f"sum_transp={sum_center:.6e} min_transp={min_center:.6e} "
                f"neg_count={neg_count}"
            )
            if neg_count > 0:
                neg_idx = np.where(center_mask)[0][center_transp < 0.0]
                for line_i in neg_idx[:10]:
                    rec = catalog.records[int(line_i)]
                    print(
                        f"  NEG TRANSP line: wl={float(rec.wavelength):.6f} "
                        f"loggf={float(rec.log_gf):.3f} "
                        f"transp={float(transp[line_i, target_depth]):.6e}"
                    )

    # Pre-compute arrays for JIT kernel if Numba is available
    if NUMBA_AVAILABLE:
        # Pre-compute kappa0, adamp, doppler_widths, gamma values, wcon, wtail for all lines/depths
        n_lines = len(catalog.records)

        # Initialize arrays
        kappa0_array = np.zeros((n_lines, n_depths), dtype=np.float64)
        adamp_array = np.zeros((n_lines, n_depths), dtype=np.float64)
        doppler_widths_array = np.zeros((n_lines, n_depths), dtype=np.float64)
        gamma_rad_array = np.asarray(catalog.gamma_rad, dtype=np.float64)
        gamma_stark_array = np.asarray(catalog.gamma_stark, dtype=np.float64)
        gamma_vdw_array = np.asarray(catalog.gamma_vdw, dtype=np.float64)
        line_types_array = np.asarray(catalog.line_types, dtype=np.int8)
        wcon_array = np.zeros(n_lines * n_depths, dtype=np.float64)  # Flattened
        wtail_array = np.zeros(n_lines * n_depths, dtype=np.float64)  # Flattened
        kapmin_ref_array = np.zeros((n_lines, n_depths), dtype=np.float64)

        # Cache populations per element
        population_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # Get Voigt tables
        voigt_tables = tables.voigt_tables()
        h0tab = voigt_tables.h0tab
        h1tab = voigt_tables.h1tab
        h2tab = voigt_tables.h2tab

        # Pre-compute all values
        center_indices_full = None
        if continuum_absorption_full is not None and wavelength_grid_full is not None:
            center_indices_full = _nearest_grid_indices(
                wavelength_grid_full,
                (
                    catalog.index_wavelength
                    if hasattr(catalog, "index_wavelength")
                    else catalog.wavelength
                ),
            )
        for line_idx in range(n_lines):
            record = catalog.records[line_idx]
            line_wavelength = record.wavelength
            element = record.element
            nelion = record.ion_stage

            # Get populations from NPZ (required)
            if element not in population_cache:
                atomic_number = _element_atomic_number(element)
                if atomic_number is None or atmosphere.population_per_ion is None:
                    continue  # Skip elements not in NPZ
                elem_idx = atomic_number - 1
                if elem_idx >= atmosphere.population_per_ion.shape[2]:
                    continue
                pop_densities = atmosphere.population_per_ion[:, :, elem_idx]
                dop_velocity = atmosphere.doppler_per_ion[:, :, elem_idx]
                population_cache[element] = (pop_densities, dop_velocity)
            else:
                pop_densities, dop_velocity = population_cache[element]

            for depth_idx in range(n_depths):
                if valid_mask is not None and not valid_mask[line_idx, depth_idx]:
                    continue

                transp_val = transp[line_idx, depth_idx]
                if transp_val <= 0.0:
                    continue

                # Get population and Doppler for this ion stage
                if nelion > pop_densities.shape[1]:
                    continue

                pop_val = pop_densities[depth_idx, nelion - 1]
                # dop_velocity is 2D: (n_depths, n_ion_stages)
                if dop_velocity.ndim > 1:
                    dop_val = (
                        dop_velocity[depth_idx, nelion - 1]
                        if nelion <= dop_velocity.shape[1]
                        else dop_velocity[depth_idx, 0]
                    )
                else:
                    dop_val = dop_velocity[depth_idx]

                if pop_val <= 0.0 or dop_val <= 0.0:
                    continue

                # Compute doppler width
                doppler_width = dop_val * line_wavelength
                doppler_widths_array[line_idx, depth_idx] = doppler_width

                # Compute damping parameter (gamma_* already linear)
                gamma_rad = catalog.gamma_rad[line_idx]
                gamma_stark = catalog.gamma_stark[line_idx]
                gamma_vdw = catalog.gamma_vdw[line_idx]

                xne = atmosphere.electron_density[depth_idx]
                if populations is not None:
                    state = populations.layers[depth_idx]
                    txnxn = state.txnxn
                else:
                    # Fallback: compute TXNXN from atmosphere (matches populations.py formula)
                    xnf_h = (
                        atmosphere.xnf_h[depth_idx]
                        if atmosphere.xnf_h is not None
                        else 0.0
                    )
                    xnf_he1 = (
                        atmosphere.xnf_he1[depth_idx]
                        if atmosphere.xnf_he1 is not None
                        else 0.0
                    )
                    xnf_h2 = (
                        atmosphere.xnf_h2[depth_idx]
                        if atmosphere.xnf_h2 is not None
                        else 0.0
                    )
                    temp = atmosphere.temperature[depth_idx]
                    txnxn = (xnf_h + 0.42 * xnf_he1 + 0.85 * xnf_h2) * (
                        temp / 10_000.0
                    ) ** 0.3

                # Fortran synthe.for: ADAMP = (GAMMAR + GAMMAS*XNE + GAMMAW*TXNXN) / DOPPLE
                # GAMMA* values are already normalized by (4*pi*freq) in rgfall.
                dopple = (
                    doppler_width / line_wavelength if line_wavelength > 0 else 1e-6
                )
                if dopple > 0 and line_wavelength > 0:
                    gamma_total = gamma_rad + gamma_stark * xne + gamma_vdw * txnxn
                    adamp = gamma_total / dopple
                else:
                    adamp = 0.0

                adamp = max(adamp, 1e-12)
                adamp_array[line_idx, depth_idx] = adamp

                # Depth-specific KAPMIN reference at the line center.
                if use_cutoff:
                    if (
                        continuum_absorption_full is not None
                        and wavelength_grid_full is not None
                        and center_indices_full is not None
                    ):
                        full_idx = int(center_indices_full[line_idx])
                        full_idx = max(
                            0, min(full_idx, continuum_absorption_full.shape[1] - 1)
                        )
                        kapmin_ref_array[line_idx, depth_idx] = (
                            continuum_absorption_full[depth_idx, full_idx] * cutoff
                        )
                    else:
                        center_idx = int(line_indices_wing[line_idx])
                        center_idx = max(0, min(center_idx, n_wavelengths - 1))
                        kapmin_ref_array[line_idx, depth_idx] = (
                            continuum_absorption[depth_idx, center_idx] * cutoff
                        )

                # Recover kappa0 from TRANSP
                if record.line_type == 1:
                    # Autoionizing line uses KAPPA0 directly (no Voigt center scaling)
                    kappa0 = transp_val
                else:
                    if adamp < 0.2:
                        voigt_center = 1.0 - 1.128 * adamp
                    else:
                        voigt_center = voigt_profile(0.0, adamp)

                    if voigt_center > 0:
                        kappa0 = transp_val / voigt_center
                    else:
                        kappa0 = transp_val

                kappa0_array[line_idx, depth_idx] = kappa0

                # Compute WCON/WTAIL if metal_tables available
                if metal_tables is not None and populations is not None:
                    state = populations.layers[depth_idx]
                    from ..engine.opacity import _compute_continuum_limits

                    wcon, wtail = _compute_continuum_limits(
                        ncon=state.ncon if hasattr(state, "ncon") else 0,
                        nelion=nelion,
                        nelionx=state.nelionx if hasattr(state, "nelionx") else 0,
                        emerge_val=state.emerge if hasattr(state, "emerge") else 0.0,
                        emerge_h_val=(
                            state.emerge_h if hasattr(state, "emerge_h") else 0.0
                        ),
                        metal_tables=metal_tables,
                        ifvac=1,
                    )
                    idx_wcon = line_idx * n_depths + depth_idx
                    if wcon is not None and wcon > 0.0:
                        wcon_array[idx_wcon] = wcon
                        if wtail is not None and wtail > 0.0:
                            wtail_array[idx_wcon] = wtail

        # Optional debug: inspect wing reach for a specific line/wave/depth.
        debug_line_wl = os.getenv("PY_DEBUG_WING_LINE_WL")
        debug_line_depth = os.getenv("PY_DEBUG_WING_LINE_DEPTH")
        debug_target_wave = os.getenv("PY_DEBUG_WING_TARGET_WAVE")
        if debug_line_wl and debug_line_depth and debug_target_wave:
            try:
                line_wl_val = float(debug_line_wl)
                depth_val = int(debug_line_depth) - 1
                target_wave_val = float(debug_target_wave)
            except ValueError:
                line_wl_val = None
                depth_val = None
                target_wave_val = None
            if (
                line_wl_val is not None
                and depth_val is not None
                and target_wave_val is not None
                and 0 <= depth_val < n_depths
            ):
                line_idx_dbg = int(np.argmin(np.abs(catalog.wavelength - line_wl_val)))
                target_idx_dbg = int(
                    np.argmin(np.abs(wavelength_grid - target_wave_val))
                )
                doppler_width_dbg = doppler_widths_array[line_idx_dbg, depth_val]
                dopple_dbg = (
                    doppler_width_dbg / line_wl_val if line_wl_val > 0.0 else 0.0
                )
                adamp_dbg = adamp_array[line_idx_dbg, depth_val]
                kappa0_dbg = kappa0_array[line_idx_dbg, depth_val]
                kapmin_dbg = kapmin_ref_array[line_idx_dbg, depth_val]
                center_idx_dbg = int(line_indices_wing[line_idx_dbg])
                resolu_dbg = 300000.0
                if n_wavelengths > 1:
                    ratio_dbg = wavelength_grid[1] / wavelength_grid[0]
                    if ratio_dbg > 1.0:
                        resolu_dbg = 1.0 / (ratio_dbg - 1.0)
                n10dop_dbg = int(10.0 * dopple_dbg * resolu_dbg)
                dvoigt_dbg = 1.0 / (dopple_dbg * resolu_dbg) if dopple_dbg > 0 else 1.0
                profile_n10 = 0.0
                if n10dop_dbg > 0:
                    if adamp_dbg < 0.2:
                        vsteps_dbg = 200.0
                        tabstep_dbg = vsteps_dbg * dvoigt_dbg
                        tabi_dbg = 0.5 + tabstep_dbg * float(n10dop_dbg)
                        idx_dbg = int(tabi_dbg)
                        idx_dbg = max(0, min(idx_dbg, h0tab.size - 1))
                        profile_n10 = kappa0_dbg * (
                            h0tab[idx_dbg] + adamp_dbg * h1tab[idx_dbg]
                        )
                    else:
                        x_step_dbg = float(n10dop_dbg) * dvoigt_dbg
                        voigt_dbg = voigt_profile(x_step_dbg, adamp_dbg)
                        profile_n10 = kappa0_dbg * voigt_dbg
                if n10dop_dbg > 0 and profile_n10 > 0.0 and kapmin_dbg > 0.0:
                    x_far_dbg = profile_n10 * float(n10dop_dbg) ** 2
                    maxstep_dbg = int(np.sqrt(x_far_dbg / kapmin_dbg) + 1.0)
                else:
                    maxstep_dbg = 0
                offset_dbg = abs(center_idx_dbg - target_idx_dbg)
                min_profile_dbg = float(os.getenv("PY_DEBUG_WING_MIN_PROFILE", "1e-6"))
                profile_at_offset = 0.0
                if offset_dbg > 0 and dopple_dbg > 0.0:
                    if adamp_dbg < 0.2:
                        vsteps_dbg = 200.0
                        tabstep_dbg = vsteps_dbg * dvoigt_dbg
                        tabi_dbg = 0.5 + tabstep_dbg * float(offset_dbg)
                        idx_dbg = int(tabi_dbg)
                        idx_dbg = max(0, min(idx_dbg, h0tab.size - 1))
                        profile_at_offset = kappa0_dbg * (
                            h0tab[idx_dbg] + adamp_dbg * h1tab[idx_dbg]
                        )
                    else:
                        x_offset_dbg = float(offset_dbg) * dvoigt_dbg
                        profile_at_offset = kappa0_dbg * voigt_profile(
                            x_offset_dbg, adamp_dbg
                        )
                transp_dbg = transp[line_idx_dbg, depth_val]
                valid_dbg = (
                    valid_mask[line_idx_dbg, depth_val]
                    if valid_mask is not None
                    else True
                )
                wcon_dbg = -1.0
                wtail_dbg = -1.0
                if wcon_array.size > 0:
                    idx_wcon_dbg = line_idx_dbg * n_depths + depth_val
                    if idx_wcon_dbg < wcon_array.size:
                        wcon_dbg = wcon_array[idx_wcon_dbg]
                    if idx_wcon_dbg < wtail_array.size:
                        wtail_dbg = wtail_array[idx_wcon_dbg]
                print(
                    "PY_DEBUG_WING_LINE: "
                    f"line_wl={catalog.wavelength[line_idx_dbg]:.6f} "
                    f"depth={depth_val + 1} target_wave={target_wave_val:.6f} "
                    f"center_idx={center_idx_dbg} target_idx={target_idx_dbg} "
                    f"offset={offset_dbg} n10dop={n10dop_dbg} "
                    f"kapmin={kapmin_dbg:.6e} profile_n10={profile_n10:.6e} "
                    f"maxstep={maxstep_dbg} dopple={dopple_dbg:.6e} "
                    f"adamp={adamp_dbg:.6e} kappa0={kappa0_dbg:.6e} "
                    f"transp={transp_dbg:.6e} valid={valid_dbg} "
                    f"wcon={wcon_dbg:.6e} wtail={wtail_dbg:.6e} "
                    f"profile_offset={profile_at_offset:.6e} "
                    f"min_profile={min_profile_dbg:.6e}"
                )

        # Call JIT kernel
        line_wavelengths_array = np.asarray(catalog.wavelength, dtype=np.float64)
        line_indices_array = np.asarray(line_indices_wing, dtype=np.int64)

        debug_wave = os.getenv("PY_DEBUG_ASYNTH_WAVE")
        debug_depth = os.getenv("PY_DEBUG_ASYNTH_DEPTH")
        if debug_wave and debug_depth:
            try:
                target_wave = float(debug_wave)
                target_depth = int(debug_depth) - 1
            except ValueError:
                target_wave = None
                target_depth = -1
            if (
                target_wave is not None
                and 0 <= target_depth < n_depths
                and n_wavelengths > 0
            ):
                idx_target = int(np.argmin(np.abs(wavelength_grid - target_wave)))
                freq_target = C_LIGHT_NM / wavelength_grid[idx_target]
                auto_mask = line_types_array == 1
                total_auto = 0.0
                neg_auto = []
                for line_idx in np.where(auto_mask)[0]:
                    if (
                        valid_mask is not None
                        and not valid_mask[line_idx, target_depth]
                    ):
                        continue
                    center_idx = line_indices_wing[line_idx]
                    if (
                        center_idx < -MAX_PROFILE_STEPS
                        or center_idx > n_wavelengths - 1 + MAX_PROFILE_STEPS
                    ):
                        continue
                    offset = idx_target - center_idx
                    if offset == 0 or abs(offset) > MAX_PROFILE_STEPS:
                        continue
                    gamma_rad = gamma_rad_array[line_idx]
                    bshore = gamma_vdw_array[line_idx]
                    ashore = gamma_stark_array[line_idx]
                    if gamma_rad <= 0.0 or bshore <= 0.0:
                        continue
                    freq_line = C_LIGHT_NM / line_wavelengths_array[line_idx]
                    epsil = 2.0 * (freq_target - freq_line) / gamma_rad
                    kappa0 = kappa0_array[line_idx, target_depth]
                    if kappa0 <= 0.0:
                        continue
                    kappa = (
                        kappa0
                        * (ashore * epsil + bshore)
                        / (epsil * epsil + 1.0)
                        / bshore
                    )
                    total_auto += kappa
                    if kappa < 0.0:
                        neg_auto.append((line_idx, kappa))
                print(
                    f"PY_DEBUG_ASYNTH_AUTO: wave={float(wavelength_grid[idx_target]):.6f} "
                    f"depth={target_depth + 1} auto_lines={int(np.sum(auto_mask))} "
                    f"neg_auto={len(neg_auto)} total_auto={total_auto:.6e}"
                )
                for line_idx, kappa in neg_auto[:10]:
                    rec = catalog.records[int(line_idx)]
                    print(
                        f"  AUTO NEG: wl={float(rec.wavelength):.6f} "
                        f"loggf={float(rec.log_gf):.3f} kappa={kappa:.6e}"
                    )

        max_profile_steps = int(MAX_PROFILE_STEPS)
        debug_idx = -1
        debug_depth = -1
        debug_wave_env = os.getenv("PY_DEBUG_ASYNTH_WAVE")
        debug_depth_env = os.getenv("PY_DEBUG_ASYNTH_DEPTH")
        if debug_wave_env and debug_depth_env:
            try:
                debug_idx = int(
                    np.argmin(np.abs(wavelength_grid - float(debug_wave_env)))
                )
                debug_depth = int(debug_depth_env) - 1
            except ValueError:
                debug_idx = -1
                debug_depth = -1
        debug_line_env = os.getenv("PY_DEBUG_ASYNTH_LINE")
        if debug_line_env and debug_idx >= 0 and 0 <= debug_depth < n_depths:
            try:
                debug_line_idx = int(debug_line_env)
            except ValueError:
                debug_line_idx = -1
            if 0 <= debug_line_idx < n_lines:
                line_wl = line_wavelengths_array[debug_line_idx]
                center_idx = line_indices_wing[debug_line_idx]
                offset = debug_idx - center_idx
                adamp = adamp_array[debug_line_idx, debug_depth]
                kappa0 = kappa0_array[debug_line_idx, debug_depth]
                doppler_width = doppler_widths_array[debug_line_idx, debug_depth]
                dopple = doppler_width / line_wl if line_wl > 0.0 else 0.0
                wcon_dbg = -1.0
                wtail_dbg = -1.0
                if wcon_array.size > 0:
                    idx_wcon_dbg = debug_line_idx * n_depths + debug_depth
                    if idx_wcon_dbg < wcon_array.size:
                        wcon_dbg = wcon_array[idx_wcon_dbg]
                    if idx_wcon_dbg < wtail_array.size:
                        wtail_dbg = wtail_array[idx_wcon_dbg]
                wave_target = float(wavelength_grid[debug_idx])
                skip_by_wcon = wcon_dbg > 0.0 and wave_target < wcon_dbg
                taper_factor = 1.0
                if (
                    wcon_dbg > 0.0
                    and wtail_dbg > 0.0
                    and wave_target < wtail_dbg
                    and wtail_dbg > wcon_dbg
                ):
                    taper_factor = (wave_target - wcon_dbg) / (wtail_dbg - wcon_dbg)
                resolu_local = 300000.0
                if n_wavelengths > 1:
                    ratio = wavelength_grid[1] / wavelength_grid[0]
                    if ratio > 1.0:
                        resolu_local = 1.0 / (ratio - 1.0)
                dvoigt = 1.0 / (dopple * resolu_local) if dopple > 0.0 else 0.0
                if adamp < 0.2:
                    vsteps = 200.0
                    tabstep = vsteps * dvoigt
                    tabi = 0.5 + tabstep * float(abs(offset))
                    idx_tab = int(tabi)
                    if idx_tab < 0:
                        idx_tab = 0
                    x_offset = float(abs(offset)) * dvoigt
                    if x_offset > 10.0:
                        profile_val = kappa0 * (0.5642 * adamp / (x_offset * x_offset))
                    else:
                        if idx_tab >= h0tab.size:
                            idx_tab = h0tab.size - 1
                        profile_val = kappa0 * (h0tab[idx_tab] + adamp * h1tab[idx_tab])
                else:
                    x_offset = float(offset) * dvoigt
                    voigt_val = _voigt_profile_jit(x_offset, adamp, h0tab, h1tab, h2tab)
                    profile_val = kappa0 * voigt_val
                rec = catalog.records[int(debug_line_idx)]
                print(
                    f"PY_DEBUG_ASYNTH_LINE: line={debug_line_idx} wl={line_wl:.6f} "
                    f"elem={rec.element} ion={int(rec.ion_stage)} "
                    f"type={int(rec.line_type)} depth={debug_depth + 1} "
                    f"offset={offset} kappa0={kappa0:.6e} adamp={adamp:.6e} "
                    f"dopple={dopple:.6e} profile={profile_val:.6e} "
                    f"wcon={wcon_dbg:.6e} wtail={wtail_dbg:.6e} "
                    f"wave={wave_target:.6f} skip_wcon={int(skip_by_wcon)} "
                    f"taper={taper_factor:.6e}"
                )

        debug_top_env = os.getenv("PY_DEBUG_ASYNTH_TOP")
        debug_type_env = os.getenv("PY_DEBUG_ASYNTH_TYPE")
        if (
            debug_top_env
            and debug_idx >= 0
            and 0 <= debug_depth < n_depths
            and n_wavelengths > 0
        ):
            try:
                debug_top_n = max(1, int(debug_top_env))
            except ValueError:
                debug_top_n = 20
            debug_type = None
            if debug_type_env:
                try:
                    debug_type = int(debug_type_env)
                except ValueError:
                    debug_type = None

            wave_target = float(wavelength_grid[debug_idx])
            stim_target = 1.0 - math.exp(-freq_grid[debug_idx] * hkt[debug_depth])
            cont_cut = 0.0
            if continuum_absorption is not None:
                cont_cut = float(continuum_absorption[debug_depth, debug_idx]) * cutoff
            resolu_local = 300000.0
            if n_wavelengths > 1:
                ratio = wavelength_grid[1] / wavelength_grid[0]
                if ratio > 1.0:
                    resolu_local = 1.0 / (ratio - 1.0)

            top_entries = []
            for line_idx in range(n_lines):
                if valid_mask is not None and not valid_mask[line_idx, debug_depth]:
                    continue
                if debug_type is not None:
                    if int(line_types_array[line_idx]) != debug_type:
                        continue
                kappa0 = kappa0_array[line_idx, debug_depth]
                if kappa0 <= 0.0:
                    continue
                line_wl = line_wavelengths_array[line_idx]
                doppler_width = doppler_widths_array[line_idx, debug_depth]
                if line_wl <= 0.0 or doppler_width <= 0.0:
                    continue
                dopple = doppler_width / line_wl
                if dopple <= 0.0:
                    continue
                center_idx = int(line_indices_wing[line_idx])
                offset = debug_idx - center_idx
                adamp = adamp_array[line_idx, debug_depth]

                dvoigt = 1.0 / (dopple * resolu_local) if dopple > 0.0 else 0.0
                if adamp < 0.2:
                    vsteps = 200.0
                    tabstep = vsteps * dvoigt
                    tabi = 0.5 + tabstep * float(abs(offset))
                    idx_tab = int(tabi)
                    if idx_tab < 0:
                        idx_tab = 0
                    x_offset = float(abs(offset)) * dvoigt
                    if x_offset > 10.0:
                        profile_val = kappa0 * (0.5642 * adamp / (x_offset * x_offset))
                    else:
                        if idx_tab >= h0tab.size:
                            idx_tab = h0tab.size - 1
                        profile_val = kappa0 * (h0tab[idx_tab] + adamp * h1tab[idx_tab])
                else:
                    x_offset = float(offset) * dvoigt
                    voigt_val = _voigt_profile_jit(x_offset, adamp, h0tab, h1tab, h2tab)
                    profile_val = kappa0 * voigt_val

                idx_wcon = line_idx * n_depths + debug_depth
                if idx_wcon < wcon_array.size:
                    wcon_val = wcon_array[idx_wcon]
                    if wcon_val > 0.0 and wave_target < wcon_val:
                        continue
                    wtail_val = (
                        wtail_array[idx_wcon] if idx_wcon < wtail_array.size else 0.0
                    )
                    if (
                        wcon_val > 0.0
                        and wtail_val > 0.0
                        and wave_target < wtail_val
                        and wtail_val > wcon_val
                    ):
                        profile_val *= (wave_target - wcon_val) / (wtail_val - wcon_val)

                if cont_cut > 0.0 and profile_val < cont_cut:
                    continue

                contrib = profile_val * stim_target
                if contrib > 0.0:
                    rec = catalog.records[int(line_idx)]
                    top_entries.append(
                        (
                            contrib,
                            line_idx,
                            line_wl,
                            rec.element,
                            int(rec.ion_stage),
                            float(rec.log_gf),
                            int(rec.line_type),
                            kappa0,
                            adamp,
                        )
                    )

            top_entries.sort(key=lambda item: item[0], reverse=True)
            total = float(sum(item[0] for item in top_entries))
            print(
                f"PY_DEBUG_ASYNTH_TOP: wave={wave_target:.6f} depth={debug_depth + 1} "
                f"lines={len(top_entries)} total={total:.6e} top_n={debug_top_n}"
            )
            for entry in top_entries[:debug_top_n]:
                (
                    contrib,
                    line_idx,
                    line_wl,
                    element,
                    ion_stage,
                    log_gf,
                    line_type,
                    kappa0,
                    adamp,
                ) = entry
                print(
                    f"  line={line_idx} wl={line_wl:.6f} elem={element} ion={ion_stage} "
                    f"type={line_type} loggf={log_gf:.3f} kappa0={kappa0:.3e} "
                    f"adamp={adamp:.3e} contrib={contrib:.6e}"
                )

        use_wcon_debug = wcon_array.size > 0
        debug_wing_wave_env = os.getenv("PY_DEBUG_WING_WAVE")
        if debug_wing_wave_env and n_wavelengths > 0:
            try:
                debug_wing_wave = float(debug_wing_wave_env)
            except ValueError:
                debug_wing_wave = None
            if debug_wing_wave is not None:
                idx_target = int(np.argmin(np.abs(wavelength_grid - debug_wing_wave)))
                wave_target = float(wavelength_grid[idx_target])
                depth_env = os.getenv("PY_DEBUG_WING_DEPTHS")
                if depth_env:
                    depth_list = []
                    for item in depth_env.split(","):
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

                max_offset_env = os.getenv("PY_DEBUG_WING_OFFSET_MAX")
                wl_window_env = os.getenv("PY_DEBUG_WING_WL_WINDOW")
                top_n_env = os.getenv("PY_DEBUG_WING_TOP_N")
                min_profile_env = os.getenv("PY_DEBUG_WING_MIN_PROFILE")
                max_offset = int(max_offset_env) if max_offset_env else None
                wl_window = float(wl_window_env) if wl_window_env else None
                top_n = int(top_n_env) if top_n_env else 50
                min_profile = float(min_profile_env) if min_profile_env else 0.0
                debug_candidates = os.getenv("PY_DEBUG_WING_CANDIDATES") == "1"
                debug_fortran_profile = os.getenv("PY_DEBUG_WING_FORTRAN") == "1"

                if max_offset is None and wl_window is None:
                    print(
                        "PY_DEBUG_WING: set PY_DEBUG_WING_OFFSET_MAX or "
                        "PY_DEBUG_WING_WL_WINDOW to limit scan; skipping."
                    )
                else:
                    resolu_local = 300000.0
                    if n_wavelengths > 1:
                        ratio = wavelength_grid[1] / wavelength_grid[0]
                        if ratio > 1.0:
                            resolu_local = 1.0 / (ratio - 1.0)

                    for depth_idx in depth_list:
                        if not (0 <= depth_idx < n_depths):
                            continue
                        hits = []
                        center_hits = []
                        for line_idx in range(n_lines):
                            if (
                                valid_mask is not None
                                and not valid_mask[line_idx, depth_idx]
                            ):
                                continue
                            center_idx = int(line_indices_wing[line_idx])
                            offset = idx_target - center_idx
                            if max_offset is not None and abs(offset) > max_offset:
                                continue
                            line_wl = line_wavelengths_array[line_idx]
                            if (
                                wl_window is not None
                                and abs(line_wl - wave_target) > wl_window
                            ):
                                continue

                            line_type = int(line_types_array[line_idx])
                            kappa0 = kappa0_array[line_idx, depth_idx]
                            if kappa0 <= 0.0:
                                continue
                            adamp = adamp_array[line_idx, depth_idx]
                            doppler_width = doppler_widths_array[line_idx, depth_idx]
                            dopple = doppler_width / line_wl if line_wl > 0.0 else 0.0

                            wcon = -1.0
                            wtail = -1.0
                            if use_wcon_debug:
                                idx_wcon = line_idx * n_depths + depth_idx
                                if idx_wcon < wcon_array.size:
                                    wcon_val = wcon_array[idx_wcon]
                                    if wcon_val > 0.0:
                                        wcon = wcon_val
                                        if idx_wcon < wtail_array.size:
                                            wtail_val = wtail_array[idx_wcon]
                                            if wtail_val > 0.0:
                                                wtail = wtail_val

                            profile_val = None
                            n10dop = None
                            maxstep = None
                            profile_at_n10dop = None
                            kapmin_ref = None
                            fortran_profile_val = None
                            fortran_n10dop = None
                            fortran_maxstep = None
                            fortran_profile_n10dop = None
                            fortran_cutoff_hit = None
                            if offset == 0:
                                center_val = transp[line_idx, depth_idx]
                                if abs(center_val) < min_profile:
                                    continue
                                rec = catalog.records[int(line_idx)]
                                cont_center = (
                                    kapmin_ref_array[line_idx, depth_idx] / cutoff
                                    if use_cutoff and cutoff > 0.0
                                    else float("nan")
                                )
                                center_hits.append(
                                    (
                                        abs(center_val),
                                        center_val,
                                        line_idx,
                                        line_wl,
                                        rec.element,
                                        int(rec.ion_stage),
                                        int(rec.line_type),
                                        kappa0,
                                        adamp,
                                        dopple,
                                        cont_center,
                                    )
                                )
                                continue
                            if line_type == 1:
                                gamma_rad = gamma_rad_array[line_idx]
                                bshore = gamma_vdw_array[line_idx]
                                ashore = gamma_stark_array[line_idx]
                                if gamma_rad <= 0.0 or bshore <= 0.0:
                                    continue
                                freq = C_LIGHT_NM / wave_target
                                freq_line = C_LIGHT_NM / line_wl
                                epsil = 2.0 * (freq - freq_line) / gamma_rad
                                profile_val = (
                                    kappa0
                                    * (ashore * epsil + bshore)
                                    / (epsil * epsil + 1.0)
                                    / bshore
                                )
                                maxstep = max_profile_steps
                                profile_at_n10dop = None
                                kapmin_ref = None
                            else:
                                if dopple <= 0.0:
                                    continue
                                n10dop = int(10.0 * dopple * resolu_local)
                                if n10dop == 0:
                                    n10dop = 1
                                dvoigt = 1.0 / (dopple * resolu_local)
                                kapmin_ref = (
                                    kapmin_ref_array[line_idx, depth_idx]
                                    if use_cutoff
                                    else 0.0
                                )
                                nstep_cutoff = n10dop
                                profile_at_n10dop = 0.0
                                vsteps = 200.0
                                tabstep = vsteps * dvoigt
                                for nstep in range(1, n10dop + 1):
                                    if adamp < 0.2:
                                        tabi = 0.5 + tabstep * float(nstep)
                                        idx_tab = int(tabi)
                                        if idx_tab < 0:
                                            idx_tab = 0
                                        elif idx_tab >= h0tab.size:
                                            idx_tab = h0tab.size - 1
                                        step_profile = kappa0 * (
                                            h0tab[idx_tab] + adamp * h1tab[idx_tab]
                                        )
                                    else:
                                        x_step = float(nstep) * dvoigt
                                        voigt_val = _voigt_profile_jit(
                                            x_step, adamp, h0tab, h1tab, h2tab
                                        )
                                        step_profile = kappa0 * voigt_val
                                    if nstep == n10dop:
                                        profile_at_n10dop = step_profile
                                    if use_cutoff and step_profile < kapmin_ref:
                                        nstep_cutoff = nstep
                                        break
                                else:
                                    nstep_cutoff = -1

                                if nstep_cutoff != -1:
                                    maxstep = nstep_cutoff
                                    use_far_wing = False
                                    x_far = 0.0
                                else:
                                    use_far_wing = True
                                    if (
                                        n10dop > 0
                                        and profile_at_n10dop > 0.0
                                        and kapmin_ref > 0.0
                                    ):
                                        x_far = profile_at_n10dop * float(n10dop) ** 2
                                        maxstep = int(np.sqrt(x_far / kapmin_ref) + 1.0)
                                    else:
                                        x_far = 0.0
                                        maxstep = 0
                                    if maxstep > max_profile_steps:
                                        maxstep = max_profile_steps

                                if debug_fortran_profile:
                                    (
                                        fortran_profile_val,
                                        fortran_n10dop,
                                        fortran_maxstep,
                                        fortran_profile_n10dop,
                                        fortran_cutoff_hit,
                                    ) = _compute_fortran_profile_steps(
                                        offset=offset,
                                        kappa0=kappa0,
                                        adamp=adamp,
                                        dopple=dopple,
                                        resolu=resolu_local,
                                        kapmin_ref=kapmin_ref,
                                        h0tab=h0tab,
                                        h1tab=h1tab,
                                        h2tab=h2tab,
                                        max_profile_steps=max_profile_steps,
                                    )

                                if debug_candidates:
                                    include = (
                                        abs(offset) <= maxstep
                                        if maxstep is not None
                                        else False
                                    )
                                    print(
                                        "PY_DEBUG_WING_CAND: "
                                        f"depth={depth_idx + 1} wl={line_wl:.6f} "
                                        f"offset={offset} kappa0={kappa0:.3e} "
                                        f"adamp={adamp:.3e} dopple={dopple:.3e} "
                                        f"n10dop={n10dop} maxstep={maxstep} "
                                        f"profile_n10dop={profile_at_n10dop:.3e} "
                                        f"kapmin={kapmin_ref:.3e} include={include}"
                                    )
                                if abs(offset) > maxstep:
                                    continue
                                if use_far_wing and abs(offset) > n10dop:
                                    profile_val = x_far / float(abs(offset)) ** 2
                                else:
                                    if adamp < 0.2:
                                        tabi = 0.5 + tabstep * float(abs(offset))
                                        idx_tab = int(tabi)
                                        if idx_tab < 0:
                                            idx_tab = 0
                                        elif idx_tab >= h0tab.size:
                                            idx_tab = h0tab.size - 1
                                        profile_val = kappa0 * (
                                            h0tab[idx_tab] + adamp * h1tab[idx_tab]
                                        )
                                    else:
                                        x_offset = float(abs(offset)) * dvoigt
                                        voigt_val = _voigt_profile_jit(
                                            x_offset, adamp, h0tab, h1tab, h2tab
                                        )
                                        profile_val = kappa0 * voigt_val

                            if profile_val is None:
                                continue
                            if wcon > 0.0 and wave_target < wcon:
                                continue
                            if wtail > 0.0 and wcon > 0.0 and wave_target < wtail:
                                taper = (wave_target - wcon) / max(wtail - wcon, 1e-10)
                                profile_val = profile_val * taper

                            if abs(profile_val) < min_profile:
                                continue

                            rec = catalog.records[int(line_idx)]
                            cont_center = (
                                kapmin_ref_array[line_idx, depth_idx] / cutoff
                                if use_cutoff and cutoff > 0.0
                                else float("nan")
                            )
                            hits.append(
                                (
                                    abs(profile_val),
                                    profile_val,
                                    line_idx,
                                    offset,
                                    line_wl,
                                    rec.element,
                                    int(rec.ion_stage),
                                    int(rec.line_type),
                                    kappa0,
                                    adamp,
                                    dopple,
                                    cont_center,
                                    n10dop,
                                    maxstep,
                                    profile_at_n10dop,
                                    kapmin_ref,
                                    fortran_profile_val,
                                    fortran_n10dop,
                                    fortran_maxstep,
                                    fortran_profile_n10dop,
                                    fortran_cutoff_hit,
                                )
                            )

                        hits.sort(key=lambda x: x[0], reverse=True)
                        center_hits.sort(key=lambda x: x[0], reverse=True)
                        print(
                            "PY_DEBUG_WING: "
                            f"wave={wave_target:.6f} idx={idx_target} depth={depth_idx + 1} "
                            f"hits={len(hits)} showing={min(top_n, len(hits))}"
                        )
                        if center_hits:
                            print(
                                "PY_DEBUG_CENTER: "
                                f"wave={wave_target:.6f} idx={idx_target} depth={depth_idx + 1} "
                                f"hits={len(center_hits)} showing={min(top_n, len(center_hits))}"
                            )
                            for (
                                _abs_val,
                                center_val,
                                line_idx,
                                line_wl,
                                element,
                                ion_stage,
                                line_type,
                                kappa0,
                                adamp,
                                dopple,
                                cont_center,
                            ) in center_hits[:top_n]:
                                print(
                                    "PY_DEBUG_CENTER_HIT: "
                                    f"depth={depth_idx + 1} line={line_idx} "
                                    f"center_idx={int(line_indices_wing[line_idx])} "
                                    f"idx={idx_target} "
                                    f"kappa={center_val:.6e} adamp={adamp:.6e} "
                                    f"kappa0={kappa0:.6e} wl={line_wl:.6f} "
                                    f"elem={element} ion={ion_stage} "
                                    f"type={line_type} dopple={dopple:.6e} "
                                    f"cont={cont_center:.6e}"
                                )
                        for (
                            _abs_val,
                            profile_val,
                            line_idx,
                            offset,
                            line_wl,
                            element,
                            ion_stage,
                            line_type,
                            kappa0,
                            adamp,
                            dopple,
                            cont_center,
                            n10dop,
                            maxstep,
                            profile_at_n10dop,
                            kapmin_ref,
                            fortran_profile_val,
                            fortran_n10dop,
                            fortran_maxstep,
                            fortran_profile_n10dop,
                            fortran_cutoff_hit,
                        ) in hits[:top_n]:
                            print(
                                "PY_DEBUG_WING_HIT: "
                                f"depth={depth_idx + 1} line={line_idx} "
                                f"center_idx={int(line_indices_wing[line_idx])} "
                                f"idx={idx_target} offset={offset} "
                                f"profile={profile_val:.6e} adamp={adamp:.6e} "
                                f"kappa0={kappa0:.6e} wl={line_wl:.6f} "
                                f"elem={element} ion={ion_stage} "
                                f"type={line_type} dopple={dopple:.6e} "
                                f"cont={cont_center:.6e} "
                                f"n10dop={n10dop} maxstep={maxstep} "
                                f"profile_n10dop={profile_at_n10dop} "
                                f"kapmin={kapmin_ref} "
                                f"ft_profile={fortran_profile_val} "
                                f"ft_n10dop={fortran_n10dop} "
                                f"ft_maxstep={fortran_maxstep} "
                                f"ft_profile_n10dop={fortran_profile_n10dop} "
                                f"ft_cutoff={fortran_cutoff_hit}"
                            )
        _compute_asynth_wings_kernel(
            asynth,
            wavelength_grid,
            transp,
            (
                valid_mask
                if valid_mask is not None
                else np.ones((n_lines, n_depths), dtype=np.bool_)
            ),
            line_wavelengths_array,
            line_indices_array,
            line_types_array,
            stim_factors,
            kappa0_array,
            adamp_array,
            doppler_widths_array,
            gamma_rad_array,
            gamma_stark_array,
            gamma_vdw_array,
            kapmin_ref_array,
            (
                continuum_absorption
                if use_cutoff
                else np.zeros((n_depths, n_wavelengths), dtype=np.float64)
            ),
            wcon_array,
            wtail_array,
            cutoff,
            max_profile_steps,
            h0tab,
            h1tab,
            h2tab,
        )

        # Center contributions were already added before kernel call
        # Kernel only adds wing contributions

        # Fortran synthe.for line 94: ASYNTH(J)=TRANSP(J,I)*(1.-EXP(-FREQ*HKT(J)))
        # Apply stimulated emission factor after center+wing accumulation.
        asynth *= stim_grid

    else:
        # Numba is required for performance - no pure Python fallback
        raise RuntimeError(
            "Numba is required for ASYNTH wing computation. "
            "Please install numba: pip install numba"
        )

    # DEBUG: Summary statistics for ASYNTH
    print(f"\nDEBUG: ASYNTH statistics:")
    print(f"  ASYNTH shape: {asynth.shape}")
    print(f"  ASYNTH non-zero count: {np.count_nonzero(asynth):,}")
    print(f"  ASYNTH max: {float(np.max(asynth)):.6e}")
    asynth_nonzero = asynth[asynth > 0] if np.any(asynth > 0) else np.array([0.0])
    print(f"  ASYNTH min (non-zero): {float(np.min(asynth_nonzero)):.6e}")
    print(f"  ASYNTH mean (non-zero): {float(np.mean(asynth_nonzero)):.6e}")
    print(f"  Values > 1e10: {np.sum(asynth > 1e10):,}")
    print(f"  Values > 1e20: {np.sum(asynth > 1e20):,}")
    print(f"  Values > 1e24: {np.sum(asynth > 1e24):,}")

    debug_wave_env = os.getenv("PY_DEBUG_ASYNTH_WAVE")
    debug_depth_env = os.getenv("PY_DEBUG_ASYNTH_DEPTH")
    if debug_wave_env and debug_depth_env:
        try:
            debug_wave_val = float(debug_wave_env)
            debug_depth_val = int(debug_depth_env) - 1
            if 0 <= debug_depth_val < n_depths:
                idx_target = int(np.argmin(np.abs(wavelength_grid - debug_wave_val)))
                print(
                    f"PY_DEBUG_ASYNTH_TOTAL: wave={float(wavelength_grid[idx_target]):.6f} "
                    f"depth={debug_depth_val + 1} asynth={float(asynth[debug_depth_val, idx_target]):.6e}"
                )
        except ValueError:
            pass

    return asynth
