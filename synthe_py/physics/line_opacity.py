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


@jit(nopython=True)
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


@jit(
    nopython=True, parallel=False, cache=False
)  # CRITICAL: parallel=False to avoid race conditions
def _compute_asynth_wings_kernel(
    asynth: np.ndarray,
    wavelength_grid: np.ndarray,
    transp: np.ndarray,
    valid_mask: np.ndarray,
    line_wavelengths: np.ndarray,
    line_indices: np.ndarray,
    stim_factors: np.ndarray,
    kappa0_values: np.ndarray,
    adamp_values: np.ndarray,
    doppler_widths: np.ndarray,
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

        # CRITICAL FIX: Match Fortran synthe.for line 301 EXACTLY
        # IF(NBUFF.LT.1.OR.NBUFF.GT.LENGTH)GO TO 320
        # Fortran SKIPS lines whose centers are outside the grid - NO wing computation!
        # Previous Python code had "DELLIM handling" for margin lines which adds extra
        # opacity at grid boundaries, causing ~60x deeper absorption at grid start.
        if center_idx < 0 or center_idx >= n_wavelengths:
            continue  # Skip lines outside grid entirely (matching Fortran)

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

            if doppler_width <= 0.0:
                continue

            # CRITICAL FIX: Compute N10DOP to match Fortran behavior exactly
            # Fortran synthe.for line 311: N10DOP = 10 * (DOPPLE * RESOLU)
            # DOPPLE is dimensionless: doppler_width / line_wavelength
            dopple = doppler_width / line_wavelength if line_wavelength > 0.0 else 1e-10
            n10dop = int(10.0 * dopple * resolu)

            # If N10DOP = 0, Fortran skips ALL wing contributions for this line
            # This is the critical fix - without it, Python computes huge spurious wings
            if n10dop == 0:
                continue

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

            # For MAXSTEP estimation, we still use a reference kapmin (at line center)
            kapmin_ref = 0.0
            if use_cutoff:
                kapmin_ref = continuum_absorption[depth_idx, center_idx] * cutoff

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
            for nstep in range(1, n10dop + 1):
                x_step = float(nstep) * dvoigt
                voigt_val = _voigt_profile_jit(x_step, adamp, h0tab, h1tab, h2tab)
                profile_val = kappa0 * voigt_val  # No stim_factor here
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
            # New code (CORRECT - matches Fortran XLINOP):
            maxstep = max_profile_steps  # Always use full range, let per-step cutoff terminate

            # Phase 3: Apply profile to both red and blue wings
            while offset <= maxstep and (red_active or blue_active):
                # Compute profile value for this offset
                # CRITICAL FIX (Dec 2025): Use FULL VOIGT at ALL wing steps!
                # Fortran XLINOP (synthe.for lines 761-762, 780-781) uses:
                #   VVOIGT=ABS(WAVE-WL)/DOPWL
                #   KAPPA=KAPPA0*VOIGT(VVOIGT,ADAMP)
                # The 1/x^2 approximation (fort.12 path) underestimates far-wing opacity
                # compared to full Voigt, causing narrower wings at surface layers.
                x_offset = float(offset) * dvoigt
                voigt_val = _voigt_profile_jit(x_offset, adamp, h0tab, h1tab, h2tab)
                profile_val = kappa0 * voigt_val * stim_factor

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

                            # XLINOP per-step cutoff (line 767):
                            # IF(KAPPA.LT.CONTINUUM(IBUFF)*CUTOFF)GO TO 212
                            # Check BEFORE adding - exit red wing if below threshold
                            if use_cutoff:
                                kapmin_at_idx = (
                                    continuum_absorption[depth_idx, idx] * cutoff
                                )
                                if value_red < kapmin_at_idx:
                                    red_active = False
                                    # Skip this step but continue processing blue wing
                                else:
                                    asynth[depth_idx, idx] += value_red
                            else:
                                asynth[depth_idx, idx] += value_red

                # Process blue wing
                # Fortran XLINOP (lines 784-785): Add FIRST, then check for exit
                if blue_active:
                    idx = center_idx - offset
                    if idx < 0:
                        blue_active = False
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

                            # Check AFTER adding - exit blue wing if below threshold
                            if use_cutoff:
                                kapmin_at_idx = (
                                    continuum_absorption[depth_idx, idx] * cutoff
                                )
                                if value_blue < kapmin_at_idx:
                                    blue_active = False

                offset += 1


def compute_transp(
    catalog: "LineCatalog",
    populations: "Populations",
    atmosphere: "AtmosphereModel",
    cutoff: float = 1e-3,
    continuum_absorption: Optional[np.ndarray] = None,
    wavelength_grid: Optional[np.ndarray] = None,
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

    center_indices = _nearest_grid_indices(wavelength_grid, catalog.wavelength)
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

    # Process each line
    for line_idx in range(n_lines):
        if line_idx % log_interval == 0:
            logger.info(
                f"TRANSP progress: {line_idx:,}/{n_lines:,} lines ({100.0*line_idx/n_lines:.1f}%)"
            )

        record = catalog.records[line_idx]
        element = record.element
        nelion = record.ion_stage

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
            wavelength_nm = record.wavelength
            freq_hz = C_LIGHT_NM / wavelength_nm  # Frequency in Hz
            gf_linear = catalog.gf[
                line_idx
            ]  # Linear gf (already converted from log_gf)
            cgf = CGF_CONSTANT * gf_linear / freq_hz  # CONGF conversion

            # PRE-BOLTZMANN CUTOFF CHECK (matches Fortran synthe.for lines 262-267)
            # Fortran: KAPMIN = CONTINUUM(MIN(MAX(NBUFF,1),LENGTH)) * CUTOFF
            # Then: IF(KAPPA0.LT.KAPMIN)GO TO 350
            kappa0_pre_boltz = cgf * xnfdop

            # Compute KAPMIN dynamically (matching Fortran exactly)
            # Fortran: KAPMIN = CONTINUUM(MIN(MAX(NBUFF,1),LENGTH)) * CUTOFF
            # No fallback - Fortran always uses this formula
            center_idx = center_indices[line_idx]
            clamped_idx = max(0, min(center_idx, n_wavelengths - 1))
            kapmin = continuum_absorption[depth_idx, clamped_idx] * cutoff

            # DEBUG: Track statistics (only for first depth to avoid spam)
            is_near_300nm = DEBUG_WL_MIN <= record.wavelength <= DEBUG_WL_MAX
            if depth_idx == 0:
                debug_stats["total_line_depth_pairs"] += 1
                if is_near_300nm:
                    debug_stats["lines_near_300nm"] += 1

            if kappa0_pre_boltz < kapmin:
                if depth_idx == 0:
                    debug_stats["skipped_pre_boltz"] += 1
                continue  # Skip weak lines (matches Fortran cutoff behavior)

            # Apply Boltzmann factor AFTER pre-Boltzmann cutoff (Fortran line 268-270)
            kappa0 = kappa0_pre_boltz * boltz

            # POST-BOLTZMANN CUTOFF CHECK (Fortran line 272)
            # RE-ENABLED: This matches Fortran behavior and prevents weak line accumulation
            # Note: The Ca I 551nm issue was actually due to molecular opacity, not this cutoff
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
            # Gamma values are already linear (s^-1) after catalog load.
            gamma_rad = catalog.gamma_rad[line_idx]
            gamma_stark = catalog.gamma_stark[line_idx]
            gamma_vdw = catalog.gamma_vdw[line_idx]

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

                    # Doppler width in frequency units (Hz)
                    # delta_nu_D = (c / wavelength) * dopple
                    # where c = C_LIGHT_NM (nm/s), wavelength in nm
                    delta_nu_doppler = (C_LIGHT_NM / record.wavelength) * dopple

                    # Voigt damping parameter (dimensionless)
                    # a = gamma / (4 * pi * delta_nu_D)
                    adamp = gamma_total / (4.0 * np.pi * delta_nu_doppler)
                else:
                    adamp = 0.0
            else:
                adamp = 0.0

            # Compute line center opacity using Voigt profile
            # From Fortran line 286-290:
            #   IF(ADAMP.LT..2)THEN
            #     KAPCEN=KAPPA0*(1.-1.128*ADAMP)
            #   ELSE
            #     KAPCEN=KAPPA0*VOIGT(0.,ADAMP)
            #   ENDIF
            if adamp >= 0 and kappa0 > 0:
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

    return transp, valid_mask, np.arange(n_lines)


def compute_asynth_from_transp(
    transp: np.ndarray,
    catalog: "LineCatalog",
    atmosphere: "AtmosphereModel",
    wavelength_grid: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    populations: Optional["Populations"] = None,
    cutoff: float = 1e-3,
    continuum_absorption: Optional[np.ndarray] = None,
    metal_tables: Optional["tables.MetalWingTables"] = None,
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

    line_indices = _nearest_grid_indices(wavelength_grid, catalog.wavelength)

    # Vectorized ASYNTH computation
    # Compute frequencies for all lines at once (matches Fortran line 369)
    line_freqs = C_LIGHT_NM / catalog.wavelength  # Shape: (n_lines,)

    # Compute stimulated emission factors for all line-depth pairs
    # freq * hkt: (n_lines, 1) * (1, n_depths) -> (n_lines, n_depths)
    freq_hkt = (
        line_freqs[:, np.newaxis] * hkt[np.newaxis, :]
    )  # Shape: (n_lines, n_depths)
    stim_factors = 1.0 - np.exp(-freq_hkt)  # Shape: (n_lines, n_depths)

    # Apply valid mask if provided
    if valid_mask is not None:
        stim_factors = np.where(valid_mask, stim_factors, 0.0)

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

    # Add center contributions first (TRANSP * stim_factors)
    # This matches Fortran line 368: ASYNTH(J) = TRANSP(J,I) * (1. - EXP(-FREQ*HKT(J)))
    asynth_per_line = transp * stim_factors  # Shape: (n_lines, n_depths)
    for line_idx in range(len(catalog.records)):
        center_idx = line_indices[line_idx]
        if center_idx >= 0 and center_idx < n_wavelengths:
            for depth_idx in range(n_depths):
                if valid_mask is None or valid_mask[line_idx, depth_idx]:
                    asynth[depth_idx, center_idx] += asynth_per_line[
                        line_idx, depth_idx
                    ]

    # Pre-compute arrays for JIT kernel if Numba is available
    if NUMBA_AVAILABLE:
        # Pre-compute kappa0, adamp, doppler_widths, wcon, wtail for all lines/depths
        n_lines = len(catalog.records)

        # Initialize arrays
        kappa0_array = np.zeros((n_lines, n_depths), dtype=np.float64)
        adamp_array = np.zeros((n_lines, n_depths), dtype=np.float64)
        doppler_widths_array = np.zeros((n_lines, n_depths), dtype=np.float64)
        wcon_array = np.zeros(n_lines * n_depths, dtype=np.float64)  # Flattened
        wtail_array = np.zeros(n_lines * n_depths, dtype=np.float64)  # Flattened

        # Cache populations per element
        population_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # Get Voigt tables
        voigt_tables = tables.voigt_tables()
        h0tab = voigt_tables.h0tab
        h1tab = voigt_tables.h1tab
        h2tab = voigt_tables.h2tab

        # Pre-compute all values
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

                # Compute damping parameter correctly using frequency units
                # adamp = gamma / (4 * pi * delta_nu_D)
                # where delta_nu_D = (c / wavelength) * doppler_velocity
                dopple = (
                    doppler_width / line_wavelength if line_wavelength > 0 else 1e-6
                )
                if dopple > 0 and line_wavelength > 0:
                    gamma_total = gamma_rad + gamma_stark * xne + gamma_vdw * txnxn
                    delta_nu_doppler = (C_LIGHT_NM / line_wavelength) * dopple
                    adamp = gamma_total / (4.0 * np.pi * delta_nu_doppler)
                else:
                    adamp = 0.0

                adamp = max(adamp, 1e-12)
                adamp_array[line_idx, depth_idx] = adamp

                # Recover kappa0 from TRANSP
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

        # Call JIT kernel
        line_wavelengths_array = np.asarray(catalog.wavelength, dtype=np.float64)
        line_indices_array = np.asarray(line_indices, dtype=np.int64)

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
            stim_factors,
            kappa0_array,
            adamp_array,
            doppler_widths_array,
            (
                continuum_absorption
                if use_cutoff
                else np.zeros((n_depths, n_wavelengths), dtype=np.float64)
            ),
            wcon_array,
            wtail_array,
            cutoff,
            MAX_PROFILE_STEPS,
            h0tab,
            h1tab,
            h2tab,
        )

        # Center contributions were already added before kernel call
        # Kernel only adds wing contributions

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

    return asynth
