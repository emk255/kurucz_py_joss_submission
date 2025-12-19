"""LTE radiative transfer helpers using the Kurucz JOSH solver."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Tuple

import numpy as np

from synthe_py.physics.josh_solver import solve_josh_flux

_H_PLANCK = 6.62607015e-27  # erg * s
_C_LIGHT = 2.99792458e10  # cm / s
_C_LIGHT_NM = 2.99792458e17  # nm / s
_K_BOLTZ = 1.380649e-16  # erg / K


def _planck_nu(freq: float, temperature: np.ndarray) -> np.ndarray:
    """Compute Planck function B_nu(T) using Fortran's exact formula.
    
    Fortran formula (atlas7v.for line 190):
    BNU(J) = 1.47439D-2 * FREQ15^3 * EHVKT(J) / STIM(J)
    Where FREQ15 = FREQ / 1.D15, EHVKT = exp(-FREQ*HKT), STIM = 1 - EHVKT
    
    This matches Fortran exactly to avoid any numerical precision differences.
    """
    # CRITICAL FIX: Match Fortran exactly - no clamping of temperature or STIM
    # Fortran (atlas7v.for line 186-187): EHVKT(J)=EXP(-FREQ*HKT(J)), STIM(J)=1.-EHVKT(J)
    # Fortran does NOT clamp temperature or STIM - use values directly
    freq15 = freq / 1.0e15
    hkt = _H_PLANCK / (_K_BOLTZ * temperature)
    ehvkt = np.exp(-freq * hkt)
    stim = 1.0 - ehvkt
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        bnu = 1.47439e-2 * freq15**3 * ehvkt / stim
    bnu[np.isnan(bnu)] = 0.0
    return bnu


def solve_lte_frequency(
    wavelength_nm: float,
    temperature: np.ndarray,
    column_mass: np.ndarray,
    cont_abs: np.ndarray,
    cont_scat: np.ndarray,
    line_opacity: np.ndarray,
    line_scattering: np.ndarray,
    line_source: Optional[np.ndarray] = None,
    debug: bool = False,
) -> Tuple[float, float]:
    # Filter INF/NaN values in continuum opacity (prevents TAUNU integration failure)
    # Fortran uses REAL*4 for CONTINUUM (max ~3.4e38), so filter values exceeding this
    # Reference: synthe.for line 9 (REAL*4) and line 215 (10.**CONTINUUM)
    MAX_OPACITY_REAL4 = 3.4e38
    mask = (
        (column_mass >= 0.0)
        & np.isfinite(column_mass)
        & np.isfinite(cont_abs)  # Filter INF in continuum absorption
        & np.isfinite(cont_scat)  # Filter INF in continuum scattering
        & np.isfinite(line_opacity)  # Filter INF in line opacity
        & np.isfinite(line_scattering)  # Filter INF in line scattering
        & (cont_abs < MAX_OPACITY_REAL4)  # Filter very large values (REAL*4 max)
        & (cont_scat < MAX_OPACITY_REAL4)  # Filter very large values
        & (line_opacity < MAX_OPACITY_REAL4)  # Filter very large line opacity
        & (line_scattering < MAX_OPACITY_REAL4)  # Filter very large line scattering
    )
    if not np.any(mask):
        # CRITICAL DEBUG: Why is mask all False?
        print(f"\n{'='*70}")
        print(f"CRITICAL: All layers filtered out in solve_lte_frequency!")
        print(f"{'='*70}")
        print(f"  Wavelength: {wavelength_nm:.6f} nm")
        print(f"  Total layers: {len(column_mass)}")
        print(f"  column_mass >= 0: {np.sum(column_mass >= 0.0)}")
        print(f"  isfinite(column_mass): {np.sum(np.isfinite(column_mass))}")
        print(f"  isfinite(cont_abs): {np.sum(np.isfinite(cont_abs))}")
        print(f"  isfinite(cont_scat): {np.sum(np.isfinite(cont_scat))}")
        print(f"  isfinite(line_opacity): {np.sum(np.isfinite(line_opacity))}")
        print(f"  isfinite(line_scattering): {np.sum(np.isfinite(line_scattering))}")
        print(f"  cont_abs < MAX: {np.sum(cont_abs < MAX_OPACITY_REAL4)}")
        print(f"  cont_scat < MAX: {np.sum(cont_scat < MAX_OPACITY_REAL4)}")
        print(f"  line_opacity < MAX: {np.sum(line_opacity < MAX_OPACITY_REAL4)}")
        print(f"  line_scattering < MAX: {np.sum(line_scattering < MAX_OPACITY_REAL4)}")
        print(f"  line_opacity max: {np.max(line_opacity) if line_opacity.size > 0 else 'N/A':.8E}")
        print(f"  line_scattering max: {np.max(line_scattering) if line_scattering.size > 0 else 'N/A':.8E}")
        print(f"  Mask sum: {np.sum(mask)}")
        print(f"{'='*70}\n")
        return 0.0, 0.0

    # RHOX is now read correctly from fort.5 (not fort.10's wrong "depth" field)
    # Values are already in correct units (g/cm²) - no scaling needed
    mass_raw = np.asarray(column_mass[mask], dtype=np.float64)

    # Filter out zero-depth layers (invalid layers at the end)
    valid_mask = mass_raw > 0
    if not np.any(valid_mask):
        # CRITICAL DEBUG: Why are all masses zero?
        print(f"\n{'='*70}")
        print(f"CRITICAL: All column masses are zero or negative!")
        print(f"{'='*70}")
        print(f"  Wavelength: {wavelength_nm:.6f} nm")
        print(f"  mass_raw min/max: {mass_raw.min():.8E} / {mass_raw.max():.8E}")
        print(f"  mass_raw > 0 count: {np.sum(mass_raw > 0)}")
        print(f"{'='*70}\n")
        return 0.0, 0.0

    mass_valid = mass_raw[valid_mask]
    temp_valid = np.asarray(temperature[mask][valid_mask], dtype=np.float64)
    cont_a_valid = np.asarray(cont_abs[mask][valid_mask], dtype=np.float64)
    cont_s_valid = np.asarray(cont_scat[mask][valid_mask], dtype=np.float64)
    line_a_valid = np.asarray(line_opacity[mask][valid_mask], dtype=np.float64)
    line_sig_valid = np.asarray(line_scattering[mask][valid_mask], dtype=np.float64)

    # Fortran convention: J=1 is surface (small RHOX), J=NRHOX is deep (large RHOX)
    # Fortran assumes arrays are ALREADY in correct order (surface → deep, increasing RHOX)
    # No reversal logic needed - arrays from NPZ should already be in correct order
    # After masking: index 0 = surface (smallest RHOX), index -1 = deep (largest RHOX)
    mass = mass_valid
    temp = temp_valid
    cont_a = cont_a_valid
    cont_s = cont_s_valid
    line_a = line_a_valid
    line_sig = line_sig_valid
    
    # Fortran convention: J=1 is surface (small RHOX), J=NRHOX is deep (large RHOX)
    # INTEG requires RHOX to be monotonically increasing (surface → deep)
    # Fortran does NOT check array order - it assumes arrays are already correct
    # We should match Fortran: assume arrays are in correct order, only reverse if mass is decreasing
    
    # CRITICAL FIX: Match Fortran behavior - only reverse if mass is decreasing
    # Fortran doesn't check individual opacity arrays, it just uses them as-is
    # The only reversal needed is if RHOX itself is decreasing (which shouldn't happen with correct NPZ files)
    line_a_was_reversed = False  # Track if line_a was reversed (for line_source alignment)
    if mass.size > 1:
        mass_increasing = mass[0] < mass[-1]
        
        # Fortran does NOT check individual opacity arrays - it assumes they're already in correct order
        # Only reverse ALL arrays together if mass is decreasing (which shouldn't happen with correct NPZ files)
        # This matches Fortran's behavior: arrays are assumed to be in correct order (surface → deep)
        
        # Now ensure mass is in increasing order (surface → deep) for INTEG
        # Fortran's INTEG requires RHOX to be monotonically increasing
        if not mass_increasing:
            # mass is decreasing, reverse everything (matching Fortran's expectation that RHOX increases)
            if wavelength_nm < 400.0:
                print(f"  WARNING: Mass is decreasing, reversing all arrays to match Fortran convention")
            mass = mass[::-1]
            temp = temp[::-1]
            cont_a = cont_a[::-1]
            cont_s = cont_s[::-1]
            line_a = line_a[::-1]
            line_sig = line_sig[::-1]
            if line_source is not None:
                line_source = line_source[::-1]
            # Reset line_a_was_reversed since we've reversed everything together
            line_a_was_reversed = False
    
    # CRITICAL FIX: Match Fortran behavior - no clipping of opacity arrays
    # Fortran uses REAL*8 (double precision) and doesn't clip opacity arrays
    # Arrays are already float64 from masking/conversion above, matching Fortran REAL*8
    # Only ensure arrays are finite (masking already filtered INF/NaN)

    freq = _C_LIGHT_NM / max(wavelength_nm, 1e-12)
    planck = _planck_nu(freq, temp)
    
    if line_source is not None:
        ls_full = np.asarray(line_source, dtype=np.float64)
        
        # CRITICAL FIX: Reverse line_source in FULL array if line_a was reversed
        # This must happen BEFORE masking to keep them aligned after masking
        # We do this here (after getting the full array) rather than earlier to avoid
        # issues with parameter reassignment
        if line_a_was_reversed:
            if wavelength_nm < 400.0:
                print(f"  REVERSING line_source full array (line_a was reversed)")
            ls_full = ls_full[::-1]
        
        # CRITICAL: Filter NaN/INF from line_source BEFORE masking
        # Fortran would propagate NaN/INF, but we need to filter them to prevent flux calculation failures
        # However, we should log warnings to match Fortran's behavior
        nan_mask_full = np.isnan(ls_full)
        inf_mask_full = np.isinf(ls_full)
        if np.any(nan_mask_full) or np.any(inf_mask_full):
            import logging
            logger = logging.getLogger(__name__)
            if np.any(nan_mask_full):
                logger.warning(
                    f"Line source contains {np.sum(nan_mask_full)} NaN values at wavelength {wavelength_nm:.6f} nm "
                    f"(replacing with Planck function to prevent flux failure)"
                )
            if np.any(inf_mask_full):
                logger.warning(
                    f"Line source contains {np.sum(inf_mask_full)} INF values at wavelength {wavelength_nm:.6f} nm "
                    f"(replacing with Planck function to prevent flux failure)"
                )
            # Compute Planck for full array to replace NaN/INF
            temp_full = np.asarray(temperature, dtype=np.float64)
            planck_full = _planck_nu(freq, temp_full)
            # Replace NaN/INF with Planck function (safer than propagating)
            # This is a compromise: Fortran would propagate, but we need to prevent flux failures
            ls_full = np.where(nan_mask_full | inf_mask_full, planck_full, ls_full)
        # Now apply masking
        ls = ls_full[mask][valid_mask]  # Already in correct order (aligned with line_a)
    else:
        ls = None
    
    line_src = ls if ls is not None else planck
    
    # CRITICAL FIX: line_source was already reversed in FULL array when line_a was reversed
    # So after masking, line_src should already be aligned with line_a
    # No need to reverse line_src again here - it's already in the correct order
    # The reversal happened in the full array before masking, so masking preserves alignment
    
    # CRITICAL DEBUG: Check line_source values at problematic wavelengths
    if line_source is not None and (312.36 <= wavelength_nm <= 312.64):
        print(f"\n{'='*70}")
        print(f"DEBUG: Line source values at {wavelength_nm:.6f} nm")
        print(f"{'='*70}")
        print(f"  line_a_was_reversed: {line_a_was_reversed}")
        print(f"  ls_full shape (after reversal check): {ls_full.shape if 'ls_full' in locals() else 'N/A'}")
        if 'ls_full' in locals():
            print(f"  ls_full[0] (first element): {ls_full[0]:.6e}")
            print(f"  ls_full[-1] (last element): {ls_full[-1]:.6e}")
        print(f"  ls shape (after masking): {ls.shape if ls is not None else 'None'}")
        print(f"  line_src shape: {line_src.shape}")
        print(f"  planck shape: {planck.shape}")
        print(f"  line_src[0] (surface): {line_src[0]:.6e}")
        print(f"  planck[0] (surface): {planck[0]:.6e}")
        print(f"  line_src[-1] (deep): {line_src[-1]:.6e}")
        print(f"  planck[-1] (deep): {planck[-1]:.6e}")
        print(f"  line_src[0] / planck[0]: {line_src[0] / max(planck[0], 1e-40):.6e}")
        print(f"  line_src[-1] / planck[-1]: {line_src[-1] / max(planck[-1], 1e-40):.6e}")
        print(f"  line_src == planck? {np.allclose(line_src, planck, rtol=1e-3)}")
        print(f"  Max difference: {np.abs(line_src - planck).max():.6e}")
        print(f"  Is line_src reversed? {np.isclose(line_src[0], planck[-1], rtol=1e-3)}")
        # Check alignment with line_a
        if 'line_a' in locals() and line_a.size > 0:
            print(f"  Alignment check:")
            print(f"    line_a[0] (surface): {line_a[0]:.6e}")
            print(f"    line_a[-1] (deep): {line_a[-1]:.6e}")
            # If aligned, line_src[0] should correspond to same layer as line_a[0]
            # We can't directly check this, but we can verify line_src[0] != planck[-1] if aligned
        print(f"{'='*70}\n")

    # CRITICAL DEBUG: Check if line_opacity is being filtered out incorrectly
    if line_a.size > 0 and np.any(line_opacity > 0) and np.all(line_a == 0):
        print(f"\n{'='*70}")
        print(f"CRITICAL: line_opacity has non-zero values but line_a is all zeros!")
        print(f"{'='*70}")
        print(f"  Wavelength: {wavelength_nm:.6f} nm")
        print(f"  line_opacity size: {line_opacity.size}")
        print(f"  line_opacity non-zero count: {np.count_nonzero(line_opacity)}")
        print(f"  line_opacity max: {np.max(line_opacity):.8E}")
        print(f"  line_a size: {line_a.size}")
        print(f"  line_a non-zero count: {np.count_nonzero(line_a)}")
        print(f"  mask sum: {np.sum(mask)}")
        print(f"  valid_mask sum: {np.sum(valid_mask)}")
        if line_opacity.size > 0:
            print(f"  line_opacity[0] (before mask): {line_opacity[0]:.8E}")
            print(f"  line_opacity[0] < MAX_OPACITY_REAL4? {line_opacity[0] < MAX_OPACITY_REAL4}")
            print(f"  isfinite(line_opacity[0])? {np.isfinite(line_opacity[0])}")
            # Check mask conditions for first layer
            print(f"  mask[0] breakdown:")
            print(f"    column_mass[0] >= 0: {column_mass[0] >= 0.0}")
            print(f"    isfinite(column_mass[0]): {np.isfinite(column_mass[0])}")
            print(f"    isfinite(cont_abs[0]): {np.isfinite(cont_abs[0])}")
            print(f"    isfinite(cont_scat[0]): {np.isfinite(cont_scat[0])}")
            print(f"    isfinite(line_opacity[0]): {np.isfinite(line_opacity[0])}")
            print(f"    isfinite(line_scattering[0]): {np.isfinite(line_scattering[0])}")
            print(f"    cont_abs[0] < MAX: {cont_abs[0] < MAX_OPACITY_REAL4}")
            print(f"    cont_scat[0] < MAX: {cont_scat[0] < MAX_OPACITY_REAL4}")
            print(f"    line_opacity[0] < MAX: {line_opacity[0] < MAX_OPACITY_REAL4}")
            print(f"    line_scattering[0] < MAX: {line_scattering[0] < MAX_OPACITY_REAL4}")
            print(f"    mask[0] = {mask[0]}")
        if line_a.size > 0:
            print(f"  line_a[0] (after mask): {line_a[0]:.8E}")
        print(f"{'='*70}\n")
    
    # Debug flag: enable for problematic wavelengths
    problem_waves = [
        300.911,
        300.901,
        300.189,
        300.179,
        302.681,
        302.671,
        302.500,
        302.490,
        306.009,
        305.999,
        300.12016916,  # Add wavelength with line opacity
        300.00040572,
    ]
    debug = any(abs(wavelength_nm - w) < 0.01 for w in problem_waves)

    # CRITICAL DEBUG: Always check planck[0] for 300.00040572
    # Also enable debug for JOSH solver to get ATLAS7V debug output
    debug_wavelength = abs(wavelength_nm - 300.00040572) < 0.0001
    if debug_wavelength:
        print(f"\n{'='*70}")
        print(f"CRITICAL DEBUG: Wavelength {wavelength_nm:.8f} nm")
        print(f"{'='*70}")
        print(f"  freq = {freq:.6e} Hz")
        print(f"  planck[0] = {planck[0]:.8E}")
        print(f"  Expected planck[0] for 300.00040572 nm: 3.37116427E-08")
        print(f"  Error: {(planck[0]/3.37116427E-08 - 1)*100:+.2f}%")
        print(f"{'='*70}\n")

    # Debug: Check line opacity and source function values
    # Also enable debug if line opacity is huge (potential TAUNU overflow issue)
    huge_line_opacity = line_a[0] > 1e10 if line_a.size > 0 else False
    if debug or huge_line_opacity:
        print(f"\n{'='*70}")
        print(f"PYTHON LINE OPACITY DEBUG (wavelength {wavelength_nm:.6f} nm)")
        if huge_line_opacity:
            print(f"  *** HUGE LINE OPACITY DETECTED - ENABLING DEBUG ***")
        print(f"{'='*70}")
        print(f"  Line opacity (surface, J=1): {line_a[0]:.8E}")
        print(f"  Line opacity (deep, J={len(line_a)}): {line_a[-1]:.8E}")
        print(f"  Line opacity max: {line_a.max():.8E} at layer {np.argmax(line_a)+1}")
        print(f"  Line opacity min: {line_a.min():.8E}")
        print(f"  Continuum opacity (surface): {cont_a[0]:.8E}")
        print(f"  Continuum opacity (deep): {cont_a[-1]:.8E}")
        print(f"  Line source (surface): {line_src[0]:.8E}")
        print(f"  Line source (deep): {line_src[-1]:.8E}")
        print(f"  Planck (surface): {planck[0]:.8E}")
        print(f"  Planck (deep): {planck[-1]:.8E}")
        print(f"  Total opacity (surface): {(cont_a[0] + line_a[0]):.8E}")
        print(f"  Total opacity (deep): {(cont_a[-1] + line_a[-1]):.8E}")
        print(f"  Line/Cont ratio (surface): {line_a[0] / max(cont_a[0], 1e-40):.6f}")
        print(f"  Line/Cont ratio (deep): {line_a[-1] / max(cont_a[-1], 1e-40):.6f}")
        print(f"{'='*70}\n")

    # Enable debug for specific wavelengths to match Fortran debug output
    # Also enable debug if line opacity is huge (potential TAUNU overflow issue)
    debug_josh = debug or debug_wavelength or huge_line_opacity
    
    # CRITICAL DEBUG: Check line_a values before passing to solve_josh_flux
    if debug_josh:
        print(f"\n{'='*70}")
        print(f"BEFORE solve_josh_flux (TOTAL FLUX)")
        print(f"{'='*70}")
        print(f"  line_a size: {line_a.size}")
        print(f"  line_a[0] = {line_a[0]:.8E}" if line_a.size > 0 else "  line_a is empty")
        print(f"  line_a max = {np.max(line_a):.8E}" if line_a.size > 0 else "  N/A")
        print(f"  line_a non-zero count: {np.count_nonzero(line_a)}")
        print(f"  line_sig[0] = {line_sig[0]:.8E}" if line_sig.size > 0 else "  line_sig is empty")
        print(f"  line_sig max = {np.max(line_sig):.8E}" if line_sig.size > 0 else "  N/A")
        print(f"  line_sig non-zero count: {np.count_nonzero(line_sig)}")
        print(f"{'='*70}\n")
    
    flux_total = solve_josh_flux(
        cont_a,
        planck,
        line_a,
        line_src,
        cont_s,
        line_sig,
        mass,
        debug=debug_josh,
        debug_label=f"FLUX_TOTAL_{wavelength_nm:.8f}",
        temperature=temp,  # Pass temperature for debug output
    )
    
    # CRITICAL DEBUG: Check if flux_total is zero
    # Check for exactly zero OR very small values
    if flux_total == 0.0 or abs(flux_total) < 1e-50 or np.isnan(flux_total) or np.isinf(flux_total):
        print(f"\n{'='*70}")
        print(f"CRITICAL: flux_total is zero/invalid in solve_lte_frequency!")
        print(f"{'='*70}")
        print(f"  Wavelength: {wavelength_nm:.6f} nm")
        print(f"  flux_total = {flux_total:.8E}")
        print(f"  cont_a size: {cont_a.size}")
        print(f"  cont_a[0] = {cont_a[0]:.8E}" if cont_a.size > 0 else "  cont_a is empty")
        print(f"  planck[0] = {planck[0]:.8E}" if planck.size > 0 else "  planck is empty")
        print(f"  line_a[0] = {line_a[0]:.8E}" if line_a.size > 0 else "  line_a is empty")
        print(f"  mass size: {mass.size}")
        print(f"  mass[0] = {mass[0]:.8E}" if mass.size > 0 else "  mass is empty")
        print(f"{'='*70}\n")

    zero_line = np.zeros_like(line_a)
    zero_scatter = np.zeros_like(line_sig)
    # CRITICAL FIX: For continuum-only, SCONT should be Planck function, not scattering opacity!
    # In Fortran (atlas7v.for line 4477): SCONT(J) = BNU(J) for continuum-only
    # For continuum-only flux, use planck as scont (matching Fortran SCONT = BNU)
    # sigmac should be cont_s (scattering opacity), NOT planck!
    # CRITICAL DEBUG: Verify cont_s is actually scattering opacity, not Planck function
    if debug:
        print(f"\n  DEBUG: Before solve_josh_flux (continuum-only):")
        print(f"    cont_s[0] = {cont_s[0]:.8E}")
        print(f"    planck[0] = {planck[0]:.8E}")
        print(
            f"    cont_s[0] == planck[0]? {np.isclose(cont_s[0], planck[0], rtol=1e-6)}"
        )
        print(f"    cont_a[0] = {cont_a[0]:.8E}")
    flux_cont = solve_josh_flux(
        cont_a,
        planck,  # scont: continuum source function (Planck function)
        zero_line,
        planck,  # sline: line source function (not used since aline=0)
        cont_s,  # sigmac: continuum scattering opacity (CRITICAL FIX!)
        zero_scatter,
        mass,
        debug=debug,
        debug_label=f"FLUX_CONT_{wavelength_nm:.8f}",
        temperature=temp,  # Pass temperature for debug output
    )

    # CRITICAL DEBUG: Check if flux_cont is zero
    # Check for exactly zero OR very small values
    if flux_cont == 0.0 or abs(flux_cont) < 1e-50 or np.isnan(flux_cont) or np.isinf(flux_cont):
        print(f"\n{'='*70}")
        print(f"CRITICAL: flux_cont is zero/invalid in solve_lte_frequency!")
        print(f"{'='*70}")
        print(f"  Wavelength: {wavelength_nm:.6f} nm")
        print(f"  flux_cont = {flux_cont:.8E}")
        print(f"  cont_a size: {cont_a.size}")
        print(f"  cont_a[0] = {cont_a[0]:.8E}" if cont_a.size > 0 else "  cont_a is empty")
        print(f"  planck[0] = {planck[0]:.8E}" if planck.size > 0 else "  planck is empty")
        print(f"  mass size: {mass.size}")
        print(f"  mass[0] = {mass[0]:.8E}" if mass.size > 0 else "  mass is empty")
        print(f"{'='*70}\n")
    
    return flux_total, flux_cont


def _process_wavelength_batch(
    args: Tuple[
        int,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        bool,
    ],
) -> Tuple[int, float, float]:
    """Process a single wavelength (for parallel execution)."""
    (
        idx,
        wl,
        temperature,
        column_mass,
        cont_abs_col,
        cont_scat_col,
        line_opacity_col,
        line_scattering_col,
        line_source_col,
        debug,
    ) = args

    ft, fc = solve_lte_frequency(
        wl,
        temperature,
        column_mass,
        cont_abs_col,
        cont_scat_col,
        line_opacity_col,
        line_scattering_col,
        line_source_col,
        debug=debug,
    )
    return idx, ft, fc


def solve_lte_spectrum(
    wavelength_nm: np.ndarray,
    temperature: np.ndarray,
    column_mass: np.ndarray,
    cont_abs: np.ndarray,
    cont_scat: np.ndarray,
    line_opacity: np.ndarray,
    line_scattering: np.ndarray,
    line_source: Optional[np.ndarray] = None,
    n_workers: Optional[int] = None,
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve LTE radiative transfer for a spectrum.

    Parameters
    ----------
    wavelength_nm:
        Wavelength array (nm)
    temperature:
        Temperature array (K) for each depth
    column_mass:
        Column mass array (g/cm²) for each depth
    cont_abs:
        Continuum absorption opacity (n_depths, n_wavelengths)
    cont_scat:
        Continuum scattering opacity (n_depths, n_wavelengths)
    line_opacity:
        Line absorption opacity (n_depths, n_wavelengths)
    line_scattering:
        Line scattering opacity (n_depths, n_wavelengths)
    line_source:
        Optional line source function (n_depths, n_wavelengths)
    n_workers:
        Number of parallel workers. If None or 1, uses sequential processing.
        If > 1, uses multiprocessing.

    Returns
    -------
    flux_total:
        Total flux (HNU) for each wavelength
    flux_cont:
        Continuum flux (HNU) for each wavelength
    """
    n_points = wavelength_nm.size
    flux_total = np.zeros(n_points, dtype=np.float64)
    flux_cont = np.zeros(n_points, dtype=np.float64)

    logger = logging.getLogger(__name__)

    # Determine number of workers
    if n_workers is None:
        n_workers = 1
    elif n_workers < 1:
        n_workers = 1

    # Enable debug for first few wavelengths to compare with Fortran (if debug=False)
    debug_wavelengths = [
        300.00040572,
        300.00540576,
        300.01040589,
        300.01540609,
        300.02040638,
    ]

    # Log initial status
    logger.info(f"Solving radiative transfer for {n_points:,} wavelengths...")
    if n_points > 10000:
        logger.warning(
            f"Large wavelength grid ({n_points:,} points) - "
            f"consider using --wavelength-subsample to reduce computation time"
        )

    if n_workers > 1:
        logger.info(f"Using {n_workers} parallel workers")
    else:
        logger.info("Using sequential processing")

    # Prepare arguments for processing
    process_args = []
    for idx in range(n_points):
        wl = wavelength_nm[idx]
        # Use global debug flag, or enable for specific wavelengths if debug=False
        debug_this = debug or any(abs(wl - dwl) < 0.0001 for dwl in debug_wavelengths)
        line_src_col = line_source[:, idx] if line_source is not None else None

        process_args.append(
            (
                idx,
                wl,
                temperature,
                column_mass,
                cont_abs[:, idx],
                cont_scat[:, idx],
                line_opacity[:, idx],
                line_scattering[:, idx],
                line_src_col,
                debug_this,
            )
        )

    # Process wavelengths
    if n_workers > 1 and n_points > 100:  # Only parallelize for large grids
        # Parallel processing
        log_interval = max(1, n_points // 100)  # Log every 1%
        completed = 0

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Store wavelength values in a dict for error handling
            idx_to_wavelength = {args[0]: args[1] for args in process_args}
            futures = {
                executor.submit(_process_wavelength_batch, args): args[0]
                for args in process_args
            }

            for future in as_completed(futures):
                try:
                    idx, ft, fc = future.result()
                    flux_total[idx] = ft
                    flux_cont[idx] = fc
                    completed += 1

                    # Progress logging
                    if completed % log_interval == 0 or completed == n_points:
                        percent = 100.0 * completed / n_points
                        wl = idx_to_wavelength.get(idx, 0.0)
                        logger.info(
                            f"Progress: {completed:,}/{n_points:,} ({percent:.1f}%) - "
                            f"wavelength {wl:.2f} nm"
                        )
                except Exception as e:
                    idx = futures[future]
                    wl = idx_to_wavelength.get(idx, 0.0)
                    logger.error(
                        f"Error processing wavelength {idx} ({wl:.2f} nm): {e}"
                    )
                    # Set to zero on error
                    flux_total[idx] = 0.0
                    flux_cont[idx] = 0.0
                    completed += 1
    else:
        # Sequential processing with progress logging
        log_interval = max(1, n_points // 100)  # Log every 1%

        for idx in range(n_points):
            wl = wavelength_nm[idx]
            # Use global debug flag, or enable for specific wavelengths if debug=False
            debug_this = debug or any(abs(wl - dwl) < 0.0001 for dwl in debug_wavelengths)

            line_src_col = line_source[:, idx] if line_source is not None else None
            ft, fc = solve_lte_frequency(
                wavelength_nm[idx],
                temperature,
                column_mass,
                cont_abs[:, idx],
                cont_scat[:, idx],
                line_opacity[:, idx],
                line_scattering[:, idx],
                line_src_col,
                debug=debug_this,
            )
            flux_total[idx] = ft
            flux_cont[idx] = fc

            # Progress logging
            if (idx + 1) % log_interval == 0 or idx == n_points - 1:
                percent = 100.0 * (idx + 1) / n_points
                logger.info(
                    f"Progress: {idx+1:,}/{n_points:,} ({percent:.1f}%) - "
                    f"wavelength {wl:.2f} nm"
                )

            if debug_this:
                print(f"\n{'='*70}")
                print(f"WAVELENGTH {wl:.8f} nm - FINAL RESULTS")
                print(f"{'='*70}")
                print(f"  Flux total (HNU): {ft:.8E}")
                print(f"  Flux continuum (HNU): {fc:.8E}")
                print(f"  Ratio: {ft / max(fc, 1e-40):.6f}")
                print(f"{'='*70}\n")

    logger.info(f"Completed radiative transfer for {n_points:,} wavelengths")
    return flux_total, flux_cont
