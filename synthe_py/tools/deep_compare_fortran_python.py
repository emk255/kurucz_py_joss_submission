#!/usr/bin/env python3
"""Deep comparison of Fortran vs Python JOSH solver intermediate values."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from synthe_py.io.atmosphere import load_cached
from synthe_py.io import spectrv as spectrv_io
from synthe_py.physics import continuum, bfudge
from synthe_py.physics.josh_solver import _integ, _map1, COEFJ_DIAG
from synthe_py.physics.josh_tables import XTAU_GRID, NXTAU, COEFJ_MATRIX, CH_WEIGHTS

# Constants
EPS = 1.0e-38
ITER_TOL = 1.0e-9

def solve_josh_manual(xsbar, xalpha, verbose=False):
    """Solve JOSH iteration manually with detailed tracking."""
    xs = xsbar.copy()
    rhs = (1.0 - xalpha) * xsbar
    coefj_diag = np.asarray(COEFJ_DIAG, dtype=xs.dtype)
    diag = 1.0 - xalpha * coefj_diag
    diag = np.where(np.abs(diag) < EPS, EPS, diag)
    coefj_matrix = np.asarray(COEFJ_MATRIX, dtype=xs.dtype)
    
    iterations = []
    for iter_num in range(NXTAU):
        iferr = False
        max_error = 0.0
        xs_prev = xs.copy()
        
        for k in range(NXTAU - 1, -1, -1):
            delta = np.dot(coefj_matrix[k], xs)
            delta = (delta * xalpha[k] + rhs[k] - xs[k]) / diag[k]
            base = xs[k]
            errorx = abs(delta / base) if abs(base) > EPS else abs(delta)
            max_error = max(max_error, errorx)
            if errorx > ITER_TOL:
                iferr = True
            xs[k] = max(base + delta, EPS)
        
        iterations.append({
            'iter': iter_num + 1,
            'max_error': max_error,
            'converged': not iferr,
            'xs_sum': np.sum(xs),
            'xs_change': np.max(np.abs(xs - xs_prev)),
        })
        
        if verbose and iter_num < 5:
            print(f"  Iter {iter_num + 1}: max_error={max_error:.2E}, xs_sum={np.sum(xs):.6E}, change={np.max(np.abs(xs - xs_prev)):.2E}")
        
        if not iferr:
            break
    
    flux = float(np.dot(CH_WEIGHTS, xs))
    return xs, flux, iterations

def main():
    """Deep comparison."""
    print("=" * 80)
    print("DEEP COMPARISON: FORTRAN VS PYTHON JOSH SOLVER")
    print("=" * 80)
    
    # Load atmosphere
    atm_path = Path("synthe_py/data/at12_aaaaa_atmosphere.npz")
    atm = load_cached(atm_path)
    test_wavelength = 400.0
    
    # Compute continuum
    cont_abs, cont_scat, _, _ = continuum.build_depth_continuum(
        atm, np.array([test_wavelength])
    )
    acont = cont_abs[:, 0]
    sigmac = cont_scat[:, 0]
    
    # Compute frequency quantities
    C_LIGHT_NM = 2.99792458e17
    H_PLANCK = 6.62607015e-27
    K_BOLTZ = 1.380649e-16
    
    freq = C_LIGHT_NM / test_wavelength
    temp = atm.temperature
    planck = 1.47439e-02 * (freq / 1.0e15)**3 * np.exp(-freq * H_PLANCK / (K_BOLTZ * temp)) / np.maximum(1.0 - np.exp(-freq * H_PLANCK / (K_BOLTZ * temp)), 1e-300)
    
    # Read ASYNTH
    from synthe_py.tools.compare_step5_line_rt import read_fortran_fort29_asynth
    asynth_fortran = read_fortran_fort29_asynth(test_wavelength)
    
    # Compute arrays
    RHOXJ = 1e10
    column_mass_raw = atm.depth / 1e14
    valid_mask = column_mass_raw > 0
    mass_valid = column_mass_raw[valid_mask]
    mass_reversed = mass_valid[::-1]
    diffs = np.diff(np.append(0, mass_reversed))
    mass_cumulative = np.cumsum(np.abs(diffs))
    mass_shifted = mass_cumulative - mass_cumulative[0]
    RHOX_SCALE_FACTOR = 1.0 / 6.659004
    rhox_scaled = mass_shifted * RHOX_SCALE_FACTOR
    
    fscat = np.zeros(len(rhox_scaled))
    for j in range(len(rhox_scaled)):
        if rhox_scaled[j] / RHOXJ < 100.0:
            fscat[j] = np.exp(-rhox_scaled[j] / RHOXJ)
    
    fscat_full = np.zeros(atm.layers)
    fscat_full[valid_mask] = fscat[::-1]
    
    aline = asynth_fortran * (1.0 - fscat_full)
    sigmal = asynth_fortran * fscat_full
    
    # Load spectrv params
    fort25_path = Path("synthe/stmp_at12_aaaaa/fort.25")
    if fort25_path.exists():
        spectrv_params = spectrv_io.load(fort25_path)
    else:
        spectrv_params = spectrv_io.SpectrvParams(
            rhoxj=0.0, ph1=0.0, pc1=0.0, psi1=0.0, prddop=0.0, prdpow=0.0
        )
    
    bfudge_values, slinec = bfudge.compute_bfudge_and_slinec(
        atmosphere=atm,
        params=spectrv_params,
        bnu=planck[:, None],
        stim=np.ones((atm.layers, 1)),
        ehvkt=np.zeros((atm.layers, 1)),
    )
    sline = slinec[:, 0]
    
    # Prepare arrays
    mask = (atm.depth >= 0.0) & np.isfinite(atm.depth)
    mass_raw = np.asarray(atm.depth[mask], dtype=np.float64)
    if mass_raw.size > 0 and mass_raw.max() > 1e10:
        mass = mass_raw / 1e14
    else:
        mass = mass_raw
    
    valid_mask = mass > 0
    mass_valid = mass[valid_mask]
    mass_reversed = mass_valid[::-1]
    diffs = np.diff(np.append(0, mass_reversed))
    mass_cumulative = np.cumsum(np.abs(diffs))
    mass_shifted = mass_cumulative - mass_cumulative[0]
    RHOX_SCALE_FACTOR = 1.0 / 6.659004
    mass_scaled = mass_shifted * RHOX_SCALE_FACTOR
    
    temp_rev = atm.temperature[mask][valid_mask][::-1]
    cont_a = acont[mask][valid_mask][::-1]
    cont_s = sigmac[mask][valid_mask][::-1]
    line_a = aline[mask][valid_mask][::-1]
    line_sig = sigmal[mask][valid_mask][::-1]
    line_src = sline[mask][valid_mask][::-1]
    
    planck_rev = planck[mask][valid_mask][::-1]
    
    # With lines case
    abtot_lines = cont_a + cont_s + line_a + line_sig
    scatter_lines = cont_s + line_sig
    alpha_lines = scatter_lines / np.maximum(abtot_lines, EPS)
    denom_lines = np.maximum(cont_a + line_a, EPS)
    snubar_lines = (cont_a * planck_rev + line_a * line_src) / denom_lines
    
    start_lines = abtot_lines[0] * mass_scaled[0] if mass_scaled.size else 0.0
    taunu_lines = _integ(mass_scaled, abtot_lines, start_lines)
    if taunu_lines.size:
        taunu_lines = np.maximum.accumulate(taunu_lines)
    
    xsbar_lines, _ = _map1(taunu_lines, snubar_lines, XTAU_GRID)
    xalpha_lines, _ = _map1(taunu_lines, alpha_lines, XTAU_GRID)
    xsbar_lines = np.maximum(xsbar_lines, EPS)
    xalpha_lines = np.clip(xalpha_lines, 0.0, 1.0)
    
    if taunu_lines.size:
        mask_grid = XTAU_GRID < taunu_lines[0]
        if mask_grid.any():
            xsbar_lines[mask_grid] = max(snubar_lines[0], EPS)
            xalpha_lines[mask_grid] = np.clip(alpha_lines[0], 0.0, 1.0)
    
    print("\nSolving JOSH iteration manually...")
    xs_final, flux_raw, iters = solve_josh_manual(xsbar_lines, xalpha_lines, verbose=True)
    
    print(f"\nFinal flux (before correction): {flux_raw:.6E}")
    FLUX_CORRECTION_FACTOR = 1.0 / 8.228608
    flux_corrected = flux_raw * FLUX_CORRECTION_FACTOR
    print(f"Final flux (after correction): {flux_corrected:.6E}")
    
    # Read Fortran result
    from synthe_py.tools.compare_step5_line_rt import read_fortran_fort33
    hnu_fortran, surf_fortran, _ = read_fortran_fort33(test_wavelength)
    print(f"Fortran HNU: {hnu_fortran:.6E}")
    print(f"Ratio: {flux_corrected / hnu_fortran:.6f}")
    
    # Check if the correction factor needs adjustment
    print(f"\n" + "=" * 80)
    print("CORRECTION FACTOR ANALYSIS")
    print("=" * 80)
    print(f"Current correction: {FLUX_CORRECTION_FACTOR:.10f}")
    print(f"Required correction for lines: {hnu_fortran / flux_raw:.10f}")
    print(f"Difference: {abs(FLUX_CORRECTION_FACTOR - hnu_fortran / flux_raw):.10f}")
    print(f"Relative difference: {abs(FLUX_CORRECTION_FACTOR - hnu_fortran / flux_raw) / FLUX_CORRECTION_FACTOR * 100:.2f}%")
    
    # Check intermediate values
    print(f"\n" + "=" * 80)
    print("INTERMEDIATE VALUES CHECK")
    print("=" * 80)
    print(f"XSBAR (first 10): {xsbar_lines[:10]}")
    print(f"XALPHA (first 10): {xalpha_lines[:10]}")
    print(f"XS final (first 10): {xs_final[:10]}")
    print(f"XS final (last 10): {xs_final[-10:]}")
    
    # Check if there's a pattern in the difference
    print(f"\n" + "=" * 80)
    print("PATTERN ANALYSIS")
    print("=" * 80)
    print(f"Mean alpha: {np.mean(xalpha_lines):.6f}")
    print(f"Max alpha: {np.max(xalpha_lines):.6f}")
    print(f"Min alpha: {np.min(xalpha_lines):.6f}")
    print(f"Alpha > 0.5: {np.sum(xalpha_lines > 0.5)} layers")
    
    return True

if __name__ == "__main__":
    main()

