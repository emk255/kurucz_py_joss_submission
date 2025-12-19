#!/usr/bin/env python3
"""Detailed JOSH solver diagnostic - compare intermediate values."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from synthe_py.io.atmosphere import load_cached
from synthe_py.physics import continuum
from synthe_py.physics.josh_solver import solve_josh_flux, _integ, _map1
from synthe_py.physics.josh_tables import CH_WEIGHTS, XTAU_GRID, COEFJ_MATRIX
from synthe_py.physics.josh_solver import COEFJ_DIAG

# Constants matching Fortran exactly
C_LIGHT_NM = 2.99792458e17  # nm/s
H_PLANCK = 6.62607015e-27  # erg * s
K_BOLTZ = 1.380649e-16  # erg / K

def compute_frequency_quantities(wavelength, temperature):
    """Compute frequency quantities."""
    freq = C_LIGHT_NM / wavelength
    freq15 = freq / 1.0e15
    hkt = H_PLANCK / (K_BOLTZ * temperature)
    ehvkt = np.exp(-freq * hkt)
    stim = np.maximum(1.0 - ehvkt, 1e-300)
    bnu = 1.47439e-02 * freq15**3 * ehvkt / stim
    return bnu

def main():
    """Detailed JOSH diagnostic."""
    print("=" * 80)
    print("DETAILED JOSH SOLVER DIAGNOSTIC")
    print("=" * 80)
    
    # Load atmosphere
    atm_path = Path("synthe_py/data/at12_aaaaa_atmosphere.npz")
    atm = load_cached(atm_path)
    test_wavelength = 400.0
    
    # Step 1: Compute continuum
    cont_abs, cont_scat, _, _ = continuum.build_depth_continuum(
        atm, np.array([test_wavelength])
    )
    acont = cont_abs[:, 0]
    sigmac = cont_scat[:, 0]
    
    # Step 2: Compute frequency quantities
    bnu = compute_frequency_quantities(test_wavelength, atm.temperature)
    
    # Step 3: Set up continuum-only RT
    aline = np.zeros(atm.layers)
    sigmal = np.zeros(atm.layers)
    sline = bnu.copy()
    scont = bnu.copy()
    
    # Fix column mass units
    column_mass_raw = atm.depth
    if column_mass_raw.size > 0 and column_mass_raw.max() > 1e10:
        column_mass = column_mass_raw / 1e14
        print(f"\nColumn mass fix: scaled by 1e14")
        print(f"  Original: {column_mass_raw[0]:.6E} -> {column_mass_raw[-1]:.6E}")
        print(f"  Scaled:   {column_mass[0]:.6E} -> {column_mass[-1]:.6E}")
    else:
        column_mass = column_mass_raw
    
    # Step 4: Compute optical depth manually
    abtot = acont + aline + sigmac + sigmal
    print(f"\nOpacity values:")
    print(f"  ABTOT (surface): {abtot[0]:.6E}")
    print(f"  ABTOT (deep): {abtot[-1]:.6E}")
    
    start = abtot[0] * column_mass[0]
    taunu = _integ(column_mass, abtot, start)
    taunu = np.maximum.accumulate(taunu)
    
    print(f"\nOptical depth (tau) after fix:")
    print(f"  Surface tau: {taunu[0]:.6E}")
    print(f"  Deep tau: {taunu[-1]:.6E}")
    print(f"  Max tau: {taunu.max():.6E}")
    print(f"  Min tau: {taunu.min():.6E}")
    
    # Step 5: Check interpolation to XTAU_GRID
    snubar = (acont * scont + aline * sline) / np.maximum(acont + aline, 1e-38)
    alpha = (sigmac + sigmal) / np.maximum(abtot, 1e-38)
    
    xsbar, _ = _map1(taunu, snubar, XTAU_GRID)
    xalpha, _ = _map1(taunu, alpha, XTAU_GRID)
    xsbar = np.maximum(xsbar, 1e-38)
    xalpha = np.clip(xalpha, 0.0, 1.0)
    
    print(f"\nSource function interpolation:")
    print(f"  SNUBAR (surface): {snubar[0]:.6E}")
    print(f"  SNUBAR (deep): {snubar[-1]:.6E}")
    print(f"  XSBAR[0] (first grid point): {xsbar[0]:.6E}")
    print(f"  XSBAR[-1] (last grid point): {xsbar[-1]:.6E}")
    print(f"  XSBAR mean: {xsbar.mean():.6E}")
    
    print(f"\nAlpha (scattering fraction):")
    print(f"  ALPHA (surface): {alpha[0]:.6E}")
    print(f"  XALPHA[0]: {xalpha[0]:.6E}")
    print(f"  XALPHA[-1]: {xalpha[-1]:.6E}")
    
    # Step 6: Check XTAU_GRID coverage
    print(f"\nXTAU_GRID coverage:")
    print(f"  XTAU_GRID[0]: {XTAU_GRID[0]:.6E}")
    print(f"  XTAU_GRID[-1]: {XTAU_GRID[-1]:.6E}")
    print(f"  Surface tau vs grid: {taunu[0]:.6E} vs {XTAU_GRID[0]:.6E}")
    print(f"  Deep tau vs grid: {taunu[-1]:.6E} vs {XTAU_GRID[-1]:.6E}")
    
    # Step 7: Solve JOSH iteration manually
    xs = xsbar.copy()
    rhs = (1.0 - xalpha) * xsbar
    diag = 1.0 - xalpha * COEFJ_DIAG
    diag = np.where(np.abs(diag) < 1e-38, 1e-38, diag)
    
    print(f"\nJOSH iteration setup:")
    print(f"  Initial XS[0]: {xs[0]:.6E}")
    print(f"  RHS[0]: {rhs[0]:.6E}")
    print(f"  DIAG[0]: {diag[0]:.6E}")
    
    # Run a few iterations
    EPS = 1.0e-38
    ITER_TOL = 1.0e-5
    MAX_ITER = 51
    
    for iter_num in range(min(5, MAX_ITER)):
        iferr = False
        for k in range(51 - 1, -1, -1):
            delta = np.dot(COEFJ_MATRIX[k], xs)
            delta = (delta * xalpha[k] + rhs[k] - xs[k]) / diag[k]
            base = xs[k]
            errorx = abs(delta / base) if abs(base) > EPS else abs(delta)
            if errorx > ITER_TOL:
                iferr = True
            xs[k] = max(base + delta, EPS)
        if iter_num == 0:
            print(f"  After iteration {iter_num+1}: XS[0]={xs[0]:.6E}, error={iferr}")
        if not iferr:
            print(f"  Converged after {iter_num+1} iterations")
            break
    
    print(f"  Final XS[0]: {xs[0]:.6E}")
    print(f"  Final XS[-1]: {xs[-1]:.6E}")
    print(f"  Final XS mean: {xs.mean():.6E}")
    
    # Step 8: Compute flux
    flux = float(np.dot(CH_WEIGHTS, xs))
    print(f"\nFlux computation:")
    print(f"  CH_WEIGHTS sum: {CH_WEIGHTS.sum():.6E}")
    print(f"  CH_WEIGHTS[0]: {CH_WEIGHTS[0]:.6E}")
    print(f"  CH_WEIGHTS[-1]: {CH_WEIGHTS[-1]:.6E}")
    print(f"  Flux (dot product): {flux:.6E}")
    
    # Step 9: Compare with full solve_josh_flux
    hnu_full = solve_josh_flux(
        acont, scont, aline, sline, sigmac, sigmal, column_mass
    )
    print(f"\nFull solve_josh_flux result: {hnu_full:.6E}")
    print(f"  Match? {abs(flux - hnu_full) < 1e-10}")
    
    # Step 10: Compare with Fortran
    fortran_surf = 5.278200E-06
    print(f"\nFortran comparison:")
    print(f"  Fortran SURF(1): {fortran_surf:.6E}")
    print(f"  Python HNU(1): {hnu_full:.6E}")
    print(f"  Ratio (Python/Fortran): {hnu_full / fortran_surf:.6f}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

