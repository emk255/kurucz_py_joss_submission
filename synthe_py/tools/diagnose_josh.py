#!/usr/bin/env python3
"""Diagnose JOSH solver to find the ~100x difference."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from synthe_py.io.atmosphere import load_cached
from synthe_py.physics import continuum
from synthe_py.physics.josh_solver import solve_josh_flux
from synthe_py.physics.josh_tables import CH_WEIGHTS, XTAU_GRID

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
    """Diagnose JOSH solver step by step."""
    print("=" * 80)
    print("JOSH SOLVER DIAGNOSTIC")
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
    
    print(f"\nTest wavelength: {test_wavelength} nm")
    print(f"Atmosphere layers: {atm.layers}")
    print(f"\nSurface layer values:")
    print(f"  ACONT: {acont[0]:.6E}")
    print(f"  SIGMAC: {sigmac[0]:.6E}")
    print(f"  ALINE: {aline[0]:.6E}")
    print(f"  SIGMAL: {sigmal[0]:.6E}")
    print(f"  SLINE: {sline[0]:.6E}")
    print(f"  SCONT: {scont[0]:.6E}")
    print(f"  Column mass (depth[0]): {atm.depth[0]:.6E}")
    
    # Step 4: Compute optical depth manually
    abtot = acont + aline + sigmac + sigmal
    print(f"\n  ABTOT (surface): {abtot[0]:.6E}")
    
    # Compute optical depth
    start = abtot[0] * atm.depth[0]
    print(f"  Initial tau (ABTOT[0] * RHOX[0]): {start:.6E}")
    
    # Integrate tau
    from synthe_py.physics.josh_solver import _integ
    taunu = _integ(atm.depth, abtot, start)
    taunu = np.maximum.accumulate(taunu)
    
    print(f"\nOptical depth (tau) values:")
    print(f"  Surface tau: {taunu[0]:.6E}")
    print(f"  Deep tau: {taunu[-1]:.6E}")
    print(f"  Max tau: {taunu.max():.6E}")
    
    # Step 5: Check interpolation
    from synthe_py.physics.josh_solver import _map1
    snubar = (acont * scont + aline * sline) / np.maximum(acont + aline, 1e-38)
    xsbar, _ = _map1(taunu, snubar, XTAU_GRID)
    
    print(f"\nSource function interpolation:")
    print(f"  SNUBAR (surface): {snubar[0]:.6E}")
    print(f"  XSBAR[0] (first grid point): {xsbar[0]:.6E}")
    print(f"  XSBAR[-1] (last grid point): {xsbar[-1]:.6E}")
    
    # Step 6: Check CH_WEIGHTS
    print(f"\nCH_WEIGHTS (for flux computation):")
    print(f"  CH_WEIGHTS[0]: {CH_WEIGHTS[0]:.6E}")
    print(f"  CH_WEIGHTS sum: {CH_WEIGHTS.sum():.6E}")
    print(f"  CH_WEIGHTS max: {CH_WEIGHTS.max():.6E}")
    
    # Step 7: Call JOSH solver
    hnu_python = solve_josh_flux(
        acont=acont,
        scont=scont,
        aline=aline,
        sline=sline,
        sigmac=sigmac,
        sigmal=sigmal,
        column_mass=atm.depth,
    )
    
    print(f"\nPython JOSH result:")
    print(f"  HNU(1): {hnu_python:.6E}")
    
    # Step 8: Compare with Fortran
    print(f"\nFortran comparison:")
    print(f"  Fortran SURF(1): 5.278200E-06")
    print(f"  Ratio (Python/Fortran): {hnu_python / 5.278200E-06:.6f}")
    
    # Step 9: Check if there's a units issue
    print(f"\nUnits check:")
    print(f"  If Python flux is in different units, check:")
    print(f"  - Column mass units (should be g/cm²)")
    print(f"  - Optical depth units (should be dimensionless)")
    print(f"  - Source function units (should match BNU)")
    
    # Step 10: Manual flux computation to debug
    # After JOSH iteration, compute flux manually
    alpha = (sigmac + sigmal) / np.maximum(abtot, 1e-38)
    scatter = sigmac + sigmal
    
    # Interpolate to grid
    xalpha, _ = _map1(taunu, alpha, XTAU_GRID)
    xalpha = np.clip(xalpha, 0.0, 1.0)
    
    # Solve for XS (simplified - just use XSBAR for now)
    xs = xsbar.copy()
    rhs = (1.0 - xalpha) * xsbar
    
    # Compute flux manually
    flux_manual = float(np.dot(CH_WEIGHTS, xs))
    print(f"\nManual flux computation (using XSBAR directly):")
    print(f"  Flux: {flux_manual:.6E}")
    print(f"  Ratio to Python: {flux_manual / hnu_python:.6f}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

