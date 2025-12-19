#!/usr/bin/env python3
"""Compare intermediate JOSH solver values with Fortran."""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from synthe_py.io.atmosphere import load_cached
from synthe_py.physics import continuum
from synthe_py.physics.josh_solver import solve_josh_flux, _integ
from synthe_py.physics.josh_tables import XTAU_GRID

# Constants
C_LIGHT_NM = 2.99792458e17
H_PLANCK = 6.62607015e-27
K_BOLTZ = 1.380649e-16

def compute_bnu(wavelength, temperature):
    freq = C_LIGHT_NM / wavelength
    freq15 = freq / 1.0e15
    hkt = H_PLANCK / (K_BOLTZ * temperature)
    ehvkt = np.exp(-freq * hkt)
    stim = np.maximum(1.0 - ehvkt, 1e-300)
    bnu = 1.47439e-02 * freq15**3 * ehvkt / stim
    return bnu

def main():
    print("=" * 80)
    print("JOSH SOLVER INTERMEDIATE VALUES DIAGNOSTIC")
    print("=" * 80)
    
    atm = load_cached(Path("synthe_py/data/at12_aaaaa_atmosphere.npz"))
    test_wl = 400.0
    
    cont_abs, cont_scat, _, _ = continuum.build_depth_continuum(atm, np.array([test_wl]))
    acont = cont_abs[:, 0]
    sigmac = cont_scat[:, 0]
    abtot = acont + sigmac
    
    bnu = compute_bnu(test_wl, atm.temperature)
    aline = np.zeros(atm.layers)
    sigmal = np.zeros(atm.layers)
    sline = bnu.copy()
    scont = bnu.copy()
    
    # Use the cumulative approach that gives ratio ~8.23
    column_mass_raw = atm.depth / 1e14
    valid_mask = column_mass_raw > 0
    mass_valid = column_mass_raw[valid_mask]
    
    mass_reversed = mass_valid[::-1]
    diffs = np.diff(np.append(0, mass_reversed))
    mass_cumulative = np.cumsum(np.abs(diffs))
    mass_shifted = mass_cumulative - mass_cumulative[0]
    
    # Get arrays in correct order
    abtot_rev = abtot[valid_mask][::-1]
    acont_rev = acont[valid_mask][::-1]
    sigmac_rev = sigmac[valid_mask][::-1]
    scont_rev = scont[valid_mask][::-1]
    aline_rev = aline[valid_mask][::-1]
    sline_rev = sline[valid_mask][::-1]
    sigmal_rev = sigmal[valid_mask][::-1]
    
    # Compute optical depth
    start = abtot_rev[0] * mass_shifted[0]  # Should be ~0
    taunu = _integ(mass_shifted, abtot_rev, start)
    taunu = np.maximum.accumulate(taunu)
    
    print(f"\nOptical Depth Analysis:")
    print(f"  Surface tau: {taunu[0]:.6E}")
    print(f"  Deep tau: {taunu[-1]:.6E}")
    print(f"  RHOX range: {mass_shifted[0]:.6E} - {mass_shifted[-1]:.6E}")
    print(f"  ABTOT range: {abtot_rev[0]:.6E} - {abtot_rev[-1]:.6E}")
    
    # Read Fortran fort.33 to compare
    fort33_path = Path("synthe/stmp_at12_aaaaa/fort.33")
    if fort33_path.exists():
        print(f"\nReading Fortran fort.33...")
        with fort33_path.open() as f:
            lines = f.readlines()
        
        # Find line with wavelength 400.0
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    wl = float(parts[0])
                    if abs(wl - 400.0) < 0.1:
                        hnu_fortran = float(parts[1])
                        surf_fortran = float(parts[2])
                        # Read TAUNU values (next lines)
                        taunu_fortran = []
                        for j in range(i+1, min(i+10, len(lines))):
                            taunu_line = lines[j].strip().split()
                            if len(taunu_line) == 10:
                                taunu_fortran.extend([float(x) for x in taunu_line])
                            else:
                                break
                        
                        print(f"\nFortran values (wavelength {wl:.2f} nm):")
                        print(f"  HNU(1): {hnu_fortran:.6E}")
                        print(f"  SURF(1): {surf_fortran:.6E}")
                        print(f"  TAUNU (first 10): {taunu_fortran[:10]}")
                        print(f"  TAUNU (last 10): {taunu_fortran[-10:]}")
                        
                        # Compare
                        print(f"\nComparison:")
                        print(f"  Python surface tau: {taunu[0]:.6E}")
                        print(f"  Fortran surface tau: {taunu_fortran[0]:.6E if taunu_fortran else 'N/A'}")
                        print(f"  Python deep tau: {taunu[-1]:.6E}")
                        print(f"  Fortran deep tau: {taunu_fortran[-1]:.6E if taunu_fortran else 'N/A'}")
                        
                        # Compute flux
                        hnu_python = solve_josh_flux(acont_rev, scont_rev, aline_rev, sline_rev,
                                                     sigmac_rev, sigmal_rev, mass_shifted)
                        print(f"\n  Python HNU: {hnu_python:.6E}")
                        print(f"  Fortran SURF: {surf_fortran:.6E}")
                        print(f"  Ratio: {hnu_python / surf_fortran:.6f}")
                        break
                except (ValueError, IndexError):
                    continue
    
    print("=" * 80)

if __name__ == "__main__":
    main()

