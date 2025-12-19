#!/usr/bin/env python3
"""Diagnose different interpretations of the depth array."""

import numpy as np
import sys
from pathlib import Path

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
    print("DEPTH ARRAY INTERPRETATION DIAGNOSTIC")
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
    
    column_mass_raw = atm.depth / 1e14
    valid_mask = column_mass_raw > 0
    mass_valid = column_mass_raw[valid_mask]
    
    print(f"\nOriginal depth array (scaled):")
    print(f"  Surface (index 0): {column_mass_raw[0]:.6E}")
    print(f"  Deep (index -1): {column_mass_raw[-1]:.6E}")
    print(f"  Valid layers: {valid_mask.sum()} out of {atm.layers}")
    
    # Test different interpretations
    interpretations = []
    
    # 1. Use depth directly (reversed)
    mass_rev = mass_valid[::-1]
    if np.all(np.diff(mass_rev) >= 0):
        hnu1 = solve_josh_flux(acont[valid_mask][::-1], scont[valid_mask][::-1],
                                aline[valid_mask][::-1], sline[valid_mask][::-1],
                                sigmac[valid_mask][::-1], sigmal[valid_mask][::-1], mass_rev)
        interpretations.append(("Direct reversed", hnu1, mass_rev[0], mass_rev[-1]))
    
    # 2. Cumulative from reversed differences
    diffs = np.diff(np.append(0, mass_rev))
    mass_cum = np.cumsum(np.abs(diffs))
    hnu2 = solve_josh_flux(acont[valid_mask][::-1], scont[valid_mask][::-1],
                           aline[valid_mask][::-1], sline[valid_mask][::-1],
                           sigmac[valid_mask][::-1], sigmal[valid_mask][::-1], mass_cum)
    interpretations.append(("Cumulative from diffs", hnu2, mass_cum[0], mass_cum[-1]))
    
    # 3. Shifted cumulative (surface = 0)
    mass_shifted = mass_cum - mass_cum[0]
    hnu3 = solve_josh_flux(acont[valid_mask][::-1], scont[valid_mask][::-1],
                           aline[valid_mask][::-1], sline[valid_mask][::-1],
                           sigmac[valid_mask][::-1], sigmal[valid_mask][::-1], mass_shifted)
    interpretations.append(("Shifted cumulative", hnu3, mass_shifted[0], mass_shifted[-1]))
    
    # 4. Flipped (max - depth)
    max_depth = mass_valid.max()
    mass_flipped = max_depth - mass_valid
    if np.all(np.diff(mass_flipped) >= 0):
        hnu4 = solve_josh_flux(acont[valid_mask], scont[valid_mask],
                               aline[valid_mask], sline[valid_mask],
                               sigmac[valid_mask], sigmal[valid_mask], mass_flipped)
        interpretations.append(("Flipped (max-depth)", hnu4, mass_flipped[0], mass_flipped[-1]))
    
    fortran_surf = 5.278200E-06
    print(f"\n{'Interpretation':<30} {'HNU':<15} {'Surface RHOX':<15} {'Deep RHOX':<15} {'Ratio':<10}")
    print("-" * 80)
    for name, hnu, surf_rhox, deep_rhox in interpretations:
        ratio = hnu / fortran_surf
        print(f"{name:<30} {hnu:.6E} {surf_rhox:.6E} {deep_rhox:.6E} {ratio:.6f}")
    
    print(f"\nFortran SURF(1): {fortran_surf:.6E}")
    print("=" * 80)

if __name__ == "__main__":
    main()

