#!/usr/bin/env python3
"""Test if single-element array vs multi-element array gives different results."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from synthe_py.io.atmosphere import load_cached
from synthe_py.physics import continuum

def main():
    """Test single vs array."""
    print("=" * 80)
    print("TESTING SINGLE VS ARRAY COMPUTATION")
    print("=" * 80)
    
    # Try interleaved reading fix first
    atm_path = Path("synthe_py/data/at12_aaaaa_atmosphere_fixed_interleaved.npz")
    if not atm_path.exists():
        atm_path = Path("synthe_py/data/at12_aaaaa_atmosphere_sorted_fixed.npz")
    if not atm_path.exists():
        atm_path = Path("synthe_py/data/at12_aaaaa_atmosphere_sorted.npz")
    if not atm_path.exists():
        atm_path = Path("synthe_py/data/at12_aaaaa_atmosphere_fixed.npz")
    diag_path = Path("synthe_py/out/diagnostics_rhox_fixed.npz")
    
    atm = load_cached(atm_path)
    diag = np.load(diag_path)
    
    diag_wl = diag['wavelength']
    idx_490 = np.argmin(np.abs(diag_wl - 490.0))
    exact_wl = diag_wl[idx_490]
    
    print(f"\nTesting wavelength: {exact_wl:.9f} nm")
    
    # Test 1: Single element array
    print(f"\n1. Single-element array:")
    wl_single = np.array([exact_wl])
    cont_abs_single, cont_scat_single, _, _ = continuum.build_depth_continuum(atm, wl_single)
    print(f"   Absorption: {cont_abs_single[0, 0]:.6E} cm²/g")
    print(f"   Scattering: {cont_scat_single[0, 0]:.6E} cm²/g")
    
    # Test 2: Multi-element array (just this wavelength)
    print(f"\n2. Two-element array (same wavelength twice):")
    wl_two = np.array([exact_wl, exact_wl])
    cont_abs_two, cont_scat_two, _, _ = continuum.build_depth_continuum(atm, wl_two)
    print(f"   Absorption[0]: {cont_abs_two[0, 0]:.6E} cm²/g")
    print(f"   Absorption[1]: {cont_abs_two[0, 1]:.6E} cm²/g")
    print(f"   Scattering[0]: {cont_scat_two[0, 0]:.6E} cm²/g")
    print(f"   Scattering[1]: {cont_scat_two[0, 1]:.6E} cm²/g")
    
    # Test 3: Extract from full grid
    print(f"\n3. Extract from full grid computation:")
    cont_abs_full, cont_scat_full, _, _ = continuum.build_depth_continuum(atm, diag_wl)
    print(f"   Absorption: {cont_abs_full[0, idx_490]:.6E} cm²/g")
    print(f"   Scattering: {cont_scat_full[0, idx_490]:.6E} cm²/g")
    
    # Test 4: Diagnostics value
    print(f"\n4. Diagnostics value:")
    print(f"   Absorption: {diag['continuum_absorption'][0, idx_490]:.6E} cm²/g")
    print(f"   Scattering: {diag['continuum_scattering'][0, idx_490]:.6E} cm²/g")
    
    # Compare
    print(f"\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"Single vs Two-element:")
    print(f"  Absorption match: {np.isclose(cont_abs_single[0, 0], cont_abs_two[0, 0], rtol=1e-10)}")
    print(f"  Scattering match: {np.isclose(cont_scat_single[0, 0], cont_scat_two[0, 0], rtol=1e-10)}")
    
    print(f"\nSingle vs Full grid:")
    print(f"  Absorption match: {np.isclose(cont_abs_single[0, 0], cont_abs_full[0, idx_490], rtol=1e-10)}")
    print(f"  Scattering match: {np.isclose(cont_scat_single[0, 0], cont_scat_full[0, idx_490], rtol=1e-10)}")
    
    print(f"\nFull grid vs Diagnostics:")
    print(f"  Absorption match: {np.isclose(cont_abs_full[0, idx_490], diag['continuum_absorption'][0, idx_490], rtol=1e-10)}")
    print(f"  Scattering match: {np.isclose(cont_scat_full[0, idx_490], diag['continuum_scattering'][0, idx_490], rtol=1e-10)}")
    
    if not np.isclose(cont_abs_single[0, 0], cont_abs_full[0, idx_490], rtol=1e-10):
        print(f"\n⚠️  SINGLE VS FULL GRID DIFFER!")
        print(f"  Single: {cont_abs_single[0, 0]:.6E}")
        print(f"  Full:   {cont_abs_full[0, idx_490]:.6E}")
        print(f"  Ratio:  {cont_abs_single[0, 0] / cont_abs_full[0, idx_490]:.6f}×")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

