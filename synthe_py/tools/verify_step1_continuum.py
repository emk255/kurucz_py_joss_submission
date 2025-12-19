#!/usr/bin/env python3
"""Verify STEP 1: Continuum absorption/scattering interpolation."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from synthe_py.io.atmosphere import load_cached
from synthe_py.physics import continuum

def main():
    """Verify continuum interpolation matches Fortran."""
    print("=" * 80)
    print("STEP 1 VERIFICATION: CONTINUUM ABSORPTION/SCATTERING")
    print("=" * 80)
    
    # Try to find atmosphere file
    possible_paths = [
        Path("synthe_py/data/at12_aaaaa_atmosphere.npz"),
        Path("grids/at12_aaaaa/atm/at12_aaaaa_t05770g4.44.dat.npz"),
        Path("synthe_py/cache/at12_aaaaa_t05770g4.44.npz"),
        Path("cache/at12_aaaaa_t05770g4.44.npz"),
    ]
    
    atm_path = None
    for path in possible_paths:
        if path.exists():
            atm_path = path
            break
    
    if atm_path is None:
        print("\nERROR: Could not find atmosphere .npz file")
        print("  Tried:")
        for p in possible_paths:
            print(f"    {p}")
        print("\n  Please convert atmosphere to .npz format first")
        return False
    
    print(f"\nLoading atmosphere from: {atm_path}")
    atm = load_cached(atm_path)
    print(f"  Layers: {atm.layers}")
    print(f"  Temperature range: {atm.temperature[0]:.1f} - {atm.temperature[-1]:.1f} K")
    
    # Test wavelengths matching Fortran grid
    test_wavelengths = [400.0, 500.0, 600.0]  # nm
    
    print("\n" + "-" * 80)
    print("Python Continuum Interpolation Results:")
    print("-" * 80)
    print(f"{'Wavelength':>12} {'Layer':>6} {'Absorption':>15} {'Scattering':>15}")
    print("-" * 80)
    
    for wl in test_wavelengths:
        cont_abs, cont_scat, _, _ = continuum.build_depth_continuum(
            atm, np.array([wl])
        )
        # Show surface layer (layer 0)
        print(f"{wl:12.2f} {0:6d} {cont_abs[0, 0]:15.6E} {cont_scat[0, 0]:15.6E}")
    
    print("\n" + "-" * 80)
    print("Fortran Comparison:")
    print("-" * 80)
    print("  Need to extract values from Fortran fort.10 or fort.33 output")
    print("  Fortran computes:")
    print("    ACONT(J) = 10**(C1*CONTABS(1,IEDGE,J) + C2*CONTABS(2,IEDGE,J) + C3*CONTABS(3,IEDGE,J))")
    print("    SIGMAC(J) = 10**(C1*CONTSCAT(1,IEDGE,J) + C2*CONTSCAT(2,IEDGE,J) + C3*CONTSCAT(3,IEDGE,J))")
    
    print("\n" + "=" * 80)
    print("STEP 1 STATUS: Python computation complete")
    print("  Next: Extract Fortran values from fort.33 or fort.10 for comparison")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    main()

