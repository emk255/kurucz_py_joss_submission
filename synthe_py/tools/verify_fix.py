#!/usr/bin/env python3
"""Verify the sequential search fix produces correct opacity values."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from synthe_py.io.atmosphere import load_cached
from synthe_py.physics import continuum

def main():
    """Verify fix."""
    print("=" * 80)
    print("VERIFYING SEQUENTIAL SEARCH FIX")
    print("=" * 80)
    
    # Try sorted atmosphere first, fall back to fixed
    atm_path = Path("synthe_py/data/at12_aaaaa_atmosphere_sorted.npz")
    if not atm_path.exists():
        atm_path = Path("synthe_py/data/at12_aaaaa_atmosphere_fixed.npz")
    target_wl = 490.0
    
    atm = load_cached(atm_path)
    
    # Compute continuum opacity
    cont_abs, cont_scat, _, _ = continuum.build_depth_continuum(atm, np.array([target_wl]))
    
    print(f"\nWavelength: {target_wl} nm")
    print(f"Layer 0:")
    print(f"  Continuum absorption: {cont_abs[0, 0]:.6E} cm²/g")
    print(f"  Continuum scattering: {cont_scat[0, 0]:.6E} cm²/g")
    print(f"  Total continuum:      {cont_abs[0, 0] + cont_scat[0, 0]:.6E} cm²/g")
    
    # Expected from fort.10 direct (before fix)
    expected_abs = 1.374679E-04
    expected_scat = 9.112372E-04
    expected_total = 1.048705E-03
    
    print(f"\nExpected (from fort.10 direct, single wavelength):")
    print(f"  Continuum absorption: {expected_abs:.6E} cm²/g")
    print(f"  Continuum scattering: {expected_scat:.6E} cm²/g")
    print(f"  Total continuum:      {expected_total:.6E} cm²/g")
    
    print(f"\nComparison:")
    abs_match = np.isclose(cont_abs[0, 0], expected_abs, rtol=1e-5)
    scat_match = np.isclose(cont_scat[0, 0], expected_scat, rtol=1e-5)
    print(f"  Absorption match: {abs_match}")
    print(f"  Scattering match: {scat_match}")
    
    if abs_match and scat_match:
        print("\n✓ Fix verified! Opacity values match expected fort.10 values.")
        print("  Ready to rerun synthesis pipeline.")
    else:
        print("\n⚠️  Values don't match expected. Check edge indexing logic.")
        print(f"  Absorption ratio: {cont_abs[0, 0] / expected_abs:.6f}×")
        print(f"  Scattering ratio: {cont_scat[0, 0] / expected_scat:.6f}×")
    
    return abs_match and scat_match

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

