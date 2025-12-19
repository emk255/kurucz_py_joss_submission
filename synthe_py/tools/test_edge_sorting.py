#!/usr/bin/env python3
"""Test if sorting edge table by ABS helps."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from synthe_py.io.atmosphere import load_cached
from synthe_py.physics import continuum

def main():
    """Test edge sorting."""
    print("=" * 80)
    print("TESTING EDGE TABLE SORTING")
    print("=" * 80)
    
    atm_path = Path("synthe_py/data/at12_aaaaa_atmosphere_fixed.npz")
    target_wl = 490.0
    
    atm = load_cached(atm_path)
    wledge = np.asarray(atm.continuum_wledge, dtype=np.float64)
    
    print(f"\nOriginal edge table:")
    print(f"  Size: {len(wledge)}")
    print(f"  Is sorted: {np.all(np.diff(wledge) >= 0)}")
    print(f"  Min: {np.min(wledge):.6f}, Max: {np.max(wledge):.6f}")
    
    # Take ABS
    wledge_abs = np.abs(wledge)
    print(f"\nAfter ABS:")
    print(f"  Is sorted: {np.all(np.diff(wledge_abs) >= 0)}")
    print(f"  Min: {np.min(wledge_abs):.6f}, Max: {np.max(wledge_abs):.6f}")
    
    # Sort by ABS
    sort_idx = np.argsort(wledge_abs)
    wledge_sorted = wledge_abs[sort_idx]
    print(f"\nAfter sorting by ABS:")
    print(f"  Is sorted: {np.all(np.diff(wledge_sorted) >= 0)}")
    
    # Test computation with sorted vs unsorted
    print(f"\nTesting computation...")
    cont_abs_unsorted, cont_scat_unsorted, _, _ = continuum.build_depth_continuum(atm, np.array([target_wl]))
    
    # Temporarily replace edge table with sorted version
    original_wledge = atm.continuum_wledge
    atm.continuum_wledge = wledge_sorted
    # Also need to sort coefficients accordingly
    if atm.continuum_abs_coeff is not None:
        # Coefficients are indexed by edge, so we need to reorder them
        # This is complex - let's just check if ABS helps first
        pass
    
    print(f"\nUnsorted (current):")
    print(f"  Absorption: {cont_abs_unsorted[0, 0]:.6E} cm²/g")
    print(f"  Scattering: {cont_scat_unsorted[0, 0]:.6E} cm²/g")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

