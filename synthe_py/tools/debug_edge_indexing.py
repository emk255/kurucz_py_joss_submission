#!/usr/bin/env python3
"""Debug edge indexing to find the bug."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from synthe_py.io.atmosphere import load_cached

def main():
    """Debug edge indexing."""
    print("=" * 80)
    print("DEBUGGING EDGE INDEXING")
    print("=" * 80)
    
    atm_path = Path("synthe_py/data/at12_aaaaa_atmosphere_fixed.npz")
    diag_path = Path("synthe_py/out/diagnostics_rhox_fixed.npz")
    
    atm = load_cached(atm_path)
    diag = np.load(diag_path)
    
    diag_wl = diag['wavelength']
    idx_490 = np.argmin(np.abs(diag_wl - 490.0))
    exact_wl = diag_wl[idx_490]
    
    wledge = np.asarray(atm.continuum_wledge, dtype=np.float64)
    
    print(f"\nWavelength: {exact_wl:.9f} nm")
    print(f"Wavelength index in diagnostics grid: {idx_490}")
    
    # Test edge indexing for single wavelength
    print(f"\n1. Single wavelength edge indexing:")
    wl_single = np.array([exact_wl])
    edge_idx_single = np.searchsorted(wledge, wl_single, side="right") - 1
    edge_idx_single = np.clip(edge_idx_single, 0, wledge.size - 2)
    print(f"   Edge index: {edge_idx_single[0]}")
    print(f"   Edge range: {wledge[edge_idx_single[0]]:.6f} - {wledge[edge_idx_single[0]+1]:.6f} nm")
    
    # Test edge indexing for full grid
    print(f"\n2. Full grid edge indexing:")
    edge_idx_full = np.searchsorted(wledge, diag_wl, side="right") - 1
    edge_idx_full = np.clip(edge_idx_full, 0, wledge.size - 2)
    print(f"   Edge index at {idx_490}: {edge_idx_full[idx_490]}")
    print(f"   Edge range: {wledge[edge_idx_full[idx_490]]:.6f} - {wledge[edge_idx_full[idx_490]+1]:.6f} nm")
    
    # Check if they're different
    if edge_idx_single[0] != edge_idx_full[idx_490]:
        print(f"\n⚠️  DIFFERENT EDGE INDICES!")
        print(f"   Single: {edge_idx_single[0]}")
        print(f"   Full:   {edge_idx_full[idx_490]}")
    else:
        print(f"\n✓ Same edge index: {edge_idx_single[0]}")
    
    # Check edge values around the wavelength
    print(f"\n3. Edge values around wavelength:")
    for i in range(max(0, edge_idx_single[0]-3), min(len(wledge), edge_idx_single[0]+4)):
        marker = ""
        if i == edge_idx_single[0]:
            marker = " ← single"
        if i == edge_idx_full[idx_490]:
            marker = " ← full"
        print(f"   Edge[{i}]: {wledge[i]:.9f} nm{marker}")
    
    # Check if wavelength is near an edge boundary
    print(f"\n4. Distance to edges:")
    dist_to_left = abs(exact_wl - wledge[edge_idx_single[0]])
    dist_to_right = abs(exact_wl - wledge[edge_idx_single[0]+1])
    print(f"   Distance to left edge: {dist_to_left:.9f} nm")
    print(f"   Distance to right edge: {dist_to_right:.9f} nm")
    print(f"   Edge width: {wledge[edge_idx_single[0]+1] - wledge[edge_idx_single[0]]:.9f} nm")
    
    # Check if there are any NaN or inf values
    print(f"\n5. Checking for problematic values:")
    print(f"   Wavelength is NaN: {np.isnan(exact_wl)}")
    print(f"   Wavelength is Inf: {np.isinf(exact_wl)}")
    print(f"   Edge values contain NaN: {np.any(np.isnan(wledge))}")
    print(f"   Edge values contain Inf: {np.any(np.isinf(wledge))}")
    
    # Check if searchsorted behaves differently
    print(f"\n6. Testing searchsorted behavior:")
    pos_single = np.searchsorted(wledge, exact_wl, side="right")
    pos_full = np.searchsorted(wledge, diag_wl[idx_490], side="right")
    print(f"   Single wavelength position: {pos_single}")
    print(f"   Full grid position: {pos_full}")
    print(f"   Match: {pos_single == pos_full}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

