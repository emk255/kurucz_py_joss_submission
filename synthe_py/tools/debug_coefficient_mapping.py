#!/usr/bin/env python3
"""Debug coefficient mapping after edge sorting."""

import numpy as np
import struct
from pathlib import Path

def read_fort10_edges_and_coeffs(fort10_path):
    """Read edges and first layer coefficients from fort.10."""
    with fort10_path.open('rb') as f:
        # Skip header
        header = f.read(4)
        size = struct.unpack('<i', header)[0]
        f.read(size)
        
        # Read edges record
        header = f.read(4)
        size = struct.unpack('<i', header)[0]
        rec_edges = f.read(size)
        f.read(4)
        
        idx = 0
        nedge = struct.unpack_from('<i', rec_edges, idx)[0]
        idx += 4
        frqedg = np.frombuffer(rec_edges, dtype='<f8', count=nedge, offset=idx)
        idx += nedge * 8
        wledge = np.frombuffer(rec_edges, dtype='<f8', count=nedge, offset=idx)
        idx += nedge * 8
        cmedge = np.frombuffer(rec_edges, dtype='<f8', count=nedge, offset=idx)
        
        # Read frequency grid
        header = f.read(4)
        size = struct.unpack('<i', header)[0]
        rec_freq = f.read(size)
        f.read(4)
        num_freq = struct.unpack_from('<i', rec_freq, 0)[0]
        freqset = np.frombuffer(rec_freq, dtype='<f8', count=num_freq, offset=4)
        
        # Skip atmospheric state
        header = f.read(4)
        size = struct.unpack('<i', header)[0]
        f.read(size)
        f.read(4)
        
        # Read first layer coefficients
        rec_index = 4
        cont_total = np.frombuffer(f.read(size), dtype='<f8', count=num_freq)
        f.read(4)
        rec_index += 1
        cont_abs = np.frombuffer(f.read(size), dtype='<f8', count=num_freq)
        f.read(4)
        rec_index += 1
        cont_scat = np.frombuffer(f.read(size), dtype='<f8', count=num_freq)
        f.read(4)
        
        return wledge, frqedg, cmedge, cont_abs, cont_scat, num_freq

def main():
    """Debug coefficient mapping."""
    print("=" * 80)
    print("DEBUGGING COEFFICIENT MAPPING")
    print("=" * 80)
    
    fort10_path = Path("synthe/stmp_at12_aaaaa/fort.10")
    if not fort10_path.exists():
        print(f"ERROR: fort.10 not found: {fort10_path}")
        return
    
    wledge, frqedg, cmedge, cont_abs, cont_scat, num_freq = read_fort10_edges_and_coeffs(fort10_path)
    
    print(f"\nOriginal edges (first 10):")
    print(f"  WLEDGE: {wledge[:10]}")
    wledge_abs = np.abs(wledge)
    print(f"  ABS(WLEDGE): {wledge_abs[:10]}")
    print(f"  Is sorted by ABS: {np.all(np.diff(wledge_abs) >= 0)}")
    
    # Sort edges
    sort_idx = np.argsort(wledge_abs)
    wledge_sorted = wledge[sort_idx]
    wledge_abs_sorted = np.abs(wledge_sorted)
    
    print(f"\nAfter sorting (first 10):")
    print(f"  ABS(WLEDGE): {wledge_abs_sorted[:10]}")
    print(f"  Is sorted: {np.all(np.diff(wledge_abs_sorted) >= 0)}")
    
    # Check coefficient structure
    nedge = len(wledge)
    n_intervals = nedge - 1
    print(f"\nCoefficient structure:")
    print(f"  Number of edges: {nedge}")
    print(f"  Number of intervals: {n_intervals}")
    print(f"  Coefficients per interval: 3")
    print(f"  Total coefficients: {num_freq} (expected: {n_intervals * 3})")
    
    # Reshape coefficients
    cont_abs_reshaped = cont_abs.reshape(n_intervals, 3)
    cont_scat_reshaped = cont_scat.reshape(n_intervals, 3)
    
    print(f"\nCoefficient values (first 3 intervals, absorption):")
    for i in range(min(3, n_intervals)):
        print(f"  Interval {i}: {cont_abs_reshaped[i, :]}")
    
    # Test: Find edge for 490nm
    target_wl = 490.0
    print(f"\nTesting edge search for {target_wl} nm:")
    
    # Original order
    edge_idx_orig = np.searchsorted(wledge_abs, target_wl, side='right') - 1
    edge_idx_orig = max(0, min(edge_idx_orig, n_intervals - 1))
    print(f"  Original order: edge index {edge_idx_orig}")
    print(f"    Interval: [{wledge_abs[edge_idx_orig]:.2f}, {wledge_abs[edge_idx_orig+1]:.2f}]")
    print(f"    Coefficients: {cont_abs_reshaped[edge_idx_orig, :]}")
    
    # Sorted order
    edge_idx_sorted = np.searchsorted(wledge_abs_sorted, target_wl, side='right') - 1
    edge_idx_sorted = max(0, min(edge_idx_sorted, n_intervals - 1))
    print(f"  Sorted order: edge index {edge_idx_sorted}")
    print(f"    Interval: [{wledge_abs_sorted[edge_idx_sorted]:.2f}, {wledge_abs_sorted[edge_idx_sorted+1]:.2f}]")
    
    # Find which original interval corresponds to sorted interval edge_idx_sorted
    orig_left_edge = sort_idx[edge_idx_sorted]
    orig_right_edge = sort_idx[edge_idx_sorted + 1]
    print(f"    Original edges: {orig_left_edge} and {orig_right_edge}")
    
    # Check if they're consecutive
    if orig_right_edge == orig_left_edge + 1:
        orig_interval = orig_left_edge
        print(f"    ✓ Edges are consecutive -> original interval {orig_interval}")
        print(f"    Coefficients: {cont_abs_reshaped[orig_interval, :]}")
    else:
        print(f"    ✗ Edges are NOT consecutive!")
        print(f"    Need to find original interval with edges {orig_left_edge} and {orig_right_edge}")
        # Find it
        for orig_interval in range(n_intervals):
            if orig_interval == orig_left_edge:
                print(f"    Found: original interval {orig_interval}")
                print(f"    Coefficients: {cont_abs_reshaped[orig_interval, :]}")
                break
    
    return True

if __name__ == "__main__":
    success = main()
    import sys
    sys.exit(0 if success else 1)

