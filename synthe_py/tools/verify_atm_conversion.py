#!/usr/bin/env python3
"""Verify that NPZ generated from .atm matches ground truth from fort.10."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def compare_arrays(
    name: str,
    arr1: np.ndarray,
    arr2: np.ndarray,
    tolerance_abs: float = 1e-10,
    tolerance_rel: float = 1e-5,
) -> bool:
    """Compare two arrays and report differences.
    
    Returns True if arrays match within tolerance.
    """
    if arr1.shape != arr2.shape:
        print(f"  {name}: Shape mismatch - {arr1.shape} vs {arr2.shape}")
        return False
    
    diff = np.abs(arr1 - arr2)
    max_abs_diff = np.max(diff)
    
    # Relative error (avoid division by zero)
    denom = np.maximum(np.abs(arr2), tolerance_abs)
    rel_diff = diff / denom
    max_rel_diff = np.max(rel_diff)
    
    # For exact matches (like temperature, pressure from .atm), use stricter tolerance
    if name in ["temperature", "gas_pressure", "electron_density", "rhox", "turbulent_velocity"]:
        exact_match = max_abs_diff < tolerance_abs
        if not exact_match:
            print(f"  {name}: Max absolute diff = {max_abs_diff:.6e} (expected < {tolerance_abs})")
        return exact_match
    else:
        match = max_rel_diff < tolerance_rel
        if not match:
            print(f"  {name}: Max relative diff = {max_rel_diff:.6e} (expected < {tolerance_rel})")
            print(f"    Max absolute diff = {max_abs_diff:.6e}")
        return match


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify NPZ from .atm conversion matches ground truth"
    )
    parser.add_argument(
        "ground_truth_npz",
        type=Path,
        help="Ground truth NPZ (from convert_fort10.py)",
    )
    parser.add_argument(
        "test_npz",
        type=Path,
        help="Test NPZ (from convert_atm_to_npz.py)",
    )
    parser.add_argument(
        "--tolerance-rel",
        type=float,
        default=1e-5,
        help="Relative tolerance for comparison (default: 1e-5)",
    )
    parser.add_argument(
        "--tolerance-abs",
        type=float,
        default=1e-10,
        help="Absolute tolerance for exact matches (default: 1e-10)",
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("NPZ Conversion Verification")
    print("=" * 80)
    print(f"Ground truth: {args.ground_truth_npz}")
    print(f"Test file:   {args.test_npz}")
    print()
    
    # Load both NPZ files
    with np.load(args.ground_truth_npz) as gt_data:
        gt_keys = set(gt_data.files)
        gt_arrays = {key: gt_data[key] for key in gt_keys}
    
    with np.load(args.test_npz) as test_data:
        test_keys = set(test_data.files)
        test_arrays = {key: test_data[key] for key in test_keys}
    
    # Check for missing keys
    missing_in_test = gt_keys - test_keys
    extra_in_test = test_keys - gt_keys
    
    if missing_in_test:
        print(f"WARNING: Missing keys in test NPZ: {missing_in_test}")
    if extra_in_test:
        print(f"INFO: Extra keys in test NPZ: {extra_in_test}")
    
    print("\nComparing arrays...")
    print("-" * 80)
    
    # Compare each array
    all_match = True
    compared = 0
    
    # Basic atmosphere data (should match exactly)
    basic_keys = [
        "depth", "temperature", "tkev", "tk", "hkt", "tlog", "hckt",
        "gas_pressure", "electron_density", "xnatm", "mass_density",
        "turbulent_velocity",
    ]
    
    for key in basic_keys:
        if key not in gt_arrays or key not in test_arrays:
            if key in gt_arrays:
                print(f"  {key}: MISSING in test NPZ")
                all_match = False
            continue
        
        compared += 1
        match = compare_arrays(
            key, gt_arrays[key], test_arrays[key],
            tolerance_abs=args.tolerance_abs,
            tolerance_rel=args.tolerance_rel,
        )
        if match:
            print(f"  {key}: OK")
        else:
            all_match = False
    
    # Edge tables
    edge_keys = ["frqedg", "wledge", "cmedge"]
    for key in edge_keys:
        if key not in gt_arrays or key not in test_arrays:
            continue
        compared += 1
        match = compare_arrays(
            key, gt_arrays[key], test_arrays[key],
            tolerance_abs=args.tolerance_abs,
            tolerance_rel=args.tolerance_rel,
        )
        if match:
            print(f"  {key}: OK")
        else:
            all_match = False
    
    # Continuum coefficients (allow larger relative error)
    coeff_keys = ["cont_abs_coeff", "cont_scat_coeff", "half_edge", "delta_edge"]
    for key in coeff_keys:
        if key not in gt_arrays or key not in test_arrays:
            continue
        compared += 1
        match = compare_arrays(
            key, gt_arrays[key], test_arrays[key],
            tolerance_abs=args.tolerance_abs,
            tolerance_rel=args.tolerance_rel * 10,  # Allow 10x larger error for coefficients
        )
        if match:
            print(f"  {key}: OK")
        else:
            all_match = False
    
    # Population and Doppler data
    pop_keys = ["population_per_ion", "doppler_per_ion"]
    for key in pop_keys:
        if key not in gt_arrays or key not in test_arrays:
            continue
        compared += 1
        match = compare_arrays(
            key, gt_arrays[key], test_arrays[key],
            tolerance_abs=args.tolerance_abs,
            tolerance_rel=args.tolerance_rel * 10,
        )
        if match:
            print(f"  {key}: OK")
        else:
            all_match = False
    
    # Other arrays
    other_keys = [
        "xnf_h", "xnf_he1", "xnf_he2", "xnf_h2",
        "idmol", "momass", "freqset",
        "cont_abs", "cont_scat",
    ]
    for key in other_keys:
        if key not in gt_arrays or key not in test_arrays:
            continue
        compared += 1
        match = compare_arrays(
            key, gt_arrays[key], test_arrays[key],
            tolerance_abs=args.tolerance_abs,
            tolerance_rel=args.tolerance_rel,
        )
        if match:
            print(f"  {key}: OK")
        else:
            all_match = False
    
    print("-" * 80)
    print(f"\nCompared {compared} arrays")
    
    if all_match:
        print("\n✓ VERIFICATION PASSED: All arrays match within tolerance!")
        return 0
    else:
        print("\n✗ VERIFICATION FAILED: Some arrays do not match")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

