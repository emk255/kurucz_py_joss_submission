#!/usr/bin/env python3
"""Debug why diagnostics continuum differs from fort.10 direct computation."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from synthe_py.io.atmosphere import load_cached
from synthe_py.physics import continuum

def main():
    """Debug continuum discrepancy."""
    print("=" * 80)
    print("DEBUGGING CONTINUUM DISCREPANCY")
    print("=" * 80)
    
    atm_path = Path("synthe_py/data/at12_aaaaa_atmosphere_fixed.npz")
    diag_path = Path("synthe_py/out/diagnostics_rhox_fixed.npz")
    target_wl = 490.0
    
    # Load data
    atm = load_cached(atm_path)
    diag = np.load(diag_path)
    
    # Find wavelength index
    wl = diag['wavelength']
    idx_490 = np.argmin(np.abs(wl - target_wl))
    print(f"\nTarget wavelength: {target_wl} nm")
    print(f"Diagnostics wavelength: {wl[idx_490]:.6f} nm")
    
    # Get diagnostics values
    diag_cont_abs = diag['continuum_absorption'][0, idx_490]
    diag_cont_scat = diag.get('continuum_scattering', np.zeros_like(diag_cont_abs))[0, idx_490]
    diag_cont_total = diag_cont_abs + diag_cont_scat
    
    print(f"\nDiagnostics values (layer 0):")
    print(f"  Continuum absorption: {diag_cont_abs:.6E} cm²/g")
    print(f"  Continuum scattering: {diag_cont_scat:.6E} cm²/g")
    print(f"  Total continuum:      {diag_cont_total:.6E} cm²/g")
    
    # Compute directly from fort.10 coefficients
    cont_abs_direct, cont_scat_direct, _, _ = continuum.build_depth_continuum(
        atm, np.array([target_wl])
    )
    cont_total_direct = cont_abs_direct[0, 0] + cont_scat_direct[0, 0]
    
    print(f"\nDirect fort.10 computation (layer 0):")
    print(f"  Continuum absorption: {cont_abs_direct[0, 0]:.6E} cm²/g")
    print(f"  Continuum scattering: {cont_scat_direct[0, 0]:.6E} cm²/g")
    print(f"  Total continuum:      {cont_total_direct:.6E} cm²/g")
    
    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"Diagnostics absorption: {diag_cont_abs:.6E}")
    print(f"Direct absorption:      {cont_abs_direct[0, 0]:.6E}")
    print(f"Ratio (Diag/Direct):    {diag_cont_abs / cont_abs_direct[0, 0]:.6f}×")
    
    print(f"\nDiagnostics scattering: {diag_cont_scat:.6E}")
    print(f"Direct scattering:      {cont_scat_direct[0, 0]:.6E}")
    print(f"Ratio (Diag/Direct):    {diag_cont_scat / cont_scat_direct[0, 0]:.6f}×")
    
    print(f"\nDiagnostics total: {diag_cont_total:.6E}")
    print(f"Direct total:     {cont_total_direct:.6E}")
    print(f"Ratio (Diag/Direct): {diag_cont_total / cont_total_direct:.6f}×")
    
    # Check if hydrogen continuum is included
    print("\n" + "=" * 80)
    print("HYDROGEN CONTINUUM CHECK")
    print("=" * 80)
    print(f"atm.cont_absorption is None: {atm.cont_absorption is None}")
    print(f"atm.continuum_abs_coeff is None: {atm.continuum_abs_coeff is None}")
    
    if atm.cont_absorption is None:
        print("  → Hydrogen continuum should be added separately")
    else:
        print("  → Hydrogen continuum should be in fort.10 coefficients")
    
    # Check diagnostics keys
    print("\n" + "=" * 80)
    print("DIAGNOSTICS FILE KEYS")
    print("=" * 80)
    keys = list(diag.keys())
    for key in sorted(keys):
        val = diag[key]
        if isinstance(val, np.ndarray):
            print(f"  {key}: shape {val.shape}, dtype {val.dtype}")
        else:
            print(f"  {key}: {type(val).__name__}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

