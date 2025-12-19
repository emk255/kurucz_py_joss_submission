#!/usr/bin/env python3
"""Verify what components are included in ACONT vs what Python computes."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from synthe_py.io.atmosphere import load_cached
from synthe_py.physics import continuum

def main():
    """Verify ACONT components."""
    print("=" * 80)
    print("VERIFYING ACONT COMPONENTS")
    print("=" * 80)
    
    atm_path = Path("synthe_py/data/at12_aaaaa_atmosphere_fixed.npz")
    target_wl = 490.0
    
    atm = load_cached(atm_path)
    
    # Compute from fort.10 coefficients
    cont_abs, cont_scat, _, _ = continuum.build_depth_continuum(atm, np.array([target_wl]))
    
    print(f"\nWavelength: {target_wl} nm")
    print(f"Layer 0:")
    print(f"  Continuum absorption (from fort.10): {cont_abs[0, 0]:.6E} cm²/g")
    print(f"  Continuum scattering (from fort.10):  {cont_scat[0, 0]:.6E} cm²/g")
    print(f"  Total continuum:                     {cont_abs[0, 0] + cont_scat[0, 0]:.6E} cm²/g")
    
    # Check what Fortran ACONT should include (from atlas7v.for line 4474):
    # ACONT = A + AHYD + AHMIN + AXCONT + AHE1 + AHE2 + AC1 + AMG1 + AAL1 + ASI1 + AFE1
    # Where A = AH2P + AHEMIN + ALUKE + AHOT
    
    print("\n" + "=" * 80)
    print("FORTran ACONT COMPONENTS (from atlas7v.for)")
    print("=" * 80)
    print("ACONT = A + AHYD + AHMIN + AXCONT + AHE1 + AHE2 + AC1 + AMG1 + AAL1 + ASI1 + AFE1")
    print("Where A = AH2P + AHEMIN + ALUKE + AHOT")
    print("\nComponents:")
    print("  A = AH2P + AHEMIN + ALUKE + AHOT")
    print("  + AHYD (hydrogen)")
    print("  + AHMIN (H-)")
    print("  + AXCONT (metal continuum)")
    print("  + AHE1, AHE2 (helium)")
    print("  + AC1, AMG1, AAL1, ASI1, AFE1 (metals)")
    
    print("\n" + "=" * 80)
    print("FORTran SIGMAC COMPONENTS (from atlas7v.for line 4485)")
    print("=" * 80)
    print("SIGMAC = SIGH + SIGHE + SIGEL + SIGH2 + SIGX")
    print("\nComponents:")
    print("  SIGH (hydrogen scattering)")
    print("  SIGHE (helium scattering)")
    print("  SIGEL (electron scattering)")
    print("  SIGH2 (H2 scattering)")
    print("  SIGX (metal scattering)")
    
    # Expected Fortran values
    fortran_kappa_total = 7.156976E-04  # cm²/g (from fort.33)
    python_kappa_total = 2.580699E-04   # cm²/g (from diagnostics)
    
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"Fortran κ_total (from fort.33): {fortran_kappa_total:.6E} cm²/g")
    print(f"Python κ_total (from diagnostics): {python_kappa_total:.6E} cm²/g")
    print(f"Ratio (Fo/Py): {fortran_kappa_total / python_kappa_total:.6f}×")
    
    print(f"\nPython continuum (from fort.10): {cont_abs[0, 0] + cont_scat[0, 0]:.6E} cm²/g")
    print(f"Ratio (Fo/Python_cont): {fortran_kappa_total / (cont_abs[0, 0] + cont_scat[0, 0]):.6f}×")
    
    # Check if fort.10 ACONT matches expected
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("If fort.10 ACONT includes all components, then:")
    print(f"  fort.10 ACONT should ≈ Fortran ACONT")
    print(f"  fort.10 ACONT = {cont_abs[0, 0]:.6E} cm²/g")
    print(f"  Expected Fortran ACONT ≈ {fortran_kappa_total:.6E} cm²/g (if continuum-only)")
    print(f"  Ratio: {fortran_kappa_total / cont_abs[0, 0]:.6f}×")
    
    print("\nBut fort.10 ACONT is written by xnfpelsyn.for which calls KAPP,")
    print("which should compute the same ACONT as atlas7v.for.")
    print("\nPossible issues:")
    print("1. fort.10 coefficients are computed at different frequency points")
    print("2. fort.10 coefficients are missing some components")
    print("3. There's a unit conversion issue")
    print("4. Python is reading/interpolating incorrectly (but we verified it matches fort.10)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

