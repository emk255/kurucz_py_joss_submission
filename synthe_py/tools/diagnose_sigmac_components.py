#!/usr/bin/env python3
"""
Diagnose the 4.4% SIGMAC discrepancy between Python and Fortran.

SIGMAC = SIGH + SIGHE + SIGEL + SIGH2 + SIGX

With default IFOP settings, Fortran disables:
- SIGHE (IFOP(8)=0)  -> helium Rayleigh
- SIGH2 (IFOP(13)=0) -> H2 Rayleigh

So the main contributors are:
- SIGEL: Electron scattering (simple: 0.6653e-24 * XNE / RHO)
- SIGH: Hydrogen Rayleigh scattering (complex Gavrila tables)

This script compares these components to identify the source of discrepancy.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from synthe_py.io.atmosphere import load_cached

# Constants matching Fortran exactly
C_LIGHT_NM = 2.99792458e17  # nm/s


def compute_sigel(xne: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """
    Compute electron scattering opacity (SIGEL).
    
    Fortran formula (atlas7v.for line 9269):
        SIGEL(J) = 0.6653D-24 * XNE(J) / RHO(J)
    
    This is Thomson scattering: sigma_T * n_e / rho
    sigma_T = 6.653e-25 cm^2 (Thomson cross-section)
    """
    return 0.6653e-24 * xne / rho


def compute_sigh_fortran(wavelength_nm: float, xnfph1: np.ndarray, bhyd1: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """
    Compute hydrogen Rayleigh scattering (SIGH) using Fortran's Gavrila tables.
    
    This is a simplified version that uses the old formula to check consistency.
    Fortran's HRAYOP uses Gavrila tables which are more complex.
    
    Old simple formula (commented out in atlas7v.for lines 6946-6951):
        WAVE = 2.99792458e18 / MIN(FREQ, 2.463e15)
        WW = WAVE**2
        SIG = (5.799e-13 + 1.422e-6/WW + 2.784/(WW*WW)) / (WW*WW)
        SIGH(J) = SIG * XNFPH(J,1) * 2.0 * BHYD(J,1) / RHO(J)
    """
    freq = C_LIGHT_NM / wavelength_nm
    freq_capped = min(freq, 2.463e15)
    wave = 2.99792458e18 / freq_capped
    ww = wave ** 2
    sig = (5.799e-13 + 1.422e-6 / ww + 2.784 / (ww * ww)) / (ww * ww)
    return sig * xnfph1 * 2.0 * bhyd1 / rho


def main():
    """Compare SIGMAC components between Python and Fortran."""
    print("=" * 80)
    print("SIGMAC COMPONENT DIAGNOSIS")
    print("=" * 80)
    
    # Load Fortran-derived atmosphere
    fortran_npz_path = Path("synthe_py/data/at12_aaaaa_atmosphere_fixed_interleaved.npz")
    if not fortran_npz_path.exists():
        print(f"ERROR: {fortran_npz_path} not found")
        return False
    
    atm_fortran = load_cached(fortran_npz_path)
    print(f"\nLoaded Fortran atmosphere: {atm_fortran.layers} layers")
    
    # Load Python-computed atmosphere
    python_npz_path = Path("synthe_py/data/at12_aaaaa_t03750_fixed_ifop.npz")
    if python_npz_path.exists():
        atm_python = load_cached(python_npz_path)
        print(f"Loaded Python atmosphere: {atm_python.layers} layers")
    else:
        print(f"WARNING: Python atmosphere not found: {python_npz_path}")
        atm_python = None
    
    # Test wavelength (300 nm from the progress report)
    test_wavelength = 300.0  # nm
    freq = C_LIGHT_NM / test_wavelength
    
    print(f"\nTest wavelength: {test_wavelength:.2f} nm (frequency: {freq:.6e} Hz)")
    
    # Extract Fortran values at this wavelength
    if atm_fortran.continuum_scat_coeff is not None:
        # Find edge index for this wavelength
        wledge = np.abs(atm_fortran.continuum_wledge)
        edge_idx = np.searchsorted(wledge, test_wavelength, side="right") - 1
        edge_idx = np.clip(edge_idx, 0, len(wledge) - 2)
        
        # Interpolate scattering coefficient
        wl_left = wledge[edge_idx]
        wl_right = wledge[edge_idx + 1]
        half = atm_fortran.continuum_half_edge[edge_idx]
        delta = atm_fortran.continuum_delta_edge[edge_idx]
        
        c1 = (test_wavelength - half) * (test_wavelength - wl_right) / delta
        c2 = (wl_left - test_wavelength) * (test_wavelength - wl_right) * 2.0 / delta
        c3 = (test_wavelength - wl_left) * (test_wavelength - half) / delta
        
        scat_coeff = atm_fortran.continuum_scat_coeff
        log_scat = scat_coeff[:, edge_idx, 0] * c1 + scat_coeff[:, edge_idx, 1] * c2 + scat_coeff[:, edge_idx, 2] * c3
        sigmac_fortran = 10.0 ** log_scat
        
        print(f"\nFortran SIGMAC (interpolated from fort.10):")
        print(f"  Layer 0 (surface): {sigmac_fortran[0]:.8E} cm²/g")
        print(f"  Layer 79 (deep):   {sigmac_fortran[-1]:.8E} cm²/g")
    else:
        print("\nWARNING: Fortran atmosphere has no continuum_scat_coeff")
        sigmac_fortran = None
    
    # Compute SIGEL directly from atmosphere data
    rho = np.asarray(atm_fortran.mass_density, dtype=np.float64)
    xne = np.asarray(atm_fortran.electron_density, dtype=np.float64)
    
    sigel = compute_sigel(xne, rho)
    print(f"\nSIGEL (electron scattering):")
    print(f"  Layer 0 (surface): {sigel[0]:.8E} cm²/g")
    print(f"  Layer 79 (deep):   {sigel[-1]:.8E} cm²/g")
    
    # Compute SIGH using simple formula
    # For SIGH, we need ground-state hydrogen population and partition function
    if atm_fortran.xnf_h is not None:
        temp = np.asarray(atm_fortran.temperature, dtype=np.float64)
        xnf_h = np.asarray(atm_fortran.xnf_h, dtype=np.float64)
        
        # For cool stars (T < 10000K), most hydrogen is in ground state
        # Simple approximation: xnfph1 ≈ xnf_h (ground-state ≈ total neutral)
        # This isn't quite right but gives us a starting point
        
        # We need BHYD(J,1) which is the partition function ratio
        # From atlas7v.for, BHYD(J,1) includes the Boltzmann factor for the ground state
        # For now, assume bhyd1 ≈ 1.0 (exact value requires full partition function)
        bhyd1 = np.ones_like(xnf_h)
        
        sigh_simple = compute_sigh_fortran(test_wavelength, xnf_h, bhyd1, rho)
        print(f"\nSIGH (H Rayleigh, simple formula, assuming BHYD1=1):")
        print(f"  Layer 0 (surface): {sigh_simple[0]:.8E} cm²/g")
        print(f"  Layer 79 (deep):   {sigh_simple[-1]:.8E} cm²/g")
    else:
        print("\nWARNING: No neutral hydrogen data (xnf_h)")
        sigh_simple = np.zeros(atm_fortran.layers)
    
    # Compare with Fortran SIGMAC
    if sigmac_fortran is not None:
        sigh_implied = sigmac_fortran - sigel
        print(f"\nImplied SIGH from Fortran (SIGMAC - SIGEL):")
        print(f"  Layer 0 (surface): {sigh_implied[0]:.8E} cm²/g")
        print(f"  Layer 79 (deep):   {sigh_implied[-1]:.8E} cm²/g")
        
        # Check ratios
        print(f"\nSIGEL / SIGMAC ratio:")
        print(f"  Layer 0: {sigel[0] / sigmac_fortran[0]:.6f}")
        print(f"  Layer 79: {sigel[-1] / sigmac_fortran[-1]:.6f}")
        
        print(f"\nImplied SIGH / SIGMAC ratio:")
        print(f"  Layer 0: {sigh_implied[0] / sigmac_fortran[0]:.6f}")
        print(f"  Layer 79: {sigh_implied[-1] / sigmac_fortran[-1]:.6f}")
    
    # Now let's run the full Python kapp.py and compare
    print("\n" + "=" * 80)
    print("RUNNING PYTHON KAPP TO COMPARE")
    print("=" * 80)
    
    try:
        from synthe_py.physics.kapp import compute_kapp_full
        
        # Compute at test wavelength
        freq_array = np.array([freq])
        result = compute_kapp_full(atm_fortran, freq_array)
        
        if result is not None:
            acont_py, sigmac_py, scont_py = result
            
            print(f"\nPython kapp.py results at {test_wavelength:.2f} nm:")
            print(f"  ACONT[0]:  {acont_py[0, 0]:.8E} cm²/g")
            print(f"  SIGMAC[0]: {sigmac_py[0, 0]:.8E} cm²/g")
            
            if sigmac_fortran is not None:
                ratio = sigmac_py[0, 0] / sigmac_fortran[0]
                print(f"\n  Python SIGMAC / Fortran SIGMAC = {ratio:.6f}")
                print(f"  Discrepancy: {(ratio - 1) * 100:.2f}%")
    except Exception as e:
        print(f"\nERROR running kapp.py: {e}")
        import traceback
        traceback.print_exc()
    
    # Load Python atmosphere if available and compare
    if atm_python is not None and atm_python.continuum_scat_coeff is not None:
        print("\n" + "=" * 80)
        print("COMPARING WITH PYTHON-COMPUTED ATMOSPHERE NPZ")
        print("=" * 80)
        
        wledge_py = np.abs(atm_python.continuum_wledge)
        edge_idx = np.searchsorted(wledge_py, test_wavelength, side="right") - 1
        edge_idx = np.clip(edge_idx, 0, len(wledge_py) - 2)
        
        wl_left = wledge_py[edge_idx]
        wl_right = wledge_py[edge_idx + 1]
        half = atm_python.continuum_half_edge[edge_idx]
        delta = atm_python.continuum_delta_edge[edge_idx]
        
        c1 = (test_wavelength - half) * (test_wavelength - wl_right) / delta
        c2 = (wl_left - test_wavelength) * (test_wavelength - wl_right) * 2.0 / delta
        c3 = (test_wavelength - wl_left) * (test_wavelength - half) / delta
        
        scat_coeff = atm_python.continuum_scat_coeff
        log_scat = scat_coeff[:, edge_idx, 0] * c1 + scat_coeff[:, edge_idx, 1] * c2 + scat_coeff[:, edge_idx, 2] * c3
        sigmac_python_npz = 10.0 ** log_scat
        
        print(f"\nPython NPZ SIGMAC (from stored coefficients):")
        print(f"  Layer 0 (surface): {sigmac_python_npz[0]:.8E} cm²/g")
        print(f"  Layer 79 (deep):   {sigmac_python_npz[-1]:.8E} cm²/g")
        
        if sigmac_fortran is not None:
            ratio = sigmac_python_npz[0] / sigmac_fortran[0]
            print(f"\n  Python NPZ / Fortran SIGMAC = {ratio:.6f}")
            print(f"  Discrepancy: {(ratio - 1) * 100:.2f}%")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The 4.4% SIGMAC discrepancy could come from:
1. SIGEL calculation (unlikely - formula is identical)
2. SIGH calculation (likely - complex Gavrila tables)
3. Extra scattering terms (SIGHE, SIGH2) being included when they shouldn't

Next steps:
- Compare SIGEL values directly (should match exactly)
- Compare SIGH values (requires Gavrila table implementation check)
- Check IFOP settings to ensure SIGHE and SIGH2 are disabled
""")
    
    return True


if __name__ == "__main__":
    main()

