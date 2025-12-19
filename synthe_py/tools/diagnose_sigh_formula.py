#!/usr/bin/env python3
"""
Detailed diagnosis of SIGH (hydrogen Rayleigh scattering) discrepancy.

Fortran formula (atlas7v.for line 6932-6934):
    XSECT = 6.65D-25 * G**2
    SIGH(J) = XSECT * XNFPH(J,1) * 2. * BHYD(J,1) / RHO(J)

This compares each factor between Python and Fortran.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from synthe_py.io.atmosphere import load_cached
from synthe_py.physics.kapp import (
    HRAYOP_GAVRILAM,
    compute_hydrogen_partition_function,
    compute_ground_state_hydrogen,
)

# Constants
C_LIGHT_NM = 2.99792458e17  # nm/s
FREQ_LYMAN = 3.288051e15    # Lyman limit frequency
FREQ_STEP = 3.288051e13     # Step size for GAVRILAM


def compute_g_fortran(freq: float) -> float:
    """Compute G from Gavrila tables using exact Fortran algorithm."""
    if freq < FREQ_STEP:  # < 3.288051e13
        return HRAYOP_GAVRILAM[0] * (freq / FREQ_STEP) ** 2
    elif freq <= FREQ_LYMAN * 0.74:  # <= 0.74 * Lyman
        # Fortran: I=FREQ/3.288051D13, I=MIN(I+1,74)
        i = int(freq / FREQ_STEP)
        i = min(i + 1, 74)
        i = max(1, i)
        if i >= len(HRAYOP_GAVRILAM):
            i = len(HRAYOP_GAVRILAM) - 1
        if i > 1:
            # Fortran uses 1-based indexing: GAVRILAM(I-1) and GAVRILAM(I)
            # Python 0-based: HRAYOP_GAVRILAM[i-2] and HRAYOP_GAVRILAM[i-1]
            g = HRAYOP_GAVRILAM[i - 2] + (
                HRAYOP_GAVRILAM[i - 1] - HRAYOP_GAVRILAM[i - 2]
            ) / FREQ_STEP * (freq - (i - 1) * FREQ_STEP)
            return g
        else:
            return HRAYOP_GAVRILAM[0]
    else:
        return 15.57  # Above this range for 300nm


def main():
    """Detailed SIGH formula comparison."""
    print("=" * 80)
    print("DETAILED SIGH (HYDROGEN RAYLEIGH SCATTERING) DIAGNOSIS")
    print("=" * 80)
    
    # Load Fortran-derived atmosphere
    fortran_npz_path = Path("synthe_py/data/at12_aaaaa_atmosphere_fixed_interleaved.npz")
    if not fortran_npz_path.exists():
        print(f"ERROR: {fortran_npz_path} not found")
        return False
    
    atm = load_cached(fortran_npz_path)
    print(f"\nLoaded atmosphere: {atm.layers} layers")
    
    # Test wavelength
    test_wavelength = 300.0  # nm
    freq = C_LIGHT_NM / test_wavelength
    freq_ratio = freq / FREQ_LYMAN
    
    print(f"\nTest wavelength: {test_wavelength:.2f} nm")
    print(f"Frequency: {freq:.6e} Hz")
    print(f"Frequency / Lyman limit: {freq_ratio:.4f}")
    
    # Compute G from Gavrila tables
    g = compute_g_fortran(freq)
    xsect = 6.65e-25 * g**2
    
    print(f"\n--- Gavrila Table Lookup ---")
    print(f"G value: {g:.6f}")
    print(f"G^2: {g**2:.6f}")
    print(f"XSECT = 6.65e-25 * G^2 = {xsect:.8E} cm^2")
    
    # Get atmospheric data
    rho = np.asarray(atm.mass_density, dtype=np.float64)
    xnf_h = np.asarray(atm.xnf_h, dtype=np.float64)
    temp = np.asarray(atm.temperature, dtype=np.float64)
    
    # Compute ground-state hydrogen using Python's method
    xnfph1 = compute_ground_state_hydrogen(xnf_h, temp)
    partition_func = compute_hydrogen_partition_function(temp)
    
    print(f"\n--- Hydrogen Population (Layer 0) ---")
    print(f"XNF_H (total neutral H): {xnf_h[0]:.6E}")
    print(f"Temperature: {temp[0]:.1f} K")
    print(f"Partition function U(T): {partition_func[0]:.6f}")
    print(f"XNFPH1 (ground-state) = XNF_H / U(T): {xnfph1[0]:.6E}")
    
    # Get BHYD from atlas tables in atmosphere
    bhyd1 = np.ones(atm.layers, dtype=np.float64)
    if hasattr(atm, 'bhyd') and atm.bhyd is not None:
        bhyd = np.asarray(atm.bhyd, dtype=np.float64)
        if bhyd.shape[1] > 0:
            bhyd1 = bhyd[:, 0]
        print(f"\nBHYD table found in atmosphere!")
        print(f"BHYD(1,1) = {bhyd1[0]:.6E}")
    else:
        # Check if atlas_tables are embedded
        data = np.load(fortran_npz_path, allow_pickle=True)
        if 'bhyd' in data:
            bhyd = data['bhyd']
            if len(bhyd.shape) == 2 and bhyd.shape[1] > 0:
                bhyd1 = bhyd[:, 0]
            print(f"\nBHYD table found in NPZ file!")
            print(f"BHYD(1,1) = {bhyd1[0]:.6E}")
        else:
            print(f"\nWARNING: No BHYD table found, using 1.0")
    
    print(f"\n--- Full SIGH Computation (Layer 0) ---")
    print(f"Formula: SIGH = XSECT * XNFPH1 * 2 * BHYD1 / RHO")
    print(f"  XSECT = {xsect:.8E}")
    print(f"  XNFPH1 = {xnfph1[0]:.6E}")
    print(f"  BHYD1 = {bhyd1[0]:.6E}")
    print(f"  RHO = {rho[0]:.6E}")
    
    sigh_python = xsect * xnfph1[0] * 2.0 * bhyd1[0] / rho[0]
    print(f"\n  SIGH_Python = {sigh_python:.8E} cm^2/g")
    
    # Get Fortran SIGMAC for comparison
    if atm.continuum_scat_coeff is not None:
        wledge = np.abs(atm.continuum_wledge)
        edge_idx = np.searchsorted(wledge, test_wavelength, side="right") - 1
        edge_idx = np.clip(edge_idx, 0, len(wledge) - 2)
        
        wl_left = wledge[edge_idx]
        wl_right = wledge[edge_idx + 1]
        half = atm.continuum_half_edge[edge_idx]
        delta = atm.continuum_delta_edge[edge_idx]
        
        c1 = (test_wavelength - half) * (test_wavelength - wl_right) / delta
        c2 = (wl_left - test_wavelength) * (test_wavelength - wl_right) * 2.0 / delta
        c3 = (test_wavelength - wl_left) * (test_wavelength - half) / delta
        
        scat_coeff = atm.continuum_scat_coeff
        log_scat = scat_coeff[0, edge_idx, 0] * c1 + scat_coeff[0, edge_idx, 1] * c2 + scat_coeff[0, edge_idx, 2] * c3
        sigmac_fortran = 10.0 ** log_scat
        
        print(f"\n--- Comparison with Fortran ---")
        print(f"Fortran SIGMAC (layer 0): {sigmac_fortran:.8E} cm^2/g")
        
        # SIGEL is tiny, so SIGMAC ≈ SIGH for Fortran
        xne = np.asarray(atm.electron_density, dtype=np.float64)
        sigel = 0.6653e-24 * xne[0] / rho[0]
        sigh_fortran_implied = sigmac_fortran - sigel
        
        print(f"SIGEL (layer 0): {sigel:.8E} cm^2/g")
        print(f"Implied Fortran SIGH: {sigh_fortran_implied:.8E} cm^2/g")
        
        ratio = sigh_python / sigh_fortran_implied
        print(f"\n  SIGH_Python / SIGH_Fortran = {ratio:.6f}")
        print(f"  Discrepancy: {(ratio - 1) * 100:.2f}%")
        
        # Check which factor is off
        print(f"\n--- Factor Analysis ---")
        # If Fortran's XNFPH1 were exact, what would it be?
        # SIGH = XSECT * XNFPH1 * 2 * BHYD1 / RHO
        # XNFPH1_fortran = SIGH / (XSECT * 2 * BHYD1 / RHO)
        xnfph1_fortran_implied = sigh_fortran_implied / (xsect * 2.0 * bhyd1[0] / rho[0])
        
        print(f"Python XNFPH1: {xnfph1[0]:.6E}")
        print(f"Implied Fortran XNFPH1: {xnfph1_fortran_implied:.6E}")
        print(f"Ratio: {xnfph1[0] / xnfph1_fortran_implied:.6f}")
        
        # Check the partition function factor
        print(f"\n  If XNFPH1 = XNF_H / U(T), then:")
        print(f"  Python U(T) = {partition_func[0]:.6f}")
        implied_partition = xnf_h[0] / xnfph1_fortran_implied
        print(f"  Implied Fortran U(T) = {implied_partition:.6f}")
        print(f"  Ratio: {partition_func[0] / implied_partition:.6f}")
        
    # Now let's check what Fortran actually writes for BHYD
    print(f"\n--- Check BHYD Table in fort.10 ---")
    fort10_path = Path("synthe/stmp_at12_aaaaa/fort.10")
    if fort10_path.exists():
        # Try to read BHYD directly from fort.10 using convert_fort10's logic
        import struct
        def _read_record(handle):
            header = handle.read(4)
            if not header:
                raise EOFError
            (size,) = struct.unpack("<i", header)
            payload = handle.read(size)
            handle.read(4)  # trailer
            return payload
        
        with fort10_path.open("rb") as fh:
            records = []
            while True:
                try:
                    records.append(_read_record(fh))
                except EOFError:
                    break
        
        print(f"Read {len(records)} records from fort.10")
        
        # BHYD should be in the tail records
        # After record 4 + 4*80 layer records = 324 records for layer data
        # Then: BHYD(kw,8), BC1(kw,14), BC2(kw,6), BSI1(kw,11), BSI2(kw,10)
        rec_index = 4 + 80 * 4  # Skip header (4) and layer data (80*4)
        
        if len(records) > rec_index:
            bhyd_data = np.frombuffer(records[rec_index], dtype="<f8", count=80*8).reshape(80, 8)
            print(f"\nBHYD from fort.10:")
            print(f"  BHYD[0,0] (layer 0, column 1): {bhyd_data[0,0]:.6E}")
            print(f"  BHYD[0,1:8] (other columns): {bhyd_data[0,1:]}")
            print(f"  BHYD[79,0] (deep layer): {bhyd_data[79,0]:.6E}")
    else:
        print(f"fort.10 not found at {fort10_path}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The SIGH formula uses:
  SIGH = XSECT * XNFPH1 * 2 * BHYD1 / RHO

Where:
- XSECT = 6.65e-25 * G^2 (from Gavrila tables - should match exactly)
- XNFPH1 = ground-state hydrogen population (computed from XNF_H / U(T))
- BHYD1 = partition function factor from B-tables
- RHO = mass density (should match)

The most likely source of 4.4% discrepancy is in:
1. Partition function U(T) calculation 
2. BHYD1 values not matching

Check if Fortran uses a different partition function formula or if
BHYD already includes the partition function correction.
""")
    
    return True


if __name__ == "__main__":
    main()

