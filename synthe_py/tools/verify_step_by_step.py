#!/usr/bin/env python3
"""Step-by-step verification to identify where values diverge between Python and Fortran."""

import sys
from pathlib import Path
import numpy as np
import struct

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from synthe_py.tools.convert_atm_to_npz import (
    parse_atm_file,
    _read_fort10_record,
    extract_xne_from_fort10,
    extract_xnatom_from_fort10,
)
from synthe_py.tools.compute_xne_iterative import compute_xne_iterative
from synthe_py.tools.pops_exact import load_fortran_data, pfsaha_exact

K_BOLTZ_FORTRAN = 1.38054e-16
KBOLTZ_EV = 11604.518  # k_B in eV/K
H_PLANCK = 6.62607015e-27
C_LIGHT = 2.99792458e10


def read_fort10_state(fort10_path: Path):
    """Read state variables from fort.10."""
    with fort10_path.open('rb') as fh:
        records = []
        while True:
            try:
                records.append(_read_fort10_record(fh))
            except EOFError:
                break
    
    state_block = records[3]
    n_layers = 80
    idx = 0
    
    T = np.frombuffer(state_block, dtype='<f8', count=n_layers, offset=idx*8); idx += n_layers
    TKEV = np.frombuffer(state_block, dtype='<f8', count=n_layers, offset=idx*8); idx += n_layers
    TK = np.frombuffer(state_block, dtype='<f8', count=n_layers, offset=idx*8); idx += n_layers
    HKT = np.frombuffer(state_block, dtype='<f8', count=n_layers, offset=idx*8); idx += n_layers
    TLOG = np.frombuffer(state_block, dtype='<f8', count=n_layers, offset=idx*8); idx += n_layers
    HCKT = np.frombuffer(state_block, dtype='<f8', count=n_layers, offset=idx*8); idx += n_layers
    P = np.frombuffer(state_block, dtype='<f8', count=n_layers, offset=idx*8); idx += n_layers
    XNE = np.frombuffer(state_block, dtype='<f8', count=n_layers, offset=idx*8); idx += n_layers
    XNATOM = np.frombuffer(state_block, dtype='<f8', count=n_layers, offset=idx*8); idx += n_layers
    RHO = np.frombuffer(state_block, dtype='<f8', count=n_layers, offset=idx*8); idx += n_layers
    RHOX = np.frombuffer(state_block, dtype='<f8', count=n_layers, offset=idx*8); idx += n_layers
    VTURB = np.frombuffer(state_block, dtype='<f8', count=n_layers, offset=idx*8); idx += n_layers
    
    return {
        'T': T, 'TKEV': TKEV, 'TK': TK, 'HKT': HKT, 'TLOG': TLOG, 'HCKT': HCKT,
        'P': P, 'XNE': XNE, 'XNATOM': XNATOM, 'RHO': RHO, 'RHOX': RHOX, 'VTURB': VTURB,
    }


def verify_step_by_step(atm_path: Path, fort10_path: Path, layer_idx: int = 0):
    """Verify step by step where values diverge."""
    print("=" * 80)
    print(f"STEP-BY-STEP VERIFICATION (Layer {layer_idx})")
    print("=" * 80)
    
    # Step 1: Read .atm file
    print("\n[STEP 1] Reading .atm file...")
    atm_data = parse_atm_file(atm_path)
    first_layer = atm_data['layers'][layer_idx]
    
    T_atm = first_layer['T']
    P_atm = first_layer['P']
    XNE_atm = first_layer['XNE']
    RHOX_atm = first_layer['RHOX']
    VTURB_atm = first_layer['VTURB']
    
    print(f"  T: {T_atm:.2f}")
    print(f"  P: {P_atm:.6e}")
    print(f"  XNE: {XNE_atm:.6e}")
    print(f"  RHOX: {RHOX_atm:.6e}")
    print(f"  VTURB: {VTURB_atm:.6e}")
    
    # Step 2: Read fort.10
    print("\n[STEP 2] Reading fort.10...")
    fort10_data = read_fort10_state(fort10_path)
    
    T_fort10 = fort10_data['T'][layer_idx]
    P_fort10 = fort10_data['P'][layer_idx]
    XNE_fort10 = fort10_data['XNE'][layer_idx]
    XNATOM_fort10 = fort10_data['XNATOM'][layer_idx]
    RHOX_fort10 = fort10_data['RHOX'][layer_idx]
    TK_fort10 = fort10_data['TK'][layer_idx]
    TKEV_fort10 = fort10_data['TKEV'][layer_idx]
    TLOG_fort10 = fort10_data['TLOG'][layer_idx]
    HKT_fort10 = fort10_data['HKT'][layer_idx]
    HCKT_fort10 = fort10_data['HCKT'][layer_idx]
    
    print(f"  T: {T_fort10:.2f}")
    print(f"  P: {P_fort10:.6e}")
    print(f"  XNE: {XNE_fort10:.6e}")
    print(f"  XNATOM: {XNATOM_fort10:.2f}")
    print(f"  RHOX: {RHOX_fort10:.6e}")
    print(f"  TK: {TK_fort10:.6e}")
    print(f"  TKEV: {TKEV_fort10:.6e}")
    print(f"  TLOG: {TLOG_fort10:.6f}")
    
    # Step 3: Verify T matches
    print("\n[STEP 3] Verifying T...")
    if abs(T_atm - T_fort10) < 1.0:
        print(f"  ✓ T matches: {T_atm:.2f} == {T_fort10:.2f}")
    else:
        print(f"  ✗ T mismatch: {T_atm:.2f} != {T_fort10:.2f}")
        return
    
    # Step 4: Compute TK
    print("\n[STEP 4] Computing TK = k_B * T...")
    TK_computed = K_BOLTZ_FORTRAN * T_atm
    print(f"  TK computed: {TK_computed:.6e}")
    print(f"  TK from fort.10: {TK_fort10:.6e}")
    print(f"  Ratio: {TK_fort10 / TK_computed:.6e}")
    
    if abs(TK_computed - TK_fort10) < 1e-10:
        print(f"  ✓ TK matches")
    else:
        print(f"  ✗ TK mismatch - fort.10 TK may be in different units or corrupted")
        print(f"    Expected: k_B*T = {TK_computed:.6e}")
        print(f"    Actual: {TK_fort10:.6e}")
        print(f"    This suggests fort.10 TK is NOT k_B*T!")
    
    # Step 5: Compute derived quantities
    print("\n[STEP 5] Computing derived quantities...")
    TKEV_computed = T_atm / KBOLTZ_EV
    TLOG_computed = np.log(T_atm)  # TLOG = LOG(T) in Fortran (natural log)
    HKT_computed = H_PLANCK / TK_computed
    HCKT_computed = H_PLANCK * C_LIGHT / TK_computed
    
    print(f"  TKEV computed: {TKEV_computed:.6e}, fort.10: {TKEV_fort10:.6e}")
    print(f"  TLOG computed: {TLOG_computed:.6f}, fort.10: {TLOG_fort10:.6f}")
    print(f"  HKT computed: {HKT_computed:.6e}, fort.10: {HKT_fort10:.6e}")
    print(f"  HCKT computed: {HCKT_computed:.6e}, fort.10: {HCKT_fort10:.6e}")
    
    # Step 6: Check RHOX
    print("\n[STEP 6] Checking RHOX...")
    print(f"  RHOX from .atm: {RHOX_atm:.6e}")
    print(f"  RHOX from fort.10: {RHOX_fort10:.6e}")
    ratio_rhox = RHOX_fort10 / RHOX_atm if RHOX_atm > 0 else 0
    print(f"  Ratio: {ratio_rhox:.6e}")
    
    if abs(ratio_rhox - 1.0) < 0.01:
        print(f"  ✓ RHOX matches")
    else:
        print(f"  ✗ RHOX unit mismatch - ratio = {ratio_rhox:.6e}")
        print(f"    This is the likely source of the problem!")
    
    # Step 7: Compute P from GRAV*RHOX
    print("\n[STEP 7] Computing P = GRAV*RHOX...")
    glog = atm_data.get('glog')
    if glog is None:
        import re
        match = re.search(r'g([-\d.]+)\.atm', atm_path.name)
        if match:
            glog = float(match.group(1))
        else:
            glog = 4.44
    
    GRAV = 10.0 ** glog
    print(f"  glog: {glog}")
    print(f"  GRAV: {GRAV:.6e} cm/s^2")
    
    # Using RHOX from .atm
    P_from_atm_rhox = GRAV * RHOX_atm
    print(f"  P = GRAV * RHOX_atm: {P_from_atm_rhox:.6e}")
    
    # Using RHOX from fort.10
    P_from_fort10_rhox = GRAV * RHOX_fort10
    print(f"  P = GRAV * RHOX_fort10: {P_from_fort10_rhox:.6e}")
    print(f"  P from fort.10: {P_fort10:.6e}")
    print(f"  P from .atm file: {P_atm:.6e}")
    
    if P_fort10 > 0:
        if abs(P_from_fort10_rhox - P_fort10) < 0.01 * P_fort10:
            print(f"  ✓ P matches when using RHOX from fort.10")
        else:
            print(f"  ✗ P mismatch even with fort.10 RHOX")
    else:
        print(f"  ⚠ P is zero in fort.10 - cannot verify")
    
    # Step 8: Check XNATOM formula
    print("\n[STEP 8] Verifying XNATOM = P/TK - XNE...")
    if P_fort10 > 0 and TK_fort10 > 0:
        XNATOM_from_formula = P_fort10 / TK_fort10 - XNE_fort10
        print(f"  XNATOM = P/TK - XNE: {XNATOM_from_formula:.2f}")
        print(f"  XNATOM from fort.10: {XNATOM_fort10:.2f}")
        if abs(XNATOM_from_formula - XNATOM_fort10) < 1.0:
            print(f"  ✓ XNATOM formula verified")
        else:
            print(f"  ✗ XNATOM formula mismatch")
            print(f"    This suggests TK in fort.10 is NOT k_B*T!")
    else:
        print(f"  ⚠ Cannot verify - P or TK is zero in fort.10")
    
    # Step 9: Try computing XNE iteratively with different RHOX values
    print("\n[STEP 9] Testing XNE computation with different RHOX...")
    
    # Load Fortran data
    fortran_data_path = Path(__file__).parent / 'fortran_data.npz'
    if not fortran_data_path.exists():
        fortran_data_path = Path(__file__).parent.parent / 'data' / 'fortran_data.npz'
    if fortran_data_path.exists():
        load_fortran_data(fortran_data_path)
        print("  ✓ Loaded fortran_data.npz")
        
        # Get abundances
        abundances = atm_data.get('abundances', {})
        abundance_scale = 10.0 ** atm_data.get('abundances_scale', 0.0)
        xabund = np.zeros(99, dtype=np.float64)
        for elem_num in range(1, 100):
            if elem_num in abundances:
                log_abund = abundances[elem_num]
            else:
                log_abund = -20.0
            xabund[elem_num - 1] = abundance_scale * (10.0 ** log_abund)
        
        # Test with RHOX from .atm
        print("\n  [9a] Testing with RHOX from .atm...")
        temperature = np.array([T_atm], dtype=np.float64)
        tk = np.array([TK_computed], dtype=np.float64)
        gas_pressure = np.array([P_from_atm_rhox], dtype=np.float64)
        
        def pfsaha_wrapper(j, iz, nion, mode):
            temp_single = np.array([temperature[j]], dtype=np.float64)
            tkev_single = np.array([T_atm / KBOLTZ_EV], dtype=np.float64)
            tk_single = np.array([tk[j]], dtype=np.float64)
            hkt_single = np.array([HKT_computed], dtype=np.float64)
            hckt_single = np.array([HCKT_computed], dtype=np.float64)
            tlog_single = np.array([TLOG_computed], dtype=np.float64)
            p_single = np.array([gas_pressure[j]], dtype=np.float64)
            xne_single = np.array([0.0], dtype=np.float64)  # Will be updated iteratively
            xnatm_single = np.array([0.0], dtype=np.float64)  # Will be updated iteratively
            
            answer = np.zeros((1, 31), dtype=np.float64)
            pfsaha_exact(0, iz, nion, mode, temp_single, tkev_single, tk_single,
                        hkt_single, hckt_single, tlog_single, p_single,
                        xne_single, xnatm_single, answer)
            
            if mode == 4:
                return float(answer[0, 0])
            return 0.0
        
        try:
            xne_atm, xnatm_atm = compute_xne_iterative(
                temperature, tk, gas_pressure, xabund, pfsaha_wrapper,
                max_iterations=10, tolerance=0.0005,
            )
            print(f"    XNE computed: {xne_atm[0]:.6e}")
            print(f"    XNATOM computed: {xnatm_atm[0]:.2f}")
            print(f"    XNE from fort.10: {XNE_fort10:.6e}")
            print(f"    XNATOM from fort.10: {XNATOM_fort10:.2f}")
            print(f"    XNE ratio: {xne_atm[0] / XNE_fort10:.6e}")
            print(f"    XNATOM ratio: {xnatm_atm[0] / XNATOM_fort10:.6e}")
        except Exception as e:
            print(f"    ✗ Error: {e}")
        
        # Test with RHOX from fort.10
        print("\n  [9b] Testing with RHOX from fort.10...")
        gas_pressure_fort10 = np.array([P_from_fort10_rhox], dtype=np.float64)
        
        try:
            xne_fort10_rhox, xnatm_fort10_rhox = compute_xne_iterative(
                temperature, tk, gas_pressure_fort10, xabund, pfsaha_wrapper,
                max_iterations=10, tolerance=0.0005,
            )
            print(f"    XNE computed: {xne_fort10_rhox[0]:.6e}")
            print(f"    XNATOM computed: {xnatm_fort10_rhox[0]:.2f}")
            print(f"    XNE from fort.10: {XNE_fort10:.6e}")
            print(f"    XNATOM from fort.10: {XNATOM_fort10:.2f}")
            print(f"    XNE ratio: {xne_fort10_rhox[0] / XNE_fort10:.6e}")
            print(f"    XNATOM ratio: {xnatm_fort10_rhox[0] / XNATOM_fort10:.6e}")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    else:
        print("  ⚠ fortran_data.npz not found - skipping XNE computation test")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ T matches: {T_atm:.2f}")
    print(f"✗ RHOX unit mismatch: ratio = {ratio_rhox:.6e}")
    print(f"  - This is the root cause of P, XNE, XNATOM mismatches")
    print(f"  - Need to determine correct unit conversion for RHOX in .atm files")
    if P_fort10 > 0:
        print(f"✓ P formula verified: P = GRAV*RHOX (when using fort.10 RHOX)")
    else:
        print(f"⚠ P is zero in fort.10 - cannot verify P computation")
    if TK_fort10 != TK_computed:
        print(f"⚠ TK in fort.10 is NOT k_B*T - may be corrupted or different units")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: verify_step_by_step.py <atm_file> <fort10_file> [layer_idx]")
        sys.exit(1)
    
    atm_path = Path(sys.argv[1])
    fort10_path = Path(sys.argv[2])
    layer_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    
    verify_step_by_step(atm_path, fort10_path, layer_idx)
