#!/usr/bin/env python3
"""
Debug script to understand when and how TK is computed in Fortran.

This script traces through the Fortran code to understand:
1. When READIN is called and what it does
2. When TK gets computed (the DO 1516 loop)
3. What values are in COMMON /TEMP/ when fort.10 is written
"""

from pathlib import Path
import re

def analyze_readin_flow():
    """Analyze the READIN subroutine flow for DECK6 format."""
    atlas7v_path = Path('src/atlas7v.for')
    content = atlas7v_path.read_text()
    
    print("=" * 80)
    print("ANALYZING READIN FLOW FOR DECK6 FORMAT")
    print("=" * 80)
    
    # Find DECK6 reading section (label 1140)
    deck6_match = re.search(r'1140\s+NRHOX=FREEFF\(CARD\)', content)
    if deck6_match:
        start = deck6_match.start()
        # Get next 50 lines
        lines = content[start:start+3000].split('\n')[:50]
        print("\nDECK6 reading section (label 1140):")
        for i, line in enumerate(lines, 1):
            print(f"{i:3d}: {line}")
            if 'GO TO 98' in line:
                print("  ^^^ Returns here (GO TO 98)")
                break
    
    # Find DO 1516 loop that computes TK
    tk_compute_match = re.search(r'DO 1516 J=1,NRHOX', content)
    if tk_compute_match:
        start = tk_compute_match.start()
        lines = content[start:start+2000].split('\n')[:15]
        print("\n\nDO 1516 loop that computes TK (line ~1953):")
        for i, line in enumerate(lines, 1):
            print(f"{i:3d}: {line}")
            if '1516' in line and 'CONTINUE' in line:
                break
    
    # Check if DO 1516 is called from DECK6 path
    print("\n\nKEY FINDING:")
    print("  - DECK6 format (label 1140) reads RHOX, T, P, XNE, etc.")
    print("  - Then GO TO 98 (returns)")
    print("  - DO 1516 loop (computes TK) is NOT in the DECK6 path!")
    print("  - So TK might not be computed when xnfpelsyn reads atmosphere!")

def analyze_xnfpelsyn_flow():
    """Analyze xnfpelsyn flow to see when TK is used."""
    xnfpelsyn_path = Path('src/xnfpelsyn.for')
    content = xnfpelsyn_path.read_text()
    
    print("\n" + "=" * 80)
    print("ANALYZING XNFPELSYN FLOW")
    print("=" * 80)
    
    # Find READIN call
    readin_match = re.search(r'CALL READIN\(20\)', content)
    if readin_match:
        start = readin_match.start()
        lines = content[start:start+2000].split('\n')[:30]
        print("\nAfter READIN(20) call:")
        for i, line in enumerate(lines, 1):
            print(f"{i:3d}: {line}")
            if 'CALL POPS' in line:
                print("  ^^^ Calls POPS here")
                break
    
    # Find WRITE(10) statement
    write_match = re.search(r'WRITE\(10\)T,TKEV,TK,HKT,TLOG,HCKT,P,XNE,XNATOM', content)
    if write_match:
        start = write_match.start()
        lines = content[start-50:start+10].split('\n')
        print("\n\nBefore WRITE(10) statement:")
        for i, line in enumerate(lines, 1):
            if 'WRITE(10)' in line:
                print(f"{i:3d}: {line}  <--- WRITES TK HERE")
            else:
                print(f"{i:3d}: {line}")

def check_tk_initialization():
    """Check if TK is initialized anywhere in xnfpelsyn."""
    xnfpelsyn_path = Path('src/xnfpelsyn.for')
    content = xnfpelsyn_path.read_text()
    
    print("\n" + "=" * 80)
    print("CHECKING TK INITIALIZATION IN XNFPELSYN")
    print("=" * 80)
    
    # Search for TK assignments
    tk_assignments = re.findall(r'TK\([^)]+\)\s*=[^;]*', content)
    if tk_assignments:
        print("\nFound TK assignments:")
        for assignment in tk_assignments[:5]:
            print(f"  {assignment}")
    else:
        print("\nNo TK assignments found in xnfpelsyn.for!")
        print("  This means TK comes from COMMON /TEMP/ block")
        print("  which is shared with atlas7v.for")

def analyze_common_temp():
    """Analyze COMMON /TEMP/ usage."""
    print("\n" + "=" * 80)
    print("ANALYZING COMMON /TEMP/ BLOCK")
    print("=" * 80)
    
    atlas7v_path = Path('src/atlas7v.for')
    content = atlas7v_path.read_text()
    
    # Find COMMON /TEMP/ declaration
    common_match = re.search(r'COMMON /TEMP/T\(kw\),TKEV\(kw\),TK\(kw\)', content)
    if common_match:
        print("\nCOMMON /TEMP/ contains: T, TKEV, TK, HKT, TLOG, HCKT, ITEMP")
        print("  - This is shared between atlas7v.for and xnfpelsyn.for")
        print("  - TK is computed in atlas7v.for DO 1516 loop")
        print("  - But this loop is NOT called when reading DECK6 format!")
    
    # Check when DO 1516 is called
    print("\n\nHYPOTHESIS:")
    print("  1. xnfpelsyn calls READIN(20) which reads DECK6 format")
    print("  2. DECK6 path reads RHOX, T, P, XNE but does NOT compute TK")
    print("  3. TK in COMMON /TEMP/ is uninitialized or from previous computation")
    print("  4. When xnfpelsyn writes fort.10, it writes whatever TK is in COMMON")
    print("  5. This explains why TK in fort.10 doesn't match k_B*T!")

if __name__ == '__main__':
    analyze_readin_flow()
    analyze_xnfpelsyn_flow()
    check_tk_initialization()
    analyze_common_temp()
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Check if TK is computed elsewhere before WRITE(10)")
    print("2. Check if TK is read from .atm file (unlikely)")
    print("3. Check if TK is computed in POPS or NELECT")
    print("4. Add Fortran debugging to print TK value before WRITE(10)")

