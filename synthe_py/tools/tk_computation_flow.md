# TK Computation Flow Analysis

## Key Discovery

**DO 1516 loop IS executed when reading DECK6 format!**

### Flow Analysis

1. **xnfpelsyn calls READIN(20)**
   - MODE=20 (read previously calculated model)

2. **READIN reads DECK6 format (label 1140)**
   - Reads RHOX, T, P, XNE, ABROSS, PRAD, VTURB
   - Converts RHOX from log10 if needed
   - GO TO 98 (line 1686)

3. **Line 98: MORE=0, LETCOL=1**
   - Then line 99: READ(5,1) CARD (reads next card)
   - Continues processing until 'END' or label 1500

4. **Label 1500 (line 1918): IF(MODE.NE.1)GO TO 1510**
   - Since MODE=20, goes directly to label 1510
   - Skips parameter validation (which is for MODE=1)

5. **Label 1510 (line 1934): Processes abundances**
   - Lines 1935-1939: Process H, He, and other element abundances

6. **DO 1516 loop (line 1953): Computes TK and derived quantities**
   ```fortran
   DO 1516 J=1,NRHOX
      TK(J)=1.38054D-16*T(J)        ! TK = k_B*T
      HKT(J)=6.6256D-27/TK(J)
      HCKT(J)=HKT(J)*2.99792458D10
      TKEV(J)=8.6171D-5*T(J)
      TLOG(J)= LOG(T(J))
      XNATOM(J)=P(J)/TK(J)-XNE(J)   ! Uses TK to compute XNATOM
      RHO(J)=XNATOM(J)*WTMOLE*1.660D-24
  1516 PTURB(J)=.5*RHO(J)*VTURB(J)**2
   ```

7. **READIN continues and eventually returns**
   - Line 1972: IF(MODE.NE.1)GO TO 1575
   - Line 1999: More processing...
   - Eventually returns to xnfpelsyn

8. **xnfpelsyn continues**
   - Line 215: ITEMP=ITEMP+1
   - Line 216: CALL POPS(...) → calls NELECT
   - NELECT uses TK (line 2945: XNE(1)=P(1)/TK(1)/2)

9. **xnfpelsyn writes fort.10 (line 294)**
   - WRITE(10)T,TKEV,TK,HKT,TLOG,HCKT,P,XNE,XNATOM,RHO,RHOX,VTURB,XNFH,XNFHE,XNFH2

## The Mystery

**TK SHOULD be k_B*T = 5.1e-13, but fort.10 has TK = 0.702!**

### Possible Explanations

1. **TK gets overwritten after DO 1516**
   - But we don't see any TK assignments after DO 1516 in READIN
   - And xnfpelsyn doesn't modify TK

2. **P from .atm file is in wrong units**
   - This would make XNATOM wrong (XNATOM = P/TK - XNE)
   - But this doesn't explain why TK is wrong

3. **TK is computed differently in a different code path**
   - Maybe there's another TK computation we haven't found?

4. **Binary format issue**
   - Maybe we're reading fort.10 incorrectly?
   - But T, XNE, XNATOM seem correct...

5. **TK in fort.10 represents something else**
   - Maybe it's not k_B*T but something computed from P and XNATOM?
   - TK = P / (XNATOM + XNE) = 2.32e4 / 33000 = 0.702 ✓ **MATCHES!**

## Hypothesis

**TK in fort.10 might be computed as `TK = P / (XNATOM + XNE)` instead of `TK = k_B*T`!**

This would explain:
- Why TK = 0.702 (matches P/(XNATOM+XNE))
- Why XNATOM formula works: XNATOM = P/TK - XNE
- Why TK doesn't match k_B*T

But this is circular! We need TK to compute XNATOM, but we need XNATOM to compute TK!

Unless... maybe TK is computed iteratively? Or maybe there's a different TK variable?

## Next Steps

1. Check if there's a different TK computation after DO 1516
2. Check if TK gets modified in POPS/NELECT
3. Verify the binary reading of fort.10 is correct
4. Check if there are multiple TK variables or COMMON blocks

