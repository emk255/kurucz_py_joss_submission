# TK Computation Investigation Summary

## Critical Discovery

**HKT from fort.10 gives TK = 9.9e-15**, which is close to k_B*T (5.1e-13)!

This suggests:
- **HKT might be correct** (computed from k_B*T)
- **TK in fort.10 is wrong** (not k_B*T, but 0.702)
- **TK and HKT might be swapped or computed differently**

## Evidence from fort.10 (Layer 0)

| Variable | fort.10 Value | Expected (from T) | Ratio | Status |
|----------|---------------|-------------------|-------|--------|
| T        | 3691.30       | 3691.30           | 1.0   | ✓ Correct |
| TK       | 0.702         | 5.1e-13 (k_B*T)   | 1.38e12 | ✗ Wrong |
| TKEV     | 0.0           | 0.318             | -     | ✗ Zero |
| TLOG     | 1.14e-14      | 8.21              | -     | ✗ Wrong (looks like HCKT?) |
| HKT      | 6.67e-13      | 1.30e-14          | 51.3  | ⚠ Different |
| HCKT     | 8.23          | -                 | -     | ? |
| P        | 0.0           | 2.32e4 (from formula) | - | ✗ Zero |
| XNE      | 1.61e-4       | -                 | -     | ✓ (ground truth) |
| XNATOM   | 33000         | -                 | -     | ✓ (ground truth) |

## Key Finding: HKT → TK

**HKT = H_PLANCK / TK** (from atlas7v.for line 1955)

If we compute TK from HKT:
```
TK_from_HKT = H_PLANCK / HKT = 6.6256e-27 / 6.67e-13 = 9.93e-15
```

This is **close to k_B*T = 5.1e-13**! (ratio ~51, but same order of magnitude)

## Hypothesis

1. **HKT is computed correctly** from k_B*T (or close to it)
2. **TK in fort.10 is NOT k_B*T** - it's something else (0.702)
3. **TK might be computed from P and XNATOM**: `TK = P / (XNATOM + XNE)`
   - This gives TK = 2.32e4 / 33000 = 0.702 ✓ **MATCHES fort.10!**

## Flow Analysis (UPDATED)

1. **xnfpelsyn calls READIN(20)** → reads DECK6 format
2. **DECK6 path (label 1140)** reads RHOX, T, P, XNE → GO TO 98
3. **Line 98**: `MORE=0`, then continues reading cards
4. **Eventually reaches label 1500** (line 1918): `IF(MODE.NE.1)GO TO 1510`
5. **Since MODE=20, goes to label 1510** (line 1934): Processes abundances
6. **DO 1516 loop (line 1953) IS EXECUTED**: Computes `TK = k_B*T = 1.38054D-16*T(J)` ✓
7. **READIN returns** (line 2002)
8. **xnfpelsyn calls POPS** → calls NELECT (line 2945: `XNE(1) = P(1)/TK(1)/2`)
9. **WRITE(10)** writes whatever TK is in COMMON /TEMP/

## Critical Question (UPDATED)

**TK IS computed in DO 1516 as k_B*T, but fort.10 has TK=0.702!**

This means:
- TK should be `k_B*T = 5.1e-13` after DO 1516
- But fort.10 has `TK = 0.702`
- Something must be overwriting TK after DO 1516!

Possibilities:
1. **TK gets overwritten after DO 1516** (but we don't see any TK assignments)
2. **Binary reading issue** (unlikely - other values are correct)
3. **TK is computed differently** - maybe `TK = P / (XNATOM + XNE)` is used instead?
4. **There's a different code path** that computes TK differently
5. **COMMON block issue** - maybe TK in COMMON /TEMP/ is different from what's written?

## Next Steps

1. **Check if TK is computed in POPS/NELECT** before being used
2. **Verify if TK = P / (XNATOM + XNE)** is the actual computation
3. **Check if HKT is the correct value** and TK is derived from it
4. **Add Fortran debugging** to print TK value at each step

## Conclusion

**TK in fort.10 (0.702) is NOT k_B*T**, but it **works with the XNATOM formula**:
- `P = (XNATOM + XNE) * TK` gives P = 2.32e4 ✓
- This suggests TK might be computed as `TK = P / (XNATOM + XNE)`

But this is circular since `XNATOM = P/TK - XNE`!

The mystery remains: **How is TK computed when reading DECK6 format?**

