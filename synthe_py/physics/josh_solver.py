from __future__ import annotations

import os
import numpy as np

try:
    from numba import jit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Dummy decorator if Numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


from .josh_tables import (
    CH_WEIGHTS,
    CK_WEIGHTS,
    XTAU_GRID,
    COEFJ_MATRIX,
    COEFH_MATRIX,
    NXTAU,
)

EPS = 1.0e-38
# Match Fortran REAL*4 behavior during the XS iteration.
USE_FLOAT32_ITERATION = True
# Convergence tolerance: Match Fortran exactly (atlas7v.for line 9950: IF(ERRORX.GT..00001))
# Using the same tolerance ensures the iteration converges to the same solution
ITER_TOL = 1.0e-5  # Match Fortran's 1e-5 exactly
MAX_ITER = NXTAU
COEFJ_DIAG = np.diag(COEFJ_MATRIX)


@jit(nopython=True, cache=True)
def _josh_iteration_kernel(
    coefj_matrix: np.ndarray,
    xs: np.ndarray,
    xalpha: np.ndarray,
    xsbar_modified: np.ndarray,
    coefj_diag: np.ndarray,
    iter_tol: float,
    max_iter: int,
    eps: float,
) -> tuple[np.ndarray, int]:
    """Numba-compiled kernel for JOSH iteration loop.

    This kernel performs the scattering iteration loop, which is the main
    computational bottleneck in solve_josh_flux. Uses optimized matrix-vector
    operations for significant speedup.

    Parameters
    ----------
    coefj_matrix : np.ndarray
        COEFJ matrix (NXTAU × NXTAU)
    xs : np.ndarray
        Current XS values (modified in-place, NXTAU)
    xalpha : np.ndarray
        XALPHA values (NXTAU)
    xsbar_modified : np.ndarray
        Modified XSBAR values (NXTAU)
    coefj_diag : np.ndarray
        Diagonal of COEFJ matrix (NXTAU)
    iter_tol : float
        Iteration tolerance
    max_iter : int
        Maximum number of iterations
    eps : float
        Minimum value for XS

    Returns
    -------
    xs : np.ndarray
        Converged XS values
    num_iterations : int
        Number of iterations performed
    """
    nxtau = len(xs)
    diag = 1.0 - xalpha * coefj_diag

    for iteration in range(max_iter):
        iferr = 0
        # Fortran iterates BACKWARDS: K=NXTAU+1, then K=K-1 for KK=1..NXTAU
        # So K goes from NXTAU down to 1 (atlas7v.for lines 9856-9858)
        for k in range(nxtau - 1, -1, -1):
            # Use optimized dot product for matrix-vector multiplication
            # This is much faster than manual loop (uses BLAS)
            delxs = 0.0
            for m in range(nxtau):
                delxs += coefj_matrix[k, m] * xs[m]

            # Compute DELXS
            delxs = (delxs * xalpha[k] + xsbar_modified[k] - xs[k]) / diag[k]

            errorx = abs(delxs / xs[k]) if xs[k] != 0.0 else float("inf")
            if errorx > iter_tol:
                iferr = 1
            xs[k] = max(xs[k] + delxs, eps)

        if iferr == 0:
            return xs, iteration + 1

    return xs, max_iter


@jit(nopython=True, cache=True)
def _parcoe(f: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parabolic coefficients matching the Fortran PARCOE routine."""

    n = f.size
    a = np.zeros(n, dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    c = np.zeros(n, dtype=np.float64)

    if n == 0:
        return a, b, c
    if n == 1:
        a[0] = f[0]
        return a, b, c

    c[0] = 0.0
    denom = x[1] - x[0]
    b[0] = (f[1] - f[0]) / denom if denom != 0.0 else 0.0
    a[0] = f[0] - x[0] * b[0]

    n1 = n - 1
    c[-1] = 0.0
    denom = x[-1] - x[n1 - 1]
    b[-1] = (f[-1] - f[n1 - 1]) / denom if denom != 0.0 else 0.0
    a[-1] = f[-1] - x[-1] * b[-1]

    if n == 2:
        return a, b, c

    for j in range(1, n1):
        j1 = j - 1
        denom = x[j] - x[j1]
        d = (f[j] - f[j1]) / denom if denom != 0.0 else 0.0
        denom1 = (x[j + 1] - x[j]) * (x[j + 1] - x[j1])
        term1 = f[j + 1] / denom1 if denom1 != 0.0 else 0.0
        denom2 = x[j + 1] - x[j1]
        denom3 = x[j + 1] - x[j]
        denom4 = x[j] - x[j1]
        part = 0.0
        if denom4 != 0.0:
            t1 = f[j1] / denom2 if denom2 != 0.0 else 0.0
            t2 = f[j] / denom3 if denom3 != 0.0 else 0.0
            part = (t1 - t2) / denom4
        c[j] = term1 + part
        b[j] = d - (x[j] + x[j1]) * c[j]
        a[j] = f[j1] - x[j1] * d + x[j] * x[j1] * c[j]

    # Boundary adjustments matching the Fortran logic
    c[1] = 0.0
    denom = x[2] - x[1]
    b[1] = (f[2] - f[1]) / denom if denom != 0.0 else 0.0
    a[1] = f[1] - x[1] * b[1]

    if n > 3:
        c[2] = 0.0
        denom = x[3] - x[2]
        b[2] = (f[3] - f[2]) / denom if denom != 0.0 else 0.0
        a[2] = f[2] - x[2] * b[2]

    for j in range(1, n1):
        if c[j] == 0.0:
            continue
        j1 = min(j + 1, n - 1)
        denom = abs(c[j1]) + abs(c[j])
        wt = abs(c[j1]) / denom if denom > 0.0 else 0.0
        a[j] = a[j1] + wt * (a[j] - a[j1])
        b[j] = b[j1] + wt * (b[j] - b[j1])
        c[j] = c[j1] + wt * (c[j] - c[j1])

    a[n1 - 1] = a[-1]
    b[n1 - 1] = b[-1]
    c[n1 - 1] = c[-1]
    return a, b, c


@jit(nopython=True, cache=True)
def _integ(x: np.ndarray, f: np.ndarray, start: float) -> np.ndarray:
    """Numerical integral matching the Fortran INTEG routine."""

    n = f.size
    fint = np.zeros(n, dtype=np.float64)
    if n == 0:
        return fint

    a, b, c = _parcoe(f, x)
    fint[0] = start
    if n == 1:
        return fint

    for i in range(n - 1):
        dx = x[i + 1] - x[i]
        term = a[i] + 0.5 * b[i] * (x[i + 1] + x[i])
        term += (c[i] / 3.0) * ((x[i + 1] + x[i]) * x[i + 1] + x[i] * x[i])
        fint[i + 1] = fint[i] + term * dx
    return fint


@jit(nopython=True, cache=True)
def _deriv(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Derivative helper mirroring the Fortran DERIV routine."""

    n = f.size
    dfdx = np.zeros(n, dtype=np.float64)
    if n < 2:
        return dfdx

    dfdx[0] = (f[1] - f[0]) / (x[1] - x[0])
    dfdx[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
    if n == 2:
        return dfdx

    s = abs(x[1] - x[0]) / (x[1] - x[0]) if x[1] != x[0] else 1.0
    for j in range(1, n - 1):
        scale = max(abs(f[j - 1]), abs(f[j]), abs(f[j + 1]))
        scale = scale / abs(x[j]) if x[j] != 0.0 else scale
        if scale == 0.0:
            scale = 1.0
        d1 = (f[j + 1] - f[j]) / (x[j + 1] - x[j]) / scale
        d0 = (f[j] - f[j - 1]) / (x[j] - x[j - 1]) / scale
        tan1 = d1 / (s * np.sqrt(1.0 + d1 * d1) + 1.0)
        tan0 = d0 / (s * np.sqrt(1.0 + d0 * d0) + 1.0)
        dfdx[j] = (tan1 + tan0) / (1.0 - tan1 * tan0) * scale
    return dfdx


@jit(nopython=True)
def _map1_kernel(
    xold: np.ndarray, fold: np.ndarray, xnew: np.ndarray
) -> tuple[np.ndarray, int]:
    """Exact translation of Fortran MAP1 (atlas7v.for lines 1142-1199)."""
    nold = xold.size
    nnew = xnew.size
    fnew = np.zeros(nnew, dtype=np.float64)
    if nold == 0 or nnew == 0:
        return fnew, 0

    # Use 1-based indexing to mirror Fortran exactly.
    xold1 = np.empty(nold + 1, dtype=np.float64)
    fold1 = np.empty(nold + 1, dtype=np.float64)
    xold1[1:] = xold
    fold1[1:] = fold

    l = 2
    ll = 0
    cfor = bfor = afor = 0.0
    cbac = bbac = abac = 0.0
    a = b = c = 0.0

    for k in range(1, nnew + 1):
        xk = xnew[k - 1]
        while True:
            # Label 10
            if xk < xold1[l]:
                # Label 20
                if l == ll:
                    break
                if l == 2 or l == 3:
                    # Label 30
                    l = min(nold, l)
                    c = 0.0
                    b = (fold1[l] - fold1[l - 1]) / (xold1[l] - xold1[l - 1])
                    a = fold1[l] - xold1[l] * b
                    ll = l
                    break
                l1 = l - 1
                if l > ll + 1 or l == 3:
                    # Label 21
                    l2 = l - 2
                    d = (fold1[l1] - fold1[l2]) / (xold1[l1] - xold1[l2])
                    cbac = fold1[l] / (
                        (xold1[l] - xold1[l1]) * (xold1[l] - xold1[l2])
                    ) + (
                        fold1[l2] / (xold1[l] - xold1[l2])
                        - fold1[l1] / (xold1[l] - xold1[l1])
                    ) / (
                        xold1[l1] - xold1[l2]
                    )
                    bbac = d - (xold1[l1] + xold1[l2]) * cbac
                    abac = fold1[l2] - xold1[l2] * d + xold1[l1] * xold1[l2] * cbac
                    if l < nold:
                        # Fall through to label 25 below.
                        pass
                    else:
                        # Label 22
                        c = cbac
                        b = bbac
                        a = abac
                        ll = l
                        break
                elif l > ll + 1 or l == 4:
                    # Label 21 (same as above)
                    l2 = l - 2
                    d = (fold1[l1] - fold1[l2]) / (xold1[l1] - xold1[l2])
                    cbac = fold1[l] / (
                        (xold1[l] - xold1[l1]) * (xold1[l] - xold1[l2])
                    ) + (
                        fold1[l2] / (xold1[l] - xold1[l2])
                        - fold1[l1] / (xold1[l] - xold1[l1])
                    ) / (
                        xold1[l1] - xold1[l2]
                    )
                    bbac = d - (xold1[l1] + xold1[l2]) * cbac
                    abac = fold1[l2] - xold1[l2] * d + xold1[l1] * xold1[l2] * cbac
                    if l < nold:
                        pass
                    else:
                        # Label 22
                        c = cbac
                        b = bbac
                        a = abac
                        ll = l
                        break
                else:
                    cbac = cfor
                    bbac = bfor
                    abac = afor
                    if l == nold:
                        # Label 22
                        c = cbac
                        b = bbac
                        a = abac
                        ll = l
                        break

                # Label 25
                d = (fold1[l] - fold1[l1]) / (xold1[l] - xold1[l1])
                cfor = fold1[l + 1] / (
                    (xold1[l + 1] - xold1[l]) * (xold1[l + 1] - xold1[l1])
                ) + (
                    fold1[l1] / (xold1[l + 1] - xold1[l1])
                    - fold1[l] / (xold1[l + 1] - xold1[l])
                ) / (
                    xold1[l] - xold1[l1]
                )
                bfor = d - (xold1[l] + xold1[l1]) * cfor
                afor = fold1[l1] - xold1[l1] * d + xold1[l] * xold1[l1] * cfor
                wt = 0.0
                if abs(cfor) != 0.0:
                    wt = abs(cfor) / (abs(cfor) + abs(cbac))
                a = afor + wt * (abac - afor)
                b = bfor + wt * (bbac - bfor)
                c = cfor + wt * (cbac - cfor)
                ll = l
                break

            # Continue label 10 loop.
            l += 1
            if l > nold:
                # Label 30
                l = min(nold, l)
                c = 0.0
                b = (fold1[l] - fold1[l - 1]) / (xold1[l] - xold1[l - 1])
                a = fold1[l] - xold1[l] * b
                ll = l
                break

        fnew[k - 1] = a + (b + c * xk) * xk

    return fnew, ll - 1


def _map1(
    xold: np.ndarray, fold: np.ndarray, xnew: np.ndarray, debug: bool = False
) -> tuple[np.ndarray, int]:
    """Faithful port of the Fortran MAP1 interpolation routine.

    Wrapper around Numba-compiled kernel for performance.
    """
    if debug:
        print(f"\n{'='*70}")
        print("MAP1 DEBUG")
        print(f"{'='*70}")
        print(f"Input: XOLD size={xold.size}, XNEW size={xnew.size}")
        print(f"  XOLD[0]={xold[0]:.8E}, XOLD[-1]={xold[-1]:.8E}")
        print(f"  XNEW[0]={xnew[0]:.8E}, XNEW[-1]={xnew[-1]:.8E}")
        print(f"  FOLD[0]={fold[0]:.8E}, FOLD[-1]={fold[-1]:.8E}")

    # Always call the kernel - if Numba is available, it's JIT-compiled; otherwise it's pure Python
    fnew, maxj = _map1_kernel(xold, fold, xnew)

    if debug:
        print(f"\nMAP1 Result:")
        print(f"  FNEW[0]={fnew[0]:.8E}, FNEW[-1]={fnew[-1]:.8E}")
        print(f"  MAXJ={maxj}")
        print("=" * 70)

    return fnew, maxj


def solve_josh_flux(
    acont: np.ndarray,
    scont: np.ndarray,
    aline: np.ndarray,
    sline: np.ndarray,
    sigmac: np.ndarray,
    sigmal: np.ndarray,
    column_mass: np.ndarray,
    debug: bool = False,
    debug_label: str = "",
    temperature: np.ndarray | None = None,  # Optional: for debug output
) -> float:
    """Compute the emergent flux for a single frequency using the JOSH solver.

    Option 2: Use higher precision arithmetic when alpha (scattering) is large
    to reduce numerical errors. This is especially important when alpha > 0.1.
    """
    # CRITICAL DEBUG: Print at function entry to verify it's being called
    if debug:
        if not hasattr(solve_josh_flux, "_entry_count"):
            solve_josh_flux._entry_count = 0
        solve_josh_flux._entry_count += 1
        if solve_josh_flux._entry_count <= 5:  # Print first 5 calls
            print(
                f"\nDEBUG ENTRY #{solve_josh_flux._entry_count}: solve_josh_flux called, label={debug_label}, acont.size={acont.size}",
                flush=True,
            )

    # CRITICAL FIX: Use float64 (REAL*8) to match Fortran exactly
    # Fortran uses REAL*8 (double precision) for all opacity and flux calculations
    # No higher precision needed - match Fortran's REAL*8 exactly
    dtype = np.float64  # Match Fortran REAL*8

    acont = np.asarray(acont, dtype=dtype)
    scont = np.asarray(scont, dtype=dtype)
    aline = np.asarray(aline, dtype=dtype)
    sline = np.asarray(sline, dtype=dtype)
    sigmac = np.asarray(sigmac, dtype=dtype)
    sigmal = np.asarray(sigmal, dtype=dtype)
    rho = np.asarray(column_mass, dtype=dtype)

    # CRITICAL DEBUG: Check aline immediately after conversion
    if debug and "FLUX_TOTAL" in debug_label:
        print(f"\nDEBUG: After dtype conversion in solve_josh_flux:")
        print(f"  aline.size: {aline.size}")
        print(f"  aline[0] = {aline[0]:.8E}" if aline.size > 0 else "  aline is empty")
        print(f"  aline non-zero count: {np.count_nonzero(aline)}")
        print(f"  aline max: {np.max(aline):.8E}" if aline.size > 0 else "  N/A")

    # CRITICAL DEBUG: Always print first call to verify function is being called
    # Only print once to avoid spam
    if debug and not hasattr(solve_josh_flux, "_debug_printed"):
        print(f"\n{'='*70}")
        print(f"DEBUG: solve_josh_flux called (first time)")
        print(f"{'='*70}")
        print(f"  debug_label: {debug_label}")
        print(f"  acont.size: {acont.size}")
        print(f"  scont.size: {scont.size}")
        print(f"  rho.size: {rho.size}")
        print(f"{'='*70}\n")
        solve_josh_flux._debug_printed = True

    # CRITICAL DEBUG: Check if arrays are empty
    if acont.size == 0 or scont.size == 0 or rho.size == 0:
        if debug:
            print(f"\n{'='*70}")
            print(f"CRITICAL: Empty arrays in solve_josh_flux!")
            print(f"{'='*70}")
            print(f"  debug_label: {debug_label}")
            print(f"  acont.size: {acont.size}")
            print(f"  scont.size: {scont.size}")
            print(f"  aline.size: {aline.size}")
            print(f"  sline.size: {sline.size}")
            print(f"  sigmac.size: {sigmac.size}")
            print(f"  sigmal.size: {sigmal.size}")
            print(f"  rho.size: {rho.size}")
            print(f"{'='*70}\n")
        if acont.size == 0:
            return 0.0

    # CRITICAL DEBUG: Check what sigmac actually is when received
    if debug and "FLUX_CONT" in debug_label:
        print(f"\n  DEBUG: Inside solve_josh_flux (at start):")
        if sigmac.size > 0:
            print(f"    sigmac[0] = {sigmac[0]:.8E}")
        else:
            print(f"    sigmac[0] = N/A (empty array)")
        if scont.size > 0:
            print(f"    scont[0] = {scont[0]:.8E}")
        else:
            print(f"    scont[0] = N/A (empty array)")
        if sline.size > 0:
            print(f"    sline[0] = {sline[0]:.8E}")
        else:
            print(f"    sline[0] = N/A (empty array)")
        if sigmac.size > 0 and scont.size > 0:
            print(
                f"    sigmac[0] == scont[0]? {np.isclose(sigmac[0], scont[0], rtol=1e-6)}"
            )
        if sigmac.size > 0 and sline.size > 0:
            print(
                f"    sigmac[0] == sline[0]? {np.isclose(sigmac[0], sline[0], rtol=1e-6)}"
            )

    if acont.size == 0:
        # CRITICAL DEBUG: Why is ACONT empty?
        if debug:
            print(f"\n{'='*70}")
            print(f"CRITICAL: ACONT is empty in solve_josh_flux!")
            print(f"{'='*70}")
            print(f"  debug_label: {debug_label}")
            print(f"  acont.size: {acont.size}")
            print(f"  aline.size: {aline.size}")
            print(f"  sigmac.size: {sigmac.size}")
            print(f"  rho.size: {rho.size}")
            print(f"{'='*70}\n")
        return 0.0

    # CRITICAL FIX: Match Fortran behavior - no clipping of input arrays
    # Fortran uses REAL*8 (double precision) and doesn't clip opacity arrays
    # Only ensure arrays are float64 to match Fortran REAL*8
    # Arrays are already converted to float64 above, so no additional conversion needed

    # Compute ABTOT directly without clipping (matching Fortran)
    # NOTE: There's a remaining ~30% discrepancy at scattering-dominated wavelengths (300nm)
    # due to ALPHA values at deep optical depths being ~0.5 in Python vs ~0.003 in Fortran.
    # The root cause is still under investigation - it may be related to how continuum
    # opacity coefficients are stored/swapped in fort.10 vs what Fortran actually uses internally.
    abtot = acont + aline + sigmac + sigmal
    # Only ensure ABTOT >= EPS to prevent division by zero (Fortran also does this)
    abtot = np.maximum(abtot, EPS)

    # Optional debug: dump ABTOT components for a specific wavelength.
    debug_abtot_wave = os.getenv("PY_DEBUG_ABTOT_WAVE")
    if debug_label and debug_abtot_wave:
        try:
            target_wave = float(debug_abtot_wave)
        except ValueError:
            target_wave = None
        wl_val = None
        if debug_label.startswith("FLUX_TOTAL_") or debug_label.startswith(
            "FLUX_CONT_"
        ):
            try:
                wl_val = float(debug_label.split("_")[-1])
            except ValueError:
                wl_val = None
        if (
            wl_val is not None
            and target_wave is not None
            and abs(wl_val - target_wave) < 1e-4
        ):
            label = "TOTAL" if debug_label.startswith("FLUX_TOTAL_") else "CONT"
            print(f"\nPY_DEBUG ABTOT ARRAY {label}: WAVE={wl_val:.8f}")
            for idx in range(abtot.size):
                print(
                    f"  L={idx+1:3d} RHOX={rho[idx]:.8E} ABTOT={abtot[idx]:.8E} "
                    f"ACONT={acont[idx]:.8E} ALINE={aline[idx]:.8E} "
                    f"SIGMAC={sigmac[idx]:.8E} SIGMAL={sigmal[idx]:.8E}"
                )

    # CRITICAL: Check for INF/NaN (should be rare, but log if found)
    # Fortran would propagate INF/NaN, but we log a warning for debugging
    if np.any(~np.isfinite(abtot)):
        n_inf = np.sum(~np.isfinite(abtot))
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(
            f"Found {n_inf} INF/NaN values in ABTOT (matching Fortran behavior - will propagate)"
        )
        # Don't clip - let INF/NaN propagate like Fortran does

    scatter = sigmac + sigmal
    # CRITICAL DEBUG: Check scatter and abtot values
    if debug:
        print(f"\nDEBUG: ALPHA calculation:")
        print(f"  sigmac[0] = {sigmac[0]:.8E}")
        print(f"  sigmal[0] = {sigmal[0]:.8E}")
        print(f"  scatter[0] = {scatter[0]:.8E}")
        print(f"  acont[0] = {acont[0]:.8E}")
        print(f"  aline[0] = {aline[0]:.8E}")
        print(f"  abtot[0] = {abtot[0]:.8E}")
        print(
            f"  Expected alpha[0] = scatter[0] / abtot[0] = {scatter[0] / max(abtot[0], 1e-40):.8E}"
        )
    alpha = np.zeros_like(abtot)
    np.divide(scatter, abtot, out=alpha, where=abtot > 0.0)
    alpha = np.clip(alpha, 0.0, 1.0)
    # CRITICAL DEBUG: Check computed alpha
    if debug:
        print(f"  Computed alpha[0] = {alpha[0]:.8E}")
        print(f"  alpha[-1] = {alpha[-1]:.8E}")
        print(f"  alpha min/max: {alpha.min():.8E} / {alpha.max():.8E}")

    # CRITICAL FIX: Match Fortran SNUBAR calculation exactly (atlas7v.for line 9030-9031)
    # Fortran: SNUBAR(J)=(ACONT(J)*SCONT(J)+ALINE(J)*SLINE(J))/(ACONT(J)+ALINE(J))
    # Fortran does NOT use max(denom, EPS) - it uses denom directly
    # This preserves small ALINE effects even when ACONT + ALINE is very small
    # Match Fortran exactly: use denom directly, no maximum
    denom = acont + aline
    # Handle division by zero (should be rare, but match Fortran behavior)
    # Fortran would divide by zero if ACONT + ALINE = 0, but this shouldn't happen
    # Use EPS only to prevent actual division by zero, not to clamp small values
    snubar = np.where(denom > 0, (acont * scont + aline * sline) / denom, scont)
    # CRITICAL: Fortran does NOT clamp SNUBAR (atlas7v.for line 8175-8176)
    # SNUBAR can be negative, zero, or exceed SCONT - all are physically possible
    # NOTE: SNUBAR can exceed SCONT if SLINE > SCONT (which is physically possible)

    # Debug SNUBAR at long wavelengths (where emission is observed) or problematic wavelengths
    debug_long_wl = False
    debug_problem_wl = False
    if debug_label and "FLUX_TOTAL" in debug_label:
        # Extract wavelength from debug_label (format: FLUX_TOTAL_1385.15546931)
        try:
            wl_str = debug_label.split("_")[-1]
            wl_val = float(wl_str)
            if wl_val > 850.0:  # Long wavelength
                debug_long_wl = True
            # Also debug problematic wavelengths with huge flux ratios
            if 312.0 <= wl_val <= 313.0:  # Problematic wavelengths
                debug_problem_wl = True
        except (ValueError, IndexError):
            pass

    # IMPORTANT: Avoid unconditional per-wavelength stdout spam.
    # This is extremely slow and can dominate runtime due to I/O.
    # Preserve the ability to debug long/problem wavelengths via an explicit env flag.
    debug_snubar_env = os.getenv("PY_DEBUG_SNUBAR") == "1"
    if debug or (debug_snubar_env and (debug_long_wl or debug_problem_wl)):
        print(f"\nSNUBAR calculation:")
        print(f"  ACONT[0] = {acont[0]:.8E}")
        print(f"  ALINE[0] = {aline[0]:.8E}")
        print(f"  SCONT[0] = {scont[0]:.8E}")
        print(f"  SLINE[0] = {sline[0]:.8E}")
        print(f"  SNUBAR[0] = {snubar[0]:.8E}")
        print(f"  SNUBAR[0] / SCONT[0] = {snubar[0] / max(scont[0], 1e-40):.6f}")
        if aline[0] > 0:
            print(
                f"  ALINE[0] > 0: SNUBAR = (ACONT*SCONT + ALINE*SLINE) / (ACONT + ALINE)"
            )
            print(
                f"    Numerator: {acont[0] * scont[0]:.6e} + {aline[0] * sline[0]:.6e} = {acont[0] * scont[0] + aline[0] * sline[0]:.6e}"
            )
            print(f"    Denominator: {acont[0] + aline[0]:.6e}")
            print(f"    If SLINE > SCONT and ALINE > 0, SNUBAR > SCONT (emission)")
            # CRITICAL DEBUG: Check alignment
            print(f"  Alignment check:")
            print(f"    SLINE[0] / SCONT[0] = {sline[0] / max(scont[0], 1e-40):.6e}")
            print(
                f"    SLINE[-1] / SCONT[-1] = {sline[-1] / max(scont[-1], 1e-40):.6e}"
            )
            print(f"    ALINE[0] / ACONT[0] = {aline[0] / max(acont[0], 1e-40):.6f}")
            print(
                f"    ALINE[-1] / ACONT[-1] = {aline[-1] / max(acont[-1], 1e-40):.6f}"
            )
        print(f"  SNUBAR == SCONT? {np.allclose(snubar, scont, rtol=1e-6)}")
        print(f"  Max difference: {np.abs(snubar - scont).max():.8E}")
        # Print SCONT and SNUBAR at critical depths (65-66) for comparison with Fortran
        if "FLUX_CONT" in debug_label and scont.size > 65:
            print(
                f"\n  SCONT/SNUBAR/RHOX at critical depths (for comparison with Fortran):"
            )
            for idx in [64, 65, 66, 79]:
                if idx < scont.size:
                    temp_str = (
                        f", T = {temperature[idx]:.2f} K"
                        if temperature is not None and idx < temperature.size
                        else ""
                    )
                    rho_str = (
                        f", RHOX = {column_mass[idx]:.8E} g/cm²"
                        if idx < column_mass.size
                        else ""
                    )
                    print(
                        f"    Depth {idx}: SCONT = {scont[idx]:.8E}{temp_str}{rho_str}"
                    )
                    print(f"              SNUBAR = {snubar[idx]:.8E}")
                    print(
                        f"              SNUBAR/SCONT = {snubar[idx] / scont[idx]:.6f}"
                    )

    # CRITICAL DEBUG: Print after SNUBAR to verify we reach this point
    if debug:
        print(
            f"DEBUG: After SNUBAR calculation, rho.size={rho.size}, abtot.size={abtot.size}",
            flush=True,
        )

    # Extract wavelength from debug_label if present (format: FLUX_TOTAL_300.00040572)
    # This must be done early so wavelength_nm is available for conditional debug blocks below
    wavelength_nm = None
    if debug_label and "FLUX_TOTAL_" in debug_label:
        try:
            wavelength_nm = float(debug_label.replace("FLUX_TOTAL_", ""))
        except ValueError:
            pass
        if (
            debug
            and wavelength_nm is not None
            and abs(wavelength_nm - 418.148489) < 1e-4
        ):
            print(f"DEBUG: PY_DUMP_JOSH_ARRAYS={os.getenv('PY_DUMP_JOSH_ARRAYS')}")

    # CRITICAL FIX: Fortran convention: J=1 is surface (smallest RHOX), J=NRHOX is deep (largest RHOX)
    # Fortran's INTEG requires RHOX to be INCREASING (surface → deep): RHOX(1) < RHOX(2) < ... < RHOX(N)
    #
    # Python arrays come in surface-first order, but RHOX may be decreasing.
    # We need RHOX to be INCREASING for INTEG to work correctly.
    #
    # If RHO[0] > RHO[-1], then RHOX is decreasing and we need to reverse.
    # After reversal: rho_integ[0] = rho[-1] (deep, largest), rho_integ[-1] = rho[0] (surface, smallest)
    # But Fortran expects: RHOX(1) = surface (smallest), RHOX(N) = deep (largest)
    # So we need rho_integ[0] to be SURFACE (smallest), not deep!
    #
    # This means we need to reverse TWICE: once to get increasing order, then integrate,
    # then reverse back to get surface-first order.

    if debug:
        print(f"DEBUG: Before needs_reverse check, rho.size={rho.size}", flush=True)
    try:
        # RHOX from NPZ file should already be in correct units (g/cm²)
        # The fixed.npz file has RHOX in correct units matching Fortran expectations
        # No scaling needed - use RHOX directly

        needs_reverse = rho.size > 1 and rho[0] > rho[-1]
        if needs_reverse:
            rho_integ = rho[::-1]
            abtot_integ = abtot[::-1]
            # CRITICAL FIX: When reversed, surface is at the END of the reversed arrays
            # Fortran uses ABTOT(1)*RHOX(1) where J=1 is surface (smallest RHOX)
            # After reversal: rho_integ[-1] = rho[0] (surface), rho_integ[0] = rho[-1] (deep)
            # So we need abtot_integ[-1] * rho_integ[-1] (surface values)
            start = abtot_integ[-1] * rho_integ[-1] if rho_integ.size else 0.0
        else:
            rho_integ = rho.copy()
            abtot_integ = abtot
            start = abtot[0] * rho_integ[0] if rho_integ.size else 0.0
        if debug:
            print(
                f"DEBUG: After needs_reverse setup, rho_integ.size={rho_integ.size}, start={start:.8E}",
                flush=True,
            )
    except Exception as e:
        if debug:
            print(f"DEBUG: Exception in needs_reverse setup: {e}", flush=True)
            import traceback

            traceback.print_exc()
        return 0.0
    # DEBUG: Print ABTOT and RHOX values for comparison with Fortran
    if (
        debug
        and wavelength_nm is not None
        and (
            abs(wavelength_nm - 300.00040572) < 0.0001
            or abs(wavelength_nm - 418.148489) < 0.0001
            or abs(wavelength_nm - 403.188153) < 0.0001
        )
    ):
        print(f"\n  DEBUG TAUNU START: Wavelength {wavelength_nm:.8f} nm")
        print(f"    needs_reverse: {needs_reverse}")
        print(f"    RHOX[0] (surface): {rho_integ[0]:.8E} g/cm²")
        print(f"    RHOX[1]: {rho_integ[1]:.8E} g/cm²")
        print(f"    RHOX[2]: {rho_integ[2]:.8E} g/cm²")
        print(f"    ABTOT[0] (surface): {abtot_integ[0]:.8E} cm²/g")
        print(f"    ABTOT[1]: {abtot_integ[1]:.8E} cm²/g")
        print(f"    ABTOT[2]: {abtot_integ[2]:.8E} cm²/g")
        print(f"    START = ABTOT[0] * RHOX[0] = {start:.8E}")

        # Check if ABTOT values are reasonable
        if abtot_integ[0] > 1e10:
            print(f"    WARNING: ABTOT[0] = {abtot_integ[0]:.8E} is VERY LARGE!")
            print(f"      Expected: ~0.001-0.01 cm²/g for typical stellar atmospheres")
            print(f"      Ratio: {abtot_integ[0] / 0.003:.2e}x too large")

    # Integrate: TAUNU[1] = START, TAUNU[2] = TAUNU[1] + ..., ..., TAUNU[N] = TAUNU[N-1] + ...
    # Since RHOX is INCREASING, TAUNU accumulates UPWARD: TAUNU[0] < TAUNU[1] < ... < TAUNU[-1]
    # After integration: TAUNU[0] = surface (smallest), TAUNU[-1] = deep (largest) ✓
    if debug:
        print(
            f"DEBUG: About to call _integ, rho_integ.size={rho_integ.size}, abtot_integ.size={abtot_integ.size}",
            flush=True,
        )
    try:
        taunu = _integ(rho_integ, abtot_integ, start)
        if debug:
            if taunu.size > 0:
                print(
                    f"DEBUG: After _integ, taunu.size={taunu.size}, taunu[0]={taunu[0]:.8E}",
                    flush=True,
                )
            else:
                print(
                    f"DEBUG: After _integ, taunu.size={taunu.size}, taunu[0]=N/A",
                    flush=True,
                )
    except Exception as e:
        if debug:
            print(f"DEBUG: Exception in _integ: {e}", flush=True)
            import traceback

            traceback.print_exc()
        return 0.0  # Return zero on error

    # CRITICAL DEBUG: Check TAUNU values when line opacity is huge
    if debug and taunu.size > 0:
        print(f"\n{'='*70}")
        print(f"CRITICAL: TAUNU after integration")
        print(f"{'='*70}")
        print(f"  TAUNU[0] (surface) = {taunu[0]:.8E}")
        print(f"  TAUNU[1] = {taunu[1]:.8E}" if taunu.size > 1 else "")
        print(f"  TAUNU[2] = {taunu[2]:.8E}" if taunu.size > 2 else "")
        print(f"  TAUNU[-1] (deep) = {taunu[-1]:.8E}")
        print(f"  XTAU_GRID[-1] (max) = {XTAU_GRID[-1]:.8E}")
        print(f"  TAUNU[0] > XTAU_GRID[-1]? {taunu[0] > XTAU_GRID[-1]}")
        print(f"  ABTOT[0] (surface) = {abtot_integ[0]:.8E}")
        print(f"  ABTOT[-1] (deep) = {abtot_integ[-1]:.8E}")
        print(f"  RHOX[0] (surface) = {rho_integ[0]:.8E}")
        print(f"  RHOX[-1] (deep) = {rho_integ[-1]:.8E}")
        wl_val = None
        if debug_label:
            try:
                wl_str = debug_label.split("_")[-1]
                wl_val = float(wl_str)
            except (ValueError, IndexError):
                wl_val = None
        if wl_val is not None and (
            abs(wl_val - 418.148489) < 0.0001 or abs(wl_val - 403.188153) < 0.0001
        ):
            if taunu.size > 19:
                print(f"  TAUNU[19] = {taunu[19]:.8E}")
            if taunu.size > 71:
                print(f"  TAUNU[71] = {taunu[71]:.8E}")
            if taunu.size >= 10:
                print("  TAUNU[0:10] = " + " ".join(f"{val:.8E}" for val in taunu[:10]))
            print(f"  ALPHA[0] = {alpha[0]:.8E}")
            print(f"  SNUBAR[0] = {snubar[0]:.8E}")
            if snubar.size >= 10:
                print(
                    "  SNUBAR[0:10] = " + " ".join(f"{val:.8E}" for val in snubar[:10])
                )

    # Optional debug: dump full TAUNU and SNUBAR arrays for a specific wavelength.
    debug_taunu_snubar_wave = os.getenv("PY_DEBUG_TAUNU_SNUBAR_WAVE")
    if debug_label and debug_taunu_snubar_wave:
        try:
            target_wave = float(debug_taunu_snubar_wave)
        except ValueError:
            target_wave = None
        wl_val = None
        if debug_label.startswith("FLUX_TOTAL_") or debug_label.startswith(
            "FLUX_CONT_"
        ):
            try:
                wl_val = float(debug_label.split("_")[-1])
            except ValueError:
                wl_val = None
        if (
            wl_val is not None
            and target_wave is not None
            and abs(wl_val - target_wave) < 1e-4
        ):
            label = "TOTAL" if debug_label.startswith("FLUX_TOTAL_") else "CONT"
            print(f"\nPY_DEBUG TAUNU ARRAY {label}: WAVE={wl_val:.8f}")
            for idx, val in enumerate(taunu, start=1):
                print(f"  L={idx:3d} TAUNU={val:.8E}")
            print(f"\nPY_DEBUG SNUBAR ARRAY {label}: WAVE={wl_val:.8f}")
            for idx, val in enumerate(snubar, start=1):
                print(f"  L={idx:3d} SNUBAR={val:.8E}")
            if alpha.size >= 10:
                print("  ALPHA[0:10] = " + " ".join(f"{val:.8E}" for val in alpha[:10]))
            if abtot_integ.size > 71:
                print(f"  ABTOT[71] = {abtot_integ[71]:.8E}")
                print(f"  RHOX[71] = {rho_integ[71]:.8E}")
            if alpha.size > 71:
                print(f"  ALPHA[71] = {alpha[71]:.8E}")
            dump_flag = os.getenv("PY_DUMP_JOSH_ARRAYS") == "1"
            if dump_flag:
                dump_wave = os.getenv("PY_DUMP_JOSH_ARRAYS_WAVE")
                try:
                    target_wave = float(dump_wave) if dump_wave else wl_val
                except ValueError:
                    target_wave = wl_val
                if wl_val is not None and abs(wl_val - target_wave) < 1e-4:
                    dump_path = os.getenv("PY_DUMP_JOSH_ARRAYS_PATH")
                    if not dump_path:
                        dump_path = f"out/josh_arrays_{wl_val:.6f}.npz"
                    np.savez(
                        dump_path,
                        taunu=taunu,
                        snubar=snubar,
                        alpha=alpha,
                    )
                    print(f"  -> Dumped JOSH arrays to {dump_path}")
        if taunu[0] > XTAU_GRID[-1]:
            print(f"\n  WARNING: TAUNU[0] exceeds XTAU_GRID max!")
            print(f"    This will set MAXJ=1, which may affect flux calculation")
            print(f"    Expected: TAUNU[0] should be small for surface layer")
            print(f"    Problem: Huge line opacity causes TAUNU to be huge immediately")
        print(f"{'='*70}\n")

    # CRITICAL: After reversing rho for integration, TAUNU is in increasing order:
    #   TAUNU[0] = surface (smallest RHOX), TAUNU[-1] = deep (largest RHOX)
    #
    # But snubar and alpha are still in original order (surface-first, decreasing RHOX):
    #   snubar[0] = surface (largest RHOX), snubar[-1] = deep (smallest RHOX)
    #
    # For MAP1 to work correctly, TAUNU, SNUBAR, and ALPHA must all be in the SAME order!
    # So we need to reverse snubar and alpha to match TAUNU's order (increasing RHOX).
    if needs_reverse:
        snubar = snubar[::-1]
        alpha = alpha[::-1]
        # Now all arrays are in increasing RHOX order: [0] = surface, [-1] = deep

    # DON'T reverse back - TAUNU is already in correct order (surface → deep, increasing)
    # This matches Fortran's order: TAUNU(1) < TAUNU(2) < ... < TAUNU(NRHOX)
    if debug and taunu.size > 1:
        print(f"\nAfter _integ (before maximum.accumulate):")
        print(f"  needs_reverse = {needs_reverse}")
        if needs_reverse:
            print(f"  rho_integ[0] (surface, smallest) = {rho_integ[0]:.8E}")
            print(f"  rho_integ[-1] (deep, largest) = {rho_integ[-1]:.8E}")
        print(f"  Start value = ABTOT[0] * RHO[0] = {start:.8E}")
        print(f"  TAUNU[0] = {taunu[0]:.8E}")
        print(
            f"  TAUNU[0] matches start? {abs(taunu[0] - start) / max(abs(start), 1e-40) < 1e-10}"
        )
        print(f"  TAUNU[1] = {taunu[1]:.8E}")
        print(f"  TAUNU[2] = {taunu[2]:.8E}")
        print(f"  TAUNU[-1] = {taunu[-1]:.8E}")
        print(f"  TAUNU is increasing? {taunu[0] < taunu[-1]}")
        print(
            f"  TAUNU variation: {(taunu.max() - taunu.min()) / max(taunu.max(), 1e-40) * 100:.6f}%"
        )
        # CRITICAL DEBUG: Check INTEG calculation details
        print(f"\n  INTEG calculation details:")
        print(f"    START = ABTOT[0] * RHO[0] = {start:.8E}")
        print(f"    TAUNU[0] = START = {taunu[0]:.8E}")
        if rho_integ.size > 1:
            drho_01 = rho_integ[1] - rho_integ[0]
            abtot_avg_01 = 0.5 * (abtot_integ[0] + abtot_integ[1])
            print(f"    RHO[1] - RHO[0] = {drho_01:.8E}")
            print(f"    ABTOT[0] = {abtot_integ[0]:.8E}")
            print(f"    ABTOT[1] = {abtot_integ[1]:.8E}")
            print(f"    ABTOT_avg[0-1] = {abtot_avg_01:.8E}")
            print(
                f"    Approx integral_term = ABTOT_avg * dRHO = {abtot_avg_01 * drho_01:.8E}"
            )
            print(f"    Actual TAUNU[1] - TAUNU[0] = {taunu[1] - taunu[0]:.8E}")
        # CRITICAL DEBUG: Print ACONT and SIGMAC components for comparison with Fortran
        # For continuum-only: ABTOT = ACONT + SIGMAC (ALINE=0, SIGMAL=0)
        if needs_reverse:
            acont_0 = acont[-1] if acont.size > 0 else 0.0
            sigmac_0 = sigmac[-1] if sigmac.size > 0 else 0.0
        else:
            acont_0 = acont[0] if acont.size > 0 else 0.0
            sigmac_0 = sigmac[0] if sigmac.size > 0 else 0.0
        print(f"\n  Opacity components (for comparison with Fortran):")
        print(f"    ACONT[0] = {acont_0:.8E}")
        print(f"    SIGMAC[0] = {sigmac_0:.8E}")
        print(f"    ACONT[0] + SIGMAC[0] = {acont_0 + sigmac_0:.8E}")
        print(f"    ABTOT[0] = {abtot_integ[0]:.8E}")
        print(
            f"    Match? {abs((acont_0 + sigmac_0) - abtot_integ[0]) / max(abs(abtot_integ[0]), 1e-40) < 1e-6}"
        )
        # CRITICAL DEBUG: Check if TAUNU[1] or TAUNU[2] > XTAU_GRID[1] (which would force extrapolation)
        if XTAU_GRID.size > 1:
            xtau_grid_1 = XTAU_GRID[1]
            print(f"\n  MAP1 extrapolation check:")
            print(f"    XTAU_GRID[1] = {xtau_grid_1:.8E}")
            print(f"    TAUNU[0] > XTAU_GRID[1]? {taunu[0] > xtau_grid_1}")
            print(f"    TAUNU[1] > XTAU_GRID[1]? {taunu[1] > xtau_grid_1}")
            print(f"    TAUNU[2] > XTAU_GRID[1]? {taunu[2] > xtau_grid_1}")
            if taunu[1] > xtau_grid_1:
                print(f"    ✓ TAUNU[1] > XTAU_GRID[1] - will force extrapolation (L=2)")
            elif taunu[2] > xtau_grid_1:
                print(f"    ✓ TAUNU[2] > XTAU_GRID[1] - will force extrapolation (L=3)")
            else:
                print(f"    ✗ TAUNU[1-2] < XTAU_GRID[1] - will interpolate")
    # CRITICAL FIX: Match Fortran exactly - no monotonicity enforcement
    # Fortran computes TAUNU via INTEG and doesn't enforce monotonicity
    # TAUNU should naturally be monotonic from integration, but we match Fortran exactly
    # Remove maximum.accumulate to match Fortran behavior
    if debug and taunu.size > 1:
        print(f"\nAfter _integ (TAUNU as computed):")
        print(f"  TAUNU[0] = {taunu[0]:.8E}")
        print(f"  TAUNU[-1] = {taunu[-1]:.8E}")
        print(
            f"  TAUNU is monotonic? {np.all(np.diff(taunu) >= 0) or np.all(np.diff(taunu) <= 0)}"
        )
        # TAUNU is now in surface-first order: TAUNU[0]=surface, TAUNU[-1]=deep
        # This matches SNUBAR and ALPHA which are also surface-first

    if debug:
        print(f"\n{'='*70}")
        print(f"DEBUG JOSH_FLUX {debug_label}")
        print(f"{'='*70}")
        print(f"Surface values (layer 0):")
        print(f"  ABTOT[0] = {abtot[0]:.8E}")
        print(f"  SNUBAR[0] = {snubar[0]:.8E}")
        print(f"  ALPHA[0] = {alpha[0]:.8E}")
        print(f"  ACONT[0] = {acont[0]:.8E}")
        print(f"  ALINE[0] = {aline[0]:.8E}")
        print(f"  SIGMAC[0] = {sigmac[0]:.8E}")
        print(f"  SIGMAL[0] = {sigmal[0]:.8E}")
        print(f"  scatter[0] = SIGMAC + SIGMAL = {sigmac[0] + sigmal[0]:.8E}")
        print(
            f"  ABTOT[0] = ACONT + ALINE + SIGMAC + SIGMAL = {acont[0] + aline[0] + sigmac[0] + sigmal[0]:.8E}"
        )
        print(
            f"  Expected ALPHA = scatter/ABTOT = {(sigmac[0] + sigmal[0]) / max(abtot[0], 1e-40):.8E}"
        )
        print(f"  RHO[0] = {rho[0]:.8E}")

        # DEBUG ATLAS7V: Print TAUNU START matching Fortran format
        # Format: DEBUG ATLAS7V TAUNU START: WAVE=300.00040572 TAUNU(1)= 5.04653487E-03 TAUNU(2)= 6.60722380E-03 TAUNU(3)= 8.44791709E-03 ABTOT(1)= 2.71909514E-03 RHOX(1)= 1.85596112E+00
        if wavelength_nm is not None and taunu.size >= 3:
            print(
                f"\nDEBUG ATLAS7V TAUNU START: WAVE={wavelength_nm:.8f} "
                f"TAUNU(1)={taunu[0]:13.8E} TAUNU(2)={taunu[1]:13.8E} TAUNU(3)={taunu[2]:13.8E} "
                f"ABTOT(1)={abtot[0]:13.8E} RHOX(1)={rho[0]:13.8E}"
            )

        # DEBUG ATLAS7V SNUBAR/TAUNU END
        if wavelength_nm is not None and taunu.size > 0 and snubar.size > 0:
            nrhox = len(taunu)
            print(
                f"DEBUG ATLAS7V SNUBAR/TAUNU END: WAVE={wavelength_nm:.8f} "
                f"SNUBAR(NRHOX)={snubar[-1]:13.8E} TAUNU(NRHOX)={taunu[-1]:13.8E} NRHOX={nrhox:3d}"
            )

        # DEBUG ATLAS7V RHOX
        if wavelength_nm is not None and rho.size > 0:
            nrhox = len(rho)
            rho_idx_65 = min(
                64, nrhox - 1
            )  # Fortran uses 1-based, Python 0-based, so 65->64
            print(
                f"DEBUG ATLAS7V RHOX: WAVE={wavelength_nm:.8f} "
                f"RHOX(65)={rho[rho_idx_65]:13.8E} RHOX(NRHOX)={rho[-1]:13.8E}"
            )

        # DEBUG ATLAS7V SNUBAR DEPTHS
        if wavelength_nm is not None and snubar.size > 0:
            nrhox = len(snubar)
            snubar_idx_65 = min(64, nrhox - 1)
            snubar_idx_66 = min(65, nrhox - 1)
            print(
                f"DEBUG ATLAS7V SNUBAR DEPTHS: WAVE={wavelength_nm:.8f} "
                f"SNUBAR(65)={snubar[snubar_idx_65]:13.8E} "
                f"SNUBAR(66)={snubar[snubar_idx_66]:13.8E} "
                f"SNUBAR(NRHOX)={snubar[-1]:13.8E}"
            )
        # Print full arrays for comparison (all values, comma-separated)
        print(f"\nFull TAUNU array ({len(taunu)} values):")
        taunu_str = ", ".join(f"{v:.8E}" for v in taunu)
        print(f"  TAUNU = [{taunu_str}]")
        print(f"\nFull SNUBAR array ({len(snubar)} values):")
        snubar_str = ", ".join(f"{v:.8E}" for v in snubar)
        print(f"  SNUBAR = [{snubar_str}]")
        print(f"  Start value = {start:.8E}")
        if taunu.size > 0:
            print(f"  TAUNU[0] (surface) = {taunu[0]:.8E}")
            print(f"  TAUNU[-1] (deep) = {taunu[-1]:.8E}")
            print(
                f"  TAUNU variation: {(taunu.max() - taunu.min()) / max(taunu.max(), 1e-40) * 100:.6f}%"
            )
            print(f"  RHO[0] = {rho[0]:.8E}, RHO[-1] = {rho[-1]:.8E}")
            print(
                f"  RHO variation: {(rho.max() - rho.min()) / max(rho.max(), 1e-40) * 100:.6f}%"
            )
            print(f"  ABTOT[0] = {abtot[0]:.8E}, ABTOT[-1] = {abtot[-1]:.8E}")
            print(
                f"  ABTOT variation: {(abtot.max() - abtot.min()) / max(abtot.max(), 1e-40) * 100:.6f}%"
            )
        else:
            print(f"  TAUNU is empty!")
        print(f"  XTAU_GRID max = {XTAU_GRID[-1]:.8E}")
        print(
            f"  TAUNU[0] > XTAU_GRID max? {taunu[0] > XTAU_GRID[-1] if taunu.size > 0 else False}"
        )

    # Determine whether the scattering iteration is needed. In the Fortran JOSH flow,
    # IFSCAT=1 drives the scattering path and iteration, regardless of whether lines
    # are present. For our runs, IFSCAT is effectively 1, so iterate whenever there
    # is non-negligible scattering.
    needs_iteration = np.any(alpha > 1e-12)

    # CRITICAL: Always use scattering path (XSBAR/XALPHA) even for continuum-only
    # This matches Fortran's IFSCAT=1 behavior
    always_use_scattering_path = True

    # CRITICAL FIX: For surface flux calculation (IFSURF=1), Fortran skips ITERATION
    # but still uses the correct code path based on IFSCAT:
    #
    # - If IFSCAT=0 (no scattering): Uses MAP1(TAUNU,SNUBAR,...) directly for XS (line 8260-8262)
    #   Then skips iteration (line 8264: IF(IFSURF.EQ.1)GO TO 60)
    #
    # - If IFSCAT=1 (scattering): Uses XSBAR/XALPHA path (label 30, line 8270-8271)
    #   Applies (1-XALPHA) modification to XSBAR (line 8346)
    #   Iterates XS (lines 8348-8398)
    #   Then skips further iteration (line 8400: IF(IFSURF.EQ.1)GO TO 60)
    #
    # So for surface flux, we should:
    # 1. Use the scattering path (XSBAR/XALPHA) when scattering is present
    # 2. Skip iteration (but still use XSBAR/XALPHA calculation)
    #
    # We'll handle skipping iteration later, after XSBAR/XALPHA are calculated.
    # For now, keep needs_iteration as calculated above.

    # CRITICAL FIX: Fortran flow analysis (atlas7v_1.for lines 7115-7129):
    # - Line 7116: IF(IFSCAT.EQ.1)GO TO 30
    # - When IFSCAT=0 (no scattering):
    #   * Line 7118-7119: Sets SNU(J)=SNUBAR(J)
    #   * Line 7121: Calls MAP1(TAUNU,SNU,...) to get XS8
    #   * Line 7122-7123: Sets XS(L)=XS8(L)
    #   * Line 7124: IF(IFSURF.EQ.1)GO TO 60 (flux calculation)
    #   * So when IFSCAT=0 and IFSURF=1, Fortran uses MAP1 interpolation!
    # - When IFSCAT=1 (scattering):
    #   * Line 7128: IF(TAUNU(1).GT.XTAU8(NXTAU))MAXJ=1
    #   * Line 7129: IF(MAXJ.EQ.1)GO TO 401
    #   * So MAXJ=1 check only applies when IFSCAT=1
    #
    # In Python, `needs_iteration` corresponds to IFSCAT=1 (scattering enabled).
    # When `needs_iteration=False` (IFSCAT=0), we should ALWAYS use MAP1 interpolation,
    # even if TAUNU[0] > XTAU_GRID[-1], matching Fortran behavior.

    # Enable MAP1 debugging for problematic indices (40-50) where Python XSBAR is too small
    debug_map1_detailed = debug and "FLUX_CONT" in debug_label

    # CRITICAL FIX: Fortran flow for continuum-only (IFSCAT=0) vs scattering (IFSCAT=1):
    #
    # When IFSCAT=0 (no scattering, continuum-only) AND IFSURF=1 (surface flux):
    #   - Line 8257-8258: Sets SNU(J)=SNUBAR(J)
    #   - Line 8260: Calls MAP1(TAUNU,SNU,...) to get XS8 directly
    #   - Line 8262: Sets XS(L)=XS8(L)
    #   - Line 8264: IF(IFSURF.EQ.1)GO TO 60 (goes directly to flux calculation)
    #   - Line 8520: Uses XS directly (from MAP1(TAUNU,SNUBAR,...), NOT from XSBAR)
    #   - Does NOT call MAP1 for XSBAR/XALPHA (skips lines 8270-8271)
    #   - Does NOT apply (1-XALPHA) modification
    #
    # When IFSCAT=1 (scattering) AND IFSURF=1 (surface flux):
    #   - Line 8268: IF(TAUNU(1).GT.XTAU8(NXTAU))MAXJ=1
    #   - Line 8270-8271: Calls MAP1 for XSBAR and XALPHA
    #   - Line 8300: Initializes XS(L)=XSBAR(L)
    #   - Line 8346: Applies XSBAR(L)=(1.-XALPHA(L))*XSBAR(L)
    #   - Line 8348-8398: Iterates XS
    #   - Line 8400: IF(IFSURF.EQ.1)GO TO 60
    #   - Line 8520: Uses iterated XS
    #
    # So for continuum-only (needs_iteration=False), we should:
    #   1. Skip XSBAR/XALPHA MAP1 calls
    #   2. Use MAP1(TAUNU,SNUBAR,...) directly to get XS
    #   3. Use XS directly for flux calculation (no modification, no iteration)

    # CRITICAL FIX: Fortran ALWAYS uses IFSCAT=1 (scattering path), even for continuum-only
    # This means Fortran ALWAYS computes XSBAR and XALPHA via MAP1, then sets XS(L)=XSBAR(L)
    # Python should match this behavior by always using the scattering path
    if always_use_scattering_path or needs_iteration:
        # Scattering case (IFSCAT=1): use XSBAR/XALPHA path
        # CRITICAL: Fortran ALWAYS uses this path (IFSCAT=1), even for continuum-only
        # CRITICAL FIX: Check MAXJ=1 condition BEFORE MAP1
        # Fortran checks: IF(TAUNU(1).GT.XTAU8(NXTAU))MAXJ=1 (line 8268)
        # This check applies when IFSCAT=1 (scattering)
        # When MAXJ=1, Fortran skips MAP1 and goes to label 401 (line 8269: IF(MAXJ.EQ.1)GO TO 401)
        # For flux calculation (IFSURF=1), Fortran goes to label 60 which uses XS directly
        if taunu.size > 0 and taunu[0] > XTAU_GRID[-1]:
            maxj = 1
        else:
            maxj = 0

        # CRITICAL FIX: When MAXJ=1, Fortran skips MAP1 interpolation (line 8269: GO TO 401)
        # Instead, it sets XSBAR and XALPHA directly from SNUBAR[0] and ALPHA[0]
        # (see lines 8295-8299: when XTAU8(L) < TAUNU(1), set XSBAR(L)=SNUBAR(1))
        # Since all XTAU_GRID points are < TAUNU[0] when MAXJ=1, all XSBAR should be SNUBAR[0]
        if maxj == 1:
            # Skip MAP1 and set XSBAR/XALPHA directly (matches Fortran behavior)
            if debug:
                print(
                    f"MAXJ=1: Skipping MAP1 interpolation, setting XSBAR/XALPHA directly from SNUBAR[0]/ALPHA[0]"
                )
            xsbar = np.full(
                len(XTAU_GRID), snubar[0] if snubar.size > 0 else EPS, dtype=np.float64
            )
            xalpha = np.full(
                len(XTAU_GRID), alpha[0] if alpha.size > 0 else 0.0, dtype=np.float64
            )
            maxj_xsbar = 1
            maxj_xalpha = 1
        else:
            # Normal case: call MAP1
            # CRITICAL DEBUG: Check alpha array before MAP1
            if debug:
                print(f"\nDEBUG: Before MAP1 interpolation of ALPHA:")
                print(f"  alpha[0] = {alpha[0]:.8E}")
                print(f"  alpha[-1] = {alpha[-1]:.8E}")
                print(f"  alpha min/max: {alpha.min():.8E} / {alpha.max():.8E}")
                print(f"  taunu[0] = {taunu[0]:.8E}")
                print(f"  taunu[-1] = {taunu[-1]:.8E}")
                print(f"  XTAU_GRID[0] = {XTAU_GRID[0]:.8E}")
                print(f"  XTAU_GRID[-1] = {XTAU_GRID[-1]:.8E}")
            xsbar, maxj_xsbar = _map1(
                taunu,
                snubar,
                XTAU_GRID,
                debug=debug,
            )
            xalpha, maxj_xalpha = _map1(taunu, alpha, XTAU_GRID, debug=debug)
            # CRITICAL DEBUG: Check XSBAR after MAP1, BEFORE mask
            if debug:
                print(f"\nDEBUG: After MAP1, BEFORE mask:")
                print(f"  xsbar[0] = {xsbar[0]:.8E}")
                print(
                    f"  xsbar[17] = {xsbar[17]:.8E}"
                    if xsbar.size > 17
                    else "  xsbar[17] = N/A"
                )
                print(
                    f"  xsbar[50] = {xsbar[50]:.8E}"
                    if xsbar.size > 50
                    else "  xsbar[50] = N/A"
                )
                print(f"  xsbar[-1] = {xsbar[-1]:.8E}")
                print(f"  Expected: Fortran XSBAR(51) = 1.54421709E-13")
                if xsbar.size > 50:
                    ratio = (
                        1.54421709e-13 / xsbar[50] if xsbar[50] > 0 else float("inf")
                    )
                    print(f"  Ratio (Fortran/Python): {ratio:.2f}x")
            # CRITICAL DEBUG: Check XALPHA after MAP1
            if debug:
                print(f"\nDEBUG: After MAP1 interpolation of ALPHA:")
                print(f"  xalpha[0] = {xalpha[0]:.8E}")
                print(f"  xalpha[-1] = {xalpha[-1]:.8E}")
                print(f"  xalpha min/max: {xalpha.min():.8E} / {xalpha.max():.8E}")
                print(f"  Expected xalpha[0] ≈ alpha[0] = {alpha[0]:.8E}")
                print(f"  Difference: {abs(xalpha[0] - alpha[0]):.8E}")

        # CRITICAL FIX: Fortran overwrites MAXJ with each MAP1 call (line 8270-8271)
        # Use the result from the last MAP1 call (ALPHA) unless MAXJ was set to 1 by pre-check
        # When MAXJ=1, we skip MAP1, so keep maxj=1
        if maxj != 1:
            maxj = maxj_xalpha
        xsbar = np.maximum(xsbar, EPS)
        xalpha = np.clip(xalpha, 0.0, 1.0)

        # CRITICAL DEBUG: Check xalpha value before (1-XALPHA) modification
        if debug:
            print(f"\nDEBUG: Before (1-XALPHA) modification:")
            print(f"  xalpha[0] = {xalpha[0]:.8E}")
            print(f"  xsbar[0] = {xsbar[0]:.8E}")
            print(f"  (1 - xalpha[0]) = {1.0 - xalpha[0]:.8E}")
            print(
                f"  Expected xsbar_modified[0] = xsbar[0] * (1 - xalpha[0]) = {xsbar[0] * (1.0 - xalpha[0]):.8E}"
            )
        # Apply surface-value masking when XTAU8(L) < TAUNU(1)
        # (atlas7v.for lines ~10230-10233).
        if taunu.size > 0:
            mask = XTAU_GRID < taunu[0]
            if np.any(mask):
                if debug:
                    print(f"\nDEBUG: Applying mask for XTAU < TAUNU[0]:")
                    print(f"  mask count: {np.sum(mask)} of {len(XTAU_GRID)}")
                    print(f"  xsbar[0] (before mask) = {xsbar[0]:.8E}")
                    print(f"  xalpha[0] (before mask) = {xalpha[0]:.8E}")
                xsbar[mask] = np.maximum(snubar[0], EPS)
                xalpha[mask] = np.clip(alpha[0], 0.0, 1.0)
                if debug:
                    print(f"  xsbar[0] (after mask) = {xsbar[0]:.8E}")
                    print(f"  xalpha[0] (after mask) = {xalpha[0]:.8E}")

        # Initialize XS from XSBAR (will be modified by iteration)
        xs = xsbar.copy()

        # CRITICAL FIX: Fortran ALWAYS applies (1-XALPHA) modification (line 9100),
        # even for surface flux. This happens BEFORE the iteration check.
        # Apply (1-XALPHA) modification to XSBAR (line 9100 in atlas7v.for)
        # This must happen BEFORE iteration, matching Fortran line 9100
        # Note: XS is initialized from UNMODIFIED XSBAR, but xsbar_modified is
        # used in the iteration formula. This matches Fortran's behavior.
        xsbar_modified = xsbar * (1.0 - xalpha)

        # CRITICAL DEBUG: Check xsbar_modified after modification
        if debug:
            print(f"\nDEBUG: After (1-XALPHA) modification:")
            print(f"  xsbar_modified[0] = {xsbar_modified[0]:.8E}")
            print(f"  Expected: {xsbar[0] * (1.0 - xalpha[0]):.8E}")
            print(
                f"  Match: {abs(xsbar_modified[0] - xsbar[0] * (1.0 - xalpha[0])) / max(abs(xsbar[0] * (1.0 - xalpha[0])), 1e-40) < 1e-6}"
            )
        dump_flag = os.getenv("PY_DUMP_JOSH_ARRAYS") == "1"
        if dump_flag and debug_label and "FLUX_TOTAL_" in debug_label:
            try:
                wl_val = float(debug_label.replace("FLUX_TOTAL_", ""))
            except ValueError:
                wl_val = None
            dump_wave = os.getenv("PY_DUMP_JOSH_ARRAYS_WAVE")
            try:
                target_wave = float(dump_wave) if dump_wave else wl_val
            except ValueError:
                target_wave = wl_val
            if (
                wl_val is not None
                and target_wave is not None
                and abs(wl_val - target_wave) < 1e-4
            ):
                print(
                    "  XSBAR_MOD[0:10] = "
                    + " ".join(f"{val:.8E}" for val in xsbar_modified[:10])
                )
                print(
                    "  XALPHA[0:10] = " + " ".join(f"{val:.8E}" for val in xalpha[:10])
                )

        # CRITICAL FIX: Fortran DOES iterate for surface flux until convergence!
        # Fortran code structure (lines 9102-9154):
        #   DO 34 L=1,NXTAU
        #     DO 33 KK=1,NXTAU
        #       ... iteration code ...
        #     33 XS(K)=MAX(XS(K)+DELXS,1.E-38)
        #     39 IF(IFERR.EQ.0)GO TO 35
        #    34 CONTINUE
        #   35 IF(IFSURF.EQ.1)GO TO 60
        #
        # The iteration loop executes FIRST, then checks IFSURF AFTER convergence.
        # So Fortran iterates until convergence (IFERR=0), then checks IFSURF.
        # Python was incorrectly skipping iteration for surface flux!
        #
        # Iteration formula (line 9147):
        #   DELXS = (sum(COEFJ(K,M)*XS(M)) * XALPHA(K) + XSBAR(K) - XS(K)) / DIAG(K)
        #   XS(K) = XS(K) + DELXS
        # Where XSBAR(K) is AFTER (1-XALPHA) modification
        diag = 1.0 - xalpha * COEFJ_DIAG

        # Initialize XS from unmodified XSBAR (after masking).
        # Fortran sets XS(L)=XSBAR(L) before applying the (1-XALPHA) modification.
        xs = xsbar.copy()
        num_iterations = 0  # Initialize for debug output

        # Optional debug: mirror first-iteration K=1 sum like Fortran.
        debug_iter_wave = os.getenv("PY_DEBUG_ITER_STEP_WAVE")
        if debug_iter_wave and debug_label:
            try:
                target_wave = float(debug_iter_wave)
            except ValueError:
                target_wave = None
            wl_val = None
            if debug_label.startswith("FLUX_TOTAL_") or debug_label.startswith(
                "FLUX_CONT_"
            ):
                try:
                    wl_val = float(debug_label.split("_")[-1])
                except ValueError:
                    wl_val = None
            if (
                wl_val is not None
                and target_wave is not None
                and abs(wl_val - target_wave) < 1e-4
            ):
                xs_tmp = xs.astype(np.float32).copy()
                coefj_dbg = COEFJ_MATRIX.astype(np.float32)
                diag_dbg = diag.astype(np.float32)
                xalpha_dbg = xalpha.astype(np.float32)
                xsbar_mod_dbg = xsbar_modified.astype(np.float32)
                nxtau = xs_tmp.size
                for k in range(nxtau - 1, -1, -1):
                    sum_val = float(np.dot(coefj_dbg[k, :], xs_tmp))
                    delxs = (
                        sum_val * xalpha_dbg[k] + xsbar_mod_dbg[k] - xs_tmp[k]
                    ) / diag_dbg[k]
                    if k == 0:
                        label = (
                            "TOTAL" if debug_label.startswith("FLUX_TOTAL_") else "CONT"
                        )
                        print(
                            f"PY_DEBUG ITERATION STEP {label}: WAVE={wl_val:.8f} "
                            f"sum(COEFJ(1,M)*XS(M))={sum_val:.8E} "
                            f"XSBAR(1)={xsbar_modified[k]:.8E} XS(1)={xs_tmp[k]:.8E} "
                            f"DIAG(1)={diag[k]:.8E}"
                        )
                        break
                    xs_tmp[k] = max(xs_tmp[k] + delxs, EPS)

        # Fortran skips the iteration when MAXJ=1 (MAP1 extrapolation edge case).
        # In that regime XS should remain equal to XSBAR (after masking/modification).
        if maxj != 1:
            # DEBUG: Print iteration inputs
            debug_iteration = debug and (
                "FLUX_CONT" in debug_label
                or any(
                    key in debug_label
                    for key in (
                        "311.304",
                        "315.904",
                        "317.131",
                        "319.494",
                        "320.973",
                        "320.974",
                    )
                )
            )
            if debug_iteration:
                print(f"\n{'='*70}")
                print(f"ITERATION INPUTS (before first iteration)")
                print(f"{'='*70}")
                print(f"  XSBAR[0] (before mod) = {xsbar[0]:.8E}")
                print(f"  XSBAR[0] (after mod) = {xsbar_modified[0]:.8E}")
                print(f"  XS[0] (initialized) = {xs[0]:.8E}")
                print(f"  XALPHA[0] = {xalpha[0]:.8E}")
                print(f"  COEFJ_DIAG[0] = {COEFJ_DIAG[0]:.8E}")
                print(f"  DIAG[0] = {diag[0]:.8E}")
                print(f"  First 5 COEFJ[0,M] values:")
                for m in range(min(5, len(xs))):
                    print(f"    COEFJ[0,{m}] = {COEFJ_MATRIX[0, m]:.8E}")
                print(f"  First 5 XS[M] values (initialized from XSBAR):")
                for m in range(min(5, len(xs))):
                    print(f"    XS[{m}] = {xs[m]:.8E}")
                sum0 = float(np.dot(COEFJ_MATRIX[0, :], xs))
                delxs0 = (sum0 * xalpha[0] + xsbar_modified[0] - xs[0]) / diag[0]
                print(f"  ITER DEBUG K=0: sum={sum0:.8E} delxs={delxs0:.8E}")
                # Track K=1 with current XS state (after K>1 updates)
                if len(xs) > 1:
                    sum1 = float(np.dot(COEFJ_MATRIX[1, :], xs))
                    delxs1 = (sum1 * xalpha[1] + xsbar_modified[1] - xs[1]) / diag[1]
                    print(f"  ITER DEBUG K=1: sum={sum1:.8E} delxs={delxs1:.8E}")

            # Use Numba kernel for iteration (required for performance)
            if not NUMBA_AVAILABLE:
                raise RuntimeError(
                    "Numba is required for JOSH solver iteration. "
                    "Please install numba: pip install numba"
                )

            # Make a copy for Numba (needs writable array)
            xs_copy = xs.copy()
            if USE_FLOAT32_ITERATION:
                # Fortran uses REAL*4 arrays in the XS iteration; mirror that precision.
                coefj_f32 = COEFJ_MATRIX.astype(np.float32)
                diag_f32 = COEFJ_DIAG.astype(np.float32)
                xs_f32 = xs_copy.astype(np.float32)
                xalpha_f32 = xalpha.astype(np.float32)
                xsbar_mod_f32 = xsbar_modified.astype(np.float32)
                xs_result_f32, num_iterations = _josh_iteration_kernel(
                    coefj_f32,
                    xs_f32,
                    xalpha_f32,
                    xsbar_mod_f32,
                    diag_f32,
                    np.float32(ITER_TOL),
                    MAX_ITER,
                    np.float32(EPS),
                )
                xs[:] = xs_result_f32.astype(np.float64)
                if debug:
                    print("DEBUG: XS iteration used float32 (REAL*4) precision")
            else:
                xs_result, num_iterations = _josh_iteration_kernel(
                    COEFJ_MATRIX,
                    xs_copy,
                    xalpha,
                    xsbar_modified,
                    COEFJ_DIAG,
                    ITER_TOL,
                    MAX_ITER,
                    EPS,
                )
                xs[:] = xs_result  # Copy result back

            if debug:
                print(
                    f"DEBUG: Finished iteration loop, num_iterations={num_iterations}",
                    flush=True,
                )
                wl_val = None
                if debug_label:
                    try:
                        wl_str = debug_label.split("_")[-1]
                        wl_val = float(wl_str)
                    except (ValueError, IndexError):
                        wl_val = None
                if wl_val is not None and (
                    abs(wl_val - 418.148489) < 0.0001
                    or abs(wl_val - 403.188153) < 0.0001
                    or abs(wl_val - 319.490345) < 0.0001
                ):
                    print(f"\nPY_DEBUG XS418: WAVE={wl_val:.8f}")
                    print(f"  XS[0]={xs[0]:.8E}")
                    if xs.size > 1:
                        print(f"  XS[1]={xs[1]:.8E}")
                    print(f"  XSBAR[0]={xsbar[0]:.8E}")
                    print(f"  XALPHA[0]={xalpha[0]:.8E}")
    else:
        # IFSCAT=0 path: Direct MAP1 to XS (not used in current implementation)
        # This path is not used since always_use_scattering_path = True
        # But keep for completeness
        xs = xsbar.copy()
        num_iterations = 0  # No iteration for this path

    if debug:
        print(f"\nAfter _map1 (before mask correction):")
        print(f"  TAUNU[0] = {taunu[0] if taunu.size > 0 else 0:.8E}")
        print(f"  TAUNU[-1] = {taunu[-1] if taunu.size > 0 else 0:.8E}")
        print(f"  SNUBAR[0] = {snubar[0]:.8E}")
        print(f"  SNUBAR[-1] = {snubar[-1]:.8E}")
        print(f"  XTAU_GRID[0] = {XTAU_GRID[0]:.8E}")
        print(f"  XTAU_GRID[-1] = {XTAU_GRID[-1]:.8E}")
        print(f"  XSBAR[0] = {xsbar[0]:.8E}")
        print(f"  XSBAR[-1] = {xsbar[-1]:.8E}")
        print(f"  XSBAR unique values: {len(np.unique(xsbar))}")
        # Print full SNUBAR array for comparison with Fortran
        if "FLUX_CONT" in debug_label:
            print(f"\n  SNUBAR ARRAY (full, for comparison with Fortran):")
            for i in range(len(snubar)):
                print(f"    SNUBAR[{i}] = {snubar[i]:.8E}")
            print(f"\n  TAUNU ARRAY (full, for comparison with Fortran):")
            for i in range(len(taunu)):
                print(f"    TAUNU[{i}] = {taunu[i]:.8E}")
        if taunu.size > 0:
            mask_before = XTAU_GRID < taunu[0]
            print(f"  Points where XTAU_GRID < TAUNU[0]: {np.sum(mask_before)}")

    # Optional debug: dump XS array for a specific wavelength to compare with Fortran.
    dump_xs_wave = os.getenv("PY_DEBUG_XS_ARRAY_WAVE")
    if debug_label and dump_xs_wave:
        try:
            target_wave = float(dump_xs_wave)
        except ValueError:
            target_wave = None
        wl_val = None
        if debug_label.startswith("FLUX_TOTAL_") or debug_label.startswith(
            "FLUX_CONT_"
        ):
            try:
                wl_val = float(debug_label.split("_")[-1])
            except ValueError:
                wl_val = None
        if (
            wl_val is not None
            and target_wave is not None
            and abs(wl_val - target_wave) < 1e-4
        ):
            label = "TOTAL" if debug_label.startswith("FLUX_TOTAL_") else "CONT"
            print(f"\nPY_DEBUG XS_ARRAY {label}: WAVE={wl_val:.8f}")
            for idx, (xtau, xs_val) in enumerate(zip(XTAU_GRID, xs), start=1):
                print(f"  L={idx:3d} XTAU={xtau:.8E} XS={xs_val:.8E}")

    # When TAUNU is constant (or nearly constant) and > XTAU_GRID max, MAP1 can't
    # extrapolate properly because linear extrapolation requires TAUNU to vary.
    # However, we should still let MAP1 try to extrapolate, as Fortran does.
    # Only if MAP1 fails (returns constant values) should we use SNUBAR[0] directly.

    # Check if MAP1 returned constant values (suggesting TAUNU is constant)
    xsbar_constant = len(np.unique(xsbar)) == 1
    if debug and xsbar_constant:
        if taunu.size > 1:
            taunu_variation = (taunu.max() - taunu.min()) / max(taunu.max(), 1e-40)
            print(
                f"\nMAP1 returned constant XSBAR (TAUNU variation: {taunu_variation*100:.6f}%)"
            )
            print(f"XSBAR = {xsbar[0]:.8E} (from MAP1 extrapolation)")

    if debug:
        print(f"\nAfter MAP1 interpolation (MAXJ={maxj}):")
        print(f"  XSBAR[0] = {xsbar[0]:.8E}")
        print(f"  XSBAR[-1] = {xsbar[-1]:.8E}")
        print(f"  XALPHA[0] = {xalpha[0]:.8E}")
        print(f"  XALPHA[-1] = {xalpha[-1]:.8E}")
        print(f"  XSBAR min/max: {xsbar.min():.8E} / {xsbar.max():.8E}")
        print(f"  XALPHA min/max: {xalpha.min():.8E} / {xalpha.max():.8E}")
        # Print full XSBAR array for comparison with Fortran
        print(f"\n  XSBAR ARRAY (full):")
        for i in range(len(xsbar)):
            print(f"    XSBAR[{i}] = {xsbar[i]:.8E}")

    # For continuum-only case, XS is already initialized from MAP1(TAUNU,SNUBAR,...) above.
    # For scattering case, XS is initialized from XSBAR above.
    # MAXJ=1 has two regimes:
    # - If TAUNU(1) > XTAU_GRID[-1], Fortran runs the Feautrier MAXJ=1 path.
    # - If TAUNU(1) <= XTAU_GRID[-1], Fortran uses XS/CH directly (no Feautrier).
    if needs_iteration and maxj == 1 and taunu.size > 0 and taunu[0] > XTAU_GRID[-1]:
        tau_surface = taunu[0]
        nrhox = len(taunu)

        if debug:
            print(f"\nMAXJ=1 FEAUTRIER ITERATION: TAUNU[0] = {tau_surface:.2f}")
            print(f"  NRHOX = {nrhox}")
            print(f"  SNUBAR[0] = {snubar[0]:.6e}")
            print(f"  SNUBAR[-1] = {snubar[-1]:.6e}")
            print(f"  ALPHA[0] = {alpha[0]:.6e}")

        # Initialize SNU = SNUBAR (all depths)
        snu = snubar.copy()

        # Feautrier iteration (max NXTAU iterations, matching Fortran)
        max_feautrier_iter = NXTAU
        converged = False

        for feautrier_iter in range(max_feautrier_iter):
            error = 0.0

            # Step 1: HNU = d(SNU)/d(τ) using DERIV
            hnu = _deriv(taunu, snu)

            # Step 2: HNU = HNU / 3 (Eddington factor)
            hnu = hnu / 3.0

            # Step 3: JMINS = d(HNU)/d(τ) using DERIV
            jmins = _deriv(taunu, hnu)

            # Step 4: JNU = JMINS + SNU
            jnu = jmins + snu

            # Step 5 & 6: SNEW = (1-α)*SNUBAR + α*JNU, then SNU = SNEW
            # Track convergence error
            for j in range(nrhox):
                snew = (1.0 - alpha[j]) * snubar[j] + alpha[j] * jnu[j]
                if snew > 0:
                    error += abs(snew - snu[j]) / snew
                snu[j] = snew

            if debug and feautrier_iter == 0:
                print(f"\n  First Feautrier iteration:")
                print(f"    HNU[0] = {hnu[0]:.6e}")
                print(f"    JMINS[0] = {jmins[0]:.6e}")
                print(f"    JNU[0] = {jnu[0]:.6e}")
                print(f"    SNU[0] (new) = {snu[0]:.6e}")
                print(f"    Error = {error:.6e}")

            # Check convergence (Fortran uses 1e-5)
            if error < 1e-5:
                converged = True
                break

        # Fortran SPECTRV uses HNU(1) for the surface flux in this regime.
        flux_hnu = hnu[0]
        flux_knu = jnu[0] / 3.0

        if debug:
            print(f"\n  Feautrier result (MAXJ=1):")
            print(f"    Iterations: {feautrier_iter + 1}")
            print(f"    Converged: {converged}")
            print(f"    HNU[0] = {hnu[0]:.6e} (what Fortran SPECTRV uses)")
            print(f"    JNU[0] = {jnu[0]:.6e}")
            print(f"    KNU[0] = JNU[0]/3 = {flux_knu:.6e} (correct physics)")
            print(
                f"    Ratio KNU/HNU = {flux_knu / flux_hnu:.1f}x"
                if flux_hnu != 0
                else "    Ratio = N/A"
            )
            print(f"    Returning HNU[0] to match Fortran SPECTRV")

        dump_flag = os.getenv("PY_DUMP_JOSH_ARRAYS") == "1"
        if dump_flag and wavelength_nm is not None:
            dump_wave = os.getenv("PY_DUMP_JOSH_ARRAYS_WAVE")
            try:
                target_wave = float(dump_wave) if dump_wave else wavelength_nm
            except ValueError:
                target_wave = wavelength_nm
            if abs(wavelength_nm - target_wave) < 1e-4:
                dump_path = os.getenv("PY_DUMP_JOSH_ARRAYS_PATH")
                if not dump_path:
                    dump_path = f"out/josh_arrays_{wavelength_nm:.6f}.npz"
                np.savez(
                    dump_path,
                    taunu=taunu,
                    snubar=snubar,
                    alpha=alpha,
                )
                print(f"  -> Dumped JOSH arrays to {dump_path}")

        return flux_hnu
    elif needs_iteration and maxj == 1:
        if debug:
            print("\nMAXJ=1: Skipping Feautrier; using XS/CH flux directly")

    if debug:
        if xs.size > 0:
            print(
                f"DEBUG: Initialized xs, xs.size={xs.size}, xs[0]={xs[0]:.8E}, maxj={maxj}",
                flush=True,
            )
        else:
            print(
                f"DEBUG: Initialized xs, xs.size={xs.size}, xs[0]=N/A, maxj={maxj}",
                flush=True,
            )
        print(
            f"DEBUG: After iteration check, xs.size={xs.size}, about to calculate flux",
            flush=True,
        )
    if debug:
        if num_iterations > 0:
            print(f"\nFlux calculation (IFSURF=0): Iterated XS (lines present)")
            print(f"  Iterations performed: {num_iterations}")
        else:
            if needs_iteration:
                print(
                    f"\nFlux calculation (IFSURF=0): No iteration (surface flux, XSBAR/XALPHA path)"
                )
            else:
                print(f"\nFlux calculation (IFSURF=0): No iteration (continuum-only)")
            print(f"  Iterations performed: {num_iterations}")
        print(f"  XS[0] (before iteration): {xsbar[0]:.8E}")
        print(f"  XS[0] (after iteration): {xs[0]:.8E}")
        print(f"  XS[-1] (after iteration): {xs[-1]:.8E}")
        print(f"  XS min/max: {xs.min():.8E} / {xs.max():.8E}")
        print(f"  MAXJ = {maxj}")
        # CRITICAL DEBUG: Check if XS is using modified XSBAR
        if "xsbar_modified" in locals():
            print(f"  XSBAR_MODIFIED[0] = {xsbar_modified[0]:.8E}")
            print(
                f"  XS[0] / XSBAR_MODIFIED[0] = {xs[0] / xsbar_modified[0]:.8E}"
                if xsbar_modified[0] != 0
                else "  XS[0] / XSBAR_MODIFIED[0] = N/A"
            )
        # Print first 10 and last 10 XS values for detailed comparison
        print(f"\n  XS array (first 10, after iteration):")
        for i in range(min(10, len(xs))):
            contrib = CK_WEIGHTS[i] * xs[i] if i < len(CK_WEIGHTS) else 0
            print(
                f"    XS[{i}] = {xs[i]:.8E}, CK[{i}] = {CK_WEIGHTS[i]:.8E}, contrib = {contrib:.8E}"
            )
        if len(xs) > 20:
            print(f"  XS array (last 10, after iteration):")
            for i in range(max(10, len(xs) - 10), len(xs)):
                contrib = CK_WEIGHTS[i] * xs[i] if i < len(CK_WEIGHTS) else 0
                print(
                    f"    XS[{i}] = {xs[i]:.8E}, CK[{i}] = {CK_WEIGHTS[i]:.8E}, contrib = {contrib:.8E}"
                )

    # Targeted debug for 311-321 nm outliers: capture XS/XSBAR/XALPHA/MAXJ
    if debug and debug_label and "FLUX_TOTAL_" in debug_label:
        try:
            wl_val = float(debug_label.replace("FLUX_TOTAL_", ""))
        except ValueError:
            wl_val = None
        if wl_val is not None:
            outlier_wls = {
                320.973013,
                311.304157,
                317.130618,
                315.903591,
                315.904644,
                317.122162,
                311.305195,
                311.303120,
                319.494605,
                320.974083,
                317.131676,
                320.979432,
                320.978363,
                320.971943,
                311.302082,
                315.910962,
                320.975153,
                317.121105,
                319.493540,
                320.977293,
            }
            if any(abs(wl_val - dwl) < 0.001 for dwl in outlier_wls):
                print(
                    "DEBUG PY XS311-321:",
                    f"WL={wl_val:.8f}",
                    f"XS0={xs[0]:.8E}",
                    f"XSBAR0={xsbar[0]:.8E}",
                    f"XALPHA0={xalpha[0]:.8E}",
                    f"MAXJ={maxj}",
                    flush=True,
                )

    # For surface flux (HNU), Fortran uses CH weights at label 60.
    # CK weights are used for KNU in the IFSURF=0 branch (label 50).
    # This solver returns HNU, so use CH_WEIGHTS to match Fortran.
    flux_weights = np.asarray(CH_WEIGHTS, dtype=xs.dtype)

    # CRITICAL DEBUG: Print BEFORE flux calculation to verify we reach this point
    # Force flush to ensure output appears immediately
    if debug:
        print(f"\n{'='*70}", flush=True)
        print(f"DEBUG: About to calculate flux in solve_josh_flux", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"  debug_label: {debug_label}", flush=True)
        print(f"  xs.size: {xs.size}", flush=True)
        if xs.size > 0:
            print(f"  xs[0]: {xs[0]:.8E}", flush=True)
            print(f"  xs[-1]: {xs[-1]:.8E}", flush=True)
            print(f"  xs all zeros? {np.all(xs == 0.0)}", flush=True)
        else:
            print(f"  xs is empty", flush=True)
        print(f"  flux_weights.size: {flux_weights.size}", flush=True)
        if len(CH_WEIGHTS) > 0:
            print(f"  CH_WEIGHTS[0]: {CH_WEIGHTS[0]:.8E}", flush=True)
            print(f"  CH_WEIGHTS sum: {np.sum(CH_WEIGHTS):.8E}", flush=True)
        else:
            print(f"  CH_WEIGHTS is empty", flush=True)
        print(f"{'='*70}\n", flush=True)

    flux = float(np.dot(flux_weights, xs))

    # Also print flux result immediately
    if debug:
        print(f"DEBUG: Flux calculated = {flux:.8E}", flush=True)

    # CRITICAL DEBUG: Check if flux is zero
    # This will help diagnose why all fluxes are zero
    # Check for exactly zero OR very small values
    # Also print for first few calls to verify function is working
    if debug:
        if not hasattr(solve_josh_flux, "_call_count"):
            solve_josh_flux._call_count = 0
        solve_josh_flux._call_count += 1

        # Print debug for first 3 calls OR if flux is zero
        should_debug = (
            (solve_josh_flux._call_count <= 3)
            or (flux == 0.0)
            or abs(flux) < 1e-50
            or np.isnan(flux)
            or np.isinf(flux)
        )
    else:
        should_debug = False

    if should_debug:
        print(f"\n{'='*70}")
        print(
            f"CRITICAL: Flux calculation in solve_josh_flux (call #{solve_josh_flux._call_count})!"
        )
        print(f"{'='*70}")
        print(f"  debug_label: {debug_label}")
        print(f"  flux = {flux:.8E}")
        print(f"  XS size: {xs.size}")
        print(f"  XS min/max: {xs.min():.8E} / {xs.max():.8E}")
        print(f"  XS all zeros? {np.all(xs == 0.0)}")
        print(f"  XS first 5: {xs[:5] if xs.size >= 5 else xs}")
        print(f"  CH_WEIGHTS size: {flux_weights.size}")
        print(
            f"  CH_WEIGHTS min/max: {flux_weights.min():.8E} / {flux_weights.max():.8E}"
        )
        print(f"  CH_WEIGHTS sum: {np.sum(flux_weights):.8E}")
        print(
            f"  CH_WEIGHTS first 5: {flux_weights[:5] if flux_weights.size >= 5 else flux_weights}"
        )
        print(f"  dot product components (first 5):")
        for i in range(min(5, len(xs))):
            print(
                f"    CH[{i}] * XS[{i}] = {flux_weights[i]:.8E} * {xs[i]:.8E} = {flux_weights[i] * xs[i]:.8E}"
            )
        print(f"  TAUNU size: {taunu.size if taunu.size > 0 else 0}")
        if taunu.size > 0:
            print(f"  TAUNU[0] = {taunu[0]:.8E}")
            print(f"  TAUNU[-1] = {taunu[-1]:.8E}")
        print(f"  SNUBAR size: {snubar.size}")
        if snubar.size > 0:
            print(f"  SNUBAR[0] = {snubar[0]:.8E}")
        print(f"  MAXJ = {maxj}")
        print(f"  ACONT size: {acont.size}")
        if acont.size > 0:
            print(f"  ACONT[0] = {acont[0]:.8E}")
        print(f"{'='*70}\n")

    # CRITICAL DEBUG: Check flux calculation when MAXJ=1
    if debug and maxj == 1:
        print(f"\n{'='*70}")
        print(f"CRITICAL: MAXJ=1 flux calculation")
        print(f"{'='*70}")
        print(f"  MAXJ = {maxj}")
        print(f"  TAUNU[0] = {taunu[0]:.8E}" if taunu.size > 0 else "  TAUNU is empty")
        print(f"  XTAU_GRID[-1] = {XTAU_GRID[-1]:.8E}")
        print(f"  XSBAR[0] = {xsbar[0]:.8E}")
        print(f"  XSBAR[-1] = {xsbar[-1]:.8E}")
        print(f"  XS[0] = {xs[0]:.8E}")
        print(f"  XS[-1] = {xs[-1]:.8E}")
        print(f"  CK_WEIGHTS[0] = {flux_weights[0]:.8E}")
        print(f"  CK_WEIGHTS[-1] = {flux_weights[-1]:.8E}")
        print(f"  Flux = {flux:.8E}")
        print(f"  SNUBAR[0] = {snubar[0]:.8E}")
        print(f"  SCONT[0] = {scont[0]:.8E}")
        print(f"  ALINE[0] = {aline[0]:.8E}")
        print(f"  ACONT[0] = {acont[0]:.8E}")
        print(f"{'='*70}\n")

    if debug:
        print(f"\nFinal flux calculation:")
        print(f"  CK_WEIGHTS sum = {np.sum(flux_weights):.8E}")
        print(f"  CK_WEIGHTS[0] = {flux_weights[0]:.8E}")
        print(f"  CK_WEIGHTS[-1] = {flux_weights[-1]:.8E}")
        print(f"  Flux = dot(CK_WEIGHTS, XS) = {flux:.8E}")
        print(f"{'='*70}\n")

    # NOTE: 4π correction was temporarily applied but caused flux > continuum (physically wrong)
    # Investigation needed: Why does Python HNU(1) differ from Fortran?
    # Original investigation showed dividing by 4π gave 0.924× ratio at 490nm,
    # but full spectrum shows 4.677× error and flux > continuum (impossible).
    #
    # The Fortran code (spectrv.for) doesn't apply any empirical flux corrections after
    # the JOSH solver - it uses the raw SURF(1) value directly. We should do the same.
    #
    # TODO: Investigate root cause of flux discrepancy without applying ad-hoc corrections.

    dump_flag = os.getenv("PY_DUMP_JOSH_ARRAYS") == "1"
    if dump_flag and wavelength_nm is not None:
        dump_wave = os.getenv("PY_DUMP_JOSH_ARRAYS_WAVE")
        try:
            target_wave = float(dump_wave) if dump_wave else wavelength_nm
        except ValueError:
            target_wave = wavelength_nm
        if abs(wavelength_nm - target_wave) < 1e-4:
            dump_path = os.getenv("PY_DUMP_JOSH_ARRAYS_PATH")
            if not dump_path:
                dump_path = f"out/josh_arrays_{wavelength_nm:.6f}.npz"
            np.savez(
                dump_path,
                taunu=taunu,
                snubar=snubar,
                alpha=alpha,
                xs=xs,
            )
            print(f"  -> Dumped JOSH arrays to {dump_path}")
            print("  XS[0:10] = " + " ".join(f"{val:.8E}" for val in xs[:10]))
            print(f"  -> Dumped JOSH arrays to {dump_path}")

    return flux
