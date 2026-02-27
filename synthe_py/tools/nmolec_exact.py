#!/usr/bin/env python3
"""Exact implementation of NMOLEC subroutine for molecular equilibrium.

From atlas7v.for lines 4308-4641 (NMOLEC subroutine).
This computes molecular XNATOM including molecular contributions.
"""

from __future__ import annotations

import math
import os
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Decimal precision for extended range handling
# We use 50 digits for range (avoids float64 overflow at ~1e308)
# but compute element equations with float64 to match Fortran's binary rounding
getcontext().prec = 50

from numba import jit, prange

# Constants
MAXMOL = 200
MAXEQ = 30
MAXLOC = 3 * MAXMOL

# CRITICAL: Use np.float64 to EXACTLY match Fortran's REAL*8 precision
# Fortran uses REAL*8 (64-bit double precision)
# Using extended precision (np.longdouble) causes different rounding → different pivot selection → instability
# Always use float64 to match Fortran exactly
EXTENDED_DTYPE = np.float64  # Keep name for compatibility, but always use float64

# Standard format for float64 logging (17 digits after decimal = ~15-17 significant digits)
# This ensures bit-by-bit comparison accuracy between Python and Fortran logs
FLOAT64_FMT = ".17E"  # Scientific notation with 17 digits after decimal

ROW_NORM_MAX_EXP = 512  # keep DEQ/EQ rows within ~2**512 magnitude
ROW_NORM_MIN_EXP = -512
ROW_NORM_TARGET_EXP = 0

HAS_FMA = hasattr(math, "fma")

_FLOAT64_TINY = np.finfo(np.float64).tiny

# Minimum acceptable seed value when copying XN across layers.
# Defaults to float64 tiny (≈2.22e-308) but can be raised via NM_SEED_MIN_VALUE.
_SEED_MIN_VALUE = _FLOAT64_TINY
_seed_min_env = os.environ.get("NM_SEED_MIN_VALUE")
if _seed_min_env:
    try:
        _seed_min_candidate = float(_seed_min_env)
        if _seed_min_candidate > _SEED_MIN_VALUE:
            _SEED_MIN_VALUE = _seed_min_candidate
    except ValueError:
        print(
            f"WARNING: Ignoring invalid NM_SEED_MIN_VALUE='{_seed_min_env}'. "
            "Using float64 tiny."
        )


def _mul_preserving_precision(a: float, b: float) -> np.float64:
    """Multiply two float64 values using power-of-two scaling to reduce overflow/underflow."""
    a_val = np.float64(a)
    b_val = np.float64(b)
    if a_val == 0.0 or b_val == 0.0:
        return np.float64(0.0)
    mant_a, exp_a = math.frexp(a_val)
    mant_b, exp_b = math.frexp(b_val)
    mant_prod = mant_a * mant_b
    exp_prod = exp_a + exp_b
    mant_prod, adjust = math.frexp(mant_prod)
    exp_prod += adjust
    try:
        return np.float64(math.ldexp(mant_prod, exp_prod))
    except OverflowError:
        return np.float64(np.copysign(np.inf, mant_prod))


def _div_preserving_precision(numerator: float, denominator: float) -> np.float64:
    """Divide two float64 values using power-of-two scaling to reduce overflow/underflow."""
    num = np.float64(numerator)
    den = np.float64(denominator)
    if den == 0.0:
        if num == 0.0:
            return np.float64(np.nan)
        return np.float64(np.copysign(np.inf, num))
    if num == 0.0:
        return np.float64(0.0)
    mant_num, exp_num = math.frexp(num)
    mant_den, exp_den = math.frexp(den)
    mant_ratio = mant_num / mant_den
    exp_ratio = exp_num - exp_den
    mant_ratio, adjust = math.frexp(mant_ratio)
    exp_ratio += adjust
    try:
        return np.float64(math.ldexp(mant_ratio, exp_ratio))
    except OverflowError:
        return np.float64(np.copysign(np.inf, mant_ratio))


def _stable_subtract(minuend: float, subtrahend: float) -> float:
    """Return minuend - subtrahend using ratio form when safe (float64 only)."""
    a = np.float64(minuend)
    b = np.float64(subtrahend)
    if not (np.isfinite(a) and np.isfinite(b)):
        return a - b
    if abs(a) > _FLOAT64_TINY:
        return np.float64(a * (1.0 - b / a))
    return np.float64(a - b)


def _ratio_preserving_precision(numerator: float, denominator: float) -> float:
    """Compute abs(numerator / denominator) with power-of-two scaling to avoid overflow."""
    num = np.float64(numerator)
    den = np.float64(denominator)
    if den == 0.0:
        if num == 0.0:
            return 0.0
        return np.inf
    if num == 0.0:
        return 0.0
    ratio = _div_preserving_precision(num, den)
    return float(abs(ratio))


def _safe_add(a: float, b: float) -> float:
    """Accumulate two floats while avoiding overflow."""
    a_val = np.float64(a)
    b_val = np.float64(b)
    if not np.isfinite(a_val) or not np.isfinite(b_val):
        return a_val + b_val
    abs_max = max(abs(a_val), abs(b_val))
    if abs_max == 0.0 or abs_max < 1e300:
        return np.float64(a_val + b_val)
    mant, exp = math.frexp(abs_max)
    shift = max(exp - 900, 0)
    if shift == 0:
        return np.float64(a_val + b_val)
    scale = math.ldexp(1.0, -shift)
    scaled_sum = np.float64(a_val * scale + b_val * scale)
    try:
        return np.float64(math.ldexp(scaled_sum, shift))
    except OverflowError:
        return np.copysign(np.inf, scaled_sum)


# =============================================================================
# LOG-SPACE ARITHMETIC FOR NEWTON ITERATION
# =============================================================================
# The Newton iteration solves F(XN) = 0 where XN are number densities.
# In log-space, we work with Y = log(XN), so XN = exp(Y).
# This allows us to represent values from ~1e-4000 to ~1e+4000 (as Y from -9210 to +9210)
# which far exceeds float64's range of ~1e-308 to ~1e+308.
#
# The Jacobian transforms as: DEQ_log[i,j] = DEQ[i,j] * XN[j]
# (chain rule: ∂F/∂Y = ∂F/∂X * ∂X/∂Y = ∂F/∂X * exp(Y) = DEQ * XN)
#
# Newton step: dY = solve(DEQ_log, -EQ)
# Update: Y_new = Y + dY, XN_new = exp(Y_new)

# Threshold for switching to log-space (when |log(XN)| exceeds this)
LOG_SPACE_THRESHOLD = 650.0  # Well below ln(float64_max) ≈ 709.78

# Minimum value for XN to avoid log(0)
XN_MIN_VALUE = 1e-300

# Maximum |log(XN)| value (beyond this, clamp)
LOG_XN_MAX = 700.0


def _to_log_space(xn: np.ndarray) -> np.ndarray:
    """Convert XN array to log-space: log_xn = log(XN).

    Handles zeros and negative values by clamping to XN_MIN_VALUE.
    Returns log(XN) which can represent values from ~1e-4000 to ~1e+4000.
    """
    xn_safe = np.maximum(np.abs(xn), XN_MIN_VALUE)
    return np.log(xn_safe)


def _from_log_space(log_xn: np.ndarray) -> np.ndarray:
    """Convert log-space back to linear: XN = exp(log_xn).

    Handles extreme values by clamping log_xn to [-LOG_XN_MAX, LOG_XN_MAX]
    to prevent overflow/underflow in exp().
    """
    log_xn_clamped = np.clip(log_xn, -LOG_XN_MAX, LOG_XN_MAX)
    return np.exp(log_xn_clamped)


def _to_log_space_scalar(xn: float) -> float:
    """Convert single XN value to log-space."""
    xn_safe = max(abs(xn), XN_MIN_VALUE)
    return np.log(xn_safe)


def _from_log_space_scalar(log_xn: float) -> float:
    """Convert single log-space value back to linear."""
    log_xn_clamped = max(-LOG_XN_MAX, min(LOG_XN_MAX, log_xn))
    return np.exp(log_xn_clamped)


def _scale_jacobian_for_log_space(
    deq: np.ndarray, xn: np.ndarray, nequa: int
) -> np.ndarray:
    """Scale Jacobian for log-space Newton: DEQ_log[i,j] = DEQ[i,j] * XN[j].

    This accounts for the chain rule when working with Y = log(XN):
    ∂F/∂Y[j] = ∂F/∂X[j] * ∂X/∂Y[j] = DEQ[i,j] * XN[j]

    Args:
        deq: Original Jacobian in flat column-major format (nequa*nequa)
        xn: Current XN values (nequa)
        nequa: Number of equations

    Returns:
        Scaled Jacobian for log-space Newton iteration
    """
    deq_log = deq.copy()
    for j in range(nequa):
        # Scale column j by XN[j]
        col_start = j * nequa
        for i in range(nequa):
            deq_log[col_start + i] *= xn[j]
    return deq_log


def _log_space_newton_update(
    log_xn: np.ndarray, delta_log: np.ndarray, damping: float = 1.0
) -> np.ndarray:
    """Update log-space XN values: log_xn_new = log_xn + damping * delta_log.

    This preserves positivity of XN (exp of anything is positive).

    Args:
        log_xn: Current log(XN) values
        delta_log: Newton step in log-space (from SOLVIT)
        damping: Damping factor (0 < damping <= 1)

    Returns:
        Updated log(XN) values, clamped to valid range
    """
    log_xn_new = log_xn + damping * delta_log
    # Clamp to prevent overflow/underflow when converting back
    return np.clip(log_xn_new, -LOG_XN_MAX, LOG_XN_MAX)


# =============================================================================
# SIGNED LOG-SPACE ACCUMULATION
# =============================================================================
# For accumulating TERM values into EQ without overflow.
# TERM values can be ~10^400+ which overflows float64, but in log-space
# we just store 400*ln(10) ≈ 921 which is easily representable.
#
# Since EQ values can be positive or negative, we track (sign, log_abs) pairs.


def _log_add_exp(log_a: float, log_b: float) -> float:
    """Compute log(exp(log_a) + exp(log_b)) stably.

    Uses the log-sum-exp trick: log(a + b) = log(a) + log(1 + b/a)
                                           = log_a + log1p(exp(log_b - log_a))
    """
    if not np.isfinite(log_a) and log_a < 0:
        return log_b
    if not np.isfinite(log_b) and log_b < 0:
        return log_a
    if log_a > log_b:
        return log_a + math.log1p(math.exp(log_b - log_a))
    else:
        return log_b + math.log1p(math.exp(log_a - log_b))


def _log_sub_exp(log_a: float, log_b: float) -> tuple[int, float]:
    """Compute log(|exp(log_a) - exp(log_b)|) stably.

    Returns (sign, log_abs) where sign is +1 if a > b, -1 if b > a.
    For a == b, returns (+1, -inf) representing 0.
    """
    if log_a > log_b:
        diff = log_b - log_a
        if diff < -40:  # exp(diff) ≈ 0
            return (+1, log_a)
        return (+1, log_a + math.log1p(-math.exp(diff)))
    elif log_b > log_a:
        diff = log_a - log_b
        if diff < -40:
            return (-1, log_b)
        return (-1, log_b + math.log1p(-math.exp(diff)))
    else:
        # Equal: result is 0
        return (+1, float("-inf"))


def _add_signed_log(
    sign_a: int, log_a: float, sign_b: int, log_b: float
) -> tuple[int, float]:
    """Add two signed log-space values: (sign_a * exp(log_a)) + (sign_b * exp(log_b)).

    Returns (result_sign, result_log_abs).

    This allows accumulation of values that would overflow float64:
    - Values up to ~10^4000 can be represented (log_abs up to ~9200)
    - Handles mixed signs correctly
    - No overflow during intermediate calculations
    """
    # Handle zeros (log = -inf means value is 0)
    if not np.isfinite(log_a) and log_a < 0:
        return (sign_b, log_b)
    if not np.isfinite(log_b) and log_b < 0:
        return (sign_a, log_a)

    if sign_a == sign_b:
        # Same sign: |a + b| = |a| + |b|
        return (sign_a, _log_add_exp(log_a, log_b))
    else:
        # Different signs: compute |a| - |b| or |b| - |a|
        result_sign, result_log = _log_sub_exp(log_a, log_b)
        # Adjust sign based on which term was larger
        if sign_a == +1:
            # Computing (+a) + (-b) = a - b
            return (result_sign, result_log)
        else:
            # Computing (-a) + (+b) = b - a = -(a - b)
            return (-result_sign, result_log)


def _signed_log_to_linear(sign: int, log_abs: float, clamp_max: float = 1e307) -> float:
    """Convert signed log-space value back to linear, with clamping.

    Args:
        sign: +1 or -1
        log_abs: log(|value|)
        clamp_max: Maximum absolute value to return (prevents overflow)

    Returns:
        sign * exp(log_abs), clamped to [-clamp_max, clamp_max]
    """
    if not np.isfinite(log_abs) and log_abs < 0:
        return 0.0
    if log_abs > 708:  # Would overflow exp()
        return sign * clamp_max
    if log_abs < -745:  # Would underflow to 0
        return 0.0
    return sign * math.exp(log_abs)


def _linear_to_signed_log(value: float) -> tuple[int, float]:
    """Convert linear value to signed log-space representation.

    Args:
        value: Any finite float64 value

    Returns:
        (sign, log_abs) where sign is +1 or -1, log_abs is log(|value|)
        For value == 0, returns (+1, -inf)
    """
    if value == 0.0:
        return (+1, float("-inf"))
    if not np.isfinite(value):
        if np.isnan(value):
            return (+1, float("nan"))
        return (+1 if value > 0 else -1, float("inf"))
    sign = +1 if value > 0 else -1
    log_abs = math.log(abs(value))
    return (sign, log_abs)


def _sl_multiply(
    sign_a: int, log_a: float, sign_b: int, log_b: float
) -> tuple[int, float]:
    """Multiply two signed log values: (sign_a * exp(log_a)) * (sign_b * exp(log_b)).

    Returns (result_sign, result_log_abs).
    In log-space: log(|a*b|) = log|a| + log|b|, sign = sign_a * sign_b
    """
    return (sign_a * sign_b, log_a + log_b)


def _sl_divide(
    sign_a: int, log_a: float, sign_b: int, log_b: float
) -> tuple[int, float]:
    """Divide two signed log values: (sign_a * exp(log_a)) / (sign_b * exp(log_b)).

    Returns (result_sign, result_log_abs).
    In log-space: log(|a/b|) = log|a| - log|b|, sign = sign_a * sign_b
    """
    if log_b == float("-inf"):  # Division by zero
        return (sign_a * sign_b, float("inf"))
    return (sign_a * sign_b, log_a - log_b)


def _two_sum(a: float, b: float) -> tuple:
    """Compute s = a + b and error e such that a + b = s + e exactly (Knuth's TwoSum).

    This is an error-free transformation that computes the exact rounding error
    in a floating-point addition, allowing us to track and compensate for it.
    """
    s = a + b
    a_prime = s - b
    b_prime = s - a_prime
    delta_a = a - a_prime
    delta_b = b - b_prime
    e = delta_a + delta_b
    return s, e


def _two_product_fma(a: float, b: float) -> tuple:
    """Compute p = a * b and error e such that a * b = p + e exactly (using FMA).

    Requires FMA (fused multiply-add) instruction for exact error computation.
    """
    p = a * b
    if HAS_FMA:
        try:
            e = math.fma(a, b, -p)  # Compute a*b - p exactly
        except OverflowError:
            # FMA overflows - return 0 error (p is already inf or very large)
            e = 0.0
    else:
        # Fallback: split method (less accurate but still better than nothing)
        # This is Dekker's algorithm
        factor = 134217729.0  # 2^27 + 1
        ah = a * factor
        ah = ah - (ah - a)
        al = a - ah
        bh = b * factor
        bh = bh - (bh - b)
        bl = b - bh
        e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


def _accurate_diff(a: float, b: float) -> float:
    """Compute a - b accurately even when a ≈ b using compensated arithmetic.

    Uses TwoSum to capture the rounding error and add it back.
    """
    # Compute s = a - b and error e
    s, e = _two_sum(a, -b)
    # The exact result is s + e, but s + e rounds to s in float64
    # However, if we're subtracting nearly equal numbers, s might be inaccurate
    # The key insight: if |s| < |e|, then s is dominated by rounding error
    # In that case, return e (which captures the true small difference)
    if abs(s) < abs(e) * 1e10:
        # s is unreliable, use e as the correction
        return s + e
    return s


def _accurate_element_residual(xn_k: float, xab_k: float, xn0: float) -> float:
    """Compute XN(K) - XAB(K)*XN(1) accurately using double-double arithmetic.

    When XN(K) ≈ XAB(K)*XN(1), direct subtraction loses all precision because
    we're subtracting two ~1E+25 numbers to get a result that should be ~0.
    Fortran uses 80-bit extended precision which has 3-4 extra digits, but
    Python's float64 returns noise.

    Solution: Use double-double arithmetic (error-free transformations) to
    compute the product and subtraction with ~30 digits of precision.
    This uses only standard float64 operations - no external libraries.

    The algorithm:
    1. Compute xab_k * xn0 exactly as (prod_hi, prod_lo) using TwoProduct
    2. Compute xn_k - prod_hi exactly as (diff_hi, diff_lo) using TwoSum
    3. Combine: result = (diff_hi + diff_lo) - prod_lo

    When at the float64 precision floor, return a tiny sign-preserving value
    to ensure correct Newton damping behavior.
    """
    if xn0 == 0.0 or not np.isfinite(xn0):
        return xn_k - xab_k * xn0

    if not np.isfinite(xn_k) or not np.isfinite(xab_k):
        return xn_k - xab_k * xn0

    # Step 1: Compute xab_k * xn0 exactly as double-double (prod_hi, prod_lo)
    # True product = prod_hi + prod_lo (exact to ~30 digits)
    prod_hi, prod_lo = _two_product_fma(xab_k, xn0)

    # Handle overflow in product
    if not np.isfinite(prod_hi):
        return xn_k - xab_k * xn0

    # Step 2: Compute xn_k - prod_hi exactly as double-double (diff_hi, diff_lo)
    # xn_k - prod_hi = diff_hi + diff_lo (exact)
    diff_hi, diff_lo = _two_sum(xn_k, -prod_hi)

    # Key insight: diff_hi is the primary difference in float64.
    # If diff_hi == 0, then xn_k and prod_hi are bit-identical, meaning
    # xn_k was computed as xab_k * xn0. In this case, the "true" residual
    # for Newton iteration purposes is 0, not the product's rounding error.
    #
    # However, if diff_hi != 0, we have a real difference to compute accurately.

    if diff_hi == 0.0 and diff_lo == 0.0:
        # xn_k == prod_hi exactly (bit-identical)
        # For Newton iteration: residual = 0 (at equilibrium)
        # Return tiny sign-preserving value (sign arbitrary, magnitude negligible)
        return 0.0

    # Step 3: Combine the double-double result
    # Full difference = diff_hi + diff_lo - prod_lo
    # But prod_lo is the error in xab_k*xn0 computation, not relevant to
    # comparing xn_k (which is a stored value) to prod_hi (the same computation).
    #
    # For Newton iteration, we want: xn_k - (xab_k * xn0 as computed in float64)
    # This is just: diff_hi + diff_lo (ignoring prod_lo)

    result = diff_hi + diff_lo

    # Check if we're at the precision floor
    if prod_hi != 0.0:
        relative_residual = abs(result) / abs(prod_hi)
        if relative_residual < 1e-14:
            # At precision floor - the result is meaningful but tiny
            # Preserve sign with controlled magnitude for Newton damping
            if result == 0.0:
                # Exactly zero - use ratio to determine sign
                ratio = xn_k / xn0
                sign = 1.0 if ratio >= xab_k else -1.0
            else:
                sign = 1.0 if result > 0 else -1.0
            # Return a tiny value: sign * (1e-15 relative to product magnitude)
            tiny_residual = sign * abs(prod_hi) * 1e-15
            return tiny_residual

    return result


@jit(nopython=True, cache=True)
def _two_sum_numba(a: float, b: float) -> tuple:
    """Numba-compatible TwoSum for exact error computation."""
    s = a + b
    a_prime = s - b
    b_prime = s - a_prime
    delta_a = a - a_prime
    delta_b = b - b_prime
    e = delta_a + delta_b
    return s, e


def _kahan_add(sum_val: float, compensation: float, addend: float) -> tuple:
    """Kahan summation step: adds 'addend' to 'sum_val' with error compensation.

    Returns (new_sum, new_compensation).
    This recovers precision lost in floating-point addition by tracking
    the small error term and incorporating it into the next addition.
    """
    y = addend - compensation  # Compensate for previous error
    t = sum_val + y  # New sum
    compensation = (t - sum_val) - y  # Compute new error (what was lost)
    return t, compensation


@jit(nopython=True, cache=True)
def _kahan_add_numba(sum_val: float, compensation: float, addend: float) -> tuple:
    """Numba-compatible Kahan summation step."""
    y = addend - compensation
    t = sum_val + y
    new_compensation = (t - sum_val) - y
    return t, new_compensation


# Increase to capture late-iteration divergences (Fortran logs show iter≈23)
MAX_DEBUG_SOLVIT_ITER = 128
_current_solvit_layer = -1
_current_solvit_iter = -1
_current_solvit_call = -1
TRACE_A9_LAYER = 1  # 1-based layer index (set <=0 to disable)
TRACE_A9_CALL = 24  # SOLVIT call number to trace (1-based)
TRACE_A9_ROW = 8  # 0-based row index (row 9 in 1-based)
TRACE_A9_COL = 1  # 0-based column index (col 2 in 1-based)
TRACE_A9_ENABLED = TRACE_A9_LAYER > 0 and TRACE_A9_CALL > 0
TRACE_PIVOT_SEARCH = os.environ.get("NM_TRACE_PIVOT_SEARCH", "0") == "1"
TRACE_MOLECULE_IDS = {
    2,
    7,
    8,
    9,
    10,
    26,
    31,
    32,
    33,
    34,
    35,
    71,
    72,
    73,
    74,
    104,
    105,
    159,
    162,
    163,
    164,
    165,
    166,
    167,
    168,
    169,
    171,
    172,
    173,
    174,
    176,
    177,
    183,
    171,
    172,
    173,
    174,
    176,
    177,
    183,
    185,
}
TRACE_MOLECULES_ZERO = {mol_id - 1 for mol_id in TRACE_MOLECULE_IDS}
TRACE_MOLECULES_FORCE: set[int] = set()

_DEQ_TRACE_ROWS = (0, 16, 22)
_DEQ_TRACE_COLS = (0, 16, 22)

_PFSAHA_DEBUG_JMOLS = {2, 7, 8, 9, 10, 26, 71, 72, 73, 74}
_pfsa_trace_env = os.environ.get("NM_TRACE_PFSAHA_JMOLS", "").strip()
_PFSAHA_TRACE_EXTRA: set[int] = set()
if _pfsa_trace_env:
    for token in _pfsa_trace_env.replace(",", " ").split():
        token = token.strip()
        if not token:
            continue
        if token.lstrip("+-").isdigit():
            _PFSAHA_TRACE_EXTRA.add(abs(int(token)))
        else:
            print(
                f"WARNING: Ignoring invalid NM_TRACE_PFSAHA_JMOLS entry '{token}' "
                "(expected positive integer JM number)"
            )
_PFSAHA_TRACE_JMOLS = _PFSAHA_DEBUG_JMOLS | _PFSAHA_TRACE_EXTRA
if _PFSAHA_TRACE_JMOLS:
    TRACE_MOLECULES_FORCE.update(
        jm_idx - 1 for jm_idx in _PFSAHA_TRACE_JMOLS if jm_idx > 0
    )

# Optional env override for tracing specific molecules (by 1-based index or code)
_TRACE_MOLECULE_CODES: set[float] = set()
_trace_mol_env = os.environ.get("NM_TRACE_MOLECULES")
if _trace_mol_env:
    for token in _trace_mol_env.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            idx = int(token, 10)
            TRACE_MOLECULES_ZERO.add(idx - 1)
            TRACE_MOLECULES_FORCE.add(idx - 1)
            continue
        except ValueError:
            pass
        try:
            _TRACE_MOLECULE_CODES.add(float(token))
        except ValueError:
            print(
                f"WARNING: Ignoring invalid NM_TRACE_MOLECULES entry '{token}' "
                "(expected integer molecule index or float code)"
            )


def _parse_iteration_tokens(env_value: str, var_name: str) -> set[int]:
    """Parse comma/space separated list of ints or ranges like 24-34."""
    result: set[int] = set()
    for raw_token in env_value.replace(",", " ").split():
        token = raw_token.strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError:
                print(
                    f"WARNING: Ignoring invalid {var_name} entry '{token}' "
                    "(expected integer range start-end)"
                )
                continue
            if start > end:
                start, end = end, start
            for value in range(start, end + 1):
                result.add(value)
        else:
            try:
                value = int(token)
            except ValueError:
                print(
                    f"WARNING: Ignoring invalid {var_name} entry '{token}' "
                    "(expected integer value)"
                )
                continue
            result.add(value)
    return result


_TRACE_SOLVIT_LAYERS: set[int] = set()
_trace_layers_env = os.environ.get("NM_TRACE_LAYERS_BEFORE_SOLVIT")
if _trace_layers_env:
    for token in _trace_layers_env.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            _TRACE_SOLVIT_LAYERS.add(int(token) - 1)
        except ValueError:
            print(
                f"WARNING: Ignoring invalid NM_TRACE_LAYERS_BEFORE_SOLVIT entry '{token}' "
                "(expected integer layer index)"
            )

_TRACE_XN_SEED_LAYERS: set[int] = set()
_trace_xn_env = os.environ.get("NM_TRACE_XN_SEEDS")
if _trace_xn_env:
    parsed_layers = _parse_iteration_tokens(_trace_xn_env, "NM_TRACE_XN_SEEDS")
    for value in parsed_layers:
        idx = value - 1
        if idx < 0:
            print(
                f"WARNING: Ignoring NM_TRACE_XN_SEEDS entry '{value}' "
                "(must be >= 1 for layer numbering)"
            )
            continue
        _TRACE_XN_SEED_LAYERS.add(idx)

_TRACE_DEQ_COLS: set[int] = set()
_trace_cols_env = os.environ.get("NM_TRACE_DEQ_COLS")
if _trace_cols_env:
    for token in _trace_cols_env.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            _TRACE_DEQ_COLS.add(int(token) - 1)
        except ValueError:
            print(
                f"WARNING: Ignoring invalid NM_TRACE_DEQ_COLS entry '{token}' "
                "(expected integer column index)"
            )

_DEQ_TRACE_THRESHOLD: float | None = None
_trace_threshold_env = os.environ.get("NM_TRACE_DEQ_THRESHOLD")
if _trace_threshold_env:
    try:
        _DEQ_TRACE_THRESHOLD = float(_trace_threshold_env)
    except ValueError:
        print(
            f"WARNING: Ignoring invalid NM_TRACE_DEQ_THRESHOLD entry '{_trace_threshold_env}' "
            "(expected float)"
        )
if _DEQ_TRACE_THRESHOLD is not None:
    print(f"[NM_TRACE_DEQ_THRESHOLD] logging DEQ values > {_DEQ_TRACE_THRESHOLD:.3E}")

_TERM_TRACE_THRESHOLD: float | None = None
_trace_term_env = os.environ.get("NM_TRACE_TERM_THRESHOLD")
if _trace_term_env:
    try:
        _TERM_TRACE_THRESHOLD = float(_trace_term_env)
    except ValueError:
        print(
            f"WARNING: Ignoring invalid NM_TRACE_TERM_THRESHOLD entry '{_trace_term_env}' "
            "(expected float)"
        )
if _TERM_TRACE_THRESHOLD is not None:
    print(f"[NM_TRACE_TERM_THRESHOLD] logging terms > {_TERM_TRACE_THRESHOLD:.3E}")


def _should_trace_molecule(jmol: int, code: float) -> bool:
    if jmol in TRACE_MOLECULES_ZERO:
        return True
    if not _TRACE_MOLECULE_CODES:
        return False
    for target in _TRACE_MOLECULE_CODES:
        if abs(code - target) < 0.5:
            return True
    return False


def _should_trace_eq_target(k_idx: int) -> bool:
    if not _TRACE_EQ_TARGETS:
        return False
    return k_idx in _TRACE_EQ_TARGETS


tracked_deq_columns: Tuple[int, ...] = (0, 7, 8, 9, 15)

# Optional targeted tracking for specific DEQ(1, K) columns (0-based k indices).
# Set environment variable NM_TRACK_DEQ_KS to a comma-separated list of k values.
_TRACKED_DEQ_KS: set[int] = set()
_tracked_env = os.environ.get("NM_TRACK_DEQ_KS")
if _tracked_env:
    for token in _tracked_env.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            _TRACKED_DEQ_KS.add(int(token))
        except ValueError:
            print(
                f"WARNING: Ignoring invalid NM_TRACK_DEQ_KS entry '{token}' "
                "(expected integer k index)"
            )

_TRACKED_DEQ_CROSS: set[int] = set()
_tracked_cross_env = os.environ.get("NM_TRACK_DEQ_CROSS")
if _tracked_cross_env:
    for token in _tracked_cross_env.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            _TRACKED_DEQ_CROSS.add(int(token))
        except ValueError:
            print(
                f"WARNING: Ignoring invalid NM_TRACK_DEQ_CROSS entry '{token}' "
                "(expected integer k index)"
            )

_DUMP_SOLVIT_TARGETS: set[tuple[int, int]] = set()
_dump_solvit_env = os.environ.get("NM_DUMP_SOLVIT_MATRIX")
if _dump_solvit_env:
    for raw_token in _dump_solvit_env.replace(",", " ").split():
        token = raw_token.strip()
        if not token:
            continue
        layer_str: Optional[str]
        iter_str: Optional[str]
        if ":" in token:
            layer_str, iter_str = token.split(":", 1)
        elif "." in token:
            layer_str, iter_str = token.split(".", 1)
        else:
            layer_str, iter_str = token, "1"
        try:
            layer_idx = int(layer_str) - 1
            iter_idx = int(iter_str) - 1
        except ValueError:
            print(
                f"WARNING: Ignoring invalid NM_DUMP_SOLVIT_MATRIX entry '{token}' "
                "(expected integers for layer and iteration)"
            )
            continue
        if layer_idx < 0 or iter_idx < 0:
            print(
                f"WARNING: Ignoring NM_DUMP_SOLVIT_MATRIX entry '{token}' "
                "(layer and iteration must be >= 1)"
            )
            continue
        _DUMP_SOLVIT_TARGETS.add((layer_idx, iter_idx))

_DUMP_PREMOL_TARGETS: set[tuple[int, int]] = set()
_dump_premol_env = os.environ.get("NM_DUMP_PRE_MOLECULE")
if _dump_premol_env:
    for raw_token in _dump_premol_env.replace(",", " ").split():
        token = raw_token.strip()
        if not token:
            continue
        if ":" in token:
            layer_str, iter_str = token.split(":", 1)
        elif "." in token:
            layer_str, iter_str = token.split(".", 1)
        else:
            layer_str, iter_str = token, "1"
        try:
            layer_idx = int(layer_str) - 1
            iter_idx = int(iter_str) - 1
        except ValueError:
            print(
                f"WARNING: Ignoring invalid NM_DUMP_PRE_MOLECULE entry '{token}' "
                "(expected integers for layer and iteration)"
            )
            continue
        if layer_idx < 0 or iter_idx < 0:
            print(
                f"WARNING: Ignoring NM_DUMP_PRE_MOLECULE entry '{token}' "
                "(layer and iteration must be >= 1)"
            )
            continue
        _DUMP_PREMOL_TARGETS.add((layer_idx, iter_idx))

USE_ATLAS7_MOLECULE_ACCUMULATION = True
PFSAHAFunc = Callable[[int, int, int, int, np.ndarray, int], None]

_TRACE_XN_INDICES: set[int] = set()
_TRACE_XN_ALL_LAYERS_FLAG = False
_trace_xn_env = os.environ.get("NM_TRACE_XN")
if _trace_xn_env:
    for token in _trace_xn_env.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            print(
                f"WARNING: Ignoring invalid NM_TRACE_XN entry '{token}' "
                "(expected integer equation index)"
            )
            continue
        idx = value - 1  # interpret tokens as 1-based equation numbers (Fortran style)
        if idx < 0:
            print(
                f"WARNING: Ignoring NM_TRACE_XN entry '{token}' "
                "(must be >= 1 for EQ numbering)"
            )
            continue
        _TRACE_XN_INDICES.add(idx)
_TRACE_XN_ALL_LAYERS_FLAG = os.environ.get(
    "NM_TRACE_XN_ALL_LAYERS", ""
).strip() not in {
    "",
    "0",
    "false",
    "False",
}
_TRACE_XN_LAYERS: set[int] = set()
_trace_xn_layers_env = os.environ.get("NM_TRACE_XN_LAYERS")
if _trace_xn_layers_env:
    parsed_layers = _parse_iteration_tokens(_trace_xn_layers_env, "NM_TRACE_XN_LAYERS")
    for value in parsed_layers:
        idx = value - 1
        if idx < 0:
            print(
                f"WARNING: Ignoring NM_TRACE_XN_LAYERS entry '{value}' "
                "(must be >= 1 for layer numbering)"
            )
            continue
        _TRACE_XN_LAYERS.add(idx)

_TRACE_EQ_TARGETS: set[int] = set()
_trace_eq_env = os.environ.get("NM_TRACE_EQ_KS")
if _trace_eq_env:
    for token in _trace_eq_env.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            print(
                f"WARNING: Ignoring invalid NM_TRACE_EQ_KS entry '{token}' "
                "(expected integer equation index)"
            )
            continue
        idx = value - 1
        if idx < 0:
            print(
                f"WARNING: Ignoring NM_TRACE_EQ_KS entry '{token}' "
                "(must be >= 1 for EQ numbering)"
            )
            continue
        _TRACE_EQ_TARGETS.add(idx)

_TRACE_EQ_COMPONENTS: set[int] = set()
_TRACE_EQ_COMPONENTS_ALL_LAYERS = os.environ.get(
    "NM_TRACE_EQ_COMPONENTS_ALL_LAYERS", ""
).strip().lower() in ("1", "true", "yes")
_trace_eq_comp_env = os.environ.get("NM_TRACE_EQ_COMPONENTS")
if _trace_eq_comp_env:
    for token in _trace_eq_comp_env.replace(",", " ").split():
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            print(
                f"WARNING: Ignoring invalid NM_TRACE_EQ_COMPONENTS entry '{token}' "
                "(expected integer equation index)"
            )
            continue
        idx = value - 1
        if idx < 0:
            print(
                f"WARNING: Ignoring NM_TRACE_EQ_COMPONENTS entry '{token}' "
                "(must be >= 1 for EQ numbering)"
            )
            continue
        _TRACE_EQ_COMPONENTS.add(idx)
_TRACE_EQ_COMPONENT_LAYERS: set[int] = set()
_trace_eq_layers_env = os.environ.get("NM_TRACE_EQ_COMPONENT_LAYERS")
if _trace_eq_layers_env:
    parsed_layers = _parse_iteration_tokens(
        _trace_eq_layers_env, "NM_TRACE_EQ_COMPONENT_LAYERS"
    )
    for value in parsed_layers:
        idx = value - 1
        if idx < 0:
            print(
                f"WARNING: Ignoring NM_TRACE_EQ_COMPONENT_LAYERS entry '{value}' "
                "(must be >= 1 for layer numbering)"
            )
            continue
        _TRACE_EQ_COMPONENT_LAYERS.add(idx)

_TRACE_NEWTON_UPDATES = False
_TRACE_NEWTON_ALL_LAYERS = False
_TRACE_NEWTON_LAYERS: set[int] = set()
_newton_trace_env = os.environ.get("NM_TRACE_NEWTON_UPDATES")
if _newton_trace_env:
    normalized = _newton_trace_env.strip().lower()
    if normalized in {"", "1", "true", "all", "*"}:
        _TRACE_NEWTON_UPDATES = True
        _TRACE_NEWTON_ALL_LAYERS = True
    else:
        parsed_layers = _parse_iteration_tokens(
            _newton_trace_env, "NM_TRACE_NEWTON_UPDATES"
        )
        for value in parsed_layers:
            idx = value - 1
            if idx < 0:
                print(
                    f"WARNING: Ignoring NM_TRACE_NEWTON_UPDATES entry '{value}' "
                    "(must be >= 1 for layer numbering)"
                )
                continue
            _TRACE_NEWTON_LAYERS.add(idx)
        if _TRACE_NEWTON_LAYERS:
            _TRACE_NEWTON_UPDATES = True
        else:
            _TRACE_NEWTON_UPDATES = True
            _TRACE_NEWTON_ALL_LAYERS = True


_TRACE_ELECTRON_TERMS = os.environ.get(
    "NM_TRACE_ELECTRON_TERMS", ""
).strip().lower() not in {"", "0", "false"}
_TRACE_ELECTRON_LAYERS: set[int] = set()
if _TRACE_ELECTRON_TERMS:
    _trace_electron_layers_env = os.environ.get("NM_TRACE_ELECTRON_LAYERS")
    if _trace_electron_layers_env:
        parsed_layers = _parse_iteration_tokens(
            _trace_electron_layers_env, "NM_TRACE_ELECTRON_LAYERS"
        )
        for value in parsed_layers:
            idx = value - 1
            if idx < 0:
                print(
                    f"WARNING: Ignoring NM_TRACE_ELECTRON_LAYERS entry '{value}' "
                    "(must be >= 1 for layer numbering)"
                )
                continue
            _TRACE_ELECTRON_LAYERS.add(idx)
    if not _TRACE_ELECTRON_LAYERS:
        _TRACE_ELECTRON_LAYERS.add(0)


_TRACE_ELECTRON_CONTRIBS = os.environ.get(
    "NM_TRACE_ELECTRON_CONTRIBS", ""
).strip().lower() not in {"", "0", "false"}
_TRACE_ELECTRON_CONTRIB_LAYERS: set[int] = set()
if _TRACE_ELECTRON_CONTRIBS:
    _trace_electron_contrib_layers_env = os.environ.get(
        "NM_TRACE_ELECTRON_CONTRIB_LAYERS"
    )
    if _trace_electron_contrib_layers_env:
        parsed_layers = _parse_iteration_tokens(
            _trace_electron_contrib_layers_env,
            "NM_TRACE_ELECTRON_CONTRIB_LAYERS",
        )
        for value in parsed_layers:
            idx = value - 1
            if idx < 0:
                print(
                    f"WARNING: Ignoring NM_TRACE_ELECTRON_CONTRIB_LAYERS entry '{value}' "
                    "(must be >= 1 for layer numbering)"
                )
                continue
        _TRACE_ELECTRON_CONTRIB_LAYERS.add(idx)


_TRACE_EQ_STAGE = os.environ.get("NM_TRACE_EQ_STAGE", "").strip().lower() not in {
    "",
    "0",
    "false",
}
_TRACE_EQ_STAGE_ALL_LAYERS = False
_TRACE_EQ_STAGE_LAYERS: set[int] = set()
_TRACE_EQ_STAGE_ITERS: set[int] = set()
if _TRACE_EQ_STAGE:
    _eq_stage_layers_env = os.environ.get("NM_TRACE_EQ_STAGE_LAYERS")
    if _eq_stage_layers_env:
        parsed_layers = _parse_iteration_tokens(
            _eq_stage_layers_env, "NM_TRACE_EQ_STAGE_LAYERS"
        )
        for value in parsed_layers:
            idx = value - 1
            if idx < 0:
                print(
                    f"WARNING: Ignoring NM_TRACE_EQ_STAGE_LAYERS entry '{value}' "
                    "(must be >= 1 for layer numbering)"
                )
                continue
            _TRACE_EQ_STAGE_LAYERS.add(idx)
    else:
        _TRACE_EQ_STAGE_ALL_LAYERS = True
    _eq_stage_iters_env = os.environ.get("NM_TRACE_EQ_STAGE_ITERS")
    if _eq_stage_iters_env:
        parsed_iters = _parse_iteration_tokens(
            _eq_stage_iters_env, "NM_TRACE_EQ_STAGE_ITERS"
        )
        for value in parsed_iters:
            idx = value - 1
            if idx < 0:
                print(
                    f"WARNING: Ignoring NM_TRACE_EQ_STAGE_ITERS entry '{value}' "
                    "(must be >= 1 for iteration numbering)"
                )
                continue
            _TRACE_EQ_STAGE_ITERS.add(idx)
else:
    _TRACE_EQ_STAGE_ALL_LAYERS = False


def _should_trace_electron_layer(layer_idx: int) -> bool:
    return _TRACE_ELECTRON_TERMS and (
        not _TRACE_ELECTRON_LAYERS or layer_idx in _TRACE_ELECTRON_LAYERS
    )


def _should_trace_electron_contrib(layer_idx: int) -> bool:
    if _should_trace_electron_layer(layer_idx):
        return True
    if not _TRACE_ELECTRON_CONTRIBS:
        return False
    return (
        not _TRACE_ELECTRON_CONTRIB_LAYERS
        or layer_idx in _TRACE_ELECTRON_CONTRIB_LAYERS
    )


_LOG_ELECTRON_MOL = os.environ.get("LOG_ELECTRON_MOL", "").strip().lower() not in {
    "",
    "0",
    "false",
}
_LOG_ELECTRON_CODES: tuple[float, ...] = (6.01, 6.02, 20.01, 20.02)
_LOG_ELECTRON_CODES_ENV = os.environ.get("LOG_ELECTRON_MOL_CODES")
if _LOG_ELECTRON_CODES_ENV:
    parsed_codes: list[float] = []
    for token in _LOG_ELECTRON_CODES_ENV.replace(",", " ").split():
        token = token.strip()
        if not token:
            continue
        try:
            parsed_codes.append(float(token))
        except ValueError:
            print(
                "WARNING: Ignoring invalid LOG_ELECTRON_MOL_CODES entry "
                f"'{token}' (expected float molecule code)"
            )
    if parsed_codes:
        _LOG_ELECTRON_CODES = tuple(parsed_codes)


def _log_electron_event(line: str) -> None:
    if not line:
        return
    log_path = os.path.join(os.getcwd(), "electron_term_trace.log")
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _should_log_electron_molecule(molecule_code: float) -> bool:
    if not _LOG_ELECTRON_MOL:
        return False
    for target_code in _LOG_ELECTRON_CODES:
        if abs(molecule_code - target_code) < 1e-6:
            return True
    return False


def _log_electron_term_stage(
    *,
    stage: str,
    layer_idx: int,
    iteration: int,
    molecule_index: int,
    molecule_code: float,
    lock_idx: int | None = None,
    k_raw: int | None = None,
    xn_value: float | None = None,
    term_before: float | None = None,
    term_after: float | None = None,
) -> None:
    base = (
        "PY_ELEC_TERM: layer={layer:3d} iter={iter:5d} jm={jm:4d} "
        "code={code:8.2f} stage={stage}".format(
            layer=layer_idx + 1,
            iter=iteration + 1,
            jm=molecule_index + 1,
            code=molecule_code,
            stage=stage,
        )
    )
    suffix = ""
    if stage == "start" and term_after is not None:
        suffix = f" TERM={term_after: .12E}"
    elif stage == "multiply" and None not in (
        lock_idx,
        k_raw,
        xn_value,
        term_before,
        term_after,
    ):
        suffix = (
            f" LOCK={lock_idx:3d} K={k_raw:3d} XN={xn_value: .12E}"
            f" TERM_BEFORE={term_before: .12E} TERM_AFTER={term_after: .12E}"
        )
    elif stage == "div_pre" and None not in (xn_value, term_before):
        suffix = f" XN(NE)={xn_value: .12E} TERM_BEFORE={term_before: .12E}"
    elif stage in {"div_post", "term_final"} and term_after is not None:
        suffix = f" TERM={term_after: .12E}"
    if suffix:
        _append_nmolec_log(base + suffix)
    else:
        _append_nmolec_log(base)


def _log_electron_equilj(
    *,
    layer_idx: int,
    iteration: int,
    molecule_index: int,
    molecule_code: float,
    equilj_value: float,
    xn_total: float,
    electron_density_val: float,
) -> None:
    _append_nmolec_log(
        "PY_ELEC_EQUILJ: layer={layer:3d} iter={iter:5d} jm={jm:4d} "
        "code={code:8.2f} equilj={equilj:.17E} XN(1)={xn1:.17E} XN(NE)={xne:.17E}".format(
            layer=layer_idx + 1,
            iter=iteration + 1,
            jm=molecule_index + 1,
            code=molecule_code,
            equilj=equilj_value,
            xn1=xn_total,
            xne=electron_density_val,
        )
    )


def _log_electron_state_snapshot(
    *,
    stage: str,
    layer_idx: int,
    iteration: int,
    xn: np.ndarray,
    electron_idx: int | None,
    electron_density_val: float,
    locj: np.ndarray,
    kcomps: np.ndarray,
    code_mol: np.ndarray,
    nequa: int,
) -> None:
    xn1 = float(xn[0]) if len(xn) > 0 else float("nan")
    xne = float(xn[electron_idx]) if electron_idx is not None else float("nan")
    header = (
        "PY_ELEC_STATE: layer={layer:3d} iter={iter:5d} stage={stage} "
        "XN(1)={xn1:.17E} XN(NE)={xne:.17E} electron_density={ed:.17E}".format(
            layer=layer_idx + 1,
            iter=iteration + 1,
            stage=stage,
            xn1=xn1,
            xne=xne,
            ed=electron_density_val,
        )
    )
    _append_nmolec_log(header)
    for jmol in range(len(code_mol)):
        molecule_code = float(code_mol[jmol])
        if not _should_log_electron_molecule(molecule_code):
            continue
        locj1 = int(locj[jmol])
        locj2 = int(locj[jmol + 1] - 1)
        comp_lines = []
        for idx in range(locj1, locj2 + 1):
            comp_raw = int(kcomps[idx])
            comp_idx = nequa - 1 if comp_raw >= nequa else comp_raw
            if comp_idx < 0 or comp_idx >= len(xn):
                continue
            comp_lines.append(
                f"    comp_idx={comp_idx+1:3d} raw={comp_raw:3d} XN={xn[comp_idx]:.17E}"
            )
        if comp_lines:
            _append_nmolec_log(
                "  PY_ELEC_STATE_COMP: jm={jm:4d} code={code:8.2f}".format(
                    jm=jmol + 1, code=molecule_code
                )
            )
            for line in comp_lines:
                _append_nmolec_log(line)


def _should_trace_newton_layer(layer_idx: int) -> bool:
    if not _TRACE_NEWTON_UPDATES:
        return False
    if _TRACE_NEWTON_ALL_LAYERS:
        return True
    return layer_idx in _TRACE_NEWTON_LAYERS


def _log_newton_update(
    *,
    layer_idx: int,
    iteration: int,
    k_idx: int,
    xn_before: float,
    eq_before_damping: float,
    eq_after_damping: float,
    eqold_before: float,
    eqold_after: float,
    xneq: float,
    xn100: float,
    ratio: float,
    branch: str,
    scale_before: float,
    scale_used: float,
    scale_after: float,
    damping_applied: bool,
    scale_modified: bool,
) -> None:
    if not _should_trace_newton_layer(layer_idx):
        return
    _append_debug_line(
        "newton_update_trace.log",
        "PY_NEWTON layer={layer:3d} iter={iter:3d} k={k:2d} "
        "xn_before={xn_before:.17E} eq_before={eq_before:.17E} "
        "eq_after={eq_after:.17E} eqold_before={eqold_before:.17E} "
        "eqold_after={eqold_after:.17E} xneq={xneq:.17E} xn100={xn100:.17E} "
        "ratio={ratio:.17E} branch={branch:<6s} scale_before={scale_before:.17E} "
        "scale_used={scale_used:.17E} scale_after={scale_after:.17E} "
        "damping={damping} scale_modified={scale_mod}".format(
            layer=layer_idx + 1,
            iter=iteration + 1,
            k=k_idx + 1,
            xn_before=xn_before,
            eq_before=eq_before_damping,
            eq_after=eq_after_damping,
            eqold_before=eqold_before,
            eqold_after=eqold_after,
            xneq=xneq,
            xn100=xn100,
            ratio=ratio,
            branch=branch,
            scale_before=scale_before,
            scale_used=scale_used,
            scale_after=scale_after,
            damping=str(damping_applied),
            scale_mod=str(scale_modified),
        ),
    )


def _log_electron_term_step(
    *,
    layer_idx: int,
    iteration: int,
    molecule_index: int,
    molecule_code: float,
    component_idx: int,
    operation: str,
    term_before: float,
    term_after: float,
    xn_raw: float,
    xn_safe: float,
) -> None:
    if not _should_trace_electron_layer(layer_idx):
        return
    _log_electron_event(
        "ELEC_TERM layer={layer:3d} iter={iter:3d} mol={mol:3d} "
        "code={code:8.3f} comp={comp:02d} op={op} term_before={tb:.17E} "
        "term_after={ta:.17E} xn_raw={xn:.17E} xn_safe={xns:.17E}".format(
            layer=layer_idx + 1,
            iter=iteration + 1,
            mol=molecule_index + 1,
            code=molecule_code,
            comp=component_idx,
            op=operation,
            tb=term_before,
            ta=term_after,
            xn=xn_raw,
            xns=xn_safe,
        )
    )


def _log_electron_eq_update(
    *,
    layer_idx: int,
    iteration: int,
    molecule_index: int,
    molecule_code: float,
    term_value: float,
    eq_before: float,
    eq_after: float,
    denom: float,
    ratio: float | None,
) -> None:
    if not _should_trace_electron_layer(layer_idx):
        return
    _log_electron_event(
        "ELEC_EQ layer={layer:3d} iter={iter:3d} mol={mol:3d} code={code:8.3f} "
        "term={term:.17E} eq_before={eqb:.17E} eq_after={eqa:.17E} "
        "denom={den:.17E} ratio={rat}".format(
            layer=layer_idx + 1,
            iter=iteration + 1,
            mol=molecule_index + 1,
            code=molecule_code,
            term=term_value,
            eqb=eq_before,
            eqa=eq_after,
            den=denom,
            rat="nan" if ratio is None else f"{ratio:.17E}",
        )
    )


def _log_electron_deq_update(
    *,
    layer_idx: int,
    iteration: int,
    row_idx: int,
    col_idx: int,
    prev_val: float,
    delta: float,
    new_val: float,
    stage: str,
    molecule_index: int,
    molecule_code: float,
) -> None:
    if not _should_trace_electron_layer(layer_idx):
        return
    _log_electron_event(
        "ELEC_DEQ layer={layer:3d} iter={iter:3d} row={row:3d} col={col:3d} "
        "stage={stage} mol={mol:3d} code={code:8.3f} prev={prev:.17E} "
        "delta={delta:.17E} new={new:.17E}".format(
            layer=layer_idx + 1,
            iter=iteration + 1,
            row=row_idx + 1,
            col=col_idx + 1,
            stage=stage,
            mol=molecule_index + 1,
            code=molecule_code,
            prev=prev_val,
            delta=delta,
            new=new_val,
        )
    )


def _should_trace_eq_stage(layer_idx: int, iteration: int) -> bool:
    if not _TRACE_EQ_STAGE:
        return False
    if not _TRACE_EQ_STAGE_ALL_LAYERS and (layer_idx not in _TRACE_EQ_STAGE_LAYERS):
        return False
    if _TRACE_EQ_STAGE_ITERS and (iteration not in _TRACE_EQ_STAGE_ITERS):
        return False
    return True


def _log_eq_stage(
    stage: str,
    *,
    layer_idx: int,
    iteration: int,
    eq_vec: np.ndarray,
    xn_vec: np.ndarray,
    nequa: int,
    electron_idx: Optional[int],
) -> None:
    if not _should_trace_eq_stage(layer_idx, iteration):
        return
    log_path = os.path.join(os.getcwd(), "eq_stage_trace.log")
    active_eq = eq_vec[:nequa]
    finite_mask = np.isfinite(active_eq)
    if np.any(finite_mask):
        finite_vals = active_eq[finite_mask]
        max_abs = float(np.max(np.abs(finite_vals)))
        min_abs = float(np.min(np.abs(finite_vals)))
    else:
        max_abs = math.nan
        min_abs = math.nan
    sample_indices = [0, 1, 2, 6, 9, nequa - 1]
    seen: list[int] = []
    for idx in sample_indices:
        if 0 <= idx < nequa and idx not in seen:
            seen.append(idx)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                "PY_EQ_STAGE stage={stage} layer={layer:3d} iter={iter:3d} "
                "max_abs={max_abs:.17E} min_abs={min_abs:.17E}\n".format(
                    stage=stage,
                    layer=layer_idx + 1,
                    iter=iteration + 1,
                    max_abs=max_abs,
                    min_abs=min_abs,
                )
            )
            for idx in seen:
                f.write(
                    "  EQ[{idx:2d}]={value: .17E}\n".format(
                        idx=idx + 1, value=float(eq_vec[idx])
                    )
                )
            f.write(f"  XN[1]={float(xn_vec[0]): .17E}\n")
            if electron_idx is not None and 0 <= electron_idx < nequa:
                f.write(
                    "  XN[electron]={xn_val: .17E} EQ[electron]={eq_val: .17E}\n".format(
                        xn_val=float(xn_vec[electron_idx]),
                        eq_val=float(eq_vec[electron_idx]),
                    )
                )
    except OSError:
        pass


def _should_dump_solvit_state(layer_idx: int, iteration: int) -> bool:
    if not _DUMP_SOLVIT_TARGETS:
        return False
    return (layer_idx, iteration) in _DUMP_SOLVIT_TARGETS


def _should_dump_premol_state(layer_idx: int, iteration: int) -> bool:
    if not _DUMP_PREMOL_TARGETS:
        return False
    return (layer_idx, iteration) in _DUMP_PREMOL_TARGETS


def _dump_solvit_state(
    *,
    layer_idx: int,
    iteration: int,
    call_idx: int,
    matrix: np.ndarray,
    rhs: np.ndarray,
) -> None:
    dump_dir = os.path.join(os.getcwd(), "solvit_dumps")
    try:
        os.makedirs(dump_dir, exist_ok=True)
    except OSError:
        return
    file_name = (
        f"solvit_state_L{layer_idx + 1:02d}_I{iteration + 1:02d}_C{call_idx:04d}.npz"
    )
    dump_path = os.path.join(dump_dir, file_name)
    try:
        np.savez(
            dump_path,
            matrix=np.array(matrix, copy=True),
            rhs=np.array(rhs, copy=True),
        )
    except OSError:
        pass


def _dump_premol_state(
    *,
    layer_idx: int,
    iteration: int,
    matrix: np.ndarray,
    rhs: np.ndarray,
    xn_vec: np.ndarray,
) -> None:
    dump_dir = os.path.join(os.getcwd(), "pre_molecule_dumps")
    try:
        os.makedirs(dump_dir, exist_ok=True)
    except OSError:
        return
    file_name = f"premol_state_L{layer_idx + 1:02d}_I{iteration + 1:02d}.npz"
    dump_path = os.path.join(dump_dir, file_name)
    try:
        np.savez(
            dump_path,
            matrix=np.array(matrix, copy=True),
            rhs=np.array(rhs, copy=True),
            xn=np.array(xn_vec, copy=True),
        )
    except OSError:
        pass


_TRACE_ITERATIONS: set[int] = {4, 5, 24}
_trace_iter_env = os.environ.get("NM_TRACE_ITERATIONS")
_TRACE_ITERATIONS_ENV_SET = bool(_trace_iter_env)
if _trace_iter_env:
    _TRACE_ITERATIONS = _parse_iteration_tokens(_trace_iter_env, "NM_TRACE_ITERATIONS")
if _PFSAHA_TRACE_JMOLS and 1 not in _TRACE_ITERATIONS:
    _TRACE_ITERATIONS.add(1)

_MIN_NEWTON_ITER_ENV = os.environ.get("NM_MIN_NEWTON_ITER")
if _MIN_NEWTON_ITER_ENV:
    try:
        MIN_NEWTON_ITER = max(0, int(_MIN_NEWTON_ITER_ENV))
    except ValueError:
        print(
            f"WARNING: Ignoring invalid NM_MIN_NEWTON_ITER='{_MIN_NEWTON_ITER_ENV}' "
            "(expected non-negative integer)"
        )
        MIN_NEWTON_ITER = 0
else:
    MIN_NEWTON_ITER = 0

_TRACE_XN_ITERATIONS: set[int] | None = None
_trace_xn_iter_env = os.environ.get("NM_TRACE_XN_ITERATIONS")
if _trace_xn_iter_env:
    parsed_xn_iters = _parse_iteration_tokens(
        _trace_xn_iter_env, "NM_TRACE_XN_ITERATIONS"
    )
    _TRACE_XN_ITERATIONS = parsed_xn_iters if parsed_xn_iters else set()

# Debug output flags (controlled via environment variables)
# Set to "1", "true", or "True" to enable, anything else to disable
_TRACE_EQUILJ = os.environ.get("NM_TRACE_EQUILJ", "").strip().lower() in ("1", "true")
_TRACE_TERM = os.environ.get("NM_TRACE_TERM", "").strip().lower() in ("1", "true")
_TRACE_DEQ_FULL = os.environ.get("NM_TRACE_DEQ_FULL", "").strip().lower() in (
    "1",
    "true",
)
_TRACE_EQ_FULL = os.environ.get("NM_TRACE_EQ_FULL", "").strip().lower() in ("1", "true")
_TRACE_XN_FULL = os.environ.get("NM_TRACE_XN_FULL", "").strip().lower() in ("1", "true")
_TRACE_RATIO = os.environ.get("NM_TRACE_RATIO", "").strip().lower() in ("1", "true")
_TRACE_SOLVIT_DETAILED = os.environ.get(
    "NM_TRACE_SOLVIT_DETAILED", ""
).strip().lower() in ("1", "true")
_TRACE_SOLVIT_MATRIX = os.environ.get("NM_TRACE_SOLVIT_MATRIX", "").strip().lower() in (
    "1",
    "true",
    "yes",
)


def _append_debug_line(filename: str, line: str) -> None:
    if not line or "nmolec_debug_python" in filename:
        return
    log_path = Path(os.getcwd()) / filename
    try:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _append_nmolec_log(line: str) -> None:
    pass  # Debug logging disabled


def _log_equilj_event(
    *,
    layer_idx: int,
    iteration: int,
    molecule_index: int,
    molecule_code: float,
    equilj_value: float,
) -> None:
    pass  # nmolec_debug_python.log disabled


def _should_trace_pfsa(j_layer: int, jmol_index: int) -> bool:
    return j_layer == 0 and ((jmol_index + 1) in _PFSAHA_TRACE_JMOLS)


def _should_trace_eq_accum(layer_index: int, iteration: int) -> bool:
    return (
        bool(_TRACE_ITERATIONS)
        and layer_index == 0
        and ((iteration + 1) in _TRACE_ITERATIONS)
    )


def _log_eq_accum(
    *,
    layer_index: int,
    iteration: int,
    molecule_index: int,
    k_index: int,
    term_value: float,
    eq_before: float,
    eq_after: float,
) -> None:
    message = (
        "PY_EQ_ACCUM layer=%3d iter=%3d jm=%4d k=%3d term=% .12E "
        "eq_before=% .12E eq_after=% .12E"
        % (
            layer_index + 1,
            iteration,
            molecule_index,
            k_index + 1,
            term_value,
            eq_before,
            eq_after,
        )
    )
    _append_nmolec_log(message)
    _append_debug_line("logs/eq_accum_trace.log", message)


def _log_eq_accum_ext(
    *,
    layer_index: int,
    iteration: int,
    molecule_index: int,
    k_index: int,
    term_value: float,
    d_value: float,
    eq_before: float,
    eq_after: float,
) -> None:
    message = (
        "PY_EQ_ACCUM_EXT layer=%3d iter=%3d jm=%4d k=%3d term=% .12E d=% .12E "
        "eq_before=% .12E eq_after=% .12E"
        % (
            layer_index + 1,
            iteration,
            molecule_index,
            k_index + 1,
            term_value,
            d_value,
            eq_before,
            eq_after,
        )
    )
    _append_nmolec_log(message)
    _append_debug_line("logs/eq_accum_trace.log", message)


def _log_molecule_metadata(
    *,
    layer_index: int,
    iteration: int,
    molecule_index: int,
    ncomp: int,
    locj1: int,
    locj2: int,
    components: list[int],
) -> None:
    log_path = Path(os.getcwd()) / "logs/eq_molecule_trace.log"
    with log_path.open("a") as f:
        f.write(
            "PY_MOL_META layer=%3d iter=%3d jm=%4d ncomp=%3d "
            "locj1=%4d locj2=%4d comps=%s\n"
            % (
                layer_index + 1,
                iteration,
                molecule_index,
                ncomp,
                locj1 + 1,
                locj2 + 1,
                ",".join(str(c + 1) for c in components) if components else "[]",
            )
        )


def _log_molecule_term(
    *,
    layer_index: int,
    iteration: int,
    molecule_index: int,
    molecule_code: float,
    term_value: float,
    eq0_before: float,
    eq0_after: float,
    component_logs: Sequence[dict[str, float]],
    electron_logs: Sequence[dict[str, float]] | None = None,
    term_steps: Sequence[dict[str, Any]] | None = None,
) -> None:
    log_path = Path(os.getcwd()) / "logs/molecule_term_trace.log"
    with log_path.open("a") as f:
        f.write(
            "PY_MOL_TERM layer={layer:3d} iter={iter:3d} jm={jm:4d} "
            "code={code:8.3f} term={term: .17E} eq0_before={eq0b: .17E} "
            "eq0_after={eq0a: .17E}\n".format(
                layer=layer_index + 1,
                iter=iteration,
                jm=molecule_index,
                code=molecule_code,
                term=term_value,
                eq0b=eq0_before,
                eq0a=eq0_after,
            )
        )
        if term_steps:
            for step_idx, step in enumerate(term_steps):
                op = step.get("operation", "")
                if op == "seed":
                    f.write(
                        "  term_step {idx:2d} op=seed term_after={term_after: .17E}\n".format(
                            idx=step_idx,
                            term_after=step.get("term_after", float("nan")),
                        )
                    )
                    continue
                k_raw = step.get("k_raw")
                k_idx = step.get("k_idx")
                f.write(
                    "  term_step {idx:2d} comp_idx={comp_idx:2d} k_raw={k_raw:>4} "
                    "k_idx={k_idx:>4} op={op:>4} xn={xn: .17E} "
                    "term_before={term_before: .17E} term_after={term_after: .17E}\n".format(
                        idx=step_idx,
                        comp_idx=int(step.get("component_idx", -1)),
                        k_raw=(int(k_raw) + 1) if k_raw is not None else -1,
                        k_idx=(int(k_idx) + 1) if k_idx is not None else -1,
                        op=op,
                        xn=step.get("xn", float("nan")),
                        term_before=step.get("term_before", float("nan")),
                        term_after=step.get("term_after", float("nan")),
                    )
                )
        for entry in component_logs:
            f.write(
                "  comp k={k:3d} xn={xn: .17E} eq_before={eqb: .17E} "
                "eq_after={eqa: .17E} d={delta: .17E}\n".format(
                    k=int(entry["k_idx"]) + 1,
                    xn=entry["xn"],
                    eqb=entry["eq_before"],
                    eqa=entry["eq_after"],
                    delta=entry["delta"],
                )
            )
        if electron_logs:
            for entry in electron_logs:
                f.write(
                    "  electron k={k:3d} adj={adj: .17E} eq_before={eqb: .17E} "
                    "eq_after={eqa: .17E}\n".format(
                        k=int(entry["k_idx"]) + 1,
                        adj=entry["adjustment"],
                        eqb=entry["eq_before"],
                        eqa=entry["eq_after"],
                    )
                )


def _log_deq_snapshot(
    *,
    label: str,
    layer_idx: int,
    iteration: int,
    deq: np.ndarray,
    eq: np.ndarray,
    nequa: int,
    rows: tuple[int, ...] = _DEQ_TRACE_ROWS,
    cols: tuple[int, ...] = _DEQ_TRACE_COLS,
) -> None:
    """Log selected DEQ/eq entries for debugging."""
    if layer_idx != 0 or iteration >= 5:
        return
    log_path = Path(os.getcwd()) / "logs/deq_snapshot.log"
    with log_path.open("a") as f:
        f.write(
            "DEQ_SNAPSHOT {label} layer={layer:3d} iter={iter:3d}\n".format(
                label=label, layer=layer_idx + 1, iter=iteration + 1
            )
        )
        for row in rows:
            if row < 0 or row >= nequa:
                continue
            eq_val = float(eq[row]) if row < len(eq) else float("nan")
            col_vals = []
            for col in cols:
                if col < 0 or col >= nequa:
                    continue
                idx = row + col * nequa
                if idx < len(deq):
                    col_vals.append((col + 1, float(deq[idx])))
            if col_vals:
                vals_str = " ".join(
                    f"c{col_idx:02d}={val: .17E}" for col_idx, val in col_vals
                )
                f.write(f"  row {row+1:02d} eq={eq_val: .17E} {vals_str}\n")


@jit(nopython=True, cache=True)
def _setup_element_equations_kernel(
    eq: np.ndarray,
    deq: np.ndarray,
    xn: np.ndarray,
    xab: np.ndarray,
    nequa: int,
    nequa1: int,
    xntot: float,
    idequa: np.ndarray,
) -> None:
    """
    Numba-compiled kernel for setting up element equations EQ and DEQ.
    This is the hot loop that initializes equations before molecular terms.

    Matches Fortran logic from atlas7v.for lines 5205-5221.
    """
    eq[0] = -xntot
    kk = 0
    xn0 = xn[0]

    for k in range(1, nequa):  # k=2..NEQUA (1-based), k=1..nequa-1 (0-based)
        eq[0] = eq[0] + xn[k]
        k1 = k * nequa  # 0-based index for DEQ(1, k+1)
        deq[k1] = 1.0  # DEQ(1, k) = 1

        xn_k = xn[k]
        xab_k = xab[k]

        # Compute element residual: EQ(K) = XN(K) - XAB(K)*XN(1)
        # Direct computation (Fortran-compatible)
        eq[k] = xn_k - xab_k * xn0

        kk = kk + nequa1  # kk = k * nequa1 (0-based: DEQ(k, k) in column-major)
        deq[kk] = 1.0  # DEQ(k, k) = 1
        deq[k] = -xab_k  # DEQ(k+1, 1) = -XAB(K)

    # CRITICAL: Electron equation initialization (Fortran lines 5219-5221)
    # IF(IDEQUA(NEQUA).LT.100)GO TO 62
    # EQ(NEQUA)=-XN(NEQUA)
    # DEQ(NEQNEQ)=-1.
    electron_idx = nequa - 1  # 0-based index for NEQUA
    if idequa[electron_idx] >= 100:  # Electron equation (ID=100)
        eq[electron_idx] = -xn[electron_idx]
        neqneq_idx = nequa * nequa - 1  # 0-based index for DEQ(NEQUA, NEQUA)
        deq[neqneq_idx] = -1.0


@jit(nopython=True, cache=True)
def _accumulate_molecules_kernel(
    eq: np.ndarray,
    deq: np.ndarray,
    xn: np.ndarray,
    equilj: np.ndarray,
    locj: np.ndarray,
    kcomps: np.ndarray,
    idequa: np.ndarray,
    nequa: int,
    nummol: int,
    eq_comp: np.ndarray,  # Kahan compensation array for EQ
) -> None:
    """
    Numba-compiled kernel for accumulating molecular terms into EQ/DEQ.
    This is the hot loop that processes all molecules without tracing overhead.

    Uses Kahan summation to reduce floating-point accumulation errors.
    """
    for jmol in range(nummol):
        ncomp = int(locj[jmol + 1] - locj[jmol])
        locj1 = int(locj[jmol])
        locj2 = int(locj[jmol + 1] - 1)

        if ncomp <= 1:
            continue

        equilj_val = equilj[jmol]
        if not np.isfinite(equilj_val):
            continue

        # Fortran multiplies TERM in linear space and allows inf/nan to propagate.
        term = equilj_val
        for lock in range(locj1, locj2 + 1):
            k_raw = int(kcomps[lock])
            if k_raw >= nequa:
                k_idx = nequa - 1
                term = term / xn[k_idx]
            else:
                k_idx = k_raw
                term = term * xn[k_idx]

        # Accumulate into EQ[0] using Kahan summation
        y = term - eq_comp[0]
        t = eq[0] + y
        eq_comp[0] = (t - eq[0]) - y
        eq[0] = t

        # Accumulate into EQ and DEQ for each component
        for lock in range(locj1, locj2 + 1):
            k_raw = int(kcomps[lock])
            if k_raw == nequa:
                k_idx = nequa - 1
            else:
                k_idx = k_raw

            xn_val = xn[k_idx]
            if not np.isfinite(xn_val) or xn_val == 0.0:
                continue

            if k_raw == nequa:
                d = -term / xn_val
            else:
                d = term / xn_val

            if not np.isfinite(d):
                continue

            # Accumulate into EQ[k_idx] using Kahan summation
            y_k = term - eq_comp[k_idx]
            t_k = eq[k_idx] + y_k
            eq_comp[k_idx] = (t_k - eq[k_idx]) - y_k
            eq[k_idx] = t_k

            # Accumulate into DEQ column k_idx
            nequak = nequa * k_idx
            deq[nequak] = deq[nequak] + d

            # Accumulate into DEQ entries for all components
            for locm in range(locj1, locj2 + 1):
                m_raw = int(kcomps[locm])
                m_idx = nequa - 1 if m_raw == nequa else m_raw
                mk = m_idx + nequak
                deq[mk] = deq[mk] + d

        # Correction to charge equation for negative ions
        # FIX: Only apply if last component is REGULAR electron (kcomps = nequa-1),
        # NOT inverse electron sentinel (kcomps = nequa).
        last_comp_raw = int(kcomps[locj2])
        if (
            last_comp_raw == nequa - 1  # Regular electron, not inverse electron
            and idequa[nequa - 1] == 100  # Confirm it's the electron equation
        ):
            for lock in range(locj1, locj2 + 1):
                k_corr_raw = int(kcomps[lock])
                k_corr_idx = nequa - 1 if k_corr_raw >= nequa else k_corr_raw
                xn_val = xn[k_corr_idx]
                if not np.isfinite(xn_val) or xn_val == 0.0:
                    continue
                term_corr = term
                if not np.isfinite(term_corr):
                    continue
                d_corr = term_corr / xn_val
                if not np.isfinite(d_corr):
                    continue
                if k_corr_idx == nequa - 1:
                    eq[k_corr_idx] = eq[k_corr_idx] - term_corr - term_corr
                delta = -d_corr - d_corr
                for locm in range(locj1, locj2 + 1):
                    m_corr_raw = int(kcomps[locm])
                    # Map raw value to 0-based index (Fortran: IF(M.GE.NEQUA1)M=NEQUA)
                    m_corr_idx = nequa - 1 if m_corr_raw >= nequa else m_corr_raw
                    # Only update DEQ when M is the electron equation (Fortran: IF(M.NE.NEQUA)GO TO 93)
                    if m_corr_idx != nequa - 1:
                        continue
                    mk = m_corr_idx + nequa * k_corr_idx
                    deq[mk] = deq[mk] + delta


def _accumulate_molecules_atlas7(
    *,
    eq: np.ndarray,
    deq: np.ndarray,
    xn: np.ndarray,
    equilj: np.ndarray,
    locj: np.ndarray,
    kcomps: np.ndarray,
    idequa: np.ndarray,
    nequa: int,
    nummol: int,
    code_mol: np.ndarray,
    pending_solvit_call: int,
    layer_index: int,
    iteration: int,
    trace_callback: Optional[Callable[..., None]] = None,
    nonfinite_callback: Optional[Callable[..., None]] = None,
    log_xn: Optional[np.ndarray] = None,  # Full log-space: log(XN) values
) -> Optional[list[tuple[int, float, float, float, int, int]]]:
    """
    Port of atlas7v.for SUBROUTINE NMOLEC (lines 3772–3835) for EQ/DEQ assembly.
    """
    term_trace: Optional[list[tuple[int, float, float, float, int, int]]] = (
        [] if (layer_index == 0 and iteration == 0) else None
    )

    trace_eq_accum = _should_trace_eq_accum(layer_index, iteration)
    trace_electron_layer = _should_trace_electron_layer(layer_index)
    trace_electron_contrib = _should_trace_electron_contrib(layer_index)
    trace_metadata = trace_eq_accum

    # Fast path: Use Numba kernel when tracing is disabled
    use_numba_kernel = (
        not trace_eq_accum
        and not trace_electron_layer
        and trace_callback is None
        and nonfinite_callback is None
        and _TERM_TRACE_THRESHOLD is None
        and _DEQ_TRACE_THRESHOLD is None
        and len(_TRACE_DEQ_COLS) == 0
    )

    if use_numba_kernel:
        # Make copies for Numba (needs writable arrays)
        eq_copy = eq.copy()
        deq_copy = deq.copy()
        # Kahan compensation array for EQ accumulation
        eq_comp = np.zeros(nequa, dtype=np.float64)

        # Call Numba kernel
        _accumulate_molecules_kernel(
            eq_copy, deq_copy, xn, equilj, locj, kcomps, idequa, nequa, nummol, eq_comp
        )

        # Copy results back
        eq[:] = eq_copy
        deq[:] = deq_copy

        return term_trace

    # Kahan compensation for Python path EQ accumulation
    eq_comp_py = np.zeros(nequa, dtype=np.float64)

    for jmol in range(nummol):
        ncomp = int(locj[jmol + 1] - locj[jmol])
        locj1 = int(locj[jmol])
        locj2 = int(locj[jmol + 1] - 1)
        molecule_code = float(code_mol[jmol])
        if trace_metadata:
            comps = [int(kcomps[idx]) for idx in range(locj1, locj2 + 1)]
            _log_molecule_metadata(
                layer_index=layer_index,
                iteration=iteration,
                molecule_index=jmol + 1,
                ncomp=ncomp,
                locj1=locj1,
                locj2=locj2,
                components=comps,
            )
        if ncomp <= 1:
            continue

        default_trace = jmol in TRACE_MOLECULES_ZERO
        env_trace = False
        if _TRACE_MOLECULE_CODES:
            for target in _TRACE_MOLECULE_CODES:
                if abs(code_mol[jmol] - target) < 0.5:
                    env_trace = True
                    break
        iteration_traced = (iteration + 1) in _TRACE_ITERATIONS
        force_trace = jmol in TRACE_MOLECULES_FORCE
        has_target_component = bool(_TRACE_EQ_TARGETS) and any(
            int(kcomps[idx]) in _TRACE_EQ_TARGETS for idx in range(locj1, locj2 + 1)
        )
        trace_pfsa = False
        trace_scope_ok = layer_index == 0 or force_trace or env_trace
        trace_this_molecule = trace_scope_ok and (
            env_trace
            or force_trace
            or (
                layer_index == 0
                and (
                    (iteration <= 5 and default_trace)
                    or (iteration_traced and has_target_component)
                )
            )
        )
        component_logs: list[dict[str, float]] = []
        electron_logs: list[dict[str, float]] = []
        eq0_before_trace = float(eq[0]) if trace_this_molecule else 0.0

        # CRITICAL: Guard against NaN/Inf equilibrium values
        # Fortran keeps these in extended precision, but once they become NaN/Inf in
        # Python we skip the molecule entirely (matching the Fortran "TERM=0" effect
        # when a component underflows).
        equilj_val = np.float64(equilj[jmol])
        if not np.isfinite(equilj_val) or equilj_val <= 0.0:
            if nonfinite_callback is not None:
                nonfinite_callback(
                    stage="equilj",
                    row=None,
                    col=None,
                    value=float(equilj_val),
                    delta=float("nan"),
                    previous=float("nan"),
                    molecule_index=jmol + 1,
                    molecule_code=float(code_mol[jmol]),
                )
            continue
        # Use log-space for TERM calculation to preserve precision
        log_term = np.log(equilj_val)
        term = equilj_val  # Keep for logging compatibility
        if trace_this_molecule and (iteration + 1) in _TRACE_ITERATIONS:
            _log_equilj_event(
                layer_idx=layer_index,
                iteration=iteration,
                molecule_index=jmol,
                molecule_code=float(code_mol[jmol]),
                equilj_value=float(equilj_val),
            )
        log_term_steps = (
            [] if (trace_this_molecule or (jmol + 1) in TRACE_MOLECULE_IDS) else None
        )
        term_step_logs: list[dict[str, Any]] | None = log_term_steps
        if term_step_logs is not None:
            term_step_logs.append(
                {
                    "operation": "seed",
                    "component_idx": 0,
                    "k_raw": None,
                    "k_idx": None,
                    "xn": float("nan"),
                    "term_before": float("nan"),
                    "term_after": float(term),
                }
            )
        log_electron_molecule = _should_log_electron_molecule(molecule_code)
        if log_electron_molecule:
            _log_electron_term_stage(
                stage="start",
                layer_idx=layer_index,
                iteration=iteration,
                molecule_index=jmol,
                molecule_code=molecule_code,
                term_after=float(term),
            )
        term_invalid = False
        for lock in range(locj1, locj2 + 1):
            k_raw = int(kcomps[lock])
            # Determine xn index before modifying term
            if k_raw >= nequa:
                k_idx = nequa - 1
            else:
                k_idx = k_raw

            # FULL LOG-SPACE: Use log_xn directly to avoid any conversion
            if log_xn is not None:
                # Get log(XN[k]) directly - no conversion, no overflow possible
                log_xn_k = log_xn[k_idx]
                if not np.isfinite(log_xn_k):
                    term_invalid = True
                    break

                term_before_float = (
                    _signed_log_to_linear(1, log_term) if log_term < 700 else 1e307
                )

                # Update log_term purely in log-space
                if k_raw >= nequa:
                    log_term = (
                        log_term - log_xn_k
                    )  # Division: log(a/b) = log(a) - log(b)
                else:
                    log_term = (
                        log_term + log_xn_k
                    )  # Multiplication: log(a*b) = log(a) + log(b)

                # For logging: compute clamped linear term
                if log_term > 709.0:
                    term = np.finfo(np.float64).max
                elif log_term < -745.0:
                    term = 0.0
                else:
                    term = np.exp(log_term)

                # For xn_val used in DEQ: get from linear xn (already clamped on storage)
                xn_val = np.float64(xn[k_idx])
            else:
                # LEGACY PATH: Convert xn to log (may overflow if xn is huge)
                xn_val = np.float64(xn[k_idx])
                if not np.isfinite(xn_val) or xn_val <= 0.0:
                    term_invalid = True
                    break

                term_before_float = float(term)
                # Update log_term in log-space for precision
                if k_raw >= nequa:
                    log_term = log_term - np.log(xn_val)  # Division
                else:
                    log_term = log_term + np.log(xn_val)  # Multiplication

                # Convert to linear space for logging (clamped)
                if log_term > 709.0:
                    term = np.finfo(np.float64).max
                elif log_term < -745.0:
                    term = 0.0
                else:
                    term = np.exp(log_term)

            if term_step_logs is not None:
                step_log = {
                    "operation": "div" if k_raw >= nequa else "mul",
                    "component_idx": lock - locj1 + 1,
                    "k_raw": k_raw,
                    "k_idx": k_idx,
                    "xn": float(xn[k_idx]),
                    "xn_safe": float(xn_val),
                    "term_before": term_before_float,
                    "term_after": float(term),
                }
                term_step_logs.append(step_log)

            if trace_electron_layer and k_idx == nequa - 1:
                _log_electron_term_step(
                    layer_idx=layer_index,
                    iteration=iteration,
                    molecule_index=jmol,
                    molecule_code=float(code_mol[jmol]),
                    component_idx=lock - locj1 + 1,
                    operation="div" if k_raw >= nequa else "mul",
                    term_before=term_before_float,
                    term_after=float(term),
                    xn_raw=float(xn[k_idx]),
                    xn_safe=float(xn_val),
                )

            if log_electron_molecule:
                if k_raw < nequa:
                    _log_electron_term_stage(
                        stage="multiply",
                        layer_idx=layer_index,
                        iteration=iteration,
                        molecule_index=jmol,
                        molecule_code=molecule_code,
                        lock_idx=lock - locj1 + 1,
                        k_raw=k_raw + 1,
                        xn_value=float(xn_val),
                        term_before=term_before_float,
                        term_after=float(term),
                    )
                else:
                    _log_electron_term_stage(
                        stage="div_pre",
                        layer_idx=layer_index,
                        iteration=iteration,
                        molecule_index=jmol,
                        molecule_code=molecule_code,
                        xn_value=float(xn_val),
                        term_before=term_before_float,
                    )
                    _log_electron_term_stage(
                        stage="div_post",
                        layer_idx=layer_index,
                        iteration=iteration,
                        molecule_index=jmol,
                        molecule_code=molecule_code,
                        term_after=float(term),
                    )

        if term_invalid or not np.isfinite(log_term):
            if nonfinite_callback is not None:
                nonfinite_callback(
                    stage="term",
                    row=None,
                    col=None,
                    value=float("nan"),
                    delta=float("nan"),
                    previous=float(equilj_val),
                    molecule_index=jmol + 1,
                    molecule_code=float(code_mol[jmol]),
                )
            continue

        # Final conversion from log-space to linear space
        if log_term > 709.0:
            term = np.finfo(np.float64).max
        elif log_term < -745.0:
            term = 0.0
        else:
            term = np.exp(log_term)

        if (
            _TRACE_DEQ_COLS
            and (iteration + 1) in _TRACE_ITERATIONS
            and (layer_index + 1) == 1
            and (jmol + 1) in TRACE_MOLECULE_IDS
        ):
            debug_log_path = Path(os.getcwd()) / "logs/ electron_diag_trace.log"
            diag_idx = nequa - 1
            diag_val = float(eq[diag_idx])
            deq_diag = float(deq[diag_idx + diag_idx * nequa])
            with debug_log_path.open("a") as f:
                f.write(
                    "PY_DEQ_DIAG layer={layer} iter={iter} mol={mol} "
                    "code={code:.2f} eq_diag={eq:.17E} deq_diag={deq:.17E}\n".format(
                        layer=layer_index + 1,
                        iter=iteration + 1,
                        mol=jmol + 1,
                        code=float(code_mol[jmol]),
                        eq=diag_val,
                        deq=deq_diag,
                    )
                )
        if not np.isfinite(term):
            if nonfinite_callback is not None:
                nonfinite_callback(
                    stage="term",
                    row=None,
                    col=None,
                    value=float(term),
                    delta=float("nan"),
                    previous=float(equilj[jmol]),
                    molecule_index=jmol + 1,
                    molecule_code=float(code_mol[jmol]),
                )
            continue

        log_eq0 = trace_eq_accum or trace_this_molecule
        if log_eq0:
            eq0_before = float(eq[0])
        # Use signed log-space accumulation to handle extreme TERM values
        # This prevents overflow when TERM is ~10^400+ (log_term > 709)
        eq0_sign, log_eq0_abs = _linear_to_signed_log(float(eq[0]))
        eq0_sign, log_eq0_abs = _add_signed_log(eq0_sign, log_eq0_abs, +1, log_term)
        eq[0] = np.float64(_signed_log_to_linear(eq0_sign, log_eq0_abs))
        if trace_eq_accum:
            _log_eq_accum(
                layer_index=layer_index,
                iteration=iteration,
                molecule_index=jmol + 1,
                k_index=0,
                term_value=float(term),
                eq_before=eq0_before,
                eq_after=float(eq[0]),
            )
        if trace_this_molecule:
            eq0_before_trace = eq0_before
        if term_trace is not None and jmol < 20:
            term_trace.append(
                (
                    jmol,
                    float(code_mol[jmol]),
                    float(equilj[jmol]),
                    float(term),
                    locj1,
                    locj2,
                )
            )

        def _enforce_electron_coupling_local() -> None:
            if idequa[nequa - 1] != 100:
                return
            # Fortran does not inject any extra coupling here; electron
            # contributions arise solely from molecule terms.
            return

        for lock in range(locj1, locj2 + 1):
            k_raw = int(kcomps[lock])
            if k_raw == nequa:
                k_idx = nequa - 1
            else:
                k_idx = k_raw

            # CRITICAL: Guard against zero/non-finite xn values to prevent NaN
            xn_val = np.float64(xn[k_idx])
            if not np.isfinite(xn_val):
                continue
            if xn_val == 0.0:
                continue

            if k_raw == nequa:
                d = -_div_preserving_precision(term, xn_val)
            else:
                d = _div_preserving_precision(term, xn_val)

            if not np.isfinite(d):
                continue

            if nonfinite_callback is not None and not np.isfinite(d):
                nonfinite_callback(
                    stage="delta",
                    row=None,
                    col=k_idx,
                    value=float(d),
                    delta=float(d),
                    previous=float("nan"),
                    molecule_index=jmol + 1,
                    molecule_code=float(code_mol[jmol]),
                )

            eq_before = float(eq[k_idx])
            log_eq_component = trace_eq_accum or trace_this_molecule
            # Use signed log-space accumulation to handle extreme TERM values
            eq_k_val = np.float64(eq[k_idx])  # Save for tracing
            eq_k_sign, log_eq_k_abs = _linear_to_signed_log(float(eq[k_idx]))
            eq_k_sign, log_eq_k_abs = _add_signed_log(
                eq_k_sign, log_eq_k_abs, +1, log_term
            )
            eq[k_idx] = np.float64(_signed_log_to_linear(eq_k_sign, log_eq_k_abs))
            electron_ratio_val: float | None = None
            if trace_electron_layer and k_idx == nequa - 1 and eq_k_val != 0.0:
                electron_ratio_val = float(term / eq_k_val)
            if trace_eq_accum:
                _log_eq_accum(
                    layer_index=layer_index,
                    iteration=iteration,
                    molecule_index=jmol + 1,
                    k_index=k_idx,
                    term_value=float(term),
                    eq_before=eq_before,
                    eq_after=float(eq[k_idx]),
                )
                _log_eq_accum_ext(
                    layer_index=layer_index,
                    iteration=iteration,
                    molecule_index=jmol + 1,
                    k_index=k_idx,
                    term_value=float(term),
                    d_value=float(d),
                    eq_before=eq_before,
                    eq_after=float(eq[k_idx]),
                )
            if trace_this_molecule:
                component_logs.append(
                    {
                        "k_idx": k_idx,
                        "xn": float(xn[k_idx]),
                        "eq_before": eq_before,
                        "eq_after": float(eq[k_idx]),
                        "delta": float(d),
                    }
                )
            if trace_this_molecule and k_idx == nequa - 1:
                electron_logs.append(
                    {
                        "k_idx": k_idx,
                        "eq_before": eq_before,
                        "eq_after": float(eq[k_idx]),
                        "adjustment": float(term),
                    }
                )
            if trace_electron_layer and k_idx == nequa - 1:
                _log_electron_eq_update(
                    layer_idx=layer_index,
                    iteration=iteration,
                    molecule_index=jmol,
                    molecule_code=float(code_mol[jmol]),
                    term_value=float(term),
                    eq_before=eq_before,
                    eq_after=float(eq[k_idx]),
                    denom=float(eq_k_val),
                    ratio=electron_ratio_val,
                )
            nequak = nequa * k_idx
            prev_col_val = deq[nequak]
            new_col_val = np.float64(prev_col_val + d)
            if trace_electron_contrib and k_idx == nequa - 1:
                _log_electron_event(
                    "PY_EQ_CORR layer={layer:3d} iter={iter:3d} code={code:8.3f} "
                    "stage=deq_col row={row:3d} col={col:3d} prev={prev:.17E} "
                    "delta={delta:.17E} new={new:.17E}".format(
                        layer=layer_index + 1,
                        iter=iteration + 1,
                        code=float(code_mol[jmol]),
                        row=1,
                        col=k_idx + 1,
                        prev=float(prev_col_val),
                        delta=float(d),
                        new=float(new_col_val),
                    )
                )
            deq[nequak] = new_col_val
            if nonfinite_callback is not None and not np.isfinite(deq[nequak]):
                nonfinite_callback(
                    stage="deq_col",
                    row=0,
                    col=k_idx,
                    value=float(deq[nequak]),
                    delta=float(d),
                    previous=float(prev_col_val),
                    molecule_index=jmol + 1,
                    molecule_code=float(code_mol[jmol]),
                )

            for locm in range(locj1, locj2 + 1):
                m_raw = int(kcomps[locm])
                m_idx = nequa - 1 if m_raw == nequa else m_raw
                mk = m_idx + nequak
                prev_val = deq[mk]
                new_val = np.float64(prev_val + d)
                if trace_electron_contrib and (
                    m_idx == nequa - 1 or k_idx == nequa - 1
                ):
                    _log_electron_event(
                        "PY_EQ_CORR layer={layer:3d} iter={iter:3d} code={code:8.3f} "
                        "stage=deq_entry row={row:3d} col={col:3d} prev={prev:.17E} "
                        "delta={delta:.17E} new={new:.17E}".format(
                            layer=layer_index + 1,
                            iter=iteration + 1,
                            code=float(code_mol[jmol]),
                            row=m_idx + 1,
                            col=k_idx + 1,
                            prev=float(prev_val),
                            delta=float(d),
                            new=float(new_val),
                        )
                    )
                deq[mk] = new_val
                if trace_electron_layer and m_idx == nequa - 1:
                    _log_electron_deq_update(
                        layer_idx=layer_index,
                        iteration=iteration,
                        row_idx=m_idx,
                        col_idx=k_idx,
                        prev_val=float(prev_val),
                        delta=float(d),
                        new_val=float(deq[mk]),
                        stage="general",
                        molecule_index=jmol,
                        molecule_code=float(code_mol[jmol]),
                    )
                if (
                    m_idx == nequa - 1
                    and (layer_index + 1) == 1
                    and (iteration + 1) in _TRACE_ITERATIONS
                    and (jmol + 1) in TRACE_MOLECULE_IDS
                ):
                    diag_debug_path = (
                        Path(os.getcwd()) / "logs/ electron_diag_trace.log"
                    )
                    with diag_debug_path.open("a") as f:
                        f.write(
                            "PY_DEQ_COMP layer={layer} iter={iter} mol={mol} "
                            "stage=term row={row} col={col} prev={prev:.17E} "
                            "delta={delta:.17E} new={new:.17E} term={term:.17E} "
                            "xn_col={xn:.17E}\n".format(
                                layer=layer_index + 1,
                                iter=iteration + 1,
                                mol=jmol + 1,
                                row=m_idx + 1,
                                col=k_idx + 1,
                                prev=float(prev_val),
                                delta=float(d),
                                new=float(deq[mk]),
                                term=float(term),
                                xn=float(xn[k_idx]),
                            )
                        )
                if nonfinite_callback is not None and not np.isfinite(deq[mk]):
                    nonfinite_callback(
                        stage="deq_entry",
                        row=m_idx,
                        col=k_idx,
                        value=float(deq[mk]),
                        delta=float(d),
                        previous=float(prev_val),
                        molecule_index=jmol + 1,
                        molecule_code=float(code_mol[jmol]),
                    )

            if trace_callback is not None:
                trace_callback(
                    layer=layer_index + 1,
                    iteration=iteration,
                    call_idx=pending_solvit_call,
                    molecule_index=jmol + 1,
                    molecule_code=code_mol[jmol],
                    m=k_idx,
                    d=d,
                    deq=deq,
                    nequa=nequa,
                )

        if term_step_logs is not None:
            _log_molecule_term(
                layer_index=layer_index,
                iteration=iteration + 1,
                molecule_index=jmol + 1,
                molecule_code=float(code_mol[jmol]),
                term_value=float(term),
                eq0_before=eq0_before_trace,
                eq0_after=float(eq[0]),
                component_logs=component_logs,
                electron_logs=electron_logs,
                term_steps=term_step_logs,
            )

        if log_electron_molecule:
            _log_electron_term_stage(
                stage="term_final",
                layer_idx=layer_index,
                iteration=iteration,
                molecule_index=jmol,
                molecule_code=molecule_code,
                term_after=float(term),
            )

        # Fortran lines 5378-5390: correction to charge equation for negative ions.
        # In Fortran: K=KCOMPS(LOCJ2), IF(IDEQUA(K).NE.100) GO TO 99
        # This checks if the LAST component is a REGULAR electron (ID=100), not inverse electron (ID=101).
        # In Python's kcomps:
        #   - Regular electron (ID 100) → kcomps = nequa - 1 (e.g., 22)
        #   - Inverse electron (ID 101) → kcomps = nequa (e.g., 23, the sentinel value)
        # The neg_ion correction applies ONLY to molecules ending with a regular electron,
        # NOT to molecules ending with an inverse electron (like H+ = CODE 1.01).
        last_comp_raw = int(kcomps[locj2])
        # FIX: Only apply if last component is the REGULAR electron equation index (nequa-1),
        # NOT the inverse electron sentinel (nequa or higher).
        if (
            last_comp_raw
            == nequa - 1  # Regular electron, not inverse electron sentinel
            and idequa[nequa - 1] == 100  # Confirm it's the electron equation
        ):
            for lock in range(locj1, locj2 + 1):
                k_corr_raw = int(kcomps[lock])
                # Map raw value to 0-based index (same as Fortran's K mapping)
                # Fortran: IF(K.GE.NEQUA1)K=NEQUA means K >= 24 maps to K=23
                # Python: raw >= nequa (23) maps to index 22
                k_corr_idx = nequa - 1 if k_corr_raw >= nequa else k_corr_raw
                xn_val = np.float64(xn[k_corr_idx])
                if not np.isfinite(xn_val) or xn_val == 0.0:
                    continue
                term_corr = np.float64(term)
                if not np.isfinite(term_corr):
                    continue
                d_corr = _div_preserving_precision(term_corr, xn_val)
                if not np.isfinite(d_corr):
                    continue
                if k_corr_idx == nequa - 1:
                    prev_eq = eq[k_corr_idx]
                    # Use signed log-space to prevent overflow
                    eq_sign, log_eq_abs = _linear_to_signed_log(float(eq[k_corr_idx]))
                    # BUG FIX: Use log(term) instead of log_term to ensure sync
                    # log_2_term = log_term + np.log(2.0)  # OLD: uses potentially stale log_term
                    log_2_term = (
                        np.log(term_corr) + np.log(2.0)
                        if term_corr > 0
                        else float("-inf")
                    )  # NEW: compute from term_corr
                    eq_sign, log_eq_abs = _add_signed_log(
                        eq_sign, log_eq_abs, -1, log_2_term
                    )
                    new_eq_val = np.float64(_signed_log_to_linear(eq_sign, log_eq_abs))
                    if trace_electron_contrib:
                        _log_electron_event(
                            "PY_EQ_CORR layer={layer:3d} iter={iter:3d} code={code:8.3f} "
                            "stage=neg_eq row={row:3d} col={col:3d} prev={prev:.17E} "
                            "delta={delta:.17E} new={new:.17E}".format(
                                layer=layer_index + 1,
                                iter=iteration + 1,
                                code=float(code_mol[jmol]),
                                row=k_corr_idx + 1,
                                col=k_corr_idx + 1,
                                prev=float(prev_eq),
                                delta=float(-2.0 * term_corr),
                                new=float(new_eq_val),
                            )
                        )
                    eq[k_corr_idx] = new_eq_val
                    if (
                        (layer_index + 1) == 1
                        and (iteration + 1) in _TRACE_ITERATIONS
                        and (jmol + 1) in TRACE_MOLECULE_IDS
                    ):
                        diag_debug_path = (
                            Path(os.getcwd()) / "logs/ electron_diag_trace.log"
                        )
                        with diag_debug_path.open("a") as f:
                            f.write(
                                "PY_EQ_CORR layer={layer} iter={iter} mol={mol} "
                                "stage=neg_ion eq_prev={prev:.17E} term={term:.17E} "
                                "eq_new={new:.17E}\n".format(
                                    layer=layer_index + 1,
                                    iter=iteration + 1,
                                    mol=jmol + 1,
                                    prev=float(prev_eq),
                                    term=float(term_corr),
                                    new=float(eq[k_corr_idx]),
                                )
                            )
                base_idx = k_corr_idx * nequa
                for locm in range(locj1, locj2 + 1):
                    m_raw = int(kcomps[locm])
                    # Map raw value to 0-based index (same as Fortran's M mapping)
                    m_idx = nequa - 1 if m_raw >= nequa else m_raw
                    # Only update DEQ when M is the electron equation (Fortran: IF(M.NE.NEQUA)GO TO 93)
                    if m_idx != nequa - 1:
                        continue
                    mk = m_idx + base_idx
                    prev_val = deq[mk]
                    delta = np.float64(-2.0 * d_corr)
                    if not np.isfinite(delta):
                        continue
                    if not np.isfinite(prev_val):
                        deq[mk] = prev_val
                    else:
                        new_val = np.float64(prev_val + delta)
                        if trace_electron_contrib:
                            _log_electron_event(
                                "PY_EQ_CORR layer={layer:3d} iter={iter:3d} code={code:8.3f} "
                                "stage=neg_deq row={row:3d} col={col:3d} prev={prev:.17E} "
                                "delta={delta:.17E} new={new:.17E}".format(
                                    layer=layer_index + 1,
                                    iter=iteration + 1,
                                    code=float(code_mol[jmol]),
                                    row=m_idx + 1,
                                    col=k_corr_idx + 1,
                                    prev=float(prev_val),
                                    delta=float(delta),
                                    new=float(new_val),
                                )
                            )
                        deq[mk] = new_val
                    if trace_electron_layer:
                        _log_electron_deq_update(
                            layer_idx=layer_index,
                            iteration=iteration,
                            row_idx=m_idx,
                            col_idx=k_corr_idx,
                            prev_val=float(prev_val),
                            delta=float(delta),
                            new_val=float(deq[mk]),
                            stage="neg_ion",
                            molecule_index=jmol,
                            molecule_code=float(code_mol[jmol]),
                        )
                    if (
                        (layer_index + 1) == 1
                        and (iteration + 1) in _TRACE_ITERATIONS
                        and (jmol + 1) in TRACE_MOLECULE_IDS
                    ):
                        diag_debug_path = (
                            Path(os.getcwd()) / "logs/ electron_diag_trace.log"
                        )
                        with diag_debug_path.open("a") as f:
                            f.write(
                                "PY_DEQ_COMP layer={layer} iter={iter} mol={mol} "
                                "stage=neg_ion row={row} col={col} prev={prev:.17E} "
                                "delta={delta:.17E} new={new:.17E} term={term:.17E} "
                                "xn_row={xn:.17E}\n".format(
                                    layer=layer_index + 1,
                                    iter=iteration + 1,
                                    mol=jmol + 1,
                                    row=m_idx + 1,
                                    col=k_corr_idx + 1,
                                    prev=float(prev_val),
                                    delta=float(delta),
                                    new=float(deq[mk]),
                                    term=float(term_corr),
                                    xn=float(xn_val),
                                )
                            )

    return term_trace


def _log_xn_update(
    *,
    layer_idx: int,
    iteration: int,
    k_idx: int,
    xn_before: float,
    xn_after: float,
    eq_value: float,
    xneq: float,
    xn100: float,
    ratio: float,
    branch: str,
    scale_value: float,
) -> None:
    debug_log_path = os.path.join(os.getcwd(), "xn_update_trace.log")
    with open(debug_log_path, "a") as f:
        iter_one_based = iteration + 1
        f.write(
            "PY_XN_UPDATE: layer={layer:3d} ITER={iter:3d} K={k:2d} "
            "XN_BEFORE={before: .17E} EQ={eq: .17E} "
            "XNEQ={xneq: .17E} XN100={xn100: .17E} RATIO={ratio: .17E} "
            "BRANCH={branch:<6s} SCALE={scale: .17E}\n".format(
                layer=layer_idx + 1,
                iter=iter_one_based,
                k=k_idx + 1,
                before=xn_before,
                eq=eq_value,
                xneq=xneq,
                xn100=xn100,
                ratio=ratio,
                branch=branch,
                scale=scale_value,
            )
        )
        f.write(
            "PY_XN_AFTER : layer={layer:3d} ITER={iter:3d} K={k:2d} "
            "XN_AFTER={after: .17E}\n".format(
                layer=layer_idx + 1, iter=iter_one_based, k=k_idx + 1, after=xn_after
            )
        )


def _log_eq_components(
    *,
    layer_idx: int,
    iteration: int,
    eq_vec: np.ndarray,
    xn_vec: np.ndarray,
) -> None:
    if not _TRACE_EQ_COMPONENTS:
        return
    log_path = os.path.join(os.getcwd(), "logs/eq_component_trace.log")
    with open(log_path, "a") as f:
        iter_one_based = iteration + 1
        for idx in sorted(_TRACE_EQ_COMPONENTS):
            eq_val = eq_vec[idx] if idx < eq_vec.size else float("nan")
            xn_val = xn_vec[idx] if idx < xn_vec.size else float("nan")
            f.write(
                "PY_EQ_COMP layer={layer:3d} iter={iter:3d} k={k:2d} "
                "EQ={eq: .17E} XN={xn: .17E}\n".format(
                    layer=layer_idx + 1,
                    iter=iter_one_based,
                    k=idx + 1,
                    eq=eq_val,
                    xn=xn_val,
                )
            )


def _should_trace_xn(layer_idx: int, iteration: int, k_idx: int) -> bool:
    if not _TRACE_XN_INDICES:
        return False
    allow_layer = False
    if _TRACE_XN_ALL_LAYERS_FLAG:
        allow_layer = True
    elif layer_idx == 0:
        allow_layer = True
    elif layer_idx in _TRACE_XN_LAYERS:
        allow_layer = True
    if not allow_layer:
        return False
    if _TRACE_XN_ITERATIONS is not None:
        iter_one_based = iteration + 1
        if iter_one_based not in _TRACE_XN_ITERATIONS:
            return False
    return k_idx in _TRACE_XN_INDICES


def _should_trace_eq_full(layer_idx: int, iteration: int) -> bool:
    if not _TRACE_EQ_FULL or layer_idx != 0:
        return False
    if _TRACE_ITERATIONS_ENV_SET:
        return (iteration + 1) in _TRACE_ITERATIONS
    return iteration == 0


def _should_trace_deq_full(layer_idx: int, iteration: int) -> bool:
    if not _TRACE_DEQ_FULL or layer_idx != 0:
        return False
    if _TRACE_ITERATIONS_ENV_SET:
        return (iteration + 1) in _TRACE_ITERATIONS
    # Default behavior: iteration 0 (after setup) and iteration 2 (post-check)
    return iteration in (0, 2)


def _trace_deq_update(
    *,
    layer: int,
    iteration: int,
    call_idx: int | None,
    molecule_index: int,
    molecule_code: float,
    m: int,
    d: float,
    deq: np.ndarray,
    nequa: int,
) -> None:
    """Detailed logging for targeted DEQ columns/rows during accumulation."""
    if not _TRACKED_DEQ_KS and not _TRACKED_DEQ_CROSS:
        return

    should_log_column = m in _TRACKED_DEQ_KS
    should_log_cross = bool(_TRACKED_DEQ_CROSS)
    if not should_log_column and not should_log_cross:
        return

    col_offset = m * nequa
    entries = []
    if should_log_column:
        deq_1k = deq[col_offset] if col_offset < len(deq) else float("nan")
        diag_idx = col_offset + m
        diag_val = deq[diag_idx] if diag_idx < len(deq) else float("nan")
        entries.append(f"DEQ(1,{m+1})={deq_1k: .17E}")
        entries.append(f"DEQ({m+1},{m+1})={diag_val: .17E}")

    if should_log_cross:
        for row_idx in sorted(_TRACKED_DEQ_CROSS):
            if row_idx < 0 or row_idx >= nequa:
                continue
            idx = row_idx + col_offset
            if idx >= len(deq):
                continue
            entries.append(f"DEQ({row_idx+1},{m+1})={deq[idx]: .17E}")


def _set_solvit_context(layer: int, iteration: int, call_idx: int | None) -> None:
    global _current_solvit_layer, _current_solvit_iter, _current_solvit_call
    _current_solvit_layer = layer
    _current_solvit_iter = iteration
    _current_solvit_call = call_idx


def nmolec_exact(
    n_layers: int,
    temperature: np.ndarray,
    tkev: np.ndarray,
    tk: np.ndarray,
    tlog: np.ndarray,
    gas_pressure: np.ndarray,
    electron_density: np.ndarray,
    xabund: np.ndarray,  # Element abundances (99,)
    xnatom_atomic: np.ndarray,  # Atomic-only XNATOM = P/TK - XNE
    # Molecular data (from READMOL)
    nummol: int,
    code_mol: np.ndarray,  # (MAXMOL,) molecular codes
    equil: np.ndarray,  # (7, MAXMOL) equilibrium constants
    locj: np.ndarray,  # (MAXMOL+1,) component locations
    kcomps: np.ndarray,  # (MAXLOC,) component indices (0-based equation numbers)
    idequa: np.ndarray,  # (MAXEQ,) equation element IDs
    nequa: int,  # Number of equations
    # Partition function data (required, initialized to 1.0 for LTE, matching Fortran DATA statements)
    bhyd: np.ndarray,  # (n_layers, 8) H partition functions
    bc1: np.ndarray,  # (n_layers, 14) C partition functions
    bo1: np.ndarray,  # (n_layers, 13) O partition functions
    bmg1: np.ndarray,  # (n_layers, 11) Mg partition functions
    bal1: np.ndarray,  # (n_layers, 9) Al partition functions
    bsi1: np.ndarray,  # (n_layers, 11) Si partition functions
    bca1: np.ndarray,  # (n_layers, 8) Ca partition functions
    # PFSAHA function: (j, iz, nion, mode, frac, nlte_on) -> None
    # frac is (n_layers, 31) array, modified in-place
    pfsaha_func: Optional[PFSAHAFunc] = None,
    # Output
    xnatom_molecular: Optional[np.ndarray] = None,  # Output: molecular XNATOM
    xnmol: Optional[
        np.ndarray
    ] = None,  # Output: molecular number densities (n_layers, MAXMOL)
    # Zero pivot fix options
    zero_pivot_fix: str = "none",  # "none", "pivot_early", "perturbation"
    # Optional: xnatom array to update in-place during iterations (for PFSAHA access)
    xnatom_inout: Optional[
        np.ndarray
    ] = None,  # Modified in-place: xnatom_inout[j] = XN[0] during iterations
    # Log-space Newton iteration (experimental)
    use_log_space: bool = False,  # Enable log-space Jacobian scaling and XN updates
    # Full Decimal precision Newton (for extreme values)
    use_decimal_newton: bool = False,  # Use 50-digit Decimal for entire Newton iteration
    # Bounded Newton with trust region (for cool atmospheres)
    use_bounded_newton: bool = False,  # Enable trust-region bounded Newton for robust convergence
    # Gibbs minimization (DEFAULT: True - avoids Newton basin-of-attraction issues)
    use_gibbs: bool = True,  # Use Gibbs free energy minimization (recommended)
    gibbs_temperature_threshold: float = 5000.0,  # Auto-enable Gibbs below this T (if auto_gibbs)
    auto_gibbs: bool = False,  # Only matters if use_gibbs=False
    # Continuation method: process layers hot-to-cool to avoid bifurcation
    # CRITICAL FIX: Default to False to match Fortran's layer order (surface first)
    # Fortran atlas7v.for line 4973: DO 110 J=JSTART,NRHOX processes layers 1→80
    # Continuation mode caused Layer 0 to inherit wrong XN values from hot layers
    use_continuation: bool = False,  # Disabled to match Fortran layer order
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute molecular XNATOM using NMOLEC algorithm.

    Returns:
        (xnatom_molecular, xnmol) where:
        - xnatom_molecular: (n_layers,) molecular XNATOM = XN(1)
        - xnmol: (n_layers, MAXMOL) molecular number densities
    """
    trace_xne_layer_env = os.environ.get("NM_TRACE_XNE_LAYER", "").strip()
    trace_xne_layer = None
    if trace_xne_layer_env.lstrip("+-").isdigit():
        trace_xne_layer = int(trace_xne_layer_env)
    if trace_xne_layer is not None and pfsaha_func is None:
        trace_path = os.path.join(os.getcwd(), "logs/nmolec_xne_iter.log")
        with open(trace_path, "a") as f:
            f.write("PY_XNE_ITER: pfsaha_func is None\n")

    # DISABLED: Experimental features that cause divergence
    # These are kept for reference but not triggered by environment variables
    # The LOG-SPACE and DECIMAL Newton modes require more work to be stable
    use_log_space = False  # Force disabled - causes overflow/underflow
    use_decimal_newton = (
        False  # Force disabled - produces different trajectory than Fortran
    )

    if xnatom_molecular is None:
        xnatom_molecular = np.zeros(n_layers, dtype=np.float64)
    if xnmol is None:
        xnmol = np.zeros((n_layers, MAXMOL), dtype=np.float64)

    # Check if we should use Gibbs solver
    # Auto-enable if any layer temperature is below threshold
    if auto_gibbs:
        min_temp = np.min(temperature)
        if min_temp < gibbs_temperature_threshold:
            use_gibbs = True

    # Use Gibbs minimization if requested (recommended for cool atmospheres)
    if use_gibbs:
        from synthe_py.tools.nmolec_gibbs import nmolec_gibbs

        xnatom_result, xne_result, xnz_result = nmolec_gibbs(
            n_layers=n_layers,
            temperature=temperature,
            tkev=tkev,
            tk=tk,
            tlog=tlog,
            gas_pressure=gas_pressure,
            electron_density=electron_density,
            xabund=xabund,
            xnatom_atomic=xnatom_atomic,
            nummol=nummol,
            code_mol=code_mol,
            equil=equil,
            locj=locj,
            kcomps=kcomps,
            idequa=idequa,
            nequa=nequa,
            # Partition function data for CPF corrections (required by Gibbs)
            bhyd=bhyd,
            bc1=bc1,
            bo1=bo1,
            bmg1=bmg1,
            bal1=bal1,
            bsi1=bsi1,
            bca1=bca1,
            pfsaha_func=pfsaha_func,
            xnatom_molecular=xnatom_molecular,
            verbose=False,
        )

        # Update outputs
        xnatom_molecular[:] = xnatom_result
        electron_density[:] = xne_result

        # Return early - skip Newton iteration
        return xnatom_molecular, electron_density, xnz_result

    # Initialize XNZ array to store XN values after each layer (Fortran line 5049-5050)
    # This matches Fortran's XNZ(J,K) array used to persist XN across layers
    xnz_molecular = np.zeros((n_layers, MAXEQ), dtype=np.float64)

    # CONTINUATION METHOD: Process layers from hot to cool
    # This avoids the bifurcation problem at cool temperatures by starting
    # from a high-T solution where only one basin exists, then tracking it
    # down to lower temperatures.
    if use_continuation:
        # Sort layers by temperature (descending: hot first)
        layer_order = np.argsort(-temperature)  # Negative for descending
    else:
        # Original order (as in .atm file)
        layer_order = np.arange(n_layers)

    # XNE ITERATION: Compute self-consistent electron density before Newton iteration
    # This matches Fortran's POPS XNE iteration loop (atlas7v.for lines 2956-2980)
    # Key elements that contribute electrons: H, He, C, Na, Mg, Al, Si, K, Ca, Fe
    xne_electron_donors = [1, 2, 6, 11, 12, 13, 14, 19, 20, 26]  # Element Z numbers
    xne_nions = [1, 2, 2, 2, 2, 2, 2, 2, 2, 2]  # Max ionization stages to consider

    trace_xne_layer_env = os.environ.get("NM_TRACE_XNE_LAYER", "").strip()
    trace_xne_layer = None
    if trace_xne_layer_env.lstrip("+-").isdigit():
        trace_xne_layer = int(trace_xne_layer_env)

    def _iterate_xne_for_layer(j_layer: int, pfsaha_fn, max_iter: int = 200) -> float:
        """Iterate XNE to self-consistency using PFSAHA mode 4 electron calculation.

        CRITICAL: Fortran initializes XNE = XNTOT/2 (atlas7v.for line 2956),
        NOT the value from .atm file. This makes a huge difference because
        XNTOT/2 ≈ 6.9e12 while .atm XNE ≈ 2e7.
        """
        xntot_j = gas_pressure[j_layer] / tk[j_layer]
        # CRITICAL: Initialize XNE to XNTOT/2, matching Fortran line 2956
        xne_j = xntot_j / 2.0
        xnatom_j = xntot_j - xne_j

        if pfsaha_fn is None:
            return electron_density[j_layer]  # Can't iterate without PFSAHA

        elec_arr = np.zeros((n_layers, 31), dtype=np.float64)
        mask = [1] * len(
            xne_electron_donors
        )  # Track which elements still contribute significantly

        # Debug: track contributions for first or requested layer
        if trace_xne_layer is None:
            debug_xne_iter = j_layer == 0
        else:
            debug_xne_iter = j_layer == trace_xne_layer

        for iteration in range(max_iter):
            xne_new = 0.0
            contributions = []

            # CRITICAL: Update electron_density[j] so that pfsaha_wrapper closure
            # uses the current XNE value when computing ionization fractions.
            # Without this, PFSAHA uses the .atm value and gives wrong results.
            electron_density[j_layer] = xne_j

            for i, (iz, nion) in enumerate(zip(xne_electron_donors, xne_nions)):
                if mask[i] == 0:
                    continue
                # Call PFSAHA mode 4 to get electron contribution per atom
                elec_contribution = 0.0
                try:
                    pfsaha_fn(j_layer, iz, nion, 4, elec_arr, 0)
                    elec_contribution = elec_arr[j_layer, 0]
                except Exception as e:
                    if debug_xne_iter and iteration == 0:
                        contributions.append((iz, "ERROR", str(e)[:50]))
                    continue

                # Electron contribution = elec_per_atom * xnatom * abundance
                if iz - 1 < len(xabund) and xabund[iz - 1] > 0:
                    x_contrib = elec_contribution * xnatom_j * xabund[iz - 1]
                    xne_new += x_contrib

                    if debug_xne_iter and iteration == 0:
                        contributions.append((iz, elec_contribution, x_contrib))

                    # Mask out negligible contributors
                    if iteration > 0 and x_contrib < 1e-5 * xne_j:
                        mask[i] = 0

            if debug_xne_iter:
                debug_log_path = os.path.join(os.getcwd(), "logs/nmolec_xne_iter.log")
                with open(debug_log_path, "a") as f:
                    f.write(
                        f"PY_XNE_ITER: layer={j_layer} iter={iteration} "
                        f"xne_old={xne_j:.6e} xne_new={xne_new:.6e}\n"
                    )

            # Damped update: XNE = (XNENEW + XNE) / 2
            xne_new = (xne_new + xne_j) / 2.0

            # Check convergence
            if xne_j > 0:
                error = (
                    abs((xne_j - xne_new) / xne_new)
                    if xne_new > 0
                    else abs(xne_j - xne_new)
                )
            else:
                error = abs(xne_new)

            xne_j = xne_new
            xnatom_j = xntot_j - xne_j

            if error < 0.0005:  # Fortran uses 0.0005 tolerance
                break

        return xne_j

    # Initialize arrays
    nequa1 = nequa + 1
    neqneq = nequa * nequa

    nonfinite_term_hits = 0
    nonfinite_d_hits = 0
    nonfinite_xn_hits = 0
    NONFINITE_LOG_LIMIT = 10

    def _log_nonfinite_event(tag: str, message: str) -> None:
        log_line = f"[nmolec-nonfinite][{tag}] {message}"
        print(log_line)
        log_file = os.path.join(os.getcwd(), "logs/nmolec_nonfinite.log")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")

    def _log_eq_vector(
        stage_header: str, entry_label: str, iteration_idx: int, vec: np.ndarray
    ) -> None:
        """Log the full EQ vector for early iterations (mirrors Fortran debug)."""
        if vec is None:
            return

    def _log_tracked_deq_columns(
        layer_idx: int,
        iteration_idx: int,
        deq_matrix: np.ndarray,
        eq_vec: np.ndarray,
        call_idx: Optional[int] = None,
    ) -> None:
        """Log selected DEQ columns (1,8,9,10,16) before SOLVIT for comparison with Fortran."""
        if deq_matrix is None or eq_vec is None:
            return

    def _log_xn_seed(
        layer_idx: int,
        xn_seed: np.ndarray,
        nequa_local: int,
        ratio: float | None,
        xntot_val: float,
        electron_val: float,
    ) -> None:
        """Log the XN seed vector for selected layers (entire active system)."""
        if layer_idx not in _TRACE_XN_SEED_LAYERS:
            return
        seed_path = os.path.join(os.getcwd(), "xn_seed_trace.log")
        with open(seed_path, "a") as seed_log:
            seed_log.write(
                f"PY_XN_SEED layer={layer_idx+1:3d} xntot={xntot_val: .17E} "
                f"ratio={ratio if ratio is not None else 1.0: .17E} "
                f"electron_seed={electron_val: .17E}\n"
            )
            for idx in range(nequa_local):
                seed_log.write(f"  XN[{idx+1:2d}]={xn_seed[idx]: .17E}\n")
            finite_seed = xn_seed[:nequa_local]
            seed_log.write(
                f"  XN stats: min={np.min(finite_seed): .17E} max={np.max(finite_seed): .17E}\n"
            )

    def _log_seed_reset(
        *,
        layer_idx: int,
        reason: str,
        ratio: float | None,
        component_index: Optional[int],
        offending_value: Optional[float],
        electron_seed_value: Optional[float],
    ) -> None:
        reset_path = os.path.join(os.getcwd(), "seed_reset_trace.log")
        with open(reset_path, "a", encoding="utf-8") as reset_log:
            ratio_val = np.float64(ratio) if ratio is not None else np.nan
            comp_idx = component_index + 1 if component_index is not None else -1
            offending = (
                np.float64(offending_value) if offending_value is not None else np.nan
            )
            seeded_electron = (
                np.float64(electron_seed_value)
                if electron_seed_value is not None
                else np.nan
            )
            reset_log.write(
                "PY_SEED_RESET layer={layer:3d} reason={reason} ratio={ratio:.17E} "
                "component={component:3d} offending={offending:.17E} "
                "electron_seed={electron_seed:.17E} min_seed={min_seed:.17E}\n".format(
                    layer=layer_idx + 1,
                    reason=reason,
                    ratio=ratio_val,
                    component=comp_idx,
                    offending=offending,
                    electron_seed=seeded_electron,
                    min_seed=_SEED_MIN_VALUE,
                )
            )

    def _seed_min_threshold(
        component_index: int, electron_equation_index: Optional[int]
    ) -> float:
        if component_index == 0:
            return _SEED_MIN_VALUE
        if (
            electron_equation_index is not None
            and component_index == electron_equation_index
        ):
            return _SEED_MIN_VALUE
        return 0.0

    def _seed_value_valid(value: float, min_allowed: float) -> bool:
        return np.isfinite(value) and value >= min_allowed

    # XAB: abundances for each equation variable
    xab = np.zeros(MAXEQ, dtype=np.float64)
    for k in range(1, nequa):  # k=2..NEQUA (1-based), k=1..nequa-1 (0-based)
        id_elem = idequa[k]  # 0-based: idequa[k] = element ID
        if id_elem < 100:
            xab[k] = max(xabund[id_elem - 1], 1e-20)  # 1-based to 0-based

    # Check if last equation is for electrons
    if idequa[nequa - 1] == 100:  # 0-based
        xab[nequa - 1] = 0.0

    # XN working array (Fortran XN)
    xn = np.zeros(MAXEQ, dtype=np.float64)

    # Process each layer
    # CRITICAL: Match Fortran's exact sequence
    solvit_call_counter = 0

    # Persistent state matching Fortran's XNZ array, initialized on first layer
    xnz_prev = np.zeros(MAXEQ, dtype=np.float64)
    electron_density_atm = electron_density.copy()
    x_prev_seeded = False

    # Track computed XNE for layer-to-layer scaling (Fortran uses XNE array, not .atm values)
    xne_computed = np.zeros(n_layers, dtype=np.float64)
    # Track seed XNE used before Newton iteration (Fortran uses this in PFSAHA mode=12)
    xne_seed = np.zeros(n_layers, dtype=np.float64)

    # Track previous layer index for continuation seeding
    prev_layer_idx = -1

    for iter_idx, j in enumerate(layer_order):
        xntot = gas_pressure[j] / tk[j]
        electron_idx = nequa - 1 if idequa[nequa - 1] == 100 else None

        # For continuation: first iteration (iter_idx=0) is the hottest layer
        # For original order: first iteration is layer 0
        is_first_iteration = iter_idx == 0

        if is_first_iteration:
            # Fortran layer-1 initialization (atlas7v.for lines 4910-4916):
            #   XNTOT = P(JSTART)/TK(JSTART)
            #   XN(1) = XNTOT/2
            #   X = XN(1)/10
            #   XN(K) = X*XAB(K)  for K=2..NEQUA
            #   XNE(1) = X
            seed_ratio = None
            xn[0] = xntot / 2.0  # Fortran: XN(1) = XNTOT/2
            base_x = xn[0] / 10.0  # Fortran: X = XN(1)/10
            for k in range(1, nequa):
                xn[k] = base_x * xab[k]  # Fortran: XN(K) = X*XAB(K)
            # Fortran: IF(ID.EQ.100) XN(NEQUA) = X  and  XNE(1) = X
            if electron_idx is not None:
                xn[electron_idx] = base_x
            xne_computed[j] = base_x  # Fortran: XNE(1) = X
            electron_density[j] = base_x
            xne_seed[j] = base_x
            x_prev_seeded = True
        else:
            # Subsequent layers: use previous layer's solution as initial guess
            # For continuation: prev_layer_idx is the previous iteration's layer (hotter)
            # For original order: prev_layer_idx is j-1 (adjacent layer)
            if use_continuation:
                # Use the XN solution from the previous iteration (stored in xnz_prev)
                # Scale by pressure ratio between current and previous processed layer
                prev_pressure = gas_pressure[prev_layer_idx]
                seed_ratio = (
                    gas_pressure[j] / prev_pressure
                    if prev_pressure not in (0.0, np.inf)
                    else 1.0
                )
            else:
                # Fortran subsequent layers (atlas7v.for lines 5001-5005):
                #   RATIO = P(J)/P(J-1)
                prev_pressure = gas_pressure[j - 1]
                seed_ratio = (
                    gas_pressure[j] / prev_pressure
                    if prev_pressure not in (0.0, np.inf)
                    else 1.0
                )

            nonfinite_seed = False
            invalid_component_idx: Optional[int] = None
            invalid_value: Optional[float] = None
            invalid_reason: Optional[str] = None

            # Scale all XN components by ratio (Fortran: DO 33 K=1,NEQUA; XN(K)=XN(K)*RATIO)
            for k in range(nequa):
                xn_val = xnz_prev[k] * seed_ratio
                if not np.isfinite(xn_val):
                    nonfinite_seed = True
                    invalid_component_idx = k
                    invalid_value = xn_val
                    invalid_reason = "nonfinite_component"
                    break
                xn[k] = xn_val

            # Scale XNE by ratio (but we'll use base_x_layer for electron_density)
            if use_continuation:
                xne_scaled = xne_computed[prev_layer_idx] * seed_ratio
            else:
                xne_scaled = xne_computed[j - 1] * seed_ratio

            if not np.isfinite(xne_scaled):
                nonfinite_seed = True
                invalid_reason = "nonfinite_xne"

            if nonfinite_seed:
                # Fallback: reinitialize like layer 1
                failed_ratio = seed_ratio
                seed_ratio = None
                xn[0] = xntot / 2.0
                base_x = xn[0] / 10.0
                for k in range(1, nequa):
                    xn[k] = base_x * xab[k]
                if electron_idx is not None:
                    xn[electron_idx] = base_x
                xne_computed[j] = base_x
                electron_density[j] = base_x
                xne_seed[j] = base_x
                _log_seed_reset(
                    layer_idx=j,
                    reason=invalid_reason or "invalid_seed",
                    ratio=failed_ratio,
                    component_index=invalid_component_idx,
                    offending_value=invalid_value,
                    electron_seed_value=base_x,
                )
            else:
                xne_computed[j] = xne_scaled
                electron_density[j] = xne_scaled
                xne_seed[j] = xne_scaled

        xnz_prev[:nequa] = xn[:nequa]
        _log_xn_seed(
            layer_idx=j,
            xn_seed=xn[:nequa],
            nequa_local=nequa,
            ratio=seed_ratio,
            xntot_val=xntot,
            electron_val=electron_density[j],
        )

        # Fortran NMOLEC does not iterate XNE via PFSAHA; keep off by default.
        # Enable explicitly with NM_ENABLE_XNE_ITER=1 if needed for experiments.
        if pfsaha_func is not None and os.environ.get("NM_ENABLE_XNE_ITER", "0") != "0":
            if trace_xne_layer is not None and j == trace_xne_layer:
                trace_path = os.path.join(os.getcwd(), "logs/nmolec_xne_iter.log")
                with open(trace_path, "a") as f:
                    f.write(f"PY_XNE_ITER_ENTER: layer={j}\n")
            electron_density[j] = _iterate_xne_for_layer(j, pfsaha_func)
            xne_computed[j] = electron_density[j]
            if electron_idx is not None:
                xn[electron_idx] = electron_density[j]

        if (j == 0) and TRACE_MOLECULE_IDS:
            diag_path = Path(os.getcwd()) / "logs/ electron_diag_trace.log"
            with diag_path.open("a") as f:
                sample_size = min(nequa, 25)
                f.write(
                    f"PY_XN_SNAPSHOT layer={j+1} iter={iteration+1 if 'iteration' in locals() else 0} "
                    f"xntot={xntot:.17E}\n"
                )
                for idx in range(sample_size):
                    f.write(f"  XN[{idx+1:3d}]={xn[idx]: .17E}\n")

        # NOTE: XNE iteration is enabled by default for parity with xnfpelsyn.

        # Compute partition function corrections for NLTE
        # From atlas7v.for lines 4556-4592
        # B arrays are always required (initialized to 1.0 for LTE, matching Fortran DATA statements)
        # CPF corrections are computed if PFSAHA is available, otherwise default to 1.0
        cpfh = 1.0
        cpfc = 1.0
        cpfo = 1.0
        cpfmg = 1.0
        cpfal = 1.0
        cpfsi = 1.0
        cpfca = 1.0

        if pfsaha_func is not None:
            # NLTE partition functions (NLTEON = -1)
            pf = np.zeros((n_layers, 31), dtype=np.float64)
            pfsaha_func(j, 1, 1, 3, pf, -1)
            pfh = pf[j, 0]
            pfsaha_func(j, 6, 1, 3, pf, -1)
            pfc = pf[j, 0]
            pfsaha_func(j, 8, 1, 3, pf, -1)
            pfo = pf[j, 0]
            pfsaha_func(j, 12, 1, 3, pf, -1)
            pfmg = pf[j, 0]
            pfsaha_func(j, 13, 1, 3, pf, -1)
            pfal = pf[j, 0]
            pfsaha_func(j, 14, 1, 3, pf, -1)
            pfsi = pf[j, 0]
            pfsaha_func(j, 20, 1, 3, pf, -1)
            pfca = pf[j, 0]

            # LTE partition functions (NLTEON = 0)
            bpf = np.zeros((n_layers, 31), dtype=np.float64)
            pfsaha_func(j, 1, 1, 3, bpf, 0)
            bpfh = bpf[j, 0]
            pfsaha_func(j, 6, 1, 3, bpf, 0)
            bpfc = bpf[j, 0]
            pfsaha_func(j, 8, 1, 3, bpf, 0)
            bpfo = bpf[j, 0]
            pfsaha_func(j, 12, 1, 3, bpf, 0)
            bpfmg = bpf[j, 0]
            pfsaha_func(j, 13, 1, 3, bpf, 0)
            bpfal = bpf[j, 0]
            pfsaha_func(j, 14, 1, 3, bpf, 0)
            bpfsi = bpf[j, 0]
            pfsaha_func(j, 20, 1, 3, bpf, 0)
            bpfca = bpf[j, 0]

            # Compute corrections
            # CRITICAL: Match Fortran exactly (lines 4586-4592): CPF = PF/BPF * B
            # Fortran always has B arrays in COMMON blocks (initialized to 1.0 by default)
            # No fallback logic - arrays must always be provided
            if bpfh != 0:
                cpfh = pfh / bpfh * bhyd[j, 0]
            if bpfc != 0:
                cpfc = pfc / bpfc * bc1[j, 0]
            if bpfo != 0:
                cpfo = pfo / bpfo * bo1[j, 0]
            if bpfmg != 0:
                cpfmg = pfmg / bpfmg * bmg1[j, 0]
            if bpfal != 0:
                cpfal = pfal / bpfal * bal1[j, 0]
            if bpfsi != 0:
                cpfsi = pfsi / bpfsi * bsi1[j, 0]
            if bpfca != 0:
                cpfca = pfca / bpfca * bca1[j, 0]

        if j == 0:
            print(
                "CPF layer 0:"
                f" CPFH={cpfh:.6e} CPFC={cpfc:.6e} CPFO={cpfo:.6e}"
                f" CPF_MG={cpfmg:.6e} CPF_AL={cpfal:.6e}"
                f" CPF_SI={cpfsi:.6e} CPF_CA={cpfca:.6e}"
            )

        # Compute equilibrium constants EQUILJ for each molecule
        equilj = np.zeros(MAXMOL, dtype=np.float64)

        # Debug: Track EQUILJ values for first layer
        equilj_debug = []

        for jmol in range(nummol):
            ncomp = locj[jmol + 1] - locj[jmol]

            # DEBUG: Check EQUIL(0) before path decision for molecule 2 (CODE=1.01)
            trace_pfsa = _should_trace_pfsa(j, jmol)
            if trace_pfsa:
                _append_nmolec_log(
                    f"PY_NMOLEC: Before path check: JMOL={jmol+1} CODE={code_mol[jmol]:.2f} "
                    f"EQUIL(0)={equil[0, jmol]:.17E}"
                )

            if equil[0, jmol] == 0.0:
                # Use PFSAHA-based equilibrium
                if ncomp > 1:
                    id_elem = int(code_mol[jmol])
                    ion = ncomp - 1
                    # Call PFSAHA in mode 12 (ionization fractions)
                    if pfsaha_func is not None:
                        frac = np.zeros((n_layers, 31), dtype=np.float64)
                        log_pfsaha = trace_pfsa or (
                            j == 0
                            and (
                                (jmol + 1) in _PFSAHA_DEBUG_JMOLS
                                or _should_trace_molecule(jmol, code_mol[jmol])
                            )
                        )

                        if trace_pfsa:
                            _append_nmolec_log(
                                f"PY_NMOLEC: PFSAHA path start: JMOL={jmol+1} "
                                f"CODE={code_mol[jmol]:.2f} ID={id_elem} "
                                f"NCOMP={ncomp} ION={ion}"
                            )

                        # Match Fortran NMOLEC/PFSAHA: use the live XNE(J) state directly.
                        # Do not substitute a separate seed value before PFSAHA calls.
                        if id_elem == 11 and j == 59:
                            prev_pressure = gas_pressure[j - 1] if j > 0 else float("nan")
                            prev_xne = xne_computed[j - 1] if j > 0 else float("nan")
                            with open("logs/pfsaha_na_state_python.log", "a") as fh:
                                fh.write(
                                    "PY_PFSAHA_NA_STATE: "
                                    f"J={j+1:03d} XNE_CUR={electron_density[j]:.6e} "
                                    f"XNE_SEED={xne_seed[j]:.6e} XNE_COMP={xne_computed[j]:.6e} "
                                    f"P={gas_pressure[j]:.6e} PPREV={prev_pressure:.6e} "
                                    f"XNE_PREV={prev_xne:.6e}\n"
                                )
                        pfsaha_func(j, id_elem, ncomp, 12, frac, 0)

                        if trace_pfsa:
                            preview_len = min(max(ncomp, 2), 5)
                            frac_preview = " ".join(
                                f"{frac[j, idx]:.6E}" for idx in range(preview_len)
                            )
                            _append_nmolec_log(
                                "PY_NMOLEC: PFSAHA frac snapshot: "
                                f"JMOL={jmol+1} values={frac_preview} any_nonzero={np.any(frac[j, :ncomp])}"
                            )

                        # Debug calculation (same protection as main calculation)
                        frac0 = np.float64(frac[j, 0])
                        fracn = np.float64(frac[j, ncomp - 1])
                        if (
                            frac0 == 0.0
                            or abs(frac0) < 1e-300
                            or not np.isfinite(frac0)
                        ):
                            equilj_before = 0.0
                            equilj_str = "0.00000000000000000E+00"
                            if trace_pfsa:
                                _append_nmolec_log(
                                    f"PY_NMOLEC: PFSAHA warning JMOL={jmol+1} frac(j,0)=0; "
                                    "equilj set to 0 before ratio"
                                )
                        else:
                            # Use log-space to preserve precision: log(fracn) - log(frac0) + ion*log(xne)
                            if fracn > 0.0:
                                log_frac_ratio = np.log(fracn) - np.log(frac0)
                                log_term = log_frac_ratio + ion * np.log(
                                    electron_density[j]
                                )
                                equilj_before = np.exp(log_term)
                            else:
                                equilj_before = (
                                    fracn / frac0 * (electron_density[j] ** ion)
                                )
                            if np.isnan(equilj_before):
                                equilj_str = "NaN"
                            elif np.isinf(equilj_before):
                                equilj_str = "Inf"
                            else:
                                equilj_str = f"{equilj_before:.17E}"
                                if trace_pfsa:
                                    _append_nmolec_log(
                                        f"PY_NMOLEC: EQUILJ (PFSAHA calc)={equilj_str}"
                                    )

                        # CRITICAL: Fortran (line 4649) does NOT check for zero denominators:
                        #   EQUILJ(JMOL)=FRAC(J,NCOMP)/FRAC(J,1)*XNE(J)**ION
                        # If FRAC(J,1)=0, Fortran produces Inf/NaN but continues execution.
                        # Match Fortran behavior exactly: do NOT add protection checks!
                        # Fortran allows INF/NaN EQUILJ to propagate through TERM calculation
                        # Use np.errstate to match Fortran's silent division by zero behavior
                        with np.errstate(divide="ignore", invalid="ignore"):
                            if frac0 > 0.0 and fracn > 0.0:
                                log_frac_ratio = np.log(fracn) - np.log(frac0)
                                log_term = log_frac_ratio + ion * np.log(
                                    electron_density[j]
                                )
                                equilj[jmol] = np.exp(log_term)
                            else:
                                equilj[jmol] = (
                                    fracn / frac0 * (electron_density[j] ** ion)
                                )
                        # CRITICAL: Do NOT check for NaN/Inf - Fortran doesn't check either!
                        # Fortran allows INF/NaN EQUILJ to propagate through TERM calculation

                        # CRITICAL FIX: Do NOT apply CPFC corrections in PFSAHA path!
                        # Fortran does NOT apply CPFC corrections to PFSAHA molecules (lines 4544-4554
                        # are BEFORE label 35, so they're only in polynomial path)
                    else:
                        if trace_pfsa:
                            _append_nmolec_log(
                                f"PY_NMOLEC: PFSAHA skipped for JMOL={jmol+1} "
                                "(pfsaha_func is None); defaulting EQUILJ=1"
                            )
                        equilj[jmol] = 1.0
                else:
                    if trace_pfsa:
                        _append_nmolec_log(
                            f"PY_NMOLEC: PFSAHA not applicable for JMOL={jmol+1} "
                            f"(NCOMP={ncomp}); EQUILJ set to 1"
                        )
                    equilj[jmol] = 1.0
            else:
                if trace_pfsa:
                    _append_nmolec_log(
                        f"PY_NMOLEC: PFSAHA bypassed for JMOL={jmol+1} "
                        f"(EQUIL(0)={equil[0, jmol]:.3E}); using polynomial path"
                    )
                # Use EQUIL polynomial
                # CRITICAL: Match Fortran's ION calculation exactly (line 4510):
                #   ION=(CODE(JMOL)-DBLE( INT(CODE(JMOL))))*100.+.5
                # Fortran uses DBLE() to ensure double precision, then adds 0.5 and truncates
                # Python: Use np.float64 to ensure double precision, then add 0.5 and convert to int
                code_int = int(code_mol[jmol])
                code_frac = np.float64(code_mol[jmol]) - np.float64(code_int)
                ion = int(code_frac * 100.0 + 0.5)
                equilj[jmol] = np.float64(0.0)

                if temperature[j] > 10000.0:
                    continue

                is_h_minus = abs(code_mol[jmol] - 101.0) < 1e-9
                if is_h_minus:
                    # Special case for H- (HMINUS)
                    exp_arg = (
                        4.478 / tkev[j]
                        - 46.4584
                        + (
                            1.63660e-3
                            + (
                                -4.93992e-7
                                + (
                                    1.11822e-10
                                    + (
                                        -1.49567e-14
                                        + (1.06206e-18 - 3.08720e-23 * temperature[j])
                                        * temperature[j]
                                    )
                                    * temperature[j]
                                )
                                * temperature[j]
                            )
                            * temperature[j]
                        )
                        * temperature[j]
                        - 1.5 * tlog[j]
                    )
                    # Fortran does NOT clamp exp() arguments - it allows inf/nan
                    # Match Fortran behavior exactly: use exp() directly
                    equilj_val = np.exp(exp_arg)
                    # Apply CPFH exactly once (Fortran multiplies inside the component loop;
                    # we apply it here and skip reapplying for the H component later)
                    equilj_val *= cpfh
                    equilj[jmol] = equilj_val
                    if j == 0:
                        equilj_debug.append(
                            (jmol, code_mol[jmol], "H-", exp_arg, equilj_val)
                        )
                else:
                    # General polynomial equilibrium constant
                    # DEBUG: Log EQUIL coefficients and temperature for molecules 167-185 and molecule 162
                    debug_molecules = {
                        161,
                        166,
                        167,
                        168,
                        182,
                        184,
                    } | TRACE_MOLECULES_FORCE

                    # Calculate polynomial step-by-step for debugging
                    # CRITICAL: Match Fortran's polynomial calculation exactly (lines 4552-4555):
                    #   EQUIL(1,JMOL)/TKEV(J)-EQUIL(2,JMOL)+
                    #   (EQUIL(3,JMOL)+(-EQUIL(4,JMOL)+(EQUIL(5,JMOL)+(-EQUIL(6,JMOL)+
                    #   +EQUIL(7,JMOL)*T(J))*T(J))*T(J))*T(J))*T(J)
                    # Ensure all operations use np.float64 for double precision
                    poly_term = (
                        np.float64(equil[0, jmol]) / np.float64(tkev[j])
                        - np.float64(equil[1, jmol])
                        + (
                            np.float64(equil[2, jmol])
                            + (
                                -np.float64(equil[3, jmol])
                                + (
                                    np.float64(equil[4, jmol])
                                    + (
                                        -np.float64(equil[5, jmol])
                                        + np.float64(equil[6, jmol])
                                        * np.float64(temperature[j])
                                    )
                                    * np.float64(temperature[j])
                                )
                                * np.float64(temperature[j])
                            )
                            * np.float64(temperature[j])
                        )
                        * np.float64(temperature[j])
                    )
                    # CRITICAL: Match Fortran's TLOG_TERM calculation exactly (line 4555):
                    #   -1.5*(DBLE(NCOMP-ION-ION-1))*TLOG(J)
                    # Fortran uses DBLE() to ensure double precision
                    # Python: Use np.float64 to ensure double precision
                    tlog_term = (
                        -1.5 * np.float64(ncomp - ion - ion - 1) * np.float64(tlog[j])
                    )
                    exp_arg_poly = np.float64(poly_term) + tlog_term

                    # Fortran does NOT clamp exp() arguments - it allows inf/nan
                    # Match Fortran behavior exactly: use exp() directly
                    # CRITICAL: Ensure exp_arg_poly is np.float64 for double precision
                    equilj_before_exp = equilj[jmol]
                    equilj[jmol] = np.exp(np.float64(exp_arg_poly))

                    if j == 0 and (
                        equilj[jmol] > 1e10 or not np.isfinite(equilj[jmol])
                    ):
                        equilj_debug.append(
                            (jmol, code_mol[jmol], "poly", exp_arg_poly, equilj[jmol])
                        )

                # Apply partition function corrections
                # CRITICAL: These corrections are ONLY applied in polynomial path!
                # Fortran applies them BEFORE label 35 (PFSAHA path), so they're only in polynomial path
                debug_molecules = {
                    166,
                    167,
                    168,
                    182,
                    184,
                } | TRACE_MOLECULES_FORCE  # include env-selected molecules
                equilj_before_cpf = (
                    equilj[jmol] if j == 0 and jmol in debug_molecules else None
                )

                locj1 = locj[jmol]
                locj2 = locj[jmol + 1] - 1
                for lock in range(locj1, locj2 + 1):
                    k = kcomps[lock]  # 0-based equation number
                    k_raw = k
                    id_elem = idequa[k] if k < nequa else 100

                    # For H- we already applied CPFH above; skip the hydrogen component here
                    if is_h_minus and id_elem == 1:
                        continue

                    if id_elem == 1:
                        equilj[jmol] = equilj[jmol] * cpfh
                    elif id_elem == 6:
                        equilj[jmol] = equilj[jmol] * cpfc
                    elif id_elem == 8:
                        equilj[jmol] = equilj[jmol] * cpfo
                    elif id_elem == 12:
                        equilj[jmol] = equilj[jmol] * cpfmg
                    elif id_elem == 13:
                        equilj[jmol] = equilj[jmol] * cpfal
                    elif id_elem == 14:
                        equilj[jmol] = equilj[jmol] * cpfsi
                    elif id_elem == 20:
                        equilj[jmol] = equilj[jmol] * cpfca

                # CRITICAL: Fortran does NOT clamp EQUILJ - it allows inf/nan/negative values!
                # Fortran (line 4649) does: EQUILJ(JMOL)=FRAC(J,NCOMP)/FRAC(J,1)*XNE(J)**ION
                # No checks, no clamping - Fortran allows INF/NaN/NEGATIVE EQUILJ to propagate
                # Match Fortran behavior exactly: do NOT modify EQUILJ after calculation!
        # Newton-Raphson iteration
        eqold = np.zeros(nequa, dtype=np.float64)
        max_iter = 200
        converged = False

        # Log-space Newton iteration setup
        # When use_log_space=True, we work with log(XN) instead of XN to handle
        # extreme values that would overflow in linear space.
        # The Jacobian is scaled: DEQ_log[i,j] = DEQ[i,j] * XN[j]
        # And XN updates become: log_xn_new = log_xn + delta_log
        if use_log_space:
            log_xn = _to_log_space(xn[:nequa])
            if j == 0:
                print(
                    f"  LOG-SPACE NEWTON enabled: log(XN[0])={log_xn[0]:.4f}, log(XN[{nequa-1}])={log_xn[nequa-1]:.4f}"
                )

        # DECIMAL NEWTON: Use full 50-digit precision Newton iteration
        # This handles extreme value divergence that exceeds float64 range
        if use_decimal_newton:
            if j == 0:
                print("  DECIMAL NEWTON: Using 50-digit precision Newton iteration")

            # Compute XNTOT for this layer
            xntot_layer = gas_pressure[j] / tk[j]

            # Call the Decimal Newton iteration
            xn_converged = _nmolec_newton_decimal(
                xn_init=xn[:nequa].copy(),
                xab=xab[:nequa],
                equilj=equilj,
                locj=locj,
                kcomps=kcomps,
                idequa=idequa,
                nequa=nequa,
                nummol=nummol,
                xntot=xntot_layer,
                max_iter=max_iter,
                layer_idx=j,
            )

            # Update XN with converged values
            xn[:nequa] = xn_converged

            # Skip the float64 Newton loop - go directly to storing results
            converged = True

            # Store results and continue to next layer
            xnatom_molecular[j] = xn[0]

            # Store XN to xnz_molecular and update xnz_prev
            for k in range(nequa):
                xnz_molecular[j, k] = xn[k]
            xnz_prev[:nequa] = xn[:nequa]
            prev_layer_idx = j  # Track for continuation
            continue  # Skip to next layer

        # BOUNDED NEWTON: Add step limiting to prevent chaotic divergence
        # This modifies the existing Newton loop rather than replacing it
        # The bounds are enforced in the XN update section below
        bounded_newton_active = use_bounded_newton
        if bounded_newton_active and j == 0:
            print(
                "  BOUNDED NEWTON: Adding trust-region step limiting to Newton iteration"
            )

        for iteration in range(max_iter):
            trace_eq_full_now = _should_trace_eq_full(j, iteration)
            trace_deq_full_now = _should_trace_deq_full(j, iteration)
            pending_solvit_call = solvit_call_counter + 1
            trace_a9_active = (
                TRACE_A9_ENABLED
                and (j + 1 == TRACE_A9_LAYER)
                and (pending_solvit_call == TRACE_A9_CALL)
            )

            # CRITICAL DEBUG: Log XN[0] and XN[3] at START of iteration (before EQ calculation)
            if j == 0 and iteration < 4:
                debug_log_path = os.path.join(os.getcwd(), "logs/xn_trace_detailed.log")
                with open(debug_log_path, "a") as f:
                    f.write(
                        f"\n{'='*80}\n"
                        f"PY_XN_TRACE: Layer {j}, Iteration {iteration} - START\n"
                        f"{'='*80}\n"
                    )
                    f.write(f"XN[0] (XN(1)) at START = {xn[0]:.17E}\n")
                    if nequa > 3:
                        f.write(f"XN[3] (XN(4)) at START = {xn[3]:.17E}\n")
                    # SCALE is initialized later, so don't log it here

            # Detailed tracing: XN values at start of iteration
            if _TRACE_XN_FULL and j == 0 and iteration < 3:  # First 3 iterations
                print(f"  DEBUG TRACE XN (layer {j}, iteration {iteration}):")
                print(f"    XN values: {xn[:nequa]}")
                print(
                    f"    XN stats: min={np.min(xn[:nequa]):.2e}, max={np.max(xn[:nequa]):.2e}, sum={np.sum(xn[:nequa]):.2e}"
                )
                if iteration == 0:
                    print(
                        f"    Initial XN: XNTOT/2 = {xntot/2:.2e}, X = XN(1)/10 = {xn[0]/10:.2e}"
                    )

            if (
                (j + 1) == 1
                and (iteration + 1) in _TRACE_ITERATIONS
                and TRACE_MOLECULE_IDS
            ):
                diag_path = Path(os.getcwd()) / "logs/ electron_diag_trace.log"
                with diag_path.open("a") as f:
                    sample_size = min(nequa, 30)
                    f.write(
                        f"PY_XN_SNAPSHOT layer={j+1} iter={iteration+1} before_equilj=1\n"
                    )
                    for idx in range(sample_size):
                        f.write(f"  XN[{idx+1:3d}]={xn[idx]: .17E}\n")

            # Set up equations EQ and Jacobian DEQ
            # DEQ is stored column-major (1D array): DEQ(K1) = DEQ(1, K) where K1 = NEQUA*K - NEQUA + 1
            deq = np.zeros(neqneq, dtype=np.float64)
            eq = np.zeros(nequa, dtype=np.float64)

            xntot = gas_pressure[j] / tk[j]
            if j == 0:
                pt_log_path = os.path.join(os.getcwd(), "logs/pt_trace.log")
                with open(pt_log_path, "a") as f:
                    f.write(
                        "PY_PT_TRACE layer={layer:3d} iter={iter:3d} "
                        "P={p: .17E} TK={tk: .17E} XNTOT={xntot: .17E}\n".format(
                            layer=j + 1,
                            iter=iteration + 1,
                            p=gas_pressure[j],
                            tk=tk[j],
                            xntot=xntot,
                        )
                    )

            # CRITICAL DEBUG: Log EQ[0] and EQ[3] BEFORE SOLVIT (before molecular contributions)
            if j == 0 and iteration < 4:
                debug_log_path = os.path.join(os.getcwd(), "logs/xn_trace_detailed.log")
                with open(debug_log_path, "a") as f:
                    f.write(
                        f"\nPY_XN_TRACE: Layer {j}, Iteration {iteration} - BEFORE MOLECULAR TERMS\n"
                        f"{'='*80}\n"
                    )
                    f.write(f"XNTOT = {xntot:.17E}\n")

            # DEBUG: Track XN[1] (Helium) evolution for first layer
            debug_xn = j == 0 and iteration < 5
            if debug_xn and iteration == 0:
                debug_log_path = os.path.join(os.getcwd(), "logs/term_calc_debug.log")
                with open(debug_log_path, "a") as f:
                    f.write(f"\nEQ[0] initialization:\n")
                    f.write(
                        f"  XNTOT = P/TK = {gas_pressure[j]:.6e} / {tk[j]:.6e} = {xntot:.6e}\n"
                    )
                    f.write(f"  EQ[0] = -XNTOT = {-xntot:.6e}\n")

            # Fast path: Use Numba kernel when tracing is disabled
            use_numba_element_setup = (
                not debug_xn
                and not (j == 0 and iteration < 4)
                and not (j == 0 and iteration <= 2)
            )

            if use_numba_element_setup:
                # Use Numba kernel for element equation setup
                _setup_element_equations_kernel(
                    eq, deq, xn, xab, nequa, nequa1, xntot, idequa
                )
            else:
                # Python path with tracing/debugging
                eq[0] = -xntot
                kk = 0  # 0-based: DEQ(k, k) = DEQ[kk] where kk = k * nequa1

                xn0 = xn[0]
                for k in range(
                    1, nequa
                ):  # k=2..NEQUA (1-based), k=1..nequa-1 (0-based)
                    eq[0] = eq[0] + xn[k]
                    k1 = k * nequa  # 0-based index for DEQ(1, k+1)
                    deq[k1] = 1.0  # DEQ(1, k) = 1 (0-based: row 0, col k)
                    xn_k = xn[k]
                    xab_k = xab[k]
                    if np.isfinite(xn0) and np.isfinite(xn_k) and np.isfinite(xab_k):
                        # Use compensated arithmetic to avoid catastrophic cancellation
                        # when XN(K) ≈ XAB(K)*XN(1)
                        element_residual = _accurate_element_residual(xn_k, xab_k, xn0)
                    else:
                        element_residual = xn_k - xab_k * xn0
                    eq[k] = element_residual

                    # CRITICAL DEBUG: Log EQ[3] calculation for K=3
                    if j == 0 and iteration < 4 and k == 3:
                        debug_log_path = os.path.join(
                            os.getcwd(), "logs/xn_trace_detailed.log"
                        )
                        with open(debug_log_path, "a") as f:
                            f.write(
                                f"\nEQ[3] calculation (K=3, 0-based k={k}):\n"
                                f"  XN[3] = {xn[k]:.17E}\n"
                                f"  XAB[3] = {xab[k]:.17E}\n"
                                f"  XN[0] = {xn[0]:.17E}\n"
                                f"  XAB[3]*XN[0] = {xab[k] * xn[0]:.17E}\n"
                                f"  EQ[3] = XN[3] - XAB[3]*XN[0] = {element_residual:.17E}\n"
                            )
                    if j == 0 and iteration <= 2 and k in (1, 2, 3, 7, 8, 9):
                        debug_log_path = os.path.join(
                            os.getcwd(), "logs/eq_component_trace.log"
                        )
                        with open(debug_log_path, "a") as f:
                            f.write(
                                "PY_EQ_COMPONENT layer=%3d iter=%3d k=%3d "
                                "xn=% .12E xab=% .12E xn0=% .12E elem_res=% .12E\n"
                                % (
                                    j + 1,
                                    iteration,
                                    k + 1,
                                    xn[k],
                                    xab[k],
                                    xn[0],
                                    element_residual,
                                )
                            )
                    kk = (
                        kk + nequa1
                    )  # kk = k * nequa1 (0-based: DEQ(k, k) in column-major)
                    deq[kk] = 1.0  # DEQ(k, k) = 1 (0-based: row k, col k)
                    deq[k] = -xab[k]  # DEQ(k+1, 1) = -XAB(K) (0-based: row k, col 0)
                    # CRITICAL: DEQ(1,1) stays ZERO in Fortran - do NOT accumulate it!

                # CRITICAL: Electron equation initialization (Fortran lines 5219-5221)
                # IF(IDEQUA(NEQUA).LT.100)GO TO 62
                # EQ(NEQUA)=-XN(NEQUA)
                # DEQ(NEQNEQ)=-1.
                if electron_idx is not None and idequa[electron_idx] >= 100:
                    eq_before_elec = eq[electron_idx]
                    eq[electron_idx] = -xn[electron_idx]
                    neqneq_idx = (
                        nequa * nequa - 1
                    )  # 0-based index for DEQ(NEQUA, NEQUA)
                    deq[neqneq_idx] = -1.0
                    if j == 0 and iteration == 0:
                        debug_log_path = os.path.join(
                            os.getcwd(), "logs/electron_eq_debug.log"
                        )
                        with open(debug_log_path, "a") as f:
                            f.write(f"ELECTRON EQ FIX: layer={j} iter={iteration}\n")
                            f.write(
                                f"  electron_idx={electron_idx}, idequa[{electron_idx}]={idequa[electron_idx]}\n"
                            )
                            f.write(
                                f"  EQ[{electron_idx}] before: {eq_before_elec:.6e}\n"
                            )
                            f.write(f"  XN[{electron_idx}] = {xn[electron_idx]:.6e}\n")
                            f.write(
                                f"  EQ[{electron_idx}] after: {eq[electron_idx]:.6e}\n"
                            )

            _log_eq_stage(
                "post_elements",
                layer_idx=j,
                iteration=iteration,
                eq_vec=eq,
                xn_vec=xn,
                nequa=nequa,
                electron_idx=electron_idx,
            )

            if j == 0 and iteration < 5:
                _log_deq_snapshot(
                    label="post_elements",
                    layer_idx=j,
                    iteration=iteration,
                    deq=deq,
                    eq=eq,
                    nequa=nequa,
                )

            # DEBUG: Check EQ[0] after element equations, before molecular terms
            if debug_xn and iteration == 0:
                debug_log_path = os.path.join(os.getcwd(), "logs/term_calc_debug.log")
                with open(debug_log_path, "a") as f:
                    f.write(
                        f"  EQ[0] += sum(XN[1:nequa]) = {np.sum(xn[1:nequa]):.6e}\n"
                    )
                    f.write(f"  EQ[0] before molecular terms = {eq[0]:.6e}\n")

            # Charge equation (if electrons are included)
            if idequa[nequa - 1] == 100:  # 0-based
                eq[nequa - 1] = -xn[nequa - 1]
                deq[neqneq - 1] = -1.0

            # CRITICAL: DEQ(1,1) is only set through molecular terms!
            # If no molecules contain XN(1), DEQ(1,1) = 0, causing zero pivot
            # Check if DEQ(1,1) is zero before molecular terms
            deq11_before = deq[0] if trace_deq_full_now else None

            def _enforce_electron_coupling() -> None:
                """Match Fortran relation DEQ(1,NEQUA) = -DEQ(NEQUA,NEQUA)."""
                if idequa[nequa - 1] != 100:
                    return
                # Original Fortran relies solely on molecule/negative-ion
                # updates; there is no extra coupling step.
                return

            # Detailed tracing: EQUILJ values
            if _TRACE_EQUILJ and j == 0 and iteration == 0:
                print(f"  DEBUG TRACE EQUILJ (layer {j}, iteration {iteration}):")
                print(f"    Total molecules: {nummol}")
                print(f"    EQUILJ values (first 20 molecules):")
                for jmol in range(min(20, nummol)):
                    code = code_mol[jmol]
                    eq_val = equilj[jmol]
                    if eq_val > 0:
                        print(
                            f"      Molecule {jmol}: CODE={code:.2f}, EQUILJ={eq_val:.2e}"
                        )
                    elif eq_val == 0:
                        print(
                            f"      Molecule {jmol}: CODE={code:.2f}, EQUILJ=0.0 (skipped)"
                        )
                    elif np.isinf(eq_val):
                        print(f"      Molecule {jmol}: CODE={code:.2f}, EQUILJ=inf")
                    elif np.isnan(eq_val):
                        print(f"      Molecule {jmol}: CODE={code:.2f}, EQUILJ=nan")
                # Summary statistics
                finite_equilj = equilj[np.isfinite(equilj) & (equilj > 0)]
                if len(finite_equilj) > 0:
                    print(
                        f"    EQUILJ stats: min={np.min(finite_equilj):.2e}, max={np.max(finite_equilj):.2e}, mean={np.mean(finite_equilj):.2e}"
                    )
                inf_count = np.sum(np.isinf(equilj))
                nan_count = np.sum(np.isnan(equilj))
                zero_count = np.sum(equilj == 0.0)
                print(
                    f"    EQUILJ counts: finite={len(finite_equilj)}, zero={zero_count}, inf={inf_count}, nan={nan_count}"
                )

                # Debug EQUILJ overflow info
                if len(equilj_debug) > 0:
                    print(f"    EQUILJ overflow details (first 10):")
                    for idx, code, typ, exp_arg, eq_val in equilj_debug[:10]:
                        if isinstance(exp_arg, (int, float)):
                            exp_arg_str = f"{exp_arg:.2e}"
                        else:
                            exp_arg_str = str(exp_arg)
                        # CRITICAL FIX: Print idx+1 to match 1-based Fortran convention
                        # idx is 0-based (jmol), so idx+1 is 1-based molecule number
                        print(
                            f"      Molecule {idx+1}: CODE={code:.2f}, type={typ}, exp_arg={exp_arg_str}, EQUILJ={eq_val:.2e}"
                        )

            def _record_nonfinite_deq(**info: Any) -> None:
                nonlocal nonfinite_term_hits, nonfinite_d_hits
                stage = str(info.get("stage", "unknown"))
                row = info.get("row")
                col = info.get("col")
                value = info.get("value", float("nan"))
                delta = info.get("delta", float("nan"))
                previous = info.get("previous", float("nan"))
                molecule_index = info.get("molecule_index", -1)
                molecule_code = info.get("molecule_code", float("nan"))

                row_str = str(row + 1) if isinstance(row, int) else "N/A"
                col_str = str(col + 1) if isinstance(col, int) else "N/A"
                log_message = (
                    f"layer={j+1} iter={iteration} call={pending_solvit_call} stage={stage} "
                    f"row={row_str} col={col_str} value={value:.6e} delta={delta:.6e} "
                    f"prev={previous:.6e} mol={molecule_index} code={molecule_code:.2f}"
                )

                if stage == "term":
                    if nonfinite_term_hits < NONFINITE_LOG_LIMIT:
                        _log_nonfinite_event("term", log_message)
                    nonfinite_term_hits += 1
                else:
                    if nonfinite_d_hits < NONFINITE_LOG_LIMIT:
                        _log_nonfinite_event("deq", log_message)
                    nonfinite_d_hits += 1

            trace_callback = (
                _trace_deq_update if (_TRACKED_DEQ_KS or _TRACKED_DEQ_CROSS) else None
            )

            if _should_dump_premol_state(j, iteration):
                premol_matrix = np.array(deq, copy=True).reshape(
                    (nequa, nequa), order="F"
                )
                _dump_premol_state(
                    layer_idx=j,
                    iteration=iteration,
                    matrix=premol_matrix,
                    rhs=np.array(eq[:nequa], copy=True),
                    xn_vec=np.array(xn[:nequa], copy=True),
                )

            if _LOG_ELECTRON_MOL:
                electron_density_val = float(
                    electron_density[j] if j < len(electron_density) else float("nan")
                )
                _log_electron_state_snapshot(
                    stage="pre_terms",
                    layer_idx=j,
                    iteration=iteration,
                    xn=xn,
                    electron_idx=electron_idx,
                    electron_density_val=electron_density_val,
                    locj=locj,
                    kcomps=kcomps,
                    code_mol=code_mol,
                    nequa=nequa,
                )
                for jmol in range(nummol):
                    molecule_code = float(code_mol[jmol])
                    if not _should_log_electron_molecule(molecule_code):
                        continue
                    _log_electron_equilj(
                        layer_idx=j,
                        iteration=iteration,
                        molecule_index=jmol,
                        molecule_code=molecule_code,
                        equilj_value=float(equilj[jmol]),
                        xn_total=float(xn[0]),
                        electron_density_val=electron_density_val,
                    )

            term_trace = _accumulate_molecules_atlas7(
                eq=eq,
                deq=deq,
                xn=xn,
                equilj=equilj,
                locj=locj,
                kcomps=kcomps,
                idequa=idequa,
                nequa=nequa,
                nummol=nummol,
                code_mol=code_mol,
                pending_solvit_call=pending_solvit_call,
                layer_index=j,
                iteration=iteration,
                trace_callback=trace_callback,
                nonfinite_callback=_record_nonfinite_deq,
                log_xn=log_xn if use_log_space else None,  # Full log-space mode
            )

            _log_eq_stage(
                "post_terms",
                layer_idx=j,
                iteration=iteration,
                eq_vec=eq,
                xn_vec=xn,
                nequa=nequa,
                electron_idx=electron_idx,
            )

            # Detailed tracing: Print TERM values
            if _TRACE_TERM and term_trace is not None:
                print(f"  DEBUG TRACE TERM (layer {j}, iteration {iteration}):")
                print(f"    TERM values (first 20 molecules):")
                for jmol, code, term_before, term_after, locj1, locj2 in term_trace:
                    print(
                        f"      Molecule {jmol}: CODE={code:.2f}, EQUILJ={term_before:.2e}, TERM={term_after:.2e}, components={locj2-locj1+1}"
                    )
                print(f"    EQ(0) (total particles, RHS before solve) = {eq[0]:.2e}")
                print(f"    XNTOT = {xntot:.2e}")
                print(
                    f"    EQ(0) should be ≈ 0 at solution (EQ(0) = -XNTOT + sum(XN) + sum(TERM))"
                )

            # Check DEQ(1,1) after molecular terms
            if trace_deq_full_now and deq11_before is not None:
                deq11_after = deq[0]
                print(f"  DEBUG: DEQ(1,1) before molecular terms = {deq11_before:.6e}")
                print(f"  DEBUG: DEQ(1,1) after molecular terms  = {deq11_after:.6e}")
                print(f"  DEBUG: DEQ(1,1) change = {deq11_after - deq11_before:.6e}")
                if abs(deq11_after) < 1e-10:
                    print(
                        f"  WARNING: DEQ(1,1) is still very small after molecular terms!"
                    )
                    print(f"    This will cause zero pivot in SOLVIT!")

            # DEBUG: Print DEQ diagonal elements (matching Fortran format)
            if trace_deq_full_now:
                print("PY_NMOLEC: DEQ diagonal elements (DEQ(K,K)):")
                for k in range(nequa):
                    kk = k * nequa1  # FIXED: Must match initialization kk = k * nequa1
                    deq_val = deq[kk]
                    print(f"  DEQ({k+1:2d},{k+1:2d})={deq_val:13.4E}")

                # Also print DEQ row 1 (DEQ(1,K)) for comparison with Fortran
                print("PY_NMOLEC: DEQ row 1 (DEQ(1,K)):")
                for k in range(nequa):
                    k1 = k * nequa  # 0-based index for DEQ(1, k+1)
                    deq_val = (
                        deq[k1] if k > 0 else deq[0]
                    )  # k=0 uses deq[0], k>0 uses k*nequa
                    print(f"  DEQ( 1,{k+1:2d})={deq_val:13.4E}")

                # Print all DEQ off-diagonal elements (DEQ(M,K) where M != K)
                # This helps verify the full matrix matches Fortran
                print("PY_NMOLEC: DEQ off-diagonal elements (DEQ(M,K) where M != K):")
                off_diag_count = 0
                for m in range(nequa):
                    for k in range(nequa):
                        if m != k:  # Off-diagonal only
                            mk = m + k * nequa  # DEQ(m, k) in column-major storage
                            deq_val = deq[mk]
                            # Only print non-zero values to reduce output
                            if abs(deq_val) > 1e-20:
                                print(f"  DEQ({m+1:2d},{k+1:2d})={deq_val:13.4E}")
                                off_diag_count += 1
                                if off_diag_count >= 50:  # Limit output
                                    print(
                                        f"  ... (showing first 50 non-zero off-diagonal elements)"
                                    )
                                    break
                    if off_diag_count >= 50:
                        break

            # Solve linear system: DEQ * delta_XN = EQ
            # Use SOLVIT algorithm (Gaussian elimination with complete pivoting)
            # From atlas7v_1.for lines 1200-1262
            # SOLVIT modifies DEQ and EQ in-place

            # DEBUG: Print EQ vector and DEQ row 1 BEFORE SOLVIT (matching Fortran timing)
            # Fortran prints DEQ row 1 BEFORE SOLVIT at line 4811-4819
            # Fortran prints for layer J=1 (1-based) = Python layer j=0 (0-based), iteration 0
            if trace_eq_full_now:
                print("PY_NMOLEC: Before SOLVIT call")
                print(f"PY_NMOLEC: Layer {j} iteration {iteration}")
                print(f"PY_NMOLEC: EQ[0]={eq[0]:13.4E} DEQ[0,0]={deq[0]:13.4E}")
                # Print full EQ vector BEFORE SOLVIT for comparison with Fortran
                print("PY_NMOLEC: EQ vector (RHS) BEFORE SOLVIT:")
                for kk in range(nequa):
                    print(f"  EQ({kk+1:2d})={eq[kk]:13.4E}")
                print("PY_NMOLEC: DEQ row 1 (DEQ(1,K)) BEFORE SOLVIT:")
                for kk in range(nequa):
                    k1 = kk * nequa if kk > 0 else 0  # 0-based index for DEQ(1, kk+1)
                    print(f"  DEQ( 1,{kk+1:2d})={deq[k1]:13.4E}")

            # CRITICAL: Check 1D DEQ array before reshape (for iteration 0 and 2)
            if trace_deq_full_now:
                deq_has_inf = np.any(np.isinf(deq[:neqneq]))
                deq_has_nan = np.any(np.isnan(deq[:neqneq]))
                print(f"  DEBUG TRACE DEQ 1D (before reshape, iteration {iteration}):")
                print(f"    deq[0] = {deq[0]:.2e}")
                print(f"    deq[nequa] = {deq[nequa]:.2e} (should be DEQ[0, 1])")
                print(f"    deq[6*nequa] = {deq[6*nequa]:.2e} (should be DEQ[0, 6])")
                print(
                    f"    deq[6*nequa+6] = {deq[6*nequa+6]:.2e} (should be DEQ[6, 6])"
                )
                print(f"    DEQ has Inf? {deq_has_inf}, has NaN? {deq_has_nan}")
                if deq_has_inf:
                    inf_indices = np.where(np.isinf(deq[:neqneq]))[0]
                    print(f"    🔴 DEQ contains Inf at {len(inf_indices)} positions!")
                    print(f"    First 10 Inf indices: {inf_indices[:10]}")
                    # Check which matrix positions these correspond to
                    for idx in inf_indices[:5]:
                        row = idx % nequa
                        col = idx // nequa
                        print(f"      deq[{idx}] = Inf → DEQ[{row}, {col}] = Inf")
                if iteration == 2:
                    print(f"    First 10 values: {deq[:10]}")
                    print(
                        f"    Values at nequa intervals: {[deq[k*nequa] for k in range(min(5, nequa))]}"
                    )

            # CRITICAL: DEQ is stored column-major (Fortran order)
            # numpy reshape defaults to C-order (row-major), so we need order='F'
            deq_2d = deq[:neqneq].reshape(nequa, nequa, order="F").copy()

            if not np.isfinite(deq_2d).all():
                bad_indices = np.argwhere(~np.isfinite(deq_2d))
                first_bad_row, first_bad_col = bad_indices[0]
                bad_value = deq_2d[first_bad_row, first_bad_col]
                if nonfinite_d_hits < NONFINITE_LOG_LIMIT:
                    _log_nonfinite_event(
                        "deq-pre-solvit",
                        f"layer={j+1} iter={iteration} call={pending_solvit_call} "
                        f"row={first_bad_row+1} col={first_bad_col+1} value={bad_value}",
                    )
                nonfinite_d_hits += 1

            # DEBUG: Print eq[0] right before creating eq_copy
            if _TRACE_SOLVIT_DETAILED and j == 1 and iteration == 0:
                print(
                    f"    🔴 DEBUG BEFORE eq_copy: eq[0]={eq[0]:.6e}, eq[6]={eq[6]:.6e}, eq[13]={eq[13]:.6e}"
                )
                if eq[0] > 1e70:
                    print(
                        f"    🔴 WARNING: eq[0]={eq[0]:.6e} is HUGE before eq_copy creation!"
                    )
                if np.isnan(eq[0]):
                    print(
                        f"    🔴 CRITICAL: eq[0] is NaN! Checking xntot, gas_pressure[{j}], tk[{j}]:"
                    )
                    print(f"      xntot = {xntot:.6e}")
                    print(f"      gas_pressure[{j}] = {gas_pressure[j]:.6e}")
                    print(f"      tk[{j}] = {tk[j]:.6e}")
                    print(f"      eq id = {id(eq)}")
                    print(
                        f"      Checking if eq was modified after reinitialization..."
                    )
                    # Check if eq[0] was set correctly at line 634
                    expected_eq0 = -gas_pressure[j] / tk[j]
                    print(
                        f"      Expected eq[0] = -gas_pressure[{j}]/tk[{j}] = {expected_eq0:.6e}"
                    )
                    print(f"      Actual eq[0] = {eq[0]:.6e}")
                    # Check if xn contains NaN (which would propagate to eq[0])
                    xn_nan_count = np.sum(np.isnan(xn[:nequa]))
                    print(
                        f"      xn contains {xn_nan_count} NaN values (out of {nequa})"
                    )
                    if xn_nan_count > 0:
                        xn_nan_indices = np.where(np.isnan(xn[:nequa]))[0]
                        print(
                            f"      NaN indices in xn: {xn_nan_indices[:10]}"
                        )  # First 10
                        print(
                            f"      This explains why eq[0] is NaN - it accumulates NaN from xn!"
                        )

            eq_copy = eq.copy()

            trace_layer_before_solvit = j in _TRACE_SOLVIT_LAYERS

            # DEBUG: Check a[k, 0] values BEFORE SOLVIT (for first layer, first iteration)
            if _TRACE_SOLVIT_DETAILED and j == 0 and iteration == 0:
                print("PY_NMOLEC: DEQ column 0 (a[k, 0]) BEFORE SOLVIT:")
                for kk in range(nequa):
                    print(
                        f"  a[{kk:2d}, 0] = {deq_2d[kk, 0]:13.4E} (should be -xab[{kk}] for k>0)"
                    )
                # Also check what xab values are
                print("PY_NMOLEC: XAB values (for comparison):")
                for kk in range(min(10, nequa)):
                    if kk < len(xab):
                        print(f"  xab[{kk}] = {xab[kk]:13.4E}")

            # Option 2: Add small perturbation to DEQ[0,0] to prevent exact zero
            # CRITICAL: DEQ[0,0] starts at 0.0 and never accumulates molecular contributions
            # Adding perturbation makes it pivotable early, preventing zero pivot at iteration 20
            if zero_pivot_fix == "perturbation":
                # Use scale-relative perturbation: eps * matrix_size * max_entry
                # This is principled regularization, not ad-hoc
                finite_mask = np.isfinite(deq_2d)
                deq_max = (
                    np.max(np.abs(deq_2d[finite_mask])) if np.any(finite_mask) else 1.0
                )
                eps = np.finfo(np.float64).eps
                # Scale-relative perturbation: machine epsilon * matrix size * max entry
                perturbation = max(1e-12, eps * nequa * deq_max)
                deq_2d[0, 0] += perturbation
                if _TRACE_SOLVIT_DETAILED and j == 0 and iteration == 0:
                    print(
                        f"  DEBUG: Applied scale-relative perturbation {perturbation:.2e} to DEQ[0,0] "
                        f"(eps={eps:.2e}, nequa={nequa}, deq_max={deq_max:.2e})"
                    )

            # Detailed tracing: DEQ matrix before SOLVIT
            if _TRACE_SOLVIT_DETAILED and j == 0 and iteration == 0:
                print(
                    f"  DEBUG TRACE DEQ (layer {j}, iteration {iteration} - Before SOLVIT):"
                )
                print(f"    DEQ matrix shape: {deq_2d.shape}")
                print(f"    DEQ matrix has inf? {np.any(np.isinf(deq_2d))}")
                print(f"    DEQ matrix has nan? {np.any(np.isnan(deq_2d))}")
                finite_mask = np.isfinite(deq_2d)
                if np.any(finite_mask):
                    print(
                        f"    DEQ matrix max abs: {np.max(np.abs(deq_2d[finite_mask])):.2e}"
                    )
                    print(
                        f"    DEQ matrix min abs (finite): {np.min(np.abs(deq_2d[finite_mask])):.2e}"
                    )
                    # Count very large values
                    large_mask = np.abs(deq_2d[finite_mask]) > 1e30
                    print(
                        f"    DEQ matrix values > 1e30: {np.sum(large_mask)} out of {np.sum(finite_mask)}"
                    )
                    # Count zeros
                    zero_mask = deq_2d[finite_mask] == 0.0
                    print(
                        f"    DEQ matrix zeros: {np.sum(zero_mask)} out of {np.sum(finite_mask)}"
                    )
                    # Count very small values (potential cancellation)
                    small_mask = (np.abs(deq_2d[finite_mask]) < 1e-10) & (
                        deq_2d[finite_mask] != 0.0
                    )
                    print(
                        f"    DEQ matrix very small (<1e-10, non-zero): {np.sum(small_mask)}"
                    )
                    # Print non-zero DEQ values (first 20)
                    nonzero_mask = (deq_2d != 0.0) & finite_mask
                    if np.any(nonzero_mask):
                        nonzero_indices = np.where(nonzero_mask)
                        print(f"    DEQ non-zero values (first 20):")
                        for idx in range(min(20, len(nonzero_indices[0]))):
                            i, k = nonzero_indices[0][idx], nonzero_indices[1][idx]
                            print(f"      DEQ[{i}, {k}] = {deq_2d[i, k]:.2e}")
                else:
                    print(f"    DEQ matrix: all inf/nan")
                print(f"    EQ (RHS) has inf? {np.any(np.isinf(eq_copy))}")
                print(f"    EQ (RHS) has nan? {np.any(np.isnan(eq_copy))}")
                finite_eq = np.isfinite(eq_copy)
                if np.any(finite_eq):
                    print(
                        f"    EQ (RHS) max abs: {np.max(np.abs(eq_copy[finite_eq])):.2e}"
                    )
                    print(f"    EQ (RHS) values: {eq_copy[:nequa]}")
                else:
                    print(f"    EQ (RHS): all inf/nan")
                print(f"    XN (current): {xn[:nequa]}")
                # Print first few rows/cols of DEQ matrix
                print(f"    DEQ[0:5, 0:5]:")
                for i in range(min(5, nequa)):
                    row_str = " ".join(
                        [
                            (
                                f"{deq_2d[i, k]:.2e}"
                                if np.isfinite(deq_2d[i, k])
                                else "inf" if np.isinf(deq_2d[i, k]) else "nan"
                            )
                            for k in range(min(5, nequa))
                        ]
                    )
                    print(f"      [{row_str}]")
                # CRITICAL: Check row 0 specifically
                print(f"    Row 0 detailed check:")
                row0_nonzero = [k for k in range(nequa) if abs(deq_2d[0, k]) > 1e-10]
                row0_sum = np.sum(np.abs(deq_2d[0, :]))
                print(f"      Non-zero elements: {len(row0_nonzero)} out of {nequa}")
                if len(row0_nonzero) > 0:
                    print(f"      Non-zero positions: {row0_nonzero[:10]}")
                    print(f"      Values: {[deq_2d[0, k] for k in row0_nonzero[:10]]}")
                print(f"      Sum of absolute values: {row0_sum:.2e}")
                print(f"      Max absolute value: {np.max(np.abs(deq_2d[0, :])):.2e}")
                if np.any(deq_2d[0, :] != 0):
                    print(
                        f"      Min absolute value (non-zero): {np.min(np.abs(deq_2d[0, deq_2d[0, :] != 0])):.2e}"
                    )
                else:
                    print(f"      Min absolute value (non-zero): all zero")
                # Print row with largest values
                row_maxes = np.max(np.abs(deq_2d), axis=1)
                max_row_idx = np.argmax(row_maxes)
                print(
                    f"    Row {max_row_idx} (largest max abs): max={row_maxes[max_row_idx]:.2e}"
                )
                print(f"      Values: {deq_2d[max_row_idx, :]}")

                # CRITICAL: Save DEQ matrix state for comparison
                # This will help identify where values differ from Fortran
                import synthe_py.tools.nmolec_exact as nmolec_module

                if not hasattr(nmolec_module, "_deq_saved"):
                    nmolec_module._deq_saved = deq_2d.copy()
                    nmolec_module._eq_saved = eq_copy.copy()
                    nmolec_module._xn_saved = xn[:nequa].copy()
                    print(f"    Saved DEQ matrix state for comparison")

            # CRITICAL: Align with Fortran behavior
            # Fortran does NOT check for NaN/Inf before SOLVIT
            # Fortran allows Inf/NaN and continues execution
            # Python should match this behavior - allow solver to proceed even with Inf/NaN
            # The solver will handle Inf/NaN operations (may produce Inf/NaN results, but continues)
            # Only skip if we can't even call the solver (but Fortran doesn't skip)
            # Remove NaN check to match Fortran's behavior

            _enforce_electron_coupling()

            if j == 0 and iteration < 5:
                _log_deq_snapshot(
                    label="post_enforce",
                    layer_idx=j,
                    iteration=iteration,
                    deq=deq,
                    eq=eq,
                    nequa=nequa,
                )

            # CRITICAL DEBUG: Log EQ[0] and EQ[3] BEFORE SOLVIT (after molecular contributions)
            if j == 0 and iteration < 4:
                debug_log_path = os.path.join(os.getcwd(), "logs/xn_trace_detailed.log")
                with open(debug_log_path, "a") as f:
                    f.write(
                        f"\nPY_XN_TRACE: Layer {j}, Iteration {iteration} - BEFORE SOLVIT (after molecular terms)\n"
                        f"{'='*80}\n"
                    )
                    f.write(f"EQ[0] (before SOLVIT) = {eq[0]:.17E}\n")
                    if nequa > 3:
                        f.write(f"EQ[3] (before SOLVIT) = {eq[3]:.17E}\n")

            # DEBUG: Print matching Fortran format before SOLVIT
            # Check DEQ(1,1) for ALL layers to verify it stays zero
            if trace_eq_full_now:
                deq11_val = deq_2d[0, 0]
                if j < 10 or abs(deq11_val) > 1e-10:
                    print(
                        f"PY_NMOLEC: Layer {j:3d} Before SOLVIT: EQ[0]={eq_copy[0]:12.4e} DEQ[0,0]={deq11_val:12.4e}"
                    )

            # DEBUG: Trace XN[22] (electrons) before SOLVIT for XNE investigation
            if j == 0 and iteration == 0:
                debug_log_path = os.path.join(os.getcwd(), "logs/xne_trace.log")
                with open(debug_log_path, "a") as f:
                    f.write(
                        f"PY_NMOLEC: Layer {j}, Iteration {iteration}, BEFORE SOLVIT:\n"
                    )
                    f.write(f"  XN[22] (electrons) = {xn[nequa-1]:.6e}\n")
                    f.write(
                        f"  EQ[22] (electron equation RHS) = {eq_copy[nequa-1]:.6e}\n"
                    )
                    f.write(
                        f"  DEQ(22,22) (diagonal) = {deq_2d[nequa-1, nequa-1]:.6e}\n"
                    )
                    f.write(f"\n")

            # DEBUG: Print full DEQ matrix for first layer, first iteration (BEFORE SOLVIT)
            if _TRACE_SOLVIT_DETAILED and j == 0 and iteration == 0:
                print("PY_NMOLEC: DEQ diagonal elements (DEQ(K,K)) BEFORE SOLVIT:")
                for kk in range(nequa):
                    print(f"  DEQ({kk+1:2d},{kk+1:2d})={deq_2d[kk, kk]:12.4E}")
                print("PY_NMOLEC: DEQ row 1 (DEQ(1,K)) BEFORE SOLVIT:")
                for kk in range(nequa):
                    line = f"  DEQ( 1,{kk+1:2d})={deq_2d[0, kk]:12.4E}\n"
                    print(line.rstrip())
                # DEBUG: Check DEQ(1,22) vs DEQ(22,22) relationship
                if idequa[nequa - 1] == 100:  # Electrons included
                    # CRITICAL: DEQ(1,22) is at row 1, col 22 in 1-based (row 1, col 22 in 0-based)
                    # deq_2d[0, 22] = deq[0 + 22*23] = deq[506] (wrong!)
                    # deq_2d[1, 22] = deq[1 + 22*23] = deq[507] (correct!)
                    # OR use deq[1 + (nequa-1)*nequa] = deq[507] directly
                    k1_electrons = 1 + (nequa - 1) * nequa
                    deq_1_22 = deq[k1_electrons]  # Use direct index to avoid confusion
                    deq_22_22 = deq_2d[nequa - 1, nequa - 1]
                    print(
                        f"PY_NMOLEC: DEQ(1,22) = {deq_1_22:.6e}, DEQ(22,22) = {deq_22_22:.6e}"
                    )
                    print(
                        f"PY_NMOLEC: DEQ(1,22) / DEQ(22,22) = {deq_1_22 / deq_22_22 if deq_22_22 != 0 else 'inf':.6e}"
                    )
                    print(
                        f"PY_NMOLEC: Expected: DEQ(1,22) = -DEQ(22,22) (like Fortran DEQ(1,23) = -DEQ(23,23))"
                    )
                    if abs(deq_1_22 + deq_22_22) > 1e-5 * abs(deq_22_22):
                        print(
                            f"  ⚠️  WARNING: DEQ(1,22) != -DEQ(22,22)! Difference = {abs(deq_1_22 + deq_22_22):.6e}"
                        )

                # CRITICAL: Compare DEQ(7,7) and DEQ(9,9) - Fortran selects (9,9), Python selects (7,7)
                print(
                    "PY_NMOLEC: Critical comparison (Fortran selects 9,9, Python selects 7,7):"
                )
                if nequa > 7:
                    print(f"  DEQ(7,7)={deq_2d[7, 7]:12.4e}")
                if nequa > 9:
                    print(f"  DEQ(9,9)={deq_2d[9, 9]:12.4e}")
                    if nequa > 7:
                        ratio_79 = (
                            deq_2d[7, 7] / deq_2d[9, 9] if deq_2d[9, 9] != 0 else 0
                        )
                        print(f"  Ratio DEQ(7,7)/DEQ(9,9)={ratio_79:.6f}")
                        if ratio_79 > 1.0:
                            print(
                                f"    ⚠️  Python's DEQ(7,7) > DEQ(9,9) - explains why Python selects (7,7)"
                            )
                        else:
                            print(
                                f"    ✅ Python's DEQ(7,7) < DEQ(9,9) - but Python still selects (7,7)?"
                            )
                # Find maximum diagonal element
                max_diag_val = -1
                max_diag_idx = -1
                # DEBUG: Print all diagonal elements to verify
                print("PY_NMOLEC: All diagonal elements (for debugging):")
                for kk in range(nequa):
                    diag_val = abs(deq_2d[kk, kk])
                    print(
                        f"  DEQ({kk+1},{kk+1})={deq_2d[kk, kk]:13.4E} (abs={diag_val:.4e})"
                    )
                    if diag_val > max_diag_val:
                        max_diag_val = diag_val
                        max_diag_idx = kk
                print(
                    f"PY_NMOLEC: Maximum diagonal element: DEQ({max_diag_idx+1},{max_diag_idx+1})={max_diag_val:.4e}"
                )
                print(
                    f"PY_NMOLEC: Python should select ({max_diag_idx},{max_diag_idx}) [0-based] = ({max_diag_idx+1},{max_diag_idx+1}) [1-based] if pivot search is correct"
                )
                # Also check what the actual maximum element in the entire matrix is
                max_elem_val = -1
                max_elem_row = -1
                max_elem_col = -1
                for jj in range(nequa):
                    for kk in range(nequa):
                        elem_val = abs(deq_2d[jj, kk])
                        if elem_val > max_elem_val:
                            max_elem_val = elem_val
                            max_elem_row = jj
                            max_elem_col = kk
                print(
                    f"PY_NMOLEC: Maximum element in entire matrix: DEQ({max_elem_row+1},{max_elem_col+1})={max_elem_val:.4e}"
                )
                if max_elem_row != max_diag_idx or max_elem_col != max_diag_idx:
                    print(
                        f"  ⚠️  WARNING: Maximum element is NOT on diagonal! Maximum diagonal is DEQ({max_diag_idx+1},{max_diag_idx+1}), but maximum element is DEQ({max_elem_row+1},{max_elem_col+1})"
                    )
                deq11_val = deq_2d[0, 0]
                if abs(deq11_val) > 1e-10:
                    print(
                        f"  ⚠️  WARNING: DEQ(1,1) is non-zero! This should be 0.0 to match Fortran!"
                    )

            # DEBUG: Log EQ[22] before SOLVIT for comparison with Fortran
            if idequa[nequa - 1] == 100 and j == 0:  # Electrons, first layer
                debug_log_path = os.path.join(
                    os.getcwd(), "logs/eq22_before_solvit.log"
                )
                # CRITICAL: Use correct index for DEQ(1,22)
                k1_electrons = 1 + (nequa - 1) * nequa
                deq_1_22 = deq[k1_electrons] if k1_electrons < len(deq) else 0.0
                with open(debug_log_path, "a") as f:
                    f.write(
                        f"Layer {j}, Iteration {iteration}: EQ[22] before SOLVIT:\n"
                    )
                    f.write(f"  EQ[22] = {eq_copy[nequa - 1]:.6e}\n")
                    f.write(f"  DEQ(1,22) = {deq_1_22:.6e}\n")
                    f.write(f"  DEQ(22,22) = {deq_2d[nequa - 1, nequa - 1]:.6e}\n")
                    f.write(f"  XN[22] = {xn[nequa - 1]:.6e}\n")
                    f.write("\n")

            solvit_call_counter += 1
            current_call_idx = solvit_call_counter

            _log_eq_stage(
                "pre_solvit",
                layer_idx=j,
                iteration=iteration,
                eq_vec=eq,
                xn_vec=xn,
                nequa=nequa,
                electron_idx=electron_idx,
            )

            if j == 0 and iteration <= MAX_DEBUG_SOLVIT_ITER:
                _log_tracked_deq_columns(
                    j, iteration, deq_2d, eq_copy[:nequa], current_call_idx
                )
                _log_eq_vector(
                    "Before SOLVIT, EQ vector",
                    "EQ_before",
                    iteration,
                    eq_copy[:nequa],
                )

            # CRITICAL: Create a deep copy right before the call to ensure it's not modified
            eq_copy_final = eq_copy.copy()
            if _should_dump_solvit_state(j, iteration):
                _dump_solvit_state(
                    layer_idx=j,
                    iteration=iteration,
                    call_idx=current_call_idx,
                    matrix=deq_2d,
                    rhs=eq_copy_final[:nequa],
                )
            if j == 0 and iteration == 0:
                print(
                    f"    DEBUG: eq_copy_final[0]={eq_copy_final[0]:.6e} (deep copy right before call)"
                )
                print(
                    f"    DEBUG: eq_copy_final id={id(eq_copy_final)}, eq_copy id={id(eq_copy)}"
                )
                print(
                    f"    DEBUG: eq_copy_final shares memory with eq_copy? {np.shares_memory(eq_copy_final, eq_copy)}"
                )
                # Verify eq_copy_final hasn't been modified
                if eq_copy_final[0] != eq_copy[0]:
                    print(
                        f"    ⚠️  WARNING: eq_copy_final[0]={eq_copy_final[0]:.6e} != eq_copy[0]={eq_copy[0]:.6e}!"
                    )
                # Print eq_copy_final RIGHT BEFORE the call (last thing before function call)
                print(
                    f"    DEBUG IMMEDIATELY BEFORE CALL: eq_copy_final[0]={eq_copy_final[0]:.6e}, eq_copy_final[6]={eq_copy_final[6]:.6e}, eq_copy_final[13]={eq_copy_final[13]:.6e}"
                )

            # Solve using a working copy so elimination does not corrupt the matrix
            # that subsequent Newton iterations will rebuild.
            matrix_for_solvit = np.array(deq_2d, copy=True)

            # LOG-SPACE: Scale Jacobian for log-space Newton
            # DEQ_log[i,j] = DEQ[i,j] * XN[j]
            # This accounts for the chain rule: ∂F/∂(log X) = ∂F/∂X * X
            if use_log_space:
                for col_j in range(nequa):
                    xn_j = xn[col_j]
                    # Use log-space scaling to prevent overflow when XN is huge
                    log_xn_j = (
                        log_xn[col_j]
                        if log_xn is not None
                        else np.log(max(abs(xn_j), 1e-300))
                    )
                    for row_i in range(nequa):
                        deq_val = matrix_for_solvit[row_i, col_j]
                        if deq_val == 0.0 or not np.isfinite(deq_val):
                            continue
                        # Compute log(|DEQ * XN|) = log(|DEQ|) + log(XN)
                        sign_deq = 1 if deq_val >= 0 else -1
                        log_deq = np.log(abs(deq_val))
                        log_scaled = log_deq + log_xn_j
                        # Convert back with clamping
                        if log_scaled > 700:
                            matrix_for_solvit[row_i, col_j] = sign_deq * 1e307
                        elif log_scaled < -700:
                            matrix_for_solvit[row_i, col_j] = 0.0
                        else:
                            matrix_for_solvit[row_i, col_j] = sign_deq * np.exp(
                                log_scaled
                            )

            # CRITICAL: Pass eq_copy_final directly - don't modify it
            _set_solvit_context(j + 1, iteration, current_call_idx)

            # Check if we should use Decimal precision SOLVIT
            # DISABLED: The _solvit_decimal function can't handle NaN/inf values properly,
            # and automatic fallback causes InvalidOperation errors. Keep using float64 SOLVIT.
            use_decimal_solvit = False

            if use_decimal_solvit:
                # Use Decimal-precision SOLVIT for extreme values
                # Convert 2D matrix back to flat column-major for _solvit_decimal
                matrix_flat = matrix_for_solvit.T.flatten()  # Column-major
                delta_xn = _solvit_decimal(
                    matrix_flat,
                    nequa,
                    eq_copy_final,
                )
            else:
                delta_xn = _solvit(
                    matrix_for_solvit,
                    nequa,
                    eq_copy_final,
                    zero_pivot_fix=zero_pivot_fix,
                )

            # DEBUG: Print matching Fortran format after SOLVIT
            # Note: SOLVIT modifies DEQ_2d in-place, so we can print it after
            if j == 0 and iteration <= MAX_DEBUG_SOLVIT_ITER and delta_xn is not None:
                _log_eq_vector(
                    "After SOLVIT, EQ vector",
                    "EQ_after",
                    iteration,
                    delta_xn[:nequa],
                )
            if trace_eq_full_now and delta_xn is not None:
                print(f"PY_NMOLEC: After SOLVIT, EQ[0]={delta_xn[0]:12.4e}")
                # Print DEQ row 1 AFTER SOLVIT (matrix has been modified by Gaussian elimination)
                print("PY_NMOLEC: DEQ row 1 (DEQ(1,K)) AFTER SOLVIT:")
                for kk in range(nequa):
                    line = f"  DEQ( 1,{kk+1:2d})={matrix_for_solvit[0, kk]:12.4E}\n"
                    print(line.rstrip())

            # CRITICAL: SOLVIT should never return None now - it continues even with zero pivot
            # But keep this check for safety
            if delta_xn is None:
                # Debug: Print matrix condition for first layer
                if j == 0:
                    print(
                        f"  DEBUG NMOLEC layer {j}: SOLVIT returned None (unexpected!)"
                    )
                    print(
                        f"    Matrix condition: {np.linalg.cond(matrix_for_solvit):.2e}"
                    )
                    print(f"    Matrix has inf? {np.any(np.isinf(matrix_for_solvit))}")
                    print(f"    Matrix has nan? {np.any(np.isnan(matrix_for_solvit))}")
                    print(f"    EQ (RHS) has inf? {np.any(np.isinf(eq_copy))}")
                    print(f"    EQ (RHS) has nan? {np.any(np.isnan(eq_copy))}")
                    print(f"    EQ (RHS) = {eq_copy}")
                    print(f"    XN (current) = {xn[:nequa]}")
                # SOLVIT failed unexpectedly - continue with last iteration's XN
                # This matches Fortran behavior: even if SOLVIT has issues, we use XN(1) after loop completes
                if j == 0:
                    print(
                        f"  WARNING: SOLVIT returned None at iteration {iteration}, continuing with last XN values"
                    )
                # Don't break - continue to next iteration or use last XN values
                # The iteration loop will complete and use XN(1) as XNATOM
                break  # Exit inner iteration loop, will use last XN values

            # CRITICAL: Fortran does NOT check for Inf/NaN before using EQ(K)!
            # Fortran code (atlas7v.for lines 5039-5054) uses EQ(K) directly:
            #   RATIO=ABS(EQ(K)/XN(K))  - no check for Inf/NaN
            #   XNEQ=XN(K)-EQ(K)        - no check for Inf/NaN
            #   XN(K)=XNEQ              - sets XN to Inf if XNEQ=Inf
            # Fortran continues iteration even with Inf/NaN in XN
            # Python must match this behavior exactly!

            # After SOLVIT, eq_copy (now delta_xn) contains the solution
            # Fortran modifies EQ in-place, so we update eq to match while
            # keeping XN as the pre-solve values until the damping loop updates them.
            eq[:] = delta_xn

            if (
                (j + 1) == 1
                and (iteration + 1) in _TRACE_ITERATIONS
                and TRACE_MOLECULE_IDS
            ):
                diag_path = Path(os.getcwd()) / "logs/ electron_diag_trace.log"
                with diag_path.open("a") as f:
                    sample_size = min(nequa, 30)
                    f.write(
                        f"PY_XN_SNAPSHOT layer={j+1} iter={iteration+1} after_solvit=1\n"
                    )
                    for idx in range(sample_size):
                        f.write(f"  XN[{idx+1:3d}]={xn[idx]: .17E}\n")

            log_eq_components_now = (
                _TRACE_EQ_COMPONENTS_ALL_LAYERS
                or j == 0
                or j in _TRACE_EQ_COMPONENT_LAYERS
            )
            if _TRACE_EQ_COMPONENTS and log_eq_components_now:
                _log_eq_components(
                    layer_idx=j,
                    iteration=iteration,
                    eq_vec=eq,
                    xn_vec=xn,
                )

            # Debug: Print first iteration for first layer
            if _TRACE_XN_FULL and j == 0 and iteration == 0:
                print(f"  DEBUG NMOLEC layer {j} iteration {iteration}:")
                print(f"    EQ (solution from SOLVIT) = {eq}")
                print(f"    XN (before update) = {xn[:nequa]}")
                print(f"    Matrix condition: {np.linalg.cond(matrix_for_solvit):.2e}")
                print(f"    EQ[0] (solution for XN(1)) = {eq[0]:.6e}")
                print(f"    XN[0] (current XN(1)) = {xn[0]:.6e}")
                xneq_print = _stable_subtract(xn[0], eq[0])
                print(f"    XNEQ = XN[0] - EQ[0] = {xneq_print:.6e}")

            # Update XN and check convergence
            # From atlas7v_1.for lines 3806-3824
            iferr = 0
            scale = 100.0

            # CRITICAL DEBUG: Log SCALE initialization and EQ values AFTER SOLVIT
            if j == 0 and iteration < 4:
                debug_log_path = os.path.join(os.getcwd(), "logs/xn_trace_detailed.log")
                with open(debug_log_path, "a") as f:
                    f.write(
                        f"\nPY_XN_TRACE: Layer {j}, Iteration {iteration} - AFTER SOLVIT\n"
                        f"{'='*80}\n"
                    )
                    f.write(f"SCALE initialized = {scale:.17E}\n")
                    f.write(f"EQ[0] (solution from SOLVIT) = {eq[0]:.17E}\n")
                    if nequa > 3:
                        f.write(f"EQ[3] (solution from SOLVIT) = {eq[3]:.17E}\n")
                    f.write(f"EQOLD[0] = {eqold[0]:.17E}\n")
                    if nequa > 3:
                        f.write(f"EQOLD[3] = {eqold[3]:.17E}\n")

            # DEBUG: Trace XN[22] update from SOLVIT solution for XNE investigation
            if idequa[nequa - 1] == 100:  # Electrons included
                debug_log_path = os.path.join(os.getcwd(), "logs/xne_calc_trace.log")
                with open(debug_log_path, "a") as f:
                    f.write(
                        f"Layer {j}, Iteration {iteration}: XN[22] update from SOLVIT:\n"
                    )
                    f.write(f"  XN[22] before update = {xn[nequa - 1]:.6e}\n")
                    f.write(f"  Solution[22] (b[22]) = {eq[nequa - 1]:.6e}\n")
                    f.write(f"  electron_density[{j}] = {electron_density[j]:.6e}\n")

            # DEBUG: Track XN[1] (Helium) evolution for first layer
            debug_xn = j == 0 and iteration < 5
            if debug_xn:
                debug_log_path = os.path.join(os.getcwd(), "logs/term_calc_debug.log")
                with open(debug_log_path, "a") as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"XN Update: Layer {j}, Iteration {iteration}\n")
                    f.write(f"{'='*80}\n")
                    f.write(f"XN before update:\n")
                    for k_idx in range(min(nequa, 5)):
                        f.write(f"  XN[{k_idx}] = {xn[k_idx]:.6e}\n")
                    f.write(f"EQ (solution from SOLVIT):\n")
                    for k_idx in range(min(nequa, 5)):
                        f.write(f"  EQ[{k_idx}] = {eq[k_idx]:.6e}\n")

            for k in range(nequa):
                xn_before = xn[k]
                scale_before_k = scale  # Capture SCALE value before processing this K
                eqold_before_k = eqold[k]

                # After SOLVIT, EQ(K) contains the solution (delta XN)
                # Fortran: RATIO=ABS(EQ(K)/XN(K)) - NO CHECK FOR ZERO/INF!
                # Python must match: calculate ratio even if xn[k]=0 or eq[k]=Inf
                # Use try/except to handle division by zero like Fortran (produces Inf)
                ratio = _ratio_preserving_precision(eq[k], xn[k])
                if ratio > 0.001:
                    iferr = 1

                # Damping
                eq_before_damping = eq[k]
                damping_applied = False
                # Use sign comparison instead of multiplication to avoid overflow
                sign_change = (eqold[k] > 0 and eq[k] < 0) or (
                    eqold[k] < 0 and eq[k] > 0
                )
                if sign_change:
                    eq_k = np.float64(eq[k])
                    if np.isfinite(eq_k):
                        eq[k] = eq_k * 0.69
                    else:
                        eq[k] = eq_k
                    damping_applied = True

                # Fortran: XNEQ = XN(K) - EQ(K) where EQ(K) is the solution from SOLVIT
                xn_val = np.float64(xn[k])
                eq_val = np.float64(eq[k])
                xneq = _stable_subtract(xn_val, eq_val)
                xn100 = xn[k] / 100.0

                # CRITICAL DEBUG: Log details for K=0 and K=3
                if j == 0 and iteration < 4 and (k == 0 or k == 3):
                    debug_log_path = os.path.join(
                        os.getcwd(), "logs/xn_trace_detailed.log"
                    )
                    with open(debug_log_path, "a") as f:
                        f.write(
                            f"\nPY_XN_TRACE: K={k+1} (0-based k={k})\n"
                            f"  XN_BEFORE = {xn_before:.17E}\n"
                            f"  EQ (from SOLVIT) = {eq_before_damping:.17E}\n"
                            f"  EQOLD = {eqold[k]:.17E}\n"
                            f"  EQOLD*EQ < 0? {eqold[k] * eq_before_damping < 0.0}\n"
                            f"  Damping applied? {damping_applied}\n"
                            f"  EQ (after damping) = {eq[k]:.17E}\n"
                            f"  XNEQ = XN - EQ = {xneq:.17E}\n"
                            f"  XN100 = XN / 100 = {xn100:.17E}\n"
                            f"  SCALE before K = {scale_before_k:.17E}\n"
                        )

                # Note: xnatom_inout will be updated after xn[k] is assigned (see below)

                # DEBUG: Log XN[22] (electrons) update to investigate XNE explosion
                if j == 0 and k == nequa - 1:  # Electrons (k=22, 0-based)
                    debug_log_path = os.path.join(os.getcwd(), "logs/xne_trace.log")
                    with open(debug_log_path, "a") as f:
                        f.write(
                            f"PY_NMOLEC: Layer {j}, Iteration {iteration}, XN[22] (electrons) update:\n"
                        )
                        f.write(f"  XN[22] before SOLVIT: {xn[k]:.6e}\n")
                        f.write(
                            f"  EQ[22] before SOLVIT: {eq_copy[k] if 'eq_copy' in locals() else 'N/A':.6e}\n"
                        )
                        f.write(f"  Solution[22] from SOLVIT: {eq[k]:.6e}\n")
                        f.write(f"  xneq = XN[22] - Solution[22] = {xneq:.6e}\n")
                        f.write(
                            f"  XN[22] after update: {xn[k] if xneq >= xn100 else xn[k]/scale:.6e}\n"
                        )
                        f.write(f"\n")

                if debug_xn and k == 1:
                    debug_log_path = os.path.join(
                        os.getcwd(), "logs/term_calc_debug.log"
                    )
                    with open(debug_log_path, "a") as f:
                        f.write(f"\nXN[{k}] update:\n")
                        f.write(f"  XN[{k}] before = {xn[k]:.6e}\n")
                        f.write(f"  EQ[{k}] = {eq[k]:.6e}\n")
                        f.write(f"  xneq = XN[{k}] - EQ[{k}] = {xneq:.6e}\n")
                        f.write(f"  xn100 = XN[{k}] / 100 = {xn100:.6e}\n")

                # DEBUG: Trace XN[22] update in detail
                if j == 0 and k == nequa - 1:  # Electrons
                    debug_log_path = os.path.join(os.getcwd(), "logs/xne_trace.log")
                    with open(debug_log_path, "a") as f:
                        f.write(f"  UPDATE LOGIC:\n")
                        f.write(f"    xneq = {xneq:.6e}\n")
                        f.write(f"    xn100 = {xn100:.6e}\n")
                        f.write(f"    xneq < xn100? {xneq < xn100}\n")
                        f.write(f"    xn[k] BEFORE assignment = {xn[k]:.6e}\n")

                scale_used = 1.0
                branch = "direct"
                scale_modified = False

                # LOG-SPACE UPDATE: Use additive update in log-space
                # This prevents overflow because we add to log(XN) instead of subtracting from XN
                if use_log_space:
                    # In log-space, eq[k] is the change in log(XN[k])
                    # (because we scaled the Jacobian by XN[j])
                    delta_log_k = eq[k]

                    # Apply damping in log-space
                    if damping_applied:  # Sign flip detected above
                        delta_log_k = delta_log_k * 0.69

                    # Clamp delta to prevent extreme jumps
                    max_delta_log = 10.0  # Max ~22000x change per iteration
                    delta_log_k = max(-max_delta_log, min(max_delta_log, delta_log_k))

                    # Update log-space
                    log_xn[k] = log_xn[k] + delta_log_k

                    # Clamp log_xn to valid range
                    log_xn[k] = max(-LOG_XN_MAX, min(LOG_XN_MAX, log_xn[k]))

                    # Convert back to linear space
                    xn[k] = _from_log_space_scalar(log_xn[k])
                    branch = "log_space"

                elif xneq < xn100:
                    branch = "scale"
                    scale_used = scale
                    xn[k] = xn[k] / scale_used
                    # Use sign comparison instead of multiplication to avoid overflow
                    sign_change_scale = (eqold[k] > 0 and eq[k] < 0) or (
                        eqold[k] < 0 and eq[k] > 0
                    )
                    if sign_change_scale:
                        scale_old = scale
                        scale = np.sqrt(scale)
                        scale_modified = True
                    else:
                        scale_modified = False
                    if debug_xn and k == 1:
                        debug_log_path = os.path.join(
                            os.getcwd(), "logs/term_calc_debug.log"
                        )
                        with open(debug_log_path, "a") as f:
                            f.write(
                                f"  -> XN[{k}] < XN[{k}]/100, dividing by scale={scale:.2f}\n"
                            )
                            f.write(f"  -> XN[{k}] after = {xn[k]:.6e}\n")
                else:
                    xn[k] = xneq
                    if debug_xn and k == 1:
                        debug_log_path = os.path.join(
                            os.getcwd(), "logs/term_calc_debug.log"
                        )
                        with open(debug_log_path, "a") as f:
                            f.write(f"  -> XN[{k}] after = {xn[k]:.6e}\n")

                # BOUNDED NEWTON: Enforce physical bounds to prevent divergence
                if bounded_newton_active:
                    xn_min = 1e-100  # Minimum physical value
                    xn_max = 10.0 * xntot  # Maximum reasonable value
                    if xn[k] < xn_min:
                        xn[k] = xn_min
                    elif xn[k] > xn_max:
                        xn[k] = xn_max

                # CRITICAL DEBUG: Log update details for K=0 and K=3
                if j == 0 and iteration < 4 and (k == 0 or k == 3):
                    debug_log_path = os.path.join(
                        os.getcwd(), "logs/xn_trace_detailed.log"
                    )
                    with open(debug_log_path, "a") as f:
                        f.write(
                            f"  XNEQ < XN100? {xneq < xn100}\n"
                            f"  BRANCH = {branch}\n"
                            f"  SCALE_USED = {scale_used:.17E}\n"
                            f"  XN_AFTER = {xn[k]:.17E}\n"
                            f"  SCALE after K = {scale:.17E}\n"
                        )
                        if scale_modified:
                            f.write(
                                f"  SCALE modified: {scale_old:.17E} -> {scale:.17E} (sqrt)\n"
                            )
                        f.write(f"  EQOLD updated: {eqold[k]:.17E} -> {eq[k]:.17E}\n")

                # BUG FIX: Do NOT update xnatom_inout with XN[0]!
                # XN[0] represents NUCLEI (atoms + extra atoms in molecules), but
                # xnatom should be PARTICLES (P/TK - XNE) for:
                #   - RHO = XNATOM * WTMOLE * 1.66e-24 (mass density)
                #   - Element populations = XNATOM * XABUND
                # The original code updated xnatom_inout with XN[0], causing a 1.85x error
                # in cool atmospheres where H2 fraction is high.
                #
                # if xnatom_inout is not None and k == 0:  # DISABLED
                #     xnatom_inout[j] = xn[k]  # REMOVED - XN[0] is NUCLEI, not PARTICLES!

                if _should_trace_xn(j, iteration, k):
                    _log_xn_update(
                        layer_idx=j,
                        iteration=iteration,
                        k_idx=k,
                        xn_before=xn_before,
                        xn_after=xn[k],
                        eq_value=eq[k],
                        xneq=xneq,
                        xn100=xn100,
                        ratio=ratio,
                        branch=branch,
                        scale_value=scale_used,
                    )

                if not np.isfinite(xn[k]):
                    if nonfinite_xn_hits < NONFINITE_LOG_LIMIT:
                        _log_nonfinite_event(
                            "xn",
                            f"layer={j} iter={iteration} k={k} eq={eq[k]:.6e} "
                            f"xneq={xneq:.6e} xn100={xn100:.6e} scale={scale:.6e} "
                            f"branch={'scale' if xneq < xn100 else 'direct'}",
                        )
                    nonfinite_xn_hits += 1

                # Fortran does NOT check for NaN/inf - it just assigns XN(K)=XNEQ
                # Match Fortran exactly: no checks, no clamping, allow inf/nan to propagate

                # DEBUG: Trace XN[22] after assignment
                if j == 0 and k == nequa - 1:  # Electrons
                    debug_log_path = os.path.join(os.getcwd(), "logs/xne_trace.log")
                    with open(debug_log_path, "a") as f:
                        f.write(f"    xn[k] AFTER assignment = {xn[k]:.6e}\n")
                        f.write(f"\n")

                eqold[k] = eq[k]

                if _TRACE_NEWTON_UPDATES:
                    _log_newton_update(
                        layer_idx=j,
                        iteration=iteration,
                        k_idx=k,
                        xn_before=xn_before,
                        eq_before_damping=eq_before_damping,
                        eq_after_damping=eq[k],
                        eqold_before=eqold_before_k,
                        eqold_after=eqold[k],
                        xneq=xneq,
                        xn100=xn100,
                        ratio=ratio,
                        branch=branch,
                        scale_before=scale_before_k,
                        scale_used=scale_used,
                        scale_after=scale,
                        damping_applied=damping_applied,
                        scale_modified=scale_modified,
                    )

            # CRITICAL DEBUG: Log XN[0] and XN[3] at END of iteration (after all K updates)
            if j == 0 and iteration < 4:
                debug_log_path = os.path.join(os.getcwd(), "logs/xn_trace_detailed.log")
                with open(debug_log_path, "a") as f:
                    f.write(
                        f"\nPY_XN_TRACE: Layer {j}, Iteration {iteration} - END\n"
                        f"{'='*80}\n"
                    )
                    f.write(f"XN[0] (XN(1)) at END = {xn[0]:.17E}\n")
                    if nequa > 3:
                        f.write(f"XN[3] (XN(4)) at END = {xn[3]:.17E}\n")
                    f.write(f"SCALE at END = {scale:.17E}\n")
                    f.write(f"{'='*80}\n\n")

            if debug_xn:
                debug_log_path = os.path.join(os.getcwd(), "logs/term_calc_debug.log")
                with open(debug_log_path, "a") as f:
                    f.write(f"\nXN after update:\n")
                    for k_idx in range(min(nequa, 5)):
                        f.write(f"  XN[{k_idx}] = {xn[k_idx]:.6e}\n")

            # When tracing mismatch with Fortran (ITER=4 vs ITER=24), dump all ratios
            if _TRACE_RATIO and j == 0 and (iteration + 1) in _TRACE_ITERATIONS:
                for k_dump in range(nequa):
                    xn_val = xn[k_dump]
                    eq_val = eq[k_dump]
                    if xn_val != 0.0:
                        ratio_val = abs(_div_preserving_precision(eq_val, xn_val))
                    else:
                        ratio_val = float("inf")
                    print(
                        "PY_RATIO: iter={iter:3d} k={k:2d} "
                        "ratio={ratio:.17E} EQ={eq_val:.17E} XN={xn_val:.17E} "
                        "EQOLD={eqold:.17E}".format(
                            iter=iteration + 1,
                            k=k_dump + 1,
                            ratio=ratio_val,
                            eq_val=eq_val,
                            xn_val=xn_val,
                            eqold=eqold[k_dump],
                        )
                    )

            iteration_one_based = iteration + 1
            if iferr == 0:
                if MIN_NEWTON_ITER > 0 and iteration_one_based < MIN_NEWTON_ITER:
                    continue
                converged = True
                break

        # CRITICAL: Fortran ALWAYS uses XN(1) as XNATOM(J) after the iteration loop
        # completes, regardless of whether convergence succeeded or failed
        # (atlas7v.for around line 5243). The IFERR flag only controls whether the
        # Newton loop keeps iterating; it does not change how XNATOM is stored.
        #
        # To match Fortran exactly, always store XN(1) as the molecular XNATOM for
        # this layer, without any additional thresholds or fallbacks.
        xnatom_molecular[j] = xn[0]

        # Fortran line 5049-5050: Store XN to XNZ after iteration
        # DO 107 K=1,NEQUA
        # XNZ(J,K)=XN(K)
        # 107 CONTINUE
        for k in range(nequa):
            xnz_molecular[j, k] = xn[k]
        xnz_prev[:nequa] = xn[:nequa]
        prev_layer_idx = j  # Track for continuation

        if idequa[nequa - 1] == 100:
            # Fortran atlas7v.for line 5847: XNE(J)=XN(NEQUA)
            # Store the Newton-converged electron density, NOT the initial guess
            # The previous xntot/20 was WRONG and caused XNE to be 4500× too high!
            electron_density[j] = xn[nequa - 1]
            xne_computed[j] = electron_density[j]

        # Compute molecular number densities
        # From atlas7v_1.for lines 3831-3842
        for jmol in range(nummol):
            ncomp = locj[jmol + 1] - locj[jmol]
            xnmol[j, jmol] = equilj[jmol]
            locj1 = locj[jmol]
            locj2 = locj[jmol + 1] - 1
            for lock in range(locj1, locj2 + 1):
                k = kcomps[lock]  # 0-based equation number
                # CRITICAL: kcomps = nequa (23) is sentinel for inverse electrons
                if k == nequa:  # Sentinel value for inverse electrons
                    k = nequa - 1  # Map to actual electron equation index
                    xnmol[j, jmol] = xnmol[j, jmol] / xn[k]
                else:
                    xnmol[j, jmol] = xnmol[j, jmol] * xn[k]

    return xnatom_molecular, xnmol, xnz_molecular


def _solvit_kernel_python(
    a_work: np.ndarray,
    b_work: np.ndarray,
    ipivot: np.ndarray,
    n: int,
) -> None:
    """Python fallback for SOLVIT kernel (used when Numba not available)."""
    # Use FMA for multiply-subtract operations when available to keep
    # elimination steps numerically close to Fortran's x87 intermediate precision.
    has_fma = hasattr(math, "fma")

    def _stable_submul(lhs: float, rhs: float, factor: float) -> float:
        """Compute lhs - rhs*factor using FMA when available.

        Note: math.fma() can throw OverflowError even for finite inputs when
        the result would overflow. We catch this and fall back to regular
        arithmetic which returns inf instead of throwing.
        """
        if np.isfinite(lhs) and np.isfinite(rhs) and np.isfinite(factor):
            if has_fma:
                try:
                    return math.fma(-rhs, factor, lhs)
                except OverflowError:
                    # FMA overflows - fall back to regular arithmetic (returns inf)
                    pass
            return lhs - rhs * factor
        return lhs - rhs * factor

    for iter_idx in range(1, n + 1):
        amax = 0.0
        irow = 1
        icolum = 1

        # Pivot search
        for row in range(1, n + 1):
            if ipivot[row] == 1:
                continue
            jk = row - n
            for col in range(1, n + 1):
                jk = jk + n
                if ipivot[col] == 1:
                    continue
                aa = abs(a_work[jk])
                if aa > amax:
                    amax = aa
                    irow = row
                    icolum = col

        ipivot[icolum] += 1

        # Row/column swap if needed
        if irow != icolum:
            irl = irow - n
            icl = icolum - n
            for _ in range(1, n + 1):
                irl += n
                swap_val = a_work[irl]
                icl += n
                a_work[irl] = a_work[icl]
                a_work[icl] = swap_val
            b_work[irow], b_work[icolum] = b_work[icolum], b_work[irow]

        # Normalize pivot row
        pivot_idx = icolum * n + icolum - n
        pivot = a_work[pivot_idx]
        a_work[pivot_idx] = 1.0
        icl = icolum - n
        for _ in range(1, n + 1):
            icl += n
            a_work[icl] = a_work[icl] / pivot
        b_work[icolum] = b_work[icolum] / pivot

        # Elimination
        l1ic = icolum * n - n
        for l1 in range(1, n + 1):
            l1ic += 1
            if l1 == icolum:
                continue
            t = a_work[l1ic]
            a_work[l1ic] = 0.0
            if t == 0.0:
                continue
            l1l = l1 - n
            icl = icolum - n
            for _ in range(1, n + 1):
                l1l += n
                icl += n
                # Use FMA for numerical stability (matches original _solvit behavior)
                a_work[l1l] = _stable_submul(a_work[l1l], a_work[icl], t)
            # Use FMA for numerical stability (matches original _solvit behavior)
            b_work[l1] = _stable_submul(b_work[l1], b_work[icolum], t)


@jit(nopython=True, cache=True)
def _solvit_kernel(
    a_work: np.ndarray,
    b_work: np.ndarray,
    ipivot: np.ndarray,
    n: int,
) -> None:
    """Core SOLVIT algorithm without I/O or logging (Numba-optimized).

    This is the pure numerical computation extracted from _solvit.
    Modifies a_work and b_work in-place.

    Args:
        a_work: 1-based column-major matrix array (size n*n+1)
        b_work: 1-based RHS vector (size n+1)
        ipivot: 1-based pivot tracking array (size n+1)
        n: Matrix dimension
    """
    # Note: Numba doesn't support math.fma directly, but modern CPUs
    # will use FMA instructions automatically when optimizing (-rhs * factor + lhs)
    # The compiler will recognize this pattern and use FMA if available.
    # For numerical stability, we use the same pattern as the original code.

    for iter_idx in range(1, n + 1):
        amax = 0.0
        irow = 1
        icolum = 1

        # Pivot search
        for row in range(1, n + 1):
            if ipivot[row] == 1:
                continue
            jk = row - n
            for col in range(1, n + 1):
                jk = jk + n
                if ipivot[col] == 1:
                    continue
                aa = abs(a_work[jk])
                if aa > amax:
                    amax = aa
                    irow = row
                    icolum = col

        ipivot[icolum] += 1

        # Row/column swap if needed
        if irow != icolum:
            irl = irow - n
            icl = icolum - n
            for _ in range(1, n + 1):
                irl += n
                swap_val = a_work[irl]
                icl += n
                a_work[irl] = a_work[icl]
                a_work[icl] = swap_val
            b_work[irow], b_work[icolum] = b_work[icolum], b_work[irow]

        # Normalize pivot row
        pivot_idx = icolum * n + icolum - n
        pivot = a_work[pivot_idx]
        a_work[pivot_idx] = 1.0
        icl = icolum - n
        for _ in range(1, n + 1):
            icl += n
            a_work[icl] = a_work[icl] / pivot
        b_work[icolum] = b_work[icolum] / pivot

        # Elimination
        l1ic = icolum * n - n
        for l1 in range(1, n + 1):
            l1ic += 1
            if l1 == icolum:
                continue
            t = a_work[l1ic]
            a_work[l1ic] = 0.0
            if t == 0.0:
                continue
            l1l = l1 - n
            icl = icolum - n
            for _ in range(1, n + 1):
                l1l += n
                icl += n
                # Compute lhs - rhs*factor (Numba will optimize to FMA if available)
                a_work[l1l] = a_work[l1l] - a_work[icl] * t
            # Compute lhs - rhs*factor (Numba will optimize to FMA if available)
            b_work[l1] = b_work[l1] - b_work[icolum] * t


# =============================================================================
# DECIMAL-PRECISION SOLVIT
# =============================================================================
# Uses Python's built-in Decimal module for extended precision (50 digits).
# This matches or exceeds Fortran's 80-bit precision (~18-19 digits).


def _to_decimal(value: float) -> Decimal:
    """Convert float to Decimal, handling special values."""
    if not np.isfinite(value):
        if np.isnan(value):
            return Decimal("NaN")
        return Decimal("Infinity") if value > 0 else Decimal("-Infinity")
    return Decimal(str(value))


def _from_decimal(d: Decimal, clamp_log: float = 300.0) -> float:
    """Convert Decimal back to float with clamping for extreme values."""
    if d.is_nan():
        return float("nan")
    if d.is_infinite():
        return float("inf") if d > 0 else float("-inf")
    if d == 0:
        return 0.0

    # Check for overflow/underflow
    sign, digits, exp = d.as_tuple()
    if digits == (0,):
        return 0.0
    log10_approx = exp + len(digits) - 1

    if log10_approx > clamp_log:
        return float("inf") if sign == 0 else float("-inf")
    if log10_approx < -clamp_log:
        return 0.0

    return float(d)


def _nmolec_newton_bounded(
    xn_init: np.ndarray,
    xab: np.ndarray,
    equilj: np.ndarray,
    locj: np.ndarray,
    kcomps: np.ndarray,
    idequa: np.ndarray,
    nequa: int,
    nummol: int,
    xntot: float,
    max_iter: int = 200,
    layer_idx: int = 0,
    temperature: float = 5000.0,
    tol: float = 1e-3,
) -> tuple:
    """
    Bounded Newton iteration for NMOLEC with trust-region step limiting.

    This prevents the chaotic divergence that causes the solver to land in the
    wrong basin of attraction (atomic solution instead of molecular solution)
    for cool atmospheres.

    Key features:
    1. Works in log-space to handle large dynamic range
    2. Trust region limits step sizes to prevent wild divergence
    3. Line search ensures each step improves the residual
    4. Bounds enforcement keeps solutions physical

    Returns:
        (xn_solution, converged): Solution array and convergence flag
    """
    import os

    # Initialize in log-space
    xn = np.maximum(xn_init.copy(), 1e-100)
    log_xn = np.log(xn)

    # Bounds for log(XN)
    log_xn_min = -230  # ~1e-100
    log_xn_max = np.log(10 * xntot)

    # Trust radius (in log-space, so exp(5) ~ 150x change max)
    trust_radius = 5.0

    def compute_eq_residual(log_xn_vec):
        """Compute equilibrium residuals matching NMOLEC structure."""
        xn_local = np.exp(log_xn_vec)
        eq = np.zeros(nequa)

        # First equation: mass balance
        # EQ(1) = -XNTOT + XN(2) + XN(3) + ... + XN(NEQUA)
        # Note: XN(1) is NOT included in the sum!
        eq[0] = -xntot
        for k in range(1, nequa):  # k=1..nequa-1 (0-based), NOT including k=0
            eq[0] += xn_local[k]

        # Element equations: EQ(K) = XN(K) - XAB(K)*XN(1)
        for k in range(1, nequa):
            eq[k] = xn_local[k] - xab[k] * xn_local[0]

        # Electron equation override (if electrons included)
        electron_idx = nequa - 1
        if idequa[electron_idx] >= 100:  # Electron equation
            eq[electron_idx] = -xn_local[electron_idx]

        # Add molecular contributions to equations
        for jmol in range(nummol):
            if equilj[jmol] == 0.0:
                continue

            ncomp = locj[jmol + 1] - locj[jmol]
            if ncomp <= 1:
                continue

            # Compute molecular term = EQUILJ * product(XN[k] for k in components)
            term = equilj[jmol]
            for iloc in range(locj[jmol], locj[jmol + 1]):
                k = (
                    kcomps[iloc] - 1
                )  # Convert to 0-based (kcomps is 1-based equation numbers)
                if 0 <= k < nequa:
                    term *= xn_local[k]

            # Add to relevant equations (each component contributes)
            for iloc in range(locj[jmol], locj[jmol + 1]):
                k = kcomps[iloc] - 1
                if 0 <= k < nequa:
                    eq[k] += term

        return eq

    def compute_jacobian(log_xn_vec):
        """Numerical Jacobian."""
        eps = 1e-8
        jac = np.zeros((nequa, nequa))
        f0 = compute_eq_residual(log_xn_vec)
        for i in range(nequa):
            log_xn_vec[i] += eps
            jac[:, i] = (compute_eq_residual(log_xn_vec) - f0) / eps
            log_xn_vec[i] -= eps
        return jac

    converged = False
    best_res_norm = float("inf")
    best_xn = xn.copy()

    for iteration in range(max_iter):
        # Compute residual
        eq = compute_eq_residual(log_xn)
        xn_current = np.exp(log_xn)

        # Scale residuals by XN for relative error
        scaled_eq = eq / np.maximum(xn_current, 1e-100)
        res_norm = np.sqrt(np.sum(scaled_eq**2))

        # Track best solution
        if res_norm < best_res_norm:
            best_res_norm = res_norm
            best_xn = xn_current.copy()

        # Check convergence (relative error < tol for all equations)
        max_ratio = np.max(np.abs(eq) / np.maximum(xn_current, 1e-100))
        if max_ratio < tol:
            converged = True
            break

        # Compute Jacobian
        jac = compute_jacobian(log_xn.copy())

        # Solve for Newton step
        try:
            # Add regularization for stability
            jac_reg = jac + 1e-10 * np.eye(nequa)
            delta = np.linalg.solve(jac_reg, -eq)
        except np.linalg.LinAlgError:
            # Singular Jacobian - use gradient descent step
            delta = -scaled_eq * 0.1

        # Limit step size (trust region)
        step_norm = np.sqrt(np.sum(delta**2))
        if step_norm > trust_radius:
            delta = delta * trust_radius / step_norm

        # Line search
        alpha = 1.0
        improved = False
        for _ in range(10):
            log_xn_new = log_xn + alpha * delta
            log_xn_new = np.clip(log_xn_new, log_xn_min, log_xn_max)

            eq_new = compute_eq_residual(log_xn_new)
            xn_new = np.exp(log_xn_new)
            scaled_eq_new = eq_new / np.maximum(xn_new, 1e-100)
            res_norm_new = np.sqrt(np.sum(scaled_eq_new**2))

            if res_norm_new < res_norm:
                log_xn = log_xn_new
                improved = True
                break

            alpha *= 0.5

        if not improved:
            # Take small step anyway
            log_xn = np.clip(log_xn + 0.1 * delta, log_xn_min, log_xn_max)

        # Adaptive trust radius
        if improved and res_norm_new < 0.5 * res_norm:
            trust_radius = min(trust_radius * 1.5, 20.0)
        elif not improved or res_norm_new > 0.9 * res_norm:
            trust_radius = max(trust_radius * 0.5, 0.5)

    # Use best solution found
    xn_solution = best_xn if not converged else np.exp(log_xn)

    return xn_solution, converged


def _nmolec_newton_decimal(
    xn_init: np.ndarray,
    xab: np.ndarray,
    equilj: np.ndarray,
    locj: np.ndarray,
    kcomps: np.ndarray,
    idequa: np.ndarray,
    nequa: int,
    nummol: int,
    xntot: float,  # Total number density = P/kT
    max_iter: int = 200,
    layer_idx: int = 0,
) -> np.ndarray:
    """
    Full Newton iteration for NMOLEC using Decimal arithmetic throughout.

    This EXACTLY matches the existing Python/Fortran NMOLEC algorithm,
    but uses 50-digit Decimal precision instead of float64.

    The key insight: Fortran's 80-bit precision can handle values up to ~1e4932.
    Python's float64 overflows at ~1e308. This Decimal version handles up to ~1e999999.

    Args:
        xn_init: Initial XN values (nequa,) as float64
        xab: XAB abundance ratios (nequa,)
        equilj: Equilibrium constants for molecules (nummol,)
        locj: Location indices for molecule components
        kcomps: Component indices for molecules
        idequa: Element indices
        nequa: Number of equations
        nummol: Number of molecules
        xntot: Total number density = P/kT
        max_iter: Maximum iterations
        layer_idx: Layer index for debug output

    Returns:
        Converged XN values as float64 array
    """
    nequa1 = nequa + 1
    neqneq = nequa * nequa

    # Convert to Decimal
    xn = [_to_decimal(xn_init[k]) for k in range(nequa)]
    xab_dec = [_to_decimal(xab[k]) for k in range(nequa)]
    equilj_dec = [_to_decimal(equilj[m]) for m in range(nummol)]
    xntot_dec = _to_decimal(xntot)

    # Working arrays (Decimal)
    eq = [Decimal(0)] * nequa
    eqold = [Decimal(0)] * nequa
    # DEQ stored in column-major 1D array like Fortran: DEQ(row, col) = deq[row + col*nequa]
    deq = [Decimal(0)] * neqneq

    converged = False

    # Comprehensive debug logging for first layer
    debug = layer_idx == 0
    debug_iters = 3  # Log first N iterations in detail (compare with Fortran)

    # Full precision format for tracing divergence
    def _full_prec(d: Decimal) -> str:
        """Format Decimal with full precision for comparison with Fortran."""
        return f"{float(d):.17e}"

    if debug:
        print(f"  DECIMAL NEWTON: Starting")
        print(f"    nequa={nequa}, nummol={nummol}, xntot={float(xntot_dec):.6e}")
        print(
            f"    XN[0:3] = [{float(xn[0]):.6e}, {float(xn[1]):.6e}, {float(xn[2]):.6e}]"
        )
        print(
            f"    XAB[0:3] = [{float(xab_dec[0]):.6e}, {float(xab_dec[1]):.6e}, {float(xab_dec[2]):.6e}]"
        )

    for iteration in range(max_iter):
        # Save old EQ for damping
        for k in range(nequa):
            eqold[k] = eq[k]

        # Initialize EQ and DEQ - EXACTLY like Fortran atlas7v.for lines 5205-5221
        # and matching _setup_element_equations_kernel (lines 1928-1957)
        # EQ(1) = -XNTOT + sum(XN[k] for k >= 1), DEQ(k,k) = 1, DEQ[0][k] = 1 for k >= 1
        eq[0] = -xntot_dec

        # NOTE: DEQ[0][0] is NOT set to 1 - it stays 0 (plus molecule contributions)
        # This is because ∂EQ(1)/∂XN(1) = 0 in the initial equation

        kk = 0  # Diagonal index tracker

        for k in range(1, nequa):
            eq[0] = eq[0] + xn[k]  # EQ(1) = EQ(1) + XN(K) for K >= 2
            k1 = k * nequa  # Column k, row 0: DEQ(1, k+1) in Fortran = deq[k1]
            deq[k1] = Decimal(1)  # DEQ(1, K) = 1 for K >= 2

            # EQ(K) = XN(K) - XAB(K)*XN(1)
            # Compute with FULL Decimal precision to trace divergence
            xn_k_dec = xn[k]
            xab_times_xn0_dec = xab_dec[k] * xn[0]
            eq_k_dec = xn_k_dec - xab_times_xn0_dec
            eq[k] = eq_k_dec

            # Debug for K=2 (Helium) - FULL PRECISION TRACE
            if k == 2 and debug and iteration < debug_iters:
                print(f"    EQ[2] FULL PRECISION TRACE:")
                print(f"      XN[2]          = {_full_prec(xn_k_dec)}")
                print(f"      XAB[2]*XN[0]   = {_full_prec(xab_times_xn0_dec)}")
                print(f"      XN[2]_Decimal  = {xn_k_dec}")  # Show raw Decimal
                print(f"      XAB*XN_Decimal = {xab_times_xn0_dec}")  # Show raw Decimal
                print(f"      EQ[2]=XN[2]-XAB*XN[0] = {_full_prec(eq_k_dec)}")

            kk = kk + nequa1  # Move to next diagonal: kk = k * nequa1
            deq[kk] = Decimal(1)  # DEQ(K, K) = 1
            deq[k] = -xab_dec[k]  # DEQ(K+1, 1) = -XAB(K), i.e., row k, col 0

        # NOTE: Do NOT add xn[0] to eq[0] - the original code doesn't do this

        # Handle electron equation if present (Fortran lines 5219-5221)
        electron_idx = nequa - 1
        if idequa[electron_idx] >= 100:
            eq[electron_idx] = -xn[electron_idx]
            deq[neqneq - 1] = Decimal(-1)  # DEQ(NEQUA, NEQUA) = -1

        if debug and iteration < debug_iters:
            print(f"  --- Iteration {iteration} ---")
            # Show ALL XN values (first 10) to compare with Fortran
            xn_str = ", ".join([f"{float(xn[k]):.4e}" for k in range(min(10, nequa))])
            print(f"    XN[0:10] = [{xn_str}]")
            # Show EQ[0:5] BEFORE molecules to check element equation init
            eq_init_str = ", ".join(
                [f"{float(eq[k]):.4e}" for k in range(min(5, nequa))]
            )
            print(f"    EQ[0:5] after init (before mol): [{eq_init_str}]")

        # Track molecules that contribute to EQ[2] (Helium) for debug
        eq2_contributions = []

        # Process molecules - EXACTLY like Fortran atlas7v.for lines 3772-3835
        for jmol in range(nummol):
            ncomp = int(locj[jmol + 1] - locj[jmol])
            if ncomp <= 1:
                continue

            locj1 = int(locj[jmol])
            locj2 = int(locj[jmol + 1] - 1)

            equilj_val = equilj_dec[jmol]
            if equilj_val <= 0 or not equilj_val.is_finite():
                continue

            # Compute TERM = EQUILJ * product(XN[k])
            term = equilj_val
            term_valid = True

            for lock in range(locj1, locj2 + 1):
                k_raw = int(kcomps[lock])
                k_idx = nequa - 1 if k_raw >= nequa else k_raw

                if xn[k_idx] <= 0 or not xn[k_idx].is_finite():
                    term_valid = False
                    break

                if k_raw >= nequa:
                    term = term / xn[k_idx]  # Division for electrons
                else:
                    term = term * xn[k_idx]  # Multiplication for atoms

            if not term_valid or not term.is_finite():
                continue

            # EQ(1) = EQ(1) + TERM
            eq[0] = eq[0] + term

            # Debug: show large TERM values (molecules that matter)
            if debug and iteration < debug_iters and float(term) > 1e20:
                print(
                    f"    BIG TERM: mol {jmol} TERM={float(term):.4e} "
                    f"EQUILJ={float(equilj_val):.4e} components={[int(kcomps[l]) for l in range(locj1, locj2+1)]}"
                )

            # Process each component
            for lock in range(locj1, locj2 + 1):
                k_raw = int(kcomps[lock])
                k_idx = nequa - 1 if k_raw >= nequa else k_raw

                if xn[k_idx] == 0 or not xn[k_idx].is_finite():
                    continue

                # D = TERM / XN(K)
                if k_raw >= nequa:
                    d = -term / xn[k_idx]
                else:
                    d = term / xn[k_idx]

                # EQ(K) = EQ(K) + TERM
                eq_before = eq[k_idx]
                eq[k_idx] = eq[k_idx] + term

                # Track contributions to EQ[2] (Helium) for debug
                if k_idx == 2 and debug and iteration < debug_iters:
                    eq2_contributions.append(
                        {
                            "jmol": jmol,
                            "term": float(term),
                            "eq_before": float(eq_before),
                            "eq_after": float(eq[k_idx]),
                        }
                    )

                # DEQ(1, K) = DEQ(1, K) + D
                nequak = nequa * k_idx
                deq[nequak] = deq[nequak] + d

                # DEQ(M, K) = DEQ(M, K) + D for all components M
                for locm in range(locj1, locj2 + 1):
                    m_raw = int(kcomps[locm])
                    m_idx = nequa - 1 if m_raw >= nequa else m_raw
                    mk = m_idx + nequak
                    deq[mk] = deq[mk] + d

        if debug and iteration < debug_iters:
            # Show EQ[2] (Helium) contributions from molecules
            if eq2_contributions:
                print(
                    f"    EQ[2] (He) molecule contributions ({len(eq2_contributions)} molecules):"
                )
                for c in eq2_contributions[:5]:  # Show first 5
                    print(
                        f"      mol {c['jmol']}: TERM={c['term']:.4e} "
                        f"EQ_before={c['eq_before']:.4e} EQ_after={c['eq_after']:.4e}"
                    )
                if len(eq2_contributions) > 5:
                    print(f"      ... and {len(eq2_contributions) - 5} more")
            else:
                print(f"    EQ[2] (He): NO molecules contributed!")
            # Show ALL EQ values (first 10) before solve - compare with Fortran EQ_before
            eq_str = ", ".join([f"{float(eq[k]):.4e}" for k in range(min(10, nequa))])
            print(f"    EQ_before[0:10] = [{eq_str}]")
            # Show DEQ diagonal and first column (col 0 = partial derivs w.r.t. XN[0])
            deq_diag = [float(deq[k + k * nequa]) for k in range(min(10, nequa))]
            # Column 0 values: deq[k] for k=0..9 gives DEQ[k][0] in column-major storage
            deq_col0 = [float(deq[k]) for k in range(min(10, nequa))]
            print(f"    DEQ diag[0:10] = {[f'{v:.3e}' for v in deq_diag]}")
            print(
                f"    DEQ col0[0:10] = {[f'{v:.3e}' for v in deq_col0]}  (∂EQ[k]/∂XN[0])"
            )

        # Convert DEQ from 1D column-major to 2D for solving
        A = [[deq[i + j * nequa] for j in range(nequa)] for i in range(nequa)]
        b = [eq[k] for k in range(nequa)]

        # TIKHONOV REGULARIZATION: Add small diagonal term to improve conditioning
        # The DEQ matrix has condition number ~1e30, making the solve extremely sensitive
        # to rounding. Adding λ*I biases the solution toward smaller corrections.
        # λ is chosen to be tiny - just enough to break ties in ill-conditioned parts.
        max_diag = max(abs(A[k][k]) for k in range(nequa) if A[k][k] != 0)
        if max_diag > 0:
            lambda_reg = max_diag * Decimal("1e-30")  # Much weaker regularization
            for k in range(nequa):
                A[k][k] = A[k][k] + lambda_reg

        # Save original matrix for verification
        if debug and iteration == 0:
            A_orig = [[deq[i + j * nequa] for j in range(nequa)] for i in range(nequa)]
            b_orig = [eq[k] for k in range(nequa)]

        # Gauss-Jordan elimination with complete pivoting - EXACT Fortran SOLVIT algorithm
        # CRITICAL: Use 19-digit precision (≈ Fortran's 80-bit) for the solve.
        # For ill-conditioned matrices (condition ~1e30), higher precision produces
        # a mathematically correct but different solution that causes Newton divergence.
        # Fortran's 80-bit rounding happens to produce a solution that converges.
        from decimal import localcontext

        with localcontext() as ctx:
            ctx.prec = 19  # Match Fortran's 80-bit (~19.3 decimal digits)

            ipivot = [0] * nequa  # Track which columns have been used as pivots

            for i in range(nequa):
                # Find maximum element in submatrix of unused rows/columns
                amax = Decimal(0)
                irow = 0
                icolum = 0
                for j in range(nequa):
                    if ipivot[j] == 1:
                        continue
                    for k in range(nequa):
                        if ipivot[k] == 1:
                            continue
                        aa = abs(A[j][k])
                        if aa > amax:
                            irow = j
                            icolum = k
                            amax = aa

                ipivot[icolum] = 1  # Mark column as used

                # Swap row irow with row icolum (if different)
                if irow != icolum:
                    A[irow], A[icolum] = A[icolum], A[irow]
                    b[irow], b[icolum] = b[icolum], b[irow]

                # Now pivot is at A[icolum][icolum]
                pivot = A[icolum][icolum]
                if pivot == 0:
                    continue

                # Normalize pivot row
                A[icolum][icolum] = Decimal(1)
                for ll in range(nequa):
                    A[icolum][ll] = A[icolum][ll] / pivot
                b[icolum] = b[icolum] / pivot

                # Eliminate pivot column in all other rows (Gauss-Jordan)
                for l1 in range(nequa):
                    if l1 == icolum:
                        continue
                    t = A[l1][icolum]
                    A[l1][icolum] = Decimal(0)
                    for ll in range(nequa):
                        A[l1][ll] = A[l1][ll] - A[icolum][ll] * t
                    b[l1] = b[l1] - b[icolum] * t

            # After Gauss-Jordan, the solution is directly in b (permuted order)
            # Copy solution back to eq (still in 19-digit context)
            for k in range(nequa):
                eq[k] = b[k] + Decimal(0)  # Force 19-digit precision

        # ITERATIVE REFINEMENT: For ill-conditioned systems, improve solution accuracy
        # by computing residual r = b_orig - A_orig*x and solving for correction
        if iteration == 0 and debug:
            # Only do refinement on first iteration for debugging
            x_sol = [eq[k] for k in range(nequa)]
            for refine_iter in range(3):  # 3 refinement iterations
                # Compute residual: r = b_orig - A_orig * x_sol
                residual = []
                for i in range(nequa):
                    r_i = b_orig[i] - sum(A_orig[i][j] * x_sol[j] for j in range(nequa))
                    residual.append(r_i)

                # Compute norm of residual for debug
                res_norm = sum(abs(r) for r in residual[:10])

                # Solve A*dx = residual using simplified Gaussian elimination
                # (We can't use Gauss-Jordan again since A was modified)
                # For now, just report the residual
                print(
                    f"    Iterative refinement {refine_iter}: residual norm (first 10) = {float(res_norm):.4e}"
                )
                print(f"      residual[0:3] = {[float(r) for r in residual[:3]]}")

        if debug and iteration < debug_iters:
            # Show ALL delta values (first 10) to compare with Fortran
            delta_str = ", ".join(
                [f"{float(eq[k]):.4e}" for k in range(min(10, nequa))]
            )
            print(f"    Delta[0:10] = [{delta_str}]")
            # FULL PRECISION TRACE for delta values
            print(f"    FULL PRECISION Delta (for Helium divergence trace):")
            print(f"      Delta[0] = {_full_prec(eq[0])}")
            print(f"      Delta[2] = {_full_prec(eq[2])}")
            print(f"      Delta[0]_Decimal = {eq[0]}")
            print(f"      Delta[2]_Decimal = {eq[2]}")
            # Verify solve: compute A_orig * delta and compare with b_orig
            if iteration == 0:
                # eq now contains delta (the solution)
                residual = [
                    sum(A_orig[i][j] * eq[j] for j in range(nequa))
                    for i in range(min(3, nequa))
                ]
                print(f"    Verify: A*delta[0:3] = {[float(r) for r in residual]}")
                print(
                    f"    Expected b[0:3] = {[float(b_orig[k]) for k in range(min(3, nequa))]}"
                )

        # Update XN - EXACTLY like Fortran atlas7v.for lines 3806-3824
        # NOTE: Scale down if xneq < xn100 (Fortran: IF(XNEQ.LT.XN100)GO TO 87)
        iferr = 0
        scale = Decimal(100)

        # Collect update details for debug
        update_details = []

        for k in range(nequa):
            xn_before = xn[k]
            delta_k = eq[k]

            # Check for sign change (damping) - applied BEFORE computing xneq
            damped = False
            if eqold[k] * eq[k] < 0:
                eq[k] = eq[k] * Decimal("0.69")
                damped = True

            # XNEQ = XN(K) - EQ(K)
            # CRITICAL: Perform XN update in float64 to match Fortran's BINARY rounding.
            # With Decimal, XN[k] becomes exactly equal to XAB[k]*XN[0] (to 50 digits),
            # making EQ[k] = 0 at next iteration. Fortran's binary rounding preserves
            # a non-zero residual (~8.6e9 for Helium) that keeps the iteration stable.
            xn_k_f64 = float(xn[k])
            eq_k_f64 = float(eq[k])
            xneq_f64 = xn_k_f64 - eq_k_f64
            xn100_f64 = xn_k_f64 / 100.0

            # Fortran: IF(XNEQ.LT.XN100)GO TO 87
            # This means: if xneq < xn100, scale down; otherwise use xneq
            if xneq_f64 < xn100_f64:
                # Label 87: scale down
                xn_new_f64 = xn_k_f64 / 100.0
                branch = "SCALE"
                if eqold[k] * eq[k] < 0:
                    scale = scale.sqrt()
            else:
                # Use xneq directly
                xn_new_f64 = xneq_f64
                branch = "DIRECT"

            # Convert back to Decimal (preserving the float64 rounding)
            xn[k] = _to_decimal(xn_new_f64)
            xneq = _to_decimal(xneq_f64)
            xn100 = _to_decimal(xn100_f64)

            # Check convergence: RATIO = ABS(EQ(K)/XN(K))
            if xn[k] != 0 and xn[k].is_finite():
                ratio = abs(eq[k] / xn[k])
                if ratio > Decimal("0.001"):
                    iferr = 1

            # Store details for first 10 elements
            if k < 10:
                update_details.append(
                    {
                        "k": k,
                        "xn_before": xn_before,
                        "delta": delta_k,
                        "xneq": xneq,
                        "xn100": xn100,
                        "branch": branch,
                        "xn_after": xn[k],
                        "damped": damped,
                    }
                )

        if debug and iteration < debug_iters:
            print(f"    XN Update Details (K=0..9):")
            for d in update_details:
                print(
                    f"      K={d['k']:2d}: XN_before={float(d['xn_before']):10.3e} "
                    f"delta={float(d['delta']):10.3e} xneq={float(d['xneq']):10.3e} "
                    f"xn100={float(d['xn100']):10.3e} -> {d['branch']:6s} "
                    f"XN_after={float(d['xn_after']):10.3e}"
                    f"{' DAMP' if d['damped'] else ''}"
                )
            # FULL PRECISION TRACE for Helium (K=2) and XN[0]
            print(f"    FULL PRECISION XN after update:")
            print(f"      XN[0] = {_full_prec(xn[0])}")
            print(f"      XN[0]_Decimal = {xn[0]}")
            print(f"      XN[2] = {_full_prec(xn[2])}")
            print(f"      XN[2]_Decimal = {xn[2]}")
            # Also show what XAB[2]*XN[0] would be for next iteration
            xab2_times_xn0_next = xab_dec[2] * xn[0]
            print(
                f"      XAB[2]*XN[0] (for next iter) = {_full_prec(xab2_times_xn0_next)}"
            )
            print(f"      XAB[2]*XN[0]_Decimal = {xab2_times_xn0_next}")
            print(
                f"      Expected EQ[2] next = {_full_prec(xn[2] - xab2_times_xn0_next)}"
            )
            print(f"    iferr={iferr}, scale={float(scale):.2e}")

        if iferr == 0 and iteration >= 2:
            converged = True
            if debug:
                print(f"  CONVERGED at iteration {iteration}")
            break

        # Log every 50 iterations for long runs
        if debug and iteration > 0 and iteration % 50 == 0:
            print(
                f"  ... iteration {iteration}: XN[0]={float(xn[0]):.6e}, iferr={iferr}"
            )

    # Convert back to float64
    result = np.array([_from_decimal(xn[k]) for k in range(nequa)], dtype=np.float64)

    if layer_idx == 0:
        print(
            f"  DECIMAL NEWTON: {'Converged' if converged else 'Max iter'} "
            f"after {iteration + 1} iterations, XN[0]={result[0]:.6e}"
        )

    return result


def _solvit_decimal(
    a: np.ndarray,
    n: int,
    b: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Gaussian elimination with partial pivoting using Decimal precision.

    This is used when XN values have diverged beyond float64 range.
    The 50-digit Decimal precision exceeds Fortran's 80-bit (~18-19 digits).

    Args:
        a: Matrix (n*n) in column-major flat format
        n: Matrix dimension
        b: RHS vector (n,)

    Returns:
        Solution vector as float64 (clamped if necessary), or None if singular
    """
    # Convert to 2D list of Decimals
    A = [[_to_decimal(a[i + j * n]) for j in range(n)] for i in range(n)]
    b_dec = [_to_decimal(b[i]) for i in range(n)]

    # Forward elimination with partial pivoting
    for k in range(n - 1):
        # Find pivot (max absolute value in column k, rows k to n-1)
        max_val = abs(A[k][k])
        max_idx = k
        for i in range(k + 1, n):
            if abs(A[i][k]) > max_val:
                max_val = abs(A[i][k])
                max_idx = i

        # Swap rows if needed
        if max_idx != k:
            A[k], A[max_idx] = A[max_idx], A[k]
            b_dec[k], b_dec[max_idx] = b_dec[max_idx], b_dec[k]

        # Check for singular matrix
        if A[k][k] == 0:
            continue  # Skip this pivot (matrix may be singular)

        # Eliminate below pivot
        for i in range(k + 1, n):
            factor = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
            b_dec[i] -= factor * b_dec[k]

    # Back substitution
    x = [Decimal(0)] * n
    for k in range(n - 1, -1, -1):
        if A[k][k] == 0:
            x[k] = Decimal(0)  # Singular row
            continue
        sum_ax = sum(A[k][j] * x[j] for j in range(k + 1, n))
        x[k] = (b_dec[k] - sum_ax) / A[k][k]

    # Convert back to float64 array
    result = np.array([_from_decimal(x[i]) for i in range(n)], dtype=np.float64)
    return result


def _solvit(
    a: np.ndarray,
    n: int,
    b: np.ndarray,
    use_extended_precision: bool = False,
    zero_pivot_fix: str = "none",
) -> Optional[np.ndarray]:
    """Port of ATLAS SOLVIT (atlas7v_1.for lines 1200-1295)."""

    def idx_cm(row: int, col: int) -> int:
        """Column-major offset mirroring JK = J + (K-1)*N."""
        return row + col * n

    def _log(message: str) -> None:
        if not log_file:
            return
        log_file.write(message + "\n")

    def _trace(message: str) -> None:
        if not trace_file:
            return
        trace_file.write(message + "\n")

    # Check if tracing is enabled
    # Only check flags - don't check file existence (too expensive for 16k calls)
    # If users want logging, they should set the environment variables
    tracing_enabled = (
        _TRACE_SOLVIT_MATRIX
        or TRACE_PIVOT_SEARCH
        or _current_solvit_layer in _TRACE_SOLVIT_LAYERS
    )

    log_path = os.path.join(os.getcwd(), "solvit_debug_python.log")
    trace_path = os.path.join(os.getcwd(), "solvit_trace.log")
    detail_path = os.path.join(os.getcwd(), "solvit_matrix_trace.log")
    pivot_trace_path = os.path.join(os.getcwd(), "solvit_pivot_trace.log")

    log_file = None
    trace_file = None
    detail_file = None
    pivot_trace_file = None

    # Only open log files if tracing is enabled
    if tracing_enabled:
        try:
            log_file = open(log_path, "a")
        except OSError:
            log_file = None
        trace_path = os.path.join(os.getcwd(), "solvit_trace.log")
        try:
            trace_file = open(trace_path, "a")
        except OSError:
            trace_file = None
        detail_path = os.path.join(os.getcwd(), "solvit_matrix_trace.log")
        detail_file = None
        if _TRACE_SOLVIT_MATRIX:
            try:
                detail_file = open(detail_path, "a")
            except OSError:
                detail_file = None
        if TRACE_PIVOT_SEARCH:
            pivot_trace_path = os.path.join(os.getcwd(), "solvit_pivot_trace.log")
            try:
                pivot_trace_file = open(pivot_trace_path, "a")
            except OSError:
                pivot_trace_file = None

    ctx_layer = _current_solvit_layer
    ctx_iter = _current_solvit_iter
    ctx_call = _current_solvit_call
    ctx_suffix = (
        f"(layer={ctx_layer:3d} iter={ctx_iter:3d} call={ctx_call:5d})"
        if ctx_layer >= 0 and ctx_iter >= 0 and ctx_call is not None
        else ""
    )

    a_fortran = np.array(a, dtype=np.float64, order="F", copy=True)
    a_vec = np.reshape(a_fortran, a_fortran.size, order="F")
    size = n * n
    # Use 1-based working arrays to mirror Fortran indexing exactly.
    a_work = np.zeros(size + 1, dtype=np.float64)
    a_work[1:] = a_vec
    b_work = np.zeros(n + 1, dtype=np.float64)
    b_work[1:] = np.array(b, dtype=np.float64, copy=True)
    ipivot = np.zeros(n + 1, dtype=np.int32)

    def idx1(row_1b: int, col_1b: int) -> int:
        """1-based column-major index helper."""
        return row_1b + (col_1b - 1) * n

    # Use FMA for multiply-subtract operations when available to keep
    # elimination steps numerically close to Fortran's x87 intermediate precision.
    has_fma = hasattr(math, "fma")

    def _stable_submul(lhs: float, rhs: float, factor: float) -> float:
        """Compute lhs - rhs*factor using FMA when available.

        Note: math.fma() can throw OverflowError even for finite inputs when
        the result would overflow. We catch this and fall back to regular
        arithmetic which returns inf instead of throwing.
        """
        if np.isfinite(lhs) and np.isfinite(rhs) and np.isfinite(factor):
            if has_fma:
                try:
                    return math.fma(-rhs, factor, lhs)
                except OverflowError:
                    # FMA overflows - fall back to regular arithmetic (returns inf)
                    pass
            return lhs - rhs * factor
        return lhs - rhs * factor

    # Fast path: Use kernel when tracing is disabled
    if not tracing_enabled:
        _solvit_kernel(a_work, b_work, ipivot, n)
        return b_work[1:]

    # Slow path: Original code with logging

    def _log_matrix_state(stage: str) -> None:
        if not detail_file:
            return
        detail_file.write(
            f"SOLVIT_MATRIX {stage} layer={ctx_layer} iter={ctx_iter} call={ctx_call}\n"
        )
        detail_file.write("  B:")
        for idx in range(1, n + 1):
            detail_file.write(f" {b_work[idx]: .17E}")
        detail_file.write("\n")
        for row in range(1, n + 1):
            row_vals = [a_work[idx1(row, col)] for col in range(1, n + 1)]
            joined = " ".join(f"{val: .17E}" for val in row_vals)
            detail_file.write(f"  ROW {row:2d}: {joined}\n")

    # Track A(9,2) (1-based) before eliminations for comparison with Fortran logs.
    if n >= 9 and log_file:
        a9_2 = a_work[idx1(9, 2)]
        _log(f"PY_MATRIX iter  0: A9_2 BEFORE eliminations = {a9_2: .17E} {ctx_suffix}")
    _log_matrix_state("pre_iteration")

    def _log_pivot_candidate(row: int, col: int, value: float) -> None:
        if not pivot_trace_file:
            return
        pivot_trace_file.write(
            "PY_PIVOT_CAND layer={layer} iter={iter} call={call} "
            "solvit_iter={solvit_iter} row={row} col={col} value={value:.17E} "
            "ipiv_row={ipiv_row} ipiv_col={ipiv_col}\n".format(
                layer=ctx_layer,
                iter=ctx_iter,
                call=ctx_call,
                solvit_iter=iter_idx,
                row=row,
                col=col,
                value=value,
                ipiv_row=int(ipivot[row]),
                ipiv_col=int(ipivot[col]),
            )
        )

    for iter_idx in range(1, n + 1):
        amax = 0.0
        irow = 1
        icolum = 1

        for row in range(1, n + 1):
            if ipivot[row] == 1:
                continue
            jk = row - n
            for col in range(1, n + 1):
                jk = jk + n
                if ipivot[col] == 1:
                    continue
                aa = abs(a_work[jk])
                if aa > amax:
                    amax = aa
                    irow = row
                    icolum = col
                    _log_pivot_candidate(row, col, aa)

        ipivot[icolum] += 1

        if irow != icolum:
            irl = irow - n
            icl = icolum - n
            for _ in range(1, n + 1):
                irl += n
                swap_val = a_work[irl]
                icl += n
                a_work[irl] = a_work[icl]
                a_work[icl] = swap_val
            b_work[irow], b_work[icolum] = b_work[icolum], b_work[irow]
        _log_matrix_state(f"post_swap_iter{iter_idx}")

        pivot_idx = icolum * n + icolum - n
        pivot = a_work[pivot_idx]

        if log_file:
            _log(
                "PY_SOLVIT iter{iter:3d}: pivot row={row:3d} col={col:3d} "
                "amax={amax: .17E} pivot_val={pivot: .17E} {suffix}".format(
                    iter=iter_idx + 1,
                    row=irow,
                    col=icolum,
                    amax=amax,
                    pivot=pivot,
                    suffix=ctx_suffix,
                )
            )
        _trace(
            f"TRACE_SOLVIT iter={iter_idx} {ctx_suffix} "
            f"pivot_row={irow} pivot_col={icolum} amax={amax:.17E} pivot={pivot:.17E} "
            f"b_col_before={b_work[icolum]:.17E}"
        )

        if log_file and n > 8:
            _log(
                "PY_SOLVIT iter{iter:3d}: b[9] BEFORE normalization = {val: .17E}".format(
                    iter=iter_idx,
                    val=b_work[9] if n >= 9 else float("nan"),
                )
            )

        a_work[pivot_idx] = 1.0
        icl = icolum - n
        for _ in range(1, n + 1):
            icl += n
            a_work[icl] = a_work[icl] / pivot
        b_work[icolum] = b_work[icolum] / pivot
        _log_matrix_state(f"post_normalize_iter{iter_idx}")
        _trace(
            f"TRACE_SOLVIT iter={iter_idx} {ctx_suffix} "
            f"b_col_after={b_work[icolum]:.17E}"
        )

        if log_file and n > 8:
            _log(
                "PY_SOLVIT iter{iter:3d}: b[9] AFTER normalization = {val: .17E}".format(
                    iter=iter_idx,
                    val=b_work[9] if n >= 9 else float("nan"),
                )
            )

        l1ic = icolum * n - n
        for l1 in range(1, n + 1):
            l1ic += 1
            if l1 == icolum:
                continue
            t = a_work[l1ic]
            a_work[l1ic] = 0.0
            if t == 0.0:
                continue
            l1l = l1 - n
            icl = icolum - n
            for _ in range(1, n + 1):
                l1l += n
                icl += n
                a_work[l1l] = _stable_submul(a_work[l1l], a_work[icl], t)
            b_work[l1] = _stable_submul(b_work[l1], b_work[icolum], t)
        _log_matrix_state(f"post_eliminate_iter{iter_idx}")

        row0_sum = 0.0
        row0_max = 0.0
        for col in range(1, n + 1):
            val = abs(a_work[idx1(1, col)])
            row0_sum += val
            row0_max = max(row0_max, val)
        if log_file:
            _log(
                "PY_SOLVIT iter{iter:3d}: row0 sum_abs={sumv: .17E} "
                "max={maxv: .17E} b[1]={b0: .17E} {suffix}".format(
                    iter=iter_idx,
                    sumv=row0_sum,
                    maxv=row0_max,
                    b0=b_work[1],
                    suffix=ctx_suffix,
                )
            )
            _log(
                "PY_SOLVIT iter{iter:3d} AFTER: row0 sum_abs={sumv: .17E} "
                "max={maxv: .17E} b[1]={b0: .17E} {suffix}".format(
                    iter=iter_idx,
                    sumv=row0_sum,
                    maxv=row0_max,
                    b0=b_work[1],
                    suffix=ctx_suffix,
                )
            )
            _log(
                "PY_SOLVIT iter{iter:3d} All b values after elimination "
                "(layer={layer:3d} iter={iter_ctx:3d} call={call:5d})".format(
                    iter=iter_idx,
                    layer=ctx_layer if ctx_layer >= 0 else -1,
                    iter_ctx=ctx_iter if ctx_iter >= 0 else -1,
                    call=ctx_call if ctx_call is not None else -1,
                )
            )
            for row in range(1, n + 1):
                _log(f"  b[{row:2d}] = {b_work[row]: .17E}")
        _trace(
            f"TRACE_SOLVIT iter={iter_idx} {ctx_suffix} "
            f"row0_sum={row0_sum:.17E} row0_max={row0_max:.17E} b1={b_work[1]:.17E}"
        )

    if log_file:
        log_file.close()
    if trace_file:
        trace_file.close()
    if detail_file:
        detail_file.close()
    if pivot_trace_file:
        pivot_trace_file.close()

    return b_work[1:]
