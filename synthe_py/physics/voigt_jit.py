"""Shared Numba-compiled Voigt profile function.

This module provides a single canonical JIT-compiled implementation of the
Voigt profile H(a,v) used throughout the synthesis pipeline.  All callers
should import ``voigt_profile_jit`` from here instead of maintaining
their own copy.
"""

from __future__ import annotations

import numpy as np

from numba import jit


@jit(nopython=True, cache=True)
def voigt_profile_jit(
    v: float, a: float, h0tab: np.ndarray, h1tab: np.ndarray, h2tab: np.ndarray
) -> float:
    """JIT-compiled Voigt profile H(a, v) matching the Fortran approximation.

    Parameters
    ----------
    v : float
        Frequency displacement in Doppler units.
    a : float
        Damping parameter (ratio of Lorentz to Doppler width).
    h0tab, h1tab, h2tab : np.ndarray
        Pre-computed Voigt coefficient tables (size 2001, step=1/200).

    Returns
    -------
    float
        Voigt function value H(a, v).
    """
    # Voigt function is symmetric in v — use abs(v) for table lookup.
    iv = int(abs(v) * 200.0 + 0.5)
    iv = max(0, min(iv, h0tab.size - 1))

    if a < 0.2:
        if abs(v) > 10.0:
            return 0.5642 * a / (v * v)
        else:
            return (h2tab[iv] * a + h1tab[iv]) * a + h0tab[iv]
    elif a > 1.4 or (a + abs(v)) > 3.2:
        aa = a * a
        vv = v * v
        u = (aa + vv) * 1.4142
        voigt_val = a * 0.79788 / u
        if a <= 100.0:
            aau = aa / u
            vvu = vv / u
            uu = u * u
            voigt_val = (
                (((aau - 10.0 * vvu) * aau * 3.0 + 15.0 * vvu * vvu) + 3.0 * vv - aa)
                / uu
                + 1.0
            ) * voigt_val
        return voigt_val
    else:
        vv = v * v
        h0 = h0tab[iv]
        h1 = h1tab[iv] + h0 * 1.12838
        h2 = h2tab[iv] + h1 * 1.12838 - h0
        h3 = (1.0 - h2tab[iv]) * 0.37613 - h1 * 0.66667 * vv + h2 * 1.12838
        h4 = (3.0 * h3 - h1) * 0.37613 + h0 * 0.66667 * vv * vv
        poly_a = (((h4 * a + h3) * a + h2) * a + h1) * a + h0
        poly_b = ((-0.122727278 * a + 0.532770573) * a - 0.96284325) * a + 0.979895032
        return poly_a * poly_b









