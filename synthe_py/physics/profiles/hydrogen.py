"""Hydrogen line profiles – partial port of the Kurucz HPROF4 stack."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np

from ..populations import DepthState, HydrogenDepthState
from ..tables import fast_ex

RYDH = 3.2880515e15
C_LIGHT = 2.99792458e18
C_LIGHT_CM = 2.99792458e10
LYMAN_ALPHA_CENTER_WN = 82259.10
SQRT_PI = math.sqrt(math.pi)


@dataclass
class HydrogenTables:
    propbm: np.ndarray
    c: np.ndarray
    d: np.ndarray
    pp: np.ndarray
    beta: np.ndarray
    stalph: np.ndarray
    stwtal: np.ndarray
    istal: np.ndarray
    lnghal: np.ndarray
    stcomp: np.ndarray
    stcpwt: np.ndarray
    lncomp: np.ndarray
    cutoff_h2_plus: np.ndarray
    cutoff_h2: np.ndarray
    asum_lyman: np.ndarray
    asum: np.ndarray
    y1wtm: np.ndarray
    xknmtb: np.ndarray
    tabvi: np.ndarray
    tabh1: np.ndarray


@lru_cache(maxsize=1)
def hydrogen_tables() -> HydrogenTables:
    prob_arrays = [
        np.array(
            [
                -0.980,
                -0.967,
                -0.948,
                -0.918,
                -0.873,
                -0.968,
                -0.949,
                -0.921,
                -0.879,
                -0.821,
                -0.950,
                -0.922,
                -0.883,
                -0.830,
                -0.764,
                -0.922,
                -0.881,
                -0.830,
                -0.770,
                -0.706,
                -0.877,
                -0.823,
                -0.763,
                -0.706,
                -0.660,
                -0.806,
                -0.741,
                -0.682,
                -0.640,
                -0.625,
                -0.691,
                -0.628,
                -0.588,
                -0.577,
                -0.599,
                -0.511,
                -0.482,
                -0.484,
                -0.514,
                -0.568,
                -0.265,
                -0.318,
                -0.382,
                -0.455,
                -0.531,
                -0.013,
                -0.167,
                -0.292,
                -0.394,
                -0.478,
                0.166,
                -0.056,
                -0.216,
                -0.332,
                -0.415,
                0.251,
                0.035,
                -0.122,
                -0.237,
                -0.320,
                0.221,
                0.059,
                -0.068,
                -0.168,
                -0.247,
                0.160,
                0.055,
                -0.037,
                -0.118,
                -0.189,
                0.110,
                0.043,
                -0.022,
                -0.085,
                -0.147,
            ]
        ),
        np.array(
            [
                -0.242,
                0.060,
                0.379,
                0.671,
                0.894,
                0.022,
                0.314,
                0.569,
                0.746,
                0.818,
                0.273,
                0.473,
                0.605,
                0.651,
                0.607,
                0.432,
                0.484,
                0.489,
                0.442,
                0.343,
                0.434,
                0.366,
                0.294,
                0.204,
                0.091,
                0.304,
                0.184,
                0.079,
                -0.025,
                -0.135,
                0.167,
                0.035,
                -0.082,
                -0.189,
                -0.290,
                0.085,
                -0.061,
                -0.183,
                -0.287,
                -0.374,
                0.032,
                -0.127,
                -0.249,
                -0.344,
                -0.418,
                -0.024,
                -0.167,
                -0.275,
                -0.357,
                -0.420,
                -0.061,
                -0.170,
                -0.257,
                -0.327,
                -0.384,
                -0.047,
                -0.124,
                -0.192,
                -0.252,
                -0.306,
                -0.043,
                -0.092,
                -0.142,
                -0.190,
                -0.238,
                -0.038,
                -0.070,
                -0.107,
                -0.146,
                -0.187,
                -0.030,
                -0.049,
                -0.075,
                -0.106,
                -0.140,
            ]
        ),
        np.array(
            [
                -0.484,
                -0.336,
                -0.206,
                -0.111,
                -0.058,
                -0.364,
                -0.264,
                -0.192,
                -0.154,
                -0.144,
                -0.299,
                -0.268,
                -0.250,
                -0.244,
                -0.246,
                -0.319,
                -0.333,
                -0.337,
                -0.336,
                -0.337,
                -0.397,
                -0.414,
                -0.415,
                -0.413,
                -0.420,
                -0.456,
                -0.455,
                -0.451,
                -0.456,
                -0.478,
                -0.446,
                -0.441,
                -0.446,
                -0.469,
                -0.512,
                -0.358,
                -0.381,
                -0.415,
                -0.463,
                -0.522,
                -0.214,
                -0.288,
                -0.360,
                -0.432,
                -0.503,
                -0.063,
                -0.196,
                -0.304,
                -0.394,
                -0.468,
                0.063,
                -0.108,
                -0.237,
                -0.334,
                -0.409,
                0.151,
                -0.019,
                -0.148,
                -0.245,
                -0.319,
                0.149,
                0.016,
                -0.091,
                -0.177,
                -0.246,
                0.115,
                0.023,
                -0.056,
                -0.126,
                -0.189,
                0.078,
                0.021,
                -0.036,
                -0.091,
                -0.145,
            ]
        ),
        np.array(
            [
                -0.082,
                0.163,
                0.417,
                0.649,
                0.829,
                0.096,
                0.316,
                0.515,
                0.660,
                0.729,
                0.242,
                0.393,
                0.505,
                0.556,
                0.534,
                0.320,
                0.373,
                0.394,
                0.369,
                0.290,
                0.308,
                0.274,
                0.226,
                0.152,
                0.048,
                0.232,
                0.141,
                0.052,
                -0.046,
                -0.154,
                0.148,
                0.020,
                -0.094,
                -0.200,
                -0.299,
                0.083,
                -0.070,
                -0.195,
                -0.299,
                -0.385,
                0.031,
                -0.130,
                -0.253,
                -0.348,
                -0.422,
                -0.023,
                -0.167,
                -0.276,
                -0.359,
                -0.423,
                -0.053,
                -0.165,
                -0.254,
                -0.326,
                -0.384,
                -0.038,
                -0.119,
                -0.190,
                -0.251,
                -0.306,
                -0.034,
                -0.088,
                -0.140,
                -0.190,
                -0.239,
                -0.032,
                -0.066,
                -0.103,
                -0.144,
                -0.186,
                -0.027,
                -0.048,
                -0.075,
                -0.106,
                -0.142,
            ]
        ),
        np.array(
            [
                -0.819,
                -0.759,
                -0.689,
                -0.612,
                -0.529,
                -0.770,
                -0.707,
                -0.638,
                -0.567,
                -0.498,
                -0.721,
                -0.659,
                -0.595,
                -0.537,
                -0.488,
                -0.671,
                -0.617,
                -0.566,
                -0.524,
                -0.497,
                -0.622,
                -0.582,
                -0.547,
                -0.523,
                -0.516,
                -0.570,
                -0.545,
                -0.526,
                -0.521,
                -0.537,
                -0.503,
                -0.495,
                -0.496,
                -0.514,
                -0.551,
                -0.397,
                -0.418,
                -0.448,
                -0.492,
                -0.547,
                -0.246,
                -0.315,
                -0.384,
                -0.453,
                -0.522,
                -0.080,
                -0.210,
                -0.316,
                -0.406,
                -0.481,
                0.068,
                -0.107,
                -0.239,
                -0.340,
                -0.418,
                0.177,
                -0.006,
                -0.143,
                -0.246,
                -0.324,
                0.184,
                0.035,
                -0.082,
                -0.174,
                -0.249,
                0.146,
                0.042,
                -0.046,
                -0.123,
                -0.190,
                0.103,
                0.036,
                -0.027,
                -0.088,
                -0.146,
            ]
        ),
        np.array(
            [
                -0.073,
                0.169,
                0.415,
                0.636,
                0.809,
                0.102,
                0.311,
                0.499,
                0.639,
                0.710,
                0.232,
                0.372,
                0.479,
                0.531,
                0.514,
                0.294,
                0.349,
                0.374,
                0.354,
                0.279,
                0.278,
                0.253,
                0.212,
                0.142,
                0.040,
                0.215,
                0.130,
                0.044,
                -0.051,
                -0.158,
                0.141,
                0.015,
                -0.097,
                -0.202,
                -0.300,
                0.080,
                -0.072,
                -0.196,
                -0.299,
                -0.385,
                0.029,
                -0.130,
                -0.252,
                -0.347,
                -0.421,
                -0.022,
                -0.166,
                -0.275,
                -0.359,
                -0.423,
                -0.050,
                -0.164,
                -0.253,
                -0.325,
                -0.384,
                -0.035,
                -0.118,
                -0.189,
                -0.252,
                -0.306,
                -0.032,
                -0.087,
                -0.139,
                -0.190,
                -0.240,
                -0.029,
                -0.064,
                -0.102,
                -0.143,
                -0.185,
                -0.025,
                -0.046,
                -0.074,
                -0.106,
                -0.142,
            ]
        ),
        np.array(
            [
                0.005,
                0.128,
                0.260,
                0.389,
                0.504,
                0.004,
                0.109,
                0.220,
                0.318,
                0.389,
                -0.007,
                0.079,
                0.162,
                0.222,
                0.244,
                -0.018,
                0.041,
                0.089,
                0.106,
                0.080,
                -0.026,
                -0.003,
                0.003,
                -0.023,
                -0.086,
                -0.025,
                -0.048,
                -0.087,
                -0.148,
                -0.234,
                -0.008,
                -0.085,
                -0.165,
                -0.251,
                -0.343,
                0.018,
                -0.111,
                -0.223,
                -0.321,
                -0.407,
                0.032,
                -0.130,
                -0.255,
                -0.354,
                -0.431,
                0.014,
                -0.148,
                -0.269,
                -0.359,
                -0.427,
                -0.005,
                -0.140,
                -0.243,
                -0.323,
                -0.386,
                0.005,
                -0.095,
                -0.178,
                -0.248,
                -0.307,
                -0.002,
                -0.068,
                -0.129,
                -0.187,
                -0.241,
                -0.007,
                -0.049,
                -0.094,
                -0.139,
                -0.186,
                -0.010,
                -0.036,
                -0.067,
                -0.103,
                -0.143,
            ]
        ),
    ]

    propbm = np.stack([arr.reshape(5, 15, order="F") for arr in prob_arrays], axis=0)
    c_cols = [
        np.array([-18.396, 84.674, -96.273, 3.927, 55.191]),
        np.array([95.740, 18.489, 14.902, 24.466, 42.456]),
        np.array([-25.088, 145.882, -50.165, 7.902, 51.003]),
        np.array([93.783, 10.066, 9.224, 20.685, 40.136]),
        np.array([-19.819, 94.981, -79.606, 3.159, 52.106]),
        np.array([111.107, 11.910, 9.857, 21.371, 41.006]),
        np.array([511.318, 1.532, 4.044, 19.266, 41.812]),
    ]
    c = np.stack(c_cols, axis=1)
    d_cols = [
        np.array([11.801, 9.079, -0.651, -11.071, -26.545]),
        np.array([-6.665, -7.136, -10.605, -15.882, -23.632]),
        np.array([7.872, 5.592, -2.716, -12.180, -25.661]),
        np.array([-5.918, -6.501, -10.130, -15.588, -23.570]),
        np.array([10.938, 8.028, -1.267, -11.375, -26.047]),
        np.array([-5.899, -6.381, -10.044, -15.574, -23.644]),
        np.array([-6.070, -4.528, -8.759, -14.984, -23.956]),
    ]
    d = np.stack(d_cols, axis=1)

    pp = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    beta_grid = np.array(
        [
            1.0,
            1.259,
            1.585,
            1.995,
            2.512,
            3.162,
            3.981,
            5.012,
            6.310,
            7.943,
            10.0,
            12.59,
            15.85,
            19.95,
            25.12,
        ]
    )

    stalph = np.array(
        [
            -730.0,
            370.0,
            188.0,
            515.0,
            327.0,
            619.0,
            -772.0,
            -473.0,
            -369.0,
            120.0,
            1256.0,
            162.0,
            285.0,
            -161.0,
            -38.3,
            6.82,
            -174.0,
            -147.0,
            -101.0,
            -77.5,
            55.0,
            126.0,
            275.0,
            139.0,
            -60.0,
            3.7,
            27.0,
            -69.0,
            -42.0,
            -18.0,
            -5.5,
            -9.1,
            -33.0,
            -24.0,
        ]
    )
    stwtal = np.array(
        [
            1.0,
            2.0,
            1.0,
            2.0,
            1.0,
            2.0,
            1.0,
            2.0,
            3.0,
            1.0,
            2.0,
            1.0,
            2.0,
            1.0,
            4.0,
            6.0,
            1.0,
            2.0,
            3.0,
            4.0,
            1.0,
            2.0,
            1.0,
            2.0,
            1.0,
            4.0,
            6.0,
            1.0,
            7.0,
            6.0,
            4.0,
            4.0,
            4.0,
            5.0,
        ]
    )
    # Fortran ISTAL/LNGHAL are 1-based with 4 entries (n=1..4).
    # Keep 0-based arrays of length 4 so n-1 indexing matches Fortran.
    istal = np.array([1, 3, 10, 21])
    lnghal = np.array([2, 7, 11, 14])
    stcomp = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            468.0,
            576.0,
            -522.0,
            0.0,
            0.0,
            260.0,
            290.0,
            -33.0,
            -140.0,
            0.0,
            140.0,
            150.0,
            18.0,
            -27.0,
            -51.0,
        ]
    ).reshape(5, 4, order="F")
    stcpwt = np.array(
        [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            2.0,
            0.0,
            0.0,
            1.0,
            1.0,
            4.0,
            3.0,
            0.0,
            1.0,
            1.0,
            4.0,
            6.0,
            4.0,
        ]
    ).reshape(5, 4, order="F")
    lncomp = np.array([1, 3, 4, 5])

    cutoff_h2_plus = np.array(
        [
            -15.14,
            -15.06,
            -14.97,
            -14.88,
            -14.80,
            -14.71,
            -14.62,
            -14.53,
            -14.44,
            -14.36,
            -14.27,
            -14.18,
            -14.09,
            -14.01,
            -13.92,
            -13.83,
            -13.74,
            -13.65,
            -13.57,
            -13.48,
            -13.39,
            -13.30,
            -13.21,
            -13.13,
            -13.04,
            -12.95,
            -12.86,
            -12.77,
            -12.69,
            -12.60,
            -12.51,
            -12.40,
            -12.29,
            -12.15,
            -12.02,
            -11.90,
            -11.76,
            -11.63,
            -11.53,
            -11.41,
            -11.30,
            -11.22,
            -11.15,
            -11.09,
            -11.07,
            -11.06,
            -11.07,
            -11.09,
            -11.12,
            -11.16,
            -11.19,
            -11.21,
            -11.24,
            -11.27,
            -11.30,
            -11.33,
            -11.36,
            -11.39,
            -11.42,
            -11.45,
            -11.48,
            -11.48,
            -11.48,
            -11.48,
            -11.48,
            -11.48,
            -11.48,
            -11.48,
            -11.48,
            -11.48,
            -11.48,
            -11.48,
            -11.48,
            -11.48,
            -11.48,
            -11.48,
            -11.41,
            -11.40,
            -11.39,
            -11.38,
            -11.37,
            -11.36,
            -11.35,
            -11.34,
            -11.33,
            -11.32,
            -11.30,
            -11.29,
            -11.28,
            -11.27,
            -11.27,
            -11.27,
            -11.26,
            -11.25,
            -11.24,
            -11.23,
            -11.22,
            -11.21,
            -11.20,
            -11.19,
            -11.18,
            -11.17,
            -11.15,
            -11.14,
            -11.13,
            -11.12,
            -11.11,
            -11.10,
            -11.09,
            -11.08,
            -11.07,
        ]
    )
    cutoff_h2 = np.array(
        [
            -13.64,
            -13.52,
            -13.39,
            -13.27,
            -13.14,
            -13.01,
            -12.87,
            -12.74,
            -12.63,
            -12.56,
            -12.51,
            -12.48,
            -12.47,
            -12.49,
            -12.52,
            -12.55,
            -12.57,
            -12.61,
            -12.65,
            -12.69,
            -12.72,
            -12.76,
            -12.79,
            -12.82,
            -12.84,
            -12.85,
            -12.87,
            -12.90,
            -12.93,
            -12.94,
            -12.93,
            -12.95,
            -12.95,
            -12.96,
            -12.97,
            -12.96,
            -12.96,
            -12.95,
            -12.95,
            -12.96,
            -12.98,
            -12.99,
            -12.95,
            -12.96,
            -13.00,
            -13.00,
            -12.98,
            -12.97,
            -13.00,
            -13.00,
            -13.00,
            -13.00,
            -13.00,
            -13.00,
            -13.00,
            -13.00,
            -13.00,
            -13.00,
            -13.00,
            -13.00,
            -13.00,
            -13.00,
            -13.00,
            -13.00,
            -13.00,
            -13.00,
            -13.00,
            -13.00,
            -13.00,
            -13.00,
            -12.89,
            -12.88,
            -12.87,
            -12.86,
            -12.85,
            -12.84,
            -12.83,
            -12.81,
            -12.80,
            -12.79,
            -12.78,
            -12.76,
            -12.74,
            -12.72,
            -12.70,
            -12.68,
            -12.65,
            -12.62,
            -12.59,
            -12.56,
            -12.53,
        ]
    )

    asum_lyman = np.array(
        [
            0.0,
            6.265e8,
            1.897e8,
            8.126e7,
            4.203e7,
            2.45e7,
            1.236e7,
            8.249e6,
            5.782e6,
            4.208e6,
            3.158e6,
            2.43e6,
            1.91e6,
            1.567e6,
            1.274e6,
            1.05e6,
            8.752e5,
            7.373e5,
            6.269e5,
            5.375e5,
            4.643e5,
            4.038e5,
            3.534e5,
            3.111e5,
            2.752e5,
            2.447e5,
            2.185e5,
            1.959e5,
            1.763e5,
            1.593e5,
            1.443e5,
            1.312e5,
            1.197e5,
            1.094e5,
            1.003e5,
            9.216e4,
            8.489e4,
            7.836e4,
            7.249e4,
            6.719e4,
            6.239e4,
            5.804e4,
            5.408e4,
            5.048e4,
            4.719e4,
            4.418e4,
            4.142e4,
            3.888e4,
            3.655e4,
            3.44e4,
            3.242e4,
            3.058e4,
            2.888e4,
            2.731e4,
            2.585e4,
            2.449e4,
            2.322e4,
            2.204e4,
            2.094e4,
            1.991e4,
            1.894e4,
            1.804e4,
            1.72e4,
            1.64e4,
            1.566e4,
            1.496e4,
            1.43e4,
            1.368e4,
            1.309e4,
            1.254e4,
            1.201e4,
            1.152e4,
            1.105e4,
            1.061e4,
            1.019e4,
            9.796e3,
            9.419e3,
            9.061e3,
            8.721e3,
            8.398e3,
            8.091e3,
            7.799e3,
            7.52e3,
            7.255e3,
            7.002e3,
            6.76e3,
            6.53e3,
            6.31e3,
            6.1e3,
            5.898e3,
            5.706e3,
            5.522e3,
            5.346e3,
            5.177e3,
            5.015e3,
            4.86e3,
            4.711e3,
            4.569e3,
            4.432e3,
            4.3e3,
        ]
    )

    asum = np.array(
        [
            0.0,
            4.696e8,
            9.98e7,
            3.017e7,
            1.155e7,
            5.189e6,
            2.616e6,
            1.437e6,
            8.444e5,
            5.234e5,
            3.389e5,
            2.275e5,
            1.575e5,
            1.12e5,
            8.142e4,
            6.04e4,
            4.56e4,
            3.496e4,
            2.719e4,
            2.141e4,
            1.711e4,
            1.377e4,
            1.119e4,
            9.166e3,
            7.572e3,
            6.341e3,
            5.338e3,
            4.523e3,
            3.854e3,
            3.302e3,
            2.844e3,
            2.46e3,
            2.138e3,
            1.866e3,
            1.635e3,
            1.438e3,
            1.269e3,
            1.124e3,
            998.3,
            889.4,
            794.7,
            712.0,
            639.6,
            575.9,
            519.8,
            470.3,
            426.3,
            387.3,
            352.6,
            321.5,
            293.8,
            268.9,
            246.5,
            226.4,
            208.2,
            191.8,
            176.9,
            163.4,
            151.2,
            140.0,
            129.8,
            120.6,
            112.1,
            104.3,
            97.2,
            90.66,
            84.65,
            79.12,
            74.03,
            69.33,
            64.98,
            60.97,
            57.25,
            53.81,
            50.61,
            47.65,
            44.89,
            42.32,
            39.94,
            37.71,
            35.63,
            33.69,
            31.88,
            30.19,
            28.60,
            27.12,
            25.72,
            24.42,
            23.19,
            22.04,
            20.96,
            19.94,
            18.98,
            18.08,
            17.22,
            16.42,
        ]
    )

    y1wtm = np.array([1.0e18, 1.0e17, 1.0e16, 1.0e14]).reshape(2, 2, order="F")
    # Fortran XKNMTB(4,3) in column-major order (N=1..4, MMN=1..3).
    xknmtb = np.array(
        [
            1.716e-4,
            9.019e-3,
            0.1001,
            0.5820,
            5.235e-4,
            1.772e-2,
            0.1710,
            0.8660,
            8.912e-4,
            2.507e-2,
            0.2230,
            1.0200,
        ]
    ).reshape(4, 3, order="F")

    tabvi = np.array(
        [
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            1.6,
            1.7,
            1.8,
            1.9,
            2.0,
            2.1,
            2.2,
            2.3,
            2.4,
            2.5,
            2.6,
            2.7,
            2.8,
            2.9,
            3.0,
            3.1,
            3.2,
            3.3,
            3.4,
            3.5,
            3.6,
            3.7,
            3.8,
            3.9,
            4.0,
            4.2,
            4.4,
            4.6,
            4.8,
            5.0,
            5.2,
            5.4,
            5.6,
            5.8,
            6.0,
            6.2,
            6.4,
            6.6,
            6.8,
            7.0,
            7.2,
            7.4,
            7.6,
            7.8,
            8.0,
            8.2,
            8.4,
            8.6,
            8.8,
            9.0,
            9.2,
            9.4,
            9.6,
            9.8,
            10.0,
            10.2,
            10.4,
            10.6,
            10.8,
            11.0,
            11.2,
            11.4,
            11.6,
            11.8,
            12.0,
        ]
    )
    tabh1 = np.array(
        [
            -1.12838,
            -1.10596,
            -1.04048,
            -0.93703,
            -0.80346,
            -0.64945,
            -0.48552,
            -0.32192,
            -0.16772,
            -0.03012,
            0.08594,
            0.17789,
            0.24537,
            0.28981,
            0.31394,
            0.32130,
            0.31573,
            0.30094,
            0.28027,
            0.25648,
            0.231726,
            0.207528,
            0.184882,
            0.164341,
            0.146128,
            0.130236,
            0.116515,
            0.104739,
            0.094653,
            0.086005,
            0.078565,
            0.072129,
            0.066526,
            0.061615,
            0.057281,
            0.053430,
            0.049988,
            0.046894,
            0.044098,
            0.041561,
            0.039250,
            0.035195,
            0.031762,
            0.028824,
            0.026288,
            0.024081,
            0.022146,
            0.020441,
            0.018929,
            0.017582,
            0.016375,
            0.015291,
            0.014312,
            0.013426,
            0.012620,
            0.0118860,
            0.0112145,
            0.0105990,
            0.0100332,
            0.0095119,
            0.0090306,
            0.0085852,
            0.0081722,
            0.0077885,
            0.0074314,
            0.0070985,
            0.0067875,
            0.0064967,
            0.0062243,
            0.0059688,
            0.0057287,
            0.0055030,
            0.0052903,
            0.0050898,
            0.0049006,
            0.0047217,
            0.0045526,
            0.0043924,
            0.0042405,
            0.0040964,
            0.0039595,
        ]
    )

    return HydrogenTables(
        propbm=propbm,
        c=c,
        d=d,
        pp=pp,
        beta=beta_grid,
        stalph=stalph,
        stwtal=stwtal,
        istal=istal,
        lnghal=lnghal,
        stcomp=stcomp,
        stcpwt=stcpwt,
        lncomp=lncomp,
        cutoff_h2_plus=cutoff_h2_plus,
        cutoff_h2=cutoff_h2,
        asum_lyman=asum_lyman,
        asum=asum,
        y1wtm=y1wtm,
        xknmtb=xknmtb,
        tabvi=tabvi,
        tabh1=tabh1,
    )


def vcse1f(x: float) -> float:
    if x <= 0:
        return 0.0
    if x <= 0.01:
        return -math.log(x) - 0.577215 + x
    if x <= 1.0:
        return (
            -math.log(x)
            - 0.57721566
            + x
            * (
                0.99999193
                + x
                * (-0.24991055 + x * (0.05519968 + x * (-0.00976004 + x * 0.00107857)))
            )
        )
    if x > 30.0:
        return 0.0
    numerator = x * (x + 2.334733) + 0.25062
    denominator = (x * (x + 3.330657) + 1.681534) * x
    return numerator / denominator * fast_ex(x)


@lru_cache(maxsize=1)
def _e1_table() -> np.ndarray:
    values = np.zeros(2000, dtype=np.float64)
    for idx in range(1, 2000 + 1):
        x = idx * 0.01
        values[idx - 1] = math.exp(-x) / x
    return values


def faste1(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x < 0.5:
        return (1.0 - 0.22464 * x) * x - math.log(x) - 0.57721
    if x > 20.0:
        return 0.0
    idx = min(int(x * 100.0 + 0.5), 1999)
    return _e1_table()[idx]


def sofbeta(beta: float, p: float, n: int, m: int) -> float:
    tables = hydrogen_tables()
    corr = 1.0
    b2 = beta * beta
    sb = math.sqrt(beta)
    if beta <= 500.0:
        indx = 7
        mmn = m - n
        if n <= 3 and mmn <= 2:
            indx = 2 * (n - 1) + mmn
        indx = max(1, min(indx, 7))
        im = min(int(5.0 * p) + 1, 4)
        im = max(im, 1)
        ip = im + 1
        wtp = 5.0 * (p - tables.pp[im - 1])
        wtp = max(0.0, min(1.0, wtp))
        wtm = 1.0 - wtp
        if beta <= 25.12:
            betagrid = tables.beta
            j = int(np.searchsorted(betagrid, beta, side="right"))
            j = max(1, min(j, betagrid.size - 1))
            jm = j - 1
            jp = j
            denom = betagrid[jp] - betagrid[jm]
            if denom <= 0:
                wtb = 0.0
            else:
                wtb = (beta - betagrid[jm]) / denom
            wtbm = 1.0 - wtb
            prop = tables.propbm[indx - 1]
            cbp = prop[ip - 1, jp] * wtp + prop[im - 1, jp] * wtm
            cbm = prop[ip - 1, jm] * wtp + prop[im - 1, jm] * wtm
            corr = 1.0 + cbp * wtb + cbm * wtbm
            pr1 = 0.0
            pr2 = 0.0
            wt = max(min(0.5 * (10.0 - beta), 1.0), 0.0)
            if beta <= 10.0:
                pr1 = 8.0 / (83.0 + (2.0 + 0.95 * b2) * beta)
            if beta >= 8.0:
                pr2 = (1.5 / sb + 27.0 / b2) / b2
            return (pr1 * wt + pr2 * (1.0 - wt)) * corr
        cc = tables.c[im - 1, indx - 1] * wtp + tables.c[ip - 1, indx - 1] * wtm
        dd = tables.d[im - 1, indx - 1] * wtp + tables.d[ip - 1, indx - 1] * wtm
        corr = 1.0 + dd / (cc + beta * sb)
    return (1.5 / sb + 27.0 / b2) / b2 * corr


@lru_cache(maxsize=256)
def _hf_nm_cached(n: int, m: int) -> float:
    if m <= n:
        return 0.0
    xn = float(n)
    ginf = 0.2027 / xn**0.71
    gca = 0.124 / xn
    fkn = xn * 1.9603
    wtc = 0.45 - 2.4 / xn**3 * (xn - 1.0)
    xm = float(m)
    xmn = xm - xn
    fk = fkn * (xm / (xmn * (xm + xn))) ** 3
    xmn12 = xmn**1.2
    wt = (xmn12 - 1.0) / (xmn12 + wtc)
    fnm = fk * (1.0 - wt * ginf - (0.222 + gca / xm) * (1.0 - wt))
    return fnm


def _fine_structure(
    n: int, m: int, tables: HydrogenTables
) -> Tuple[np.ndarray, np.ndarray]:
    mmn = m - n
    xn = float(n)
    xn2 = xn * xn
    if n > 4 or m > 10:
        return np.array([0.0]), np.array([1.0])
    if mmn != 1:
        ifins = tables.lncomp[n - 1]
        offsets = tables.stcomp[:ifins, n - 1] * 1.0e7
        weights = tables.stcpwt[:ifins, n - 1] / xn2
        return offsets, weights
    ifins = tables.lnghal[n - 1]
    ipos = tables.istal[n - 1]
    offsets = tables.stalph[ipos : ipos + ifins] * 1.0e7
    weights = tables.stwtal[ipos : ipos + ifins] / xn2 / 3.0
    return offsets, weights


@lru_cache(maxsize=64)
def _fine_structure_cached(n: int, m: int) -> Tuple[np.ndarray, np.ndarray]:
    return _fine_structure(n, m, hydrogen_tables())


def hydrogen_line_profile(
    n: int, m: int, depth_state: DepthState, delta_lambda_nm: float
) -> float:
    hyd = depth_state.hydrogen
    if hyd is None:
        return 0.0
    tables = hydrogen_tables()
    mmn = m - n
    xn = float(n)
    xm = float(m)
    xn2 = xn * xn
    xm2 = xm * xm
    xm2mn2 = xm2 - xn2
    xmn2 = xm2 * xn2
    gnm = xm2mn2 / xmn2
    if mmn <= 0:
        return 0.0
    if mmn <= 3 and n <= 4:
        xknm = tables.xknmtb[n - 1, mmn - 1]
    else:
        xknm = 5.5e-5 / gnm * xmn2 / (1.0 + 0.13 / mmn)
    freqnm = RYDH * gnm
    wavenm = C_LIGHT / freqnm
    dbeta = C_LIGHT / (freqnm * freqnm * xknm)
    c1con = xknm / wavenm * gnm * xm2mn2
    c2con = (xknm / wavenm) ** 2
    # ASUM/ASUMLYMAN are Fortran 1-based arrays; convert to Python 0-based indexing.
    radamp = tables.asum[n - 1] + tables.asum[m - 1]
    if n == 1:
        radamp = tables.asum_lyman[m - 1]
    radamp /= 12.5664
    radamp /= freqnm
    resont = _hf_nm_cached(1, m) / xm / (1.0 - 1.0 / xm2)
    if n != 1:
        resont += _hf_nm_cached(1, n) / xn / (1.0 - 1.0 / xn2)
    resont *= 3.579e-24 / gnm
    vdw = 4.45e-26 / gnm * (xm2 * (7.0 * xm2 + 5.0)) ** 0.4
    hwvdw = vdw * hyd.t3nhe + 2.0 * vdw * hyd.t3nh2
    hwrad = radamp
    stark = 1.6678e-18 * freqnm * xknm
    hwres = resont * hyd.xnfph[0] * 2.0 if hyd.xnfph.size > 0 else 0.0
    hwstk = stark * hyd.fo
    hwlor = hwres + hwvdw + hwrad
    finest, finswt = _fine_structure_cached(n, m)
    wl0 = wavenm
    wl = wl0 + delta_lambda_nm * 10.0
    freq = C_LIGHT / wl
    del_freq = abs(freq - freqnm)
    dopph = max(hyd.dopph, 1e-40)
    dop = freqnm * dopph
    hfwidth = freqnm * max(dopph, hwlor, hwstk)
    ifcore = del_freq <= hfwidth

    # Match Fortran NWID selection for core handling.
    nwid = 1
    if not (dopph >= hwstk and dopph >= hwlor):
        nwid = 2
        if hwlor < hwstk:
            nwid = 3

    # Doppler core (same normalization as VOIGT in Fortran: FASTEX only).
    core = 0.0
    for offset, weight in zip(finest, finswt):
        component_freq = freqnm + offset
        d = abs(freq - component_freq) / max(dop, 1e-30)
        if d <= 7.0:
            core += fast_ex(d * d) * weight

    # Lorentz component (including Lyman-alpha special case).
    lorentz = 0.0
    hhw = freqnm * hwlor
    if n == 1 and m == 2:
        lorentz = _lyman_alpha_lorentz(
            freq=freq,
            freqnm=freqnm,
            del_freq=del_freq,
            dop=dop,
            hwres=hwres,
            hwvdw=hwvdw,
            hwrad=hwrad,
            tables=tables,
            hyd=hyd,
        )
    else:
        top = hhw
        if n == 1 and m in {3, 4, 5}:
            freq_ratio = freq / RYDH
            if m == 3 and 0.885 <= freq_ratio <= 0.890:
                top = max(hhw - freqnm * hwrad, 0.0)
            elif m == 4 and 0.936 <= freq_ratio <= 0.938:
                top = max(hhw - freqnm * hwrad, 0.0)
            elif m == 5 and 0.959 <= freq_ratio <= 0.961:
                top = max(hhw - freqnm * hwrad, 0.0)
        if hhw > 0.0:
            lorentz = top / math.pi / (del_freq * del_freq + hhw * hhw) * 1.77245 * dop

    y1num = 320.0
    if m == 2:
        y1num = 550.0
    elif m == 3:
        y1num = 380.0

    y1wht = 1.0e13
    if mmn <= 3:
        y1wht = 1.0e14
    if (
        mmn <= 2
        and n <= 2
        and tables.y1wtm.shape[0] >= n
        and tables.y1wtm.shape[1] >= mmn
    ):
        y1wht = tables.y1wtm[n - 1, mmn - 1]

    wty1 = 1.0 / (1.0 + max(depth_state.electron_density, 0.0) / max(y1wht, 1e-30))
    y1_scal = y1num * hyd.y1s * wty1 + hyd.y1b * (1.0 - wty1)
    c1 = hyd.c1d * c1con * y1_scal
    c2 = hyd.c2d * c2con

    beta = del_freq / max(hyd.fo, 1e-30) * dbeta
    y1 = c1 * beta
    y2 = c2 * beta * beta
    g1 = 6.77 * math.sqrt(max(c1, 1e-30))
    ratio = 0.0
    if c1 > 0.0 and c2 > 0.0:
        ratio = math.sqrt(c2) / max(c1, 1e-30)
    log_term = 0.0
    if ratio > 0.0:
        log_term = math.log(max(ratio, 1e-30))
    gnot = g1 * max(0.0, 0.2114 + log_term) * (1.0 - hyd.gcon1 - hyd.gcon2)
    gamma = gnot
    if y2 > 1e-4 and y1 > 1e-5:
        gamma = (
            g1
            * (0.5 * fast_ex(min(80.0, y1)) + vcse1f(y1) - 0.5 * vcse1f(y2))
            * (
                1.0
                - hyd.gcon1 / (1.0 + (90.0 * y1) ** 3)
                - hyd.gcon2 / (1.0 + 2000.0 * y1)
            )
        )
    f = 0.0
    if gamma > 0:
        f = gamma / math.pi / (gamma * gamma + beta * beta)
    prqs = sofbeta(beta, hyd.pp, n, m)
    stark_extra = 0.0
    if m <= 2:
        prqs *= 0.5
        stark_extra = _lyman_quasistatic_cutoff(
            freq=freq,
            prqs=prqs,
            hyd=hyd,
            dbeta=dbeta,
            dop=dop,
            n=n,
            m=m,
        )
    p1 = (0.9 * y1) ** 2
    fns = (p1 + 0.03 * math.sqrt(max(y1, 0.0))) / (p1 + 1.0)
    stark_core = (prqs * (1.0 + fns) + f) / max(hyd.fo, 1e-30) * dbeta * 1.77245 * dop

    # Fortran core branch uses only the dominant width component.
    if ifcore:
        if nwid == 1:
            return max(core, 0.0)
        if nwid == 2:
            return max(lorentz, 0.0)
        return max(stark_core + stark_extra, 0.0)

    return max(core + lorentz + stark_core + stark_extra, 0.0)


def _interpolate_cutoff(
    delta_wavenumber: float, table: np.ndarray, start: float, step: float
) -> float | None:
    max_delta = start + step * (table.size - 1)
    if delta_wavenumber > max_delta:
        return None
    if delta_wavenumber <= start:
        if table.size < 2:
            return table[0] if table.size else None
        frac = (delta_wavenumber - start) / step
        return table[0] + (table[1] - table[0]) * frac
    position = (delta_wavenumber - start) / step
    index = int(math.floor(position))
    frac = position - index
    if index >= table.size - 1:
        return float(table[-1])
    return float(table[index] + (table[index + 1] - table[index]) * frac)


def _lyman_alpha_lorentz(
    freq: float,
    freqnm: float,
    del_freq: float,
    dop: float,
    hwres: float,
    hwvdw: float,
    hwrad: float,
    tables: HydrogenTables,
    hyd: HydrogenDepthState,
) -> float:
    if dop <= 0.0:
        return 0.0

    hwres_near = hwres * 4.0
    hwlor_near = hwres_near + hwvdw + hwrad
    hhw_near = freqnm * max(hwlor_near, 0.0)
    freq_threshold = (LYMAN_ALPHA_CENTER_WN - 4000.0) * C_LIGHT_CM
    wavenumber = freq / C_LIGHT_CM
    delta_wn = wavenumber - LYMAN_ALPHA_CENTER_WN

    if freq > freq_threshold and hhw_near > 0.0:
        hres_term = (
            hwres_near
            * freqnm
            / math.pi
            / (del_freq * del_freq + hhw_near * hhw_near)
            * 1.77245
            * dop
        )
        hhw_use = hhw_near
    else:
        cutoff_val = 0.0
        cutoff_log = _interpolate_cutoff(delta_wn, tables.cutoff_h2, -22000.0, 200.0)
        if cutoff_log is not None and hyd.xnfph.size > 0:
            cutoff_val = (10.0 ** (cutoff_log - 14.0)) * hyd.xnfph[0] * 2.0 / C_LIGHT_CM
        hres_term = cutoff_val * 1.77245 * dop
        hwlor = hwres + hwvdw + hwrad
        hhw_use = freqnm * max(hwlor, 0.0)

    hrad_term = 0.0
    if hwrad > 0.0 and hhw_use > 0.0:
        freq_low = 2.4190611e15
        freq_high = 0.77 * RYDH
        if freq > freq_low and freq < freq_high:
            hrad_term = (
                hwrad
                * freqnm
                / math.pi
                / (del_freq * del_freq + hhw_use * hhw_use)
                * 1.77245
                * dop
            )

    hvdw_term = 0.0
    if hwvdw > 0.0 and hhw_use > 0.0:
        if freq >= 1.8e15:
            hvdw_term = (
                hwvdw
                * freqnm
                / math.pi
                / (del_freq * del_freq + hhw_use * hhw_use)
                * 1.77245
                * dop
            )

    return hres_term + hrad_term + hvdw_term


def _lyman_quasistatic_cutoff(
    freq: float,
    prqs: float,
    hyd: HydrogenDepthState,
    dbeta: float,
    dop: float,
    n: int,
    m: int,
) -> float:
    if hyd.xnfph.size < 2 or hyd.fo <= 0.0:
        return 0.0

    wavenumber_center = LYMAN_ALPHA_CENTER_WN
    wavenumber = freq / C_LIGHT_CM
    delta_wn = wavenumber - wavenumber_center

    if delta_wn < -20000.0:
        return 0.0

    extra = 0.0
    if delta_wn <= -4000.0:
        cutoff_log = _interpolate_cutoff(
            delta_wn, hydrogen_tables().cutoff_h2_plus, -15000.0, 100.0
        )
        if cutoff_log is not None:
            cutoff_val = (10.0 ** (cutoff_log - 14.0)) * hyd.xnfph[1] / C_LIGHT_CM
            extra += cutoff_val * 1.77245 * dop
    else:
        beta4000 = 4000.0 * C_LIGHT_CM / max(hyd.fo, 1e-30) * dbeta
        prqs4000 = sofbeta(beta4000, hyd.pp, n, m) * 0.5
        normalization = prqs4000 / max(hyd.fo, 1e-30) * dbeta
        cutoff4000 = (10.0 ** (-11.07 - 14.0)) * hyd.xnfph[1] / C_LIGHT_CM
        if normalization > 0.0:
            extra += (
                cutoff4000
                / normalization
                * (prqs / max(hyd.fo, 1e-30) * dbeta)
                * 1.77245
                * dop
            )

    return extra
