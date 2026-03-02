"""
Full KAPP implementation: Compute ACONT and SIGMAC from atlas_tables.

This module implements the Fortran KAPP subroutine (atlas7v.for line 4479)
which computes continuum absorption (ACONT) and scattering (SIGMAC) from
precomputed B-tables and populations.

The KAPP subroutine:
1. Calls subroutines for each species (HOP, HE1OP, HE2OP, C1OP, etc.)
2. Sums contributions: ACONT = AHYD + AHMIN + AHE1 + AHE2 + AC1 + ...
3. Computes scattering: SIGMAC = SIGH + SIGHE + SIGEL + SIGH2 + SIGX
4. Computes source function: SCONT = weighted average of source terms
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Tuple, Optional

import numpy as np

_SCALE_HMINFF = float(os.environ.get("PY_SCALE_HMINFF", "1.0"))
_SCALE_HRAYOP = float(os.environ.get("PY_SCALE_HRAYOP", "1.0"))
_SCALE_H2RAOP = float(os.environ.get("PY_SCALE_H2RAOP", "1.0"))
_SCALE_ELECOP = float(os.environ.get("PY_SCALE_ELECOP", "1.0"))

from .karsas_tables import xkarsas
from .hydrogen_wings import compute_hydrogen_continuum

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..io.atmosphere import AtmosphereModel

# Constants matching Fortran exactly
C_LIGHT_CM = 2.99792458e10  # cm/s
C_LIGHT_NM = 2.99792458e17  # nm/s
H_PLANCK = 6.62607015e-27  # erg * s
K_BOLTZ = 1.380649e-16  # erg / K
# CRITICAL: Match Fortran's TKEV calculation exactly (atlas7v.for line 1954: TKEV(J)=8.6171D-5*T(J))
KBOLTZ_EV = 8.6171e-5  # eV/K (matches Fortran: 8.6171D-5)
RYDBERG_CM = 109677.576  # cm^-1
LN10 = np.log(10.0)

# COULFF table for Coulomb free-free Gaunt factors (atlas7v.for line 4597-4612)
COULFF_Z4LOG = np.array(
    [0.0, 1.20412, 1.90849, 2.40824, 2.79588, 3.11261], dtype=np.float64
)

# HMINOP tables (atlas7v.for line 5228-5278)
HMINOP_WBF = np.array(
    [
        18.00,
        19.60,
        21.40,
        23.60,
        26.40,
        29.80,
        34.30,
        40.40,
        49.10,
        62.60,
        111.30,
        112.10,
        112.67,
        112.95,
        113.05,
        113.10,
        113.20,
        113.23,
        113.50,
        114.40,
        121.00,
        139.00,
        164.00,
        175.00,
        200.00,
        225.00,
        250.00,
        275.00,
        300.00,
        325.00,
        350.00,
        375.00,
        400.00,
        425.00,
        450.00,
        475.00,
        500.00,
        525.00,
        550.00,
        575.00,
        600.00,
        625.00,
        650.00,
        675.00,
        700.00,
        725.00,
        750.00,
        775.00,
        800.00,
        825.00,
        850.00,
        875.00,
        900.00,
        925.00,
        950.00,
        975.00,
        1000.00,
        1025.00,
        1050.00,
        1075.00,
        1100.00,
        1125.00,
        1150.00,
        1175.00,
        1200.00,
        1225.00,
        1250.00,
        1275.00,
        1300.00,
        1325.00,
        1350.00,
        1375.00,
        1400.00,
        1425.00,
        1450.00,
        1475.00,
        1500.00,
        1525.00,
        1550.00,
        1575.00,
        1600.00,
        1610.00,
        1620.00,
        1630.00,
        1643.91,
    ],
    dtype=np.float64,
)

HMINOP_BF = np.array(
    [
        0.067,
        0.088,
        0.117,
        0.155,
        0.206,
        0.283,
        0.414,
        0.703,
        1.24,
        2.33,
        11.60,
        13.90,
        24.30,
        66.70,
        95.00,
        56.60,
        20.00,
        14.60,
        8.50,
        7.10,
        5.43,
        5.91,
        7.29,
        7.918,
        9.453,
        11.08,
        12.75,
        14.46,
        16.19,
        17.92,
        19.65,
        21.35,
        23.02,
        24.65,
        26.24,
        27.77,
        29.23,
        30.62,
        31.94,
        33.17,
        34.32,
        35.37,
        36.32,
        37.17,
        37.91,
        38.54,
        39.07,
        39.48,
        39.77,
        39.95,
        40.01,
        39.95,
        39.77,
        39.48,
        39.06,
        38.53,
        37.89,
        37.13,
        36.25,
        35.28,
        34.19,
        33.01,
        31.72,
        30.34,
        28.87,
        27.33,
        25.71,
        24.02,
        22.26,
        20.46,
        18.62,
        16.74,
        14.85,
        12.95,
        11.07,
        9.211,
        7.407,
        5.677,
        4.052,
        2.575,
        1.302,
        0.8697,
        0.4974,
        0.1989,
        0.0,
    ],
    dtype=np.float64,
)

HMINOP_WAVEK = np.array(
    [
        0.50,
        0.40,
        0.35,
        0.30,
        0.25,
        0.20,
        0.18,
        0.16,
        0.14,
        0.12,
        0.10,
        0.09,
        0.08,
        0.07,
        0.06,
        0.05,
        0.04,
        0.03,
        0.02,
        0.01,
        0.008,
        0.006,
    ],
    dtype=np.float64,
)

HMINOP_THETAFF = np.array(
    [0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.8, 3.6], dtype=np.float64
)

# FFBEG: (11, 11) array - first 11 columns of FF
HMINOP_FFBEG = np.array(
    [
        [
            0.0178,
            0.0222,
            0.0308,
            0.0402,
            0.0498,
            0.0596,
            0.0695,
            0.0795,
            0.0896,
            0.131,
            0.172,
        ],
        [
            0.0228,
            0.0280,
            0.0388,
            0.0499,
            0.0614,
            0.0732,
            0.0851,
            0.0972,
            0.110,
            0.160,
            0.211,
        ],
        [
            0.0277,
            0.0342,
            0.0476,
            0.0615,
            0.0760,
            0.0908,
            0.105,
            0.121,
            0.136,
            0.199,
            0.262,
        ],
        [
            0.0364,
            0.0447,
            0.0616,
            0.0789,
            0.0966,
            0.114,
            0.132,
            0.150,
            0.169,
            0.243,
            0.318,
        ],
        [
            0.0520,
            0.0633,
            0.0859,
            0.108,
            0.131,
            0.154,
            0.178,
            0.201,
            0.225,
            0.321,
            0.418,
        ],
        [0.0791, 0.0959, 0.129, 0.161, 0.194, 0.227, 0.260, 0.293, 0.327, 0.463, 0.602],
        [0.0965, 0.117, 0.157, 0.195, 0.234, 0.272, 0.311, 0.351, 0.390, 0.549, 0.711],
        [0.121, 0.146, 0.195, 0.241, 0.288, 0.334, 0.381, 0.428, 0.475, 0.667, 0.861],
        [0.154, 0.188, 0.249, 0.309, 0.367, 0.424, 0.482, 0.539, 0.597, 0.830, 1.07],
        [0.208, 0.250, 0.332, 0.409, 0.484, 0.557, 0.630, 0.702, 0.774, 1.06, 1.36],
        [0.293, 0.354, 0.468, 0.576, 0.677, 0.777, 0.874, 0.969, 1.06, 1.45, 1.83],
    ],
    dtype=np.float64,
)

# FFEND: (11, 11) array - last 11 columns of FF
HMINOP_FFEND = np.array(
    [
        [0.358, 0.432, 0.572, 0.702, 0.825, 0.943, 1.06, 1.17, 1.28, 1.73, 2.17],
        [0.448, 0.539, 0.711, 0.871, 1.02, 1.16, 1.29, 1.43, 1.57, 2.09, 2.60],
        [0.579, 0.699, 0.924, 1.13, 1.33, 1.51, 1.69, 1.86, 2.02, 2.67, 3.31],
        [0.781, 0.940, 1.24, 1.52, 1.78, 2.02, 2.26, 2.48, 2.69, 3.52, 4.31],
        [1.11, 1.34, 1.77, 2.17, 2.53, 2.87, 3.20, 3.51, 3.80, 4.92, 5.97],
        [1.73, 2.08, 2.74, 3.37, 3.90, 4.50, 5.01, 5.50, 5.95, 7.59, 9.06],
        [3.04, 3.65, 4.80, 5.86, 6.86, 7.79, 8.67, 9.50, 10.3, 13.2, 15.6],
        [6.79, 8.16, 10.7, 13.1, 15.3, 17.4, 19.4, 21.2, 23.0, 29.5, 35.0],
        [27.0, 32.4, 42.6, 51.9, 60.7, 68.9, 76.8, 84.2, 91.4, 117.0, 140.0],
        [42.3, 50.6, 66.4, 80.8, 94.5, 107.0, 120.0, 131.0, 142.0, 183.0, 219.0],
        [75.1, 90.0, 118.0, 144.0, 168.0, 191.0, 212.0, 234.0, 253.0, 325.0, 388.0],
    ],
    dtype=np.float64,
)

# HRAYOP: Gavrila tables for hydrogen Rayleigh scattering (atlas7v.for line 5351-5420)
HRAYOP_GAVRILAM = np.array(
    [
        -0.000113,
        -0.000450,
        -0.001014,
        -0.001804,
        -0.002823,
        -0.004072,
        -0.005553,
        -0.007269,
        -0.009223,
        -0.011419,
        -0.013861,
        -0.016553,
        -0.019500,
        -0.022709,
        -0.026185,
        -0.029936,
        -0.033968,
        -0.038291,
        -0.042913,
        -0.047843,
        -0.053093,
        -0.058674,
        -0.064599,
        -0.070882,
        -0.077537,
        -0.084581,
        -0.092031,
        -0.099907,
        -0.108230,
        -0.117022,
        -0.126308,
        -0.136117,
        -0.146477,
        -0.157422,
        -0.168987,
        -0.181213,
        -0.194143,
        -0.207825,
        -0.222313,
        -0.237667,
        -0.253953,
        -0.271245,
        -0.289626,
        -0.309189,
        -0.330041,
        -0.352300,
        -0.376103,
        -0.401605,
        -0.428985,
        -0.458448,
        -0.490235,
        -0.524625,
        -0.561947,
        -0.602591,
        -0.647023,
        -0.695805,
        -0.749619,
        -0.809306,
        -0.875910,
        -0.950750,
        -1.035515,
        -1.132403,
        -1.244337,
        -1.375285,
        -1.530787,
        -1.718821,
        -1.951320,
        -2.246993,
        -2.636960,
        -3.177142,
        -3.979234,
        -5.303624,
        -7.930999,
        -15.57,
    ],
    dtype=np.float64,
)

HRAYOP_GAVRILAMAB = np.array(
    [
        15.57,
        15.382871,
        10.160646,
        7.538338,
        5.955062,
        4.890397,
        4.121176,
        3.535672,
        3.071659,
        2.691623,
        2.371483,
        2.094936,
        1.850395,
        1.629203,
        1.424526,
        1.230596,
        1.042127,
        0.853766,
        0.659460,
        0.451533,
        0.219115,
        -0.054939,
        -0.400868,
        -0.879559,
        -1.637857,
        -3.150374,
        -8.0,
    ],
    dtype=np.float64,
)

HRAYOP_GAVRILAMBC = np.array(
    [
        8.0,
        8.0,
        8.0,
        5.442077,
        4.313409,
        3.573504,
        3.043218,
        2.637983,
        2.312466,
        2.039959,
        1.803441,
        1.591244,
        1.394717,
        1.206823,
        1.021148,
        0.831020,
        0.628449,
        0.402484,
        0.136127,
        -0.200462,
        -0.667435,
        -1.410661,
        -2.906862,
        -9.0,
    ],
    dtype=np.float64,
)

HRAYOP_GAVRILAMCD = np.array(
    [
        9.0,
        9.0,
        6.145775,
        4.544224,
        3.630968,
        3.029081,
        2.593248,
        2.255265,
        1.978565,
        1.741426,
        1.529699,
        1.333240,
        1.143898,
        0.954154,
        0.755875,
        0.538760,
        0.287687,
        -0.022759,
        -0.441666,
        -1.081712,
        -2.667783,
        -2.667783,
    ],
    dtype=np.float64,
)

HRAYOP_GAVRILALYMANCONT = np.array(
    [
        2.667783,
        2.526696,
        2.408970,
        2.308970,
        2.222736,
        2.147415,
        2.080913,
        2.021653,
        1.968431,
        1.920304,
        1.876527,
        1.799739,
        1.734455,
        1.678180,
        1.629118,
        1.585943,
        1.547643,
        1.513435,
        1.482700,
        1.454941,
        1.429751,
        1.406798,
        1.385804,
        1.366536,
        1.348797,
        1.332419,
        1.317257,
        1.303187,
        1.290100,
        1.277901,
        1.266509,
        1.255848,
        1.245856,
        1.236474,
        1.227652,
        1.219344,
        1.190492,
        1.167227,
        1.148153,
        1.132293,
        1.118945,
        1.107593,
        1.097848,
        1.089413,
        1.082059,
        1.075606,
        1.069908,
        1.064850,
        1.060338,
        1.056294,
        1.052655,
        1.038936,
        1.030042,
        1.023928,
        1.019536,
        1.016269,
        1.011814,
        1.008986,
        1.007074,
        1.005720,
        1.004724,
        1.003970,
        1.003385,
        1.003140,
    ],
    dtype=np.float64,
)

HRAYOP_FGAVRILALYMANCONT = np.array(
    [
        1.00,
        1.05,
        1.10,
        1.15,
        1.20,
        1.25,
        1.30,
        1.35,
        1.40,
        1.45,
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
        4.4,
        4.8,
        5.2,
        5.6,
        6.0,
        6.4,
        6.8,
        7.2,
        7.6,
        8.0,
        8.4,
        8.8,
        9.2,
        9.6,
        10.0,
        12.0,
        14.0,
        16.0,
        18.0,
        20.0,
        24.0,
        28.0,
        32.0,
        36.0,
        40.0,
        44.0,
        48.0,
        50.0,
    ],
    dtype=np.float64,
)
COULFF_A_TABLE = np.array(
    [
        [5.53, 5.49, 5.46, 5.43, 5.40, 5.25, 5.00, 4.69, 4.48, 4.16, 3.85],
        [4.91, 4.87, 4.84, 4.80, 4.77, 4.63, 4.40, 4.13, 3.87, 3.52, 3.27],
        [4.29, 4.25, 4.22, 4.18, 4.15, 4.02, 3.80, 3.57, 3.27, 2.98, 2.70],
        [3.64, 3.61, 3.59, 3.56, 3.54, 3.41, 3.22, 2.97, 2.70, 2.45, 2.20],
        [3.00, 2.98, 2.97, 2.95, 2.94, 2.81, 2.65, 2.44, 2.21, 2.01, 1.81],
        [2.41, 2.41, 2.41, 2.41, 2.41, 2.32, 2.19, 2.02, 1.84, 1.67, 1.50],
        [1.87, 1.89, 1.91, 1.93, 1.95, 1.90, 1.80, 1.68, 1.52, 1.41, 1.30],
        [1.33, 1.39, 1.44, 1.49, 1.55, 1.56, 1.51, 1.42, 1.33, 1.25, 1.17],
        [0.90, 0.95, 1.00, 1.08, 1.17, 1.30, 1.32, 1.30, 1.20, 1.15, 1.11],
        [0.55, 0.58, 0.62, 0.70, 0.85, 1.01, 1.15, 1.18, 1.15, 1.11, 1.08],
        [0.33, 0.36, 0.39, 0.46, 0.59, 0.76, 0.97, 1.09, 1.13, 1.10, 1.08],
        [0.19, 0.21, 0.24, 0.28, 0.38, 0.53, 0.76, 0.96, 1.08, 1.09, 1.09],
    ],
    dtype=np.float64,
)

# HOTOP transition table (atlas7v.for HOTOP DATA A1..A7, 60 entries × 7 fields):
# (freq0, xsect, alpha, power, multiplier, excitation_eV, xNfpId)
HOTOP_TRANSITIONS = np.array(
    [
        (4.149945e15, 6.900000e-18, 1.000000e00, 6.000000e00, 6.000000e00, 1.371000e01, 2.000000e00),
        (4.574341e15, 2.500000e-18, 1.000000e00, 4.000000e00, 2.000000e00, 1.196000e01, 2.000000e00),
        (5.220770e15, 1.080000e-17, 1.000000e00, 4.000000e00, 1.000000e01, 9.280000e00, 2.000000e00),
        (5.222307e15, 5.350000e-18, 3.769000e00, 2.000000e00, 1.000000e00, 0.000000e00, 1.600000e01),
        (5.892577e15, 4.600000e-18, 1.950000e00, 6.000000e00, 6.000000e00, 0.000000e00, 2.000000e00),
        (6.177022e15, 3.500000e-18, 1.000000e00, 4.000000e00, 1.200000e01, 5.330000e00, 2.000000e00),
        (6.181062e15, 6.750000e-18, 3.101000e00, 5.000000e00, 1.000000e00, 4.050000e00, 6.000000e00),
        (6.701879e15, 6.650000e-18, 2.789000e00, 5.000000e00, 5.000000e00, 1.900000e00, 6.000000e00),
        (7.158382e15, 6.650000e-18, 2.860000e00, 6.000000e00, 9.000000e00, 0.000000e00, 6.000000e00),
        (7.284488e15, 3.430000e-18, 4.174000e00, 5.000000e00, 6.000000e00, 5.020000e00, 1.100000e01),
        (7.693612e15, 3.530000e-18, 3.808000e00, 5.000000e00, 1.000000e01, 3.330000e00, 1.100000e01),
        (7.885955e15, 2.320000e-18, 3.110000e00, 5.000000e00, 6.000000e00, 5.020000e00, 1.100000e01),
        (8.295079e15, 3.970000e-18, 3.033000e00, 5.000000e00, 1.000000e01, 3.330000e00, 1.100000e01),
        (8.497686e15, 7.320000e-18, 3.837000e00, 5.000000e00, 4.000000e00, 0.000000e00, 1.100000e01),
        (8.509966e15, 2.000000e-18, 1.750000e00, 7.000000e00, 3.000000e00, 1.269000e01, 3.000000e00),
        (8.572854e15, 1.680000e-18, 3.751000e00, 5.000000e00, 6.000000e00, 5.020000e00, 1.100000e01),
        (9.906370e15, 4.160000e-18, 2.717000e00, 3.000000e00, 6.000000e00, 0.000000e00, 1.700000e01),
        (1.000693e16, 2.400000e-18, 1.750000e00, 7.000000e00, 9.000000e00, 6.500000e00, 3.000000e00),
        (1.046078e16, 4.800000e-18, 1.000000e00, 4.000000e00, 1.000000e01, 1.253000e01, 7.000000e00),
        (1.067157e16, 2.710000e-18, 2.148000e00, 3.000000e00, 6.000000e00, 0.000000e00, 1.700000e01),
        (1.146734e16, 2.060000e-18, 1.626000e00, 6.000000e00, 6.000000e00, 0.000000e00, 7.000000e00),
        (1.156813e16, 5.200000e-19, 2.126000e00, 3.000000e00, 6.000000e00, 0.000000e00, 1.700000e01),
        (1.157840e16, 9.100000e-19, 4.750000e00, 4.000000e00, 1.000000e00, 0.000000e00, 3.000000e00),
        (1.177220e16, 5.300000e-18, 1.000000e00, 4.000000e00, 1.200000e01, 7.100000e00, 7.000000e00),
        (1.198813e16, 3.970000e-18, 2.780000e00, 6.000000e00, 1.000000e00, 5.350000e00, 1.200000e01),
        (1.325920e16, 3.790000e-18, 2.777000e00, 6.000000e00, 5.000000e00, 2.510000e00, 1.200000e01),
        (1.327649e16, 3.650000e-18, 2.014000e00, 6.000000e00, 9.000000e00, 0.000000e00, 1.200000e01),
        (1.361466e16, 7.000000e-18, 1.000000e00, 2.000000e00, 5.000000e00, 7.480000e00, 1.200000e01),
        (1.365932e16, 9.300000e-19, 1.500000e00, 7.000000e00, 6.000000e00, 8.000000e00, 4.000000e00),
        (1.481487e16, 1.100000e-18, 1.750000e00, 7.000000e00, 3.000000e00, 1.620000e01, 8.000000e00),
        (1.490032e16, 5.490000e-18, 3.000000e00, 5.000000e00, 1.000000e00, 6.910000e00, 1.800000e01),
        (1.533389e16, 1.800000e-18, 2.277000e00, 4.000000e00, 9.000000e00, 0.000000e00, 1.800000e01),
        (1.559452e16, 8.700000e-19, 3.000000e00, 6.000000e00, 2.000000e00, 0.000000e00, 4.000000e00),
        (1.579688e16, 4.170000e-18, 2.074000e00, 4.000000e00, 5.000000e00, 3.200000e00, 1.800000e01),
        (1.643205e16, 1.390000e-18, 2.792000e00, 5.000000e00, 5.000000e00, 3.200000e00, 1.800000e01),
        (1.656208e16, 2.500000e-18, 2.346000e00, 5.000000e00, 9.000000e00, 0.000000e00, 1.800000e01),
        (1.671401e16, 1.300000e-18, 1.750000e00, 7.000000e00, 9.000000e00, 8.350000e00, 8.000000e00),
        (1.719725e16, 1.480000e-18, 2.225000e00, 5.000000e00, 9.000000e00, 0.000000e00, 1.800000e01),
        (1.737839e16, 2.700000e-18, 1.000000e00, 4.000000e00, 1.000000e01, 1.574000e01, 1.300000e01),
        (1.871079e16, 1.270000e-18, 8.310000e-01, 6.000000e00, 6.000000e00, 0.000000e00, 1.300000e01),
        (1.873298e16, 9.100000e-19, 3.000000e00, 4.000000e00, 1.000000e00, 0.000000e00, 8.000000e00),
        (1.903597e16, 2.900000e-18, 1.000000e00, 4.000000e00, 1.200000e01, 8.880000e00, 1.300000e01),
        (2.060738e16, 4.600000e-18, 1.000000e00, 3.000000e00, 1.200000e01, 2.284000e01, 1.900000e01),
        (2.125492e16, 5.900000e-19, 1.000000e00, 6.000000e00, 6.000000e00, 9.990000e00, 9.000000e00),
        (2.162610e16, 1.690000e-18, 1.937000e00, 5.000000e00, 6.000000e00, 7.710000e00, 1.900000e01),
        (2.226127e16, 1.690000e-18, 1.841000e00, 5.000000e00, 1.000000e01, 5.080000e00, 1.900000e01),
        (2.251163e16, 9.300000e-19, 2.455000e00, 6.000000e00, 6.000000e00, 7.710000e00, 1.900000e01),
        (2.278001e16, 7.900000e-19, 1.000000e00, 6.000000e00, 9.000000e00, 1.020000e01, 1.400000e01),
        (2.317678e16, 1.650000e-18, 2.277000e00, 6.000000e00, 1.000000e01, 5.080000e00, 1.900000e01),
        (2.348946e16, 3.110000e-18, 1.963000e00, 6.000000e00, 4.000000e00, 0.000000e00, 1.900000e01),
        (2.351911e16, 7.300000e-19, 1.486000e00, 5.000000e00, 6.000000e00, 7.710000e00, 1.900000e01),
        (2.366973e16, 5.000000e-19, 1.000000e00, 4.000000e00, 2.000000e00, 0.000000e00, 9.000000e00),
        (2.507544e16, 6.900000e-19, 1.000000e00, 6.000000e00, 3.000000e00, 1.969000e01, 1.400000e01),
        (2.754065e16, 7.600000e-19, 1.000000e00, 2.000000e00, 1.000000e00, 0.000000e00, 1.400000e01),
        (2.864850e16, 1.540000e-18, 2.104000e00, 6.000000e00, 1.000000e00, 7.920000e00, 2.000000e01),
        (2.965598e16, 1.530000e-18, 2.021000e00, 6.000000e00, 5.000000e00, 3.760000e00, 2.000000e01),
        (3.054151e16, 1.400000e-18, 1.471000e00, 6.000000e00, 9.000000e00, 0.000000e00, 2.000000e01),
        (3.085141e16, 2.800000e-18, 1.000000e00, 4.000000e00, 5.000000e00, 1.101000e01, 2.000000e01),
        (3.339687e16, 3.600000e-19, 1.000000e00, 6.000000e00, 2.000000e00, 0.000000e00, 1.500000e01),
        (3.818757e16, 4.900000e-19, 1.145000e00, 6.000000e00, 6.000000e00, 0.000000e00, 2.100000e01),
    ],
    dtype=np.float64,
)


def _coulff(
    j: int,
    nz: int,
    freq: float,
    freqlg: float,
    temperature: np.ndarray,
    tlog: np.ndarray,
) -> float:
    """Compute Coulomb free-free Gaunt factor (atlas7v.for line 5057-5187).

    Parameters
    ----------
    j:
        Layer index (0-based)
    nz:
        Ion charge (1 for neutral, 2 for singly ionized, etc.)
    freq:
        Frequency in Hz
    freqlg:
        log10(frequency)
    temperature:
        Temperature array
    tlog:
        log10(temperature) array

    Returns
    -------
    coulff:
        Coulomb free-free Gaunt factor
    """
    if nz < 1 or nz > 6:
        return 1.0  # Default for unsupported charge states

    z4log = COULFF_Z4LOG[nz - 1]
    temp = temperature[j]
    tlog_j = tlog[j]

    # GAMLOG = log10(158000*Z*Z/T) * 2
    # GAMLOG = 10.39638 - TLOG(J)/1.15129 + Z4LOG(NZ)
    gamlog = 10.39638 - tlog_j / 1.15129 + z4log
    igam = max(1, min(int(gamlog + 7), 10))

    # HVKTLG = log10(h*nu/(k*T)) * 2
    # HVKTLG = (FREQLG - TLOG(J))/1.15129 - 20.63764
    hvktlg = (freqlg - tlog_j) / 1.15129 - 20.63764
    ihvkt = max(1, min(int(hvktlg + 9), 11))

    p = gamlog - float(igam - 7)
    q = hvktlg - float(ihvkt - 9)

    # Bilinear interpolation
    a_00 = COULFF_A_TABLE[igam - 1, ihvkt - 1]
    a_01 = COULFF_A_TABLE[igam - 1, ihvkt] if ihvkt < 11 else a_00
    a_10 = COULFF_A_TABLE[igam, ihvkt - 1] if igam < 10 else a_00
    a_11 = COULFF_A_TABLE[igam, ihvkt] if (igam < 10 and ihvkt < 11) else a_00

    coulff = (1.0 - p) * ((1.0 - q) * a_00 + q * a_01) + p * (
        (1.0 - q) * a_10 + q * a_11
    )

    return coulff


def _coulff_grid(nz: int, freqlg: np.ndarray, tlog: np.ndarray) -> np.ndarray:
    """Vectorized COULFF over (layer, frequency_chunk) for HOTOP."""
    if nz < 1 or nz > 6:
        return np.ones((tlog.size, freqlg.size), dtype=np.float64)

    z4log = COULFF_Z4LOG[nz - 1]
    tlog_col = tlog[:, np.newaxis]
    freqlg_row = freqlg[np.newaxis, :]

    gamlog = 10.39638 - tlog_col / 1.15129 + z4log
    hvktlg = (freqlg_row - tlog_col) / 1.15129 - 20.63764

    igam = np.clip((gamlog + 7.0).astype(np.int64), 1, 10)
    ihvkt = np.clip((hvktlg + 9.0).astype(np.int64), 1, 11)

    p = gamlog - (igam - 7.0)
    q = hvktlg - (ihvkt - 9.0)

    ig = igam - 1
    ih = ihvkt - 1

    a00 = COULFF_A_TABLE[ig, ih]
    a01_raw = COULFF_A_TABLE[ig, np.minimum(ih + 1, 10)]
    a10_raw = COULFF_A_TABLE[np.minimum(ig + 1, 11), ih]
    a11_raw = COULFF_A_TABLE[np.minimum(ig + 1, 11), np.minimum(ih + 1, 10)]

    a01 = np.where(ihvkt < 11, a01_raw, a00)
    a10 = np.where(igam < 10, a10_raw, a00)
    a11 = np.where((igam < 10) & (ihvkt < 11), a11_raw, a00)

    return (1.0 - p) * ((1.0 - q) * a00 + q * a01) + p * (
        (1.0 - q) * a10 + q * a11
    )


def _linter(xold: np.ndarray, yold: np.ndarray, xnew: np.ndarray) -> np.ndarray:
    """Linear interpolation/extrapolation (atlas7v.for line 6771-6784).

    CRITICAL: Fortran LINTER extrapolates beyond table boundaries using the
    nearest two points. It does NOT clamp to edge values!

    This is important for H- free-free opacity at wavelengths > 500nm (beyond
    the FF table), where extrapolation gives physically correct decreasing
    cross-sections.

    Parameters
    ----------
    xold:
        Sorted array of x values (increasing)
    yold:
        Corresponding y values
    xnew:
        New x values to interpolate at

    Returns
    -------
    ynew:
        Interpolated/extrapolated y values
    """
    nold = xold.size
    nnew = xnew.size
    ynew = np.zeros(nnew, dtype=np.float64)

    # Fortran LINTER algorithm (atlas7v.for lines 6771-6784):
    # IOLD=2
    # DO 2 INEW=1,NNEW
    # 1 IF(XNEW(INEW).LT.XOLD(IOLD))GO TO 2
    #   IF(IOLD.EQ.NOLD)GO TO 2
    #   IOLD=IOLD+1
    #   GO TO 1
    # 2 YNEW(INEW)=YOLD(IOLD-1)+(YOLD(IOLD)-YOLD(IOLD-1))/
    #      (XOLD(IOLD)-XOLD(IOLD-1))*(XNEW(INEW)-XOLD(IOLD-1))
    #
    # This always uses linear interpolation/extrapolation - NO clamping!

    iold = 1  # Start at index 1 (second element, 0-based = Fortran's IOLD=2)
    for inew in range(nnew):
        # Find position in xold (move iold forward until xnew[inew] < xold[iold])
        while iold < nold - 1 and xnew[inew] >= xold[iold]:
            iold += 1

        # Linear interpolation/extrapolation using points [iold-1, iold]
        # When xnew < xold[0]: iold stays at 1, uses [0,1] to extrapolate LEFT
        # When xnew > xold[-1]: iold at nold-1, uses [nold-2, nold-1] to extrapolate RIGHT
        # When within table: normal linear interpolation
        denom = xold[iold] - xold[iold - 1]
        if abs(denom) < 1e-40:
            ynew[inew] = yold[iold - 1]
        else:
            weight = (xnew[inew] - xold[iold - 1]) / denom
            ynew[inew] = yold[iold - 1] + (yold[iold] - yold[iold - 1]) * weight

    return ynew


def _map1_simple(xold: np.ndarray, fold: np.ndarray, xnew: float) -> float:
    """MAP1 for single value interpolation (used by HMINOP and HRAYOP).

    Uses the full MAP1 implementation to ensure exact matching with Fortran.
    This is a wrapper around the full _map1 function from josh_solver.
    """
    from .josh_solver import _map1

    # Convert scalar to array for full MAP1 implementation
    xnew_arr = np.array([xnew], dtype=np.float64)
    fnew_arr, _ = _map1(xold, fold, xnew_arr)
    return float(fnew_arr[0])


def _planck_nu(freq: float, temperature: np.ndarray) -> np.ndarray:
    """Compute Planck function B_nu(T) in erg/s/cm^2/Hz/steradian."""
    const_factor = 2 * H_PLANCK / C_LIGHT_CM**2
    hnu_over_kt = H_PLANCK * freq / (K_BOLTZ * temperature)

    # Handle very small hnu_over_kt (Rayleigh-Jeans limit)
    RJ_THRESHOLD = 1e-6
    bnu = np.zeros_like(temperature)

    rj_mask = hnu_over_kt < RJ_THRESHOLD
    bnu[rj_mask] = 2 * K_BOLTZ * temperature[rj_mask] * freq**2 / C_LIGHT_CM**2

    full_planck_mask = ~rj_mask
    bnu[full_planck_mask] = (
        const_factor * freq**3 / np.expm1(hnu_over_kt[full_planck_mask])
    )

    # Ensure no NaNs or Infs
    bnu[np.isnan(bnu)] = 0.0
    bnu[np.isinf(bnu)] = 0.0

    return bnu


# =============================================================================
# LUKEOP HELPER FUNCTIONS (atlas7v.for lines 8952-9259)
# =============================================================================


def _seaton(freq0: float, xsect: float, power: float, a: float, freq: float) -> float:
    """Seaton photoionization cross-section formula (atlas7v.for line 9252-9259).

    SEATON = XSECT * (A + (1-A)*(FREQ0/FREQ)) * SQRT((FREQ0/FREQ)**(2*POWER))

    Parameters
    ----------
    freq0 : float
        Threshold frequency (Hz)
    xsect : float
        Cross-section at threshold (cm²)
    power : float
        Power-law exponent
    a : float
        Asymptotic constant
    freq : float
        Frequency to evaluate at (Hz)

    Returns
    -------
    float
        Photoionization cross-section (cm²)
    """
    if freq < freq0:
        return 0.0
    ratio = freq0 / freq
    return xsect * (a + (1.0 - a) * ratio) * np.sqrt(ratio ** int(2.0 * power + 0.01))


# SI2OP Peach tables (atlas7v.for lines 9050-9073)
_SI2OP_PEACH = np.array(
    [
        # Temperature indices: 10000K, 12000K, 14000K, 16000K, 18000K, 20000K
        [-43.8941, -43.8941, -43.8941, -43.8941, -43.8941, -43.8941],  # 500 Å
        [-42.2444, -42.2444, -42.2444, -42.2444, -42.2444, -42.2444],  # 600 Å
        [-40.6054, -40.6054, -40.6054, -40.6054, -40.6054, -40.6054],  # 759 Å
        [-54.2389, -52.2906, -50.8799, -49.8033, -48.9485, -48.2490],  # 760 Å
        [-50.4108, -48.4892, -47.1090, -46.0672, -45.2510, -44.5933],  # 1905 Å
        [-52.0936, -50.0741, -48.5999, -47.4676, -46.5649, -45.8246],  # 1906 Å
        [-51.9548, -49.9371, -48.4647, -47.3340, -46.4333, -45.6947],  # 1975 Å
        [-54.2407, -51.7319, -49.9178, -48.5395, -47.4529, -46.5709],  # 1976 Å
        [-52.7355, -50.2218, -48.4059, -47.0267, -45.9402, -45.0592],  # 3245 Å
        [-53.5387, -50.9189, -49.0200, -47.5750, -46.4341, -45.5082],  # 3246 Å
        [-53.2417, -50.6234, -48.7252, -47.2810, -46.1410, -45.2153],  # 3576 Å
        [-53.5097, -50.8535, -48.9263, -47.4586, -46.2994, -45.3581],  # 3577 Å
        [-54.0561, -51.2365, -49.1980, -47.6497, -46.4302, -45.4414],  # 3900 Å
        [-53.8469, -51.0256, -48.9860, -47.4368, -46.2162, -45.2266],  # 4200 Å
    ],
    dtype=np.float64,
)

_SI2OP_FREQSI = np.array(
    [
        4.9965417e15,
        3.9466738e15,
        1.5736321e15,
        1.5171539e15,
        9.2378947e14,
        8.3825004e14,
        7.6869872e14,
    ],
    dtype=np.float64,
)

_SI2OP_FLOG = np.array(
    [
        36.32984,
        36.14752,
        35.91165,
        34.99216,
        34.95561,
        34.45951,
        34.36234,
        34.27572,
        34.20161,
    ],
    dtype=np.float64,
)

_SI2OP_TLG = np.array(
    [9.21034, 9.39266, 9.54681, 9.68034, 9.79813, 9.90349], dtype=np.float64
)


def _si2op_vectorized(
    freq: float, freqlg: float, temp: np.ndarray, tlog: np.ndarray
) -> np.ndarray:
    """Silicon II opacity (atlas7v.for lines 9043-9097).

    Returns cross-section * partition function for each layer.
    Uses Peach tables with temperature/frequency interpolation.

    Parameters
    ----------
    freq : float
        Frequency (Hz)
    freqlg : float
        Log of frequency
    temp : np.ndarray
        Temperature array (K), shape (n_layers,)
    tlog : np.ndarray
        Log of temperature array, shape (n_layers,)

    Returns
    -------
    np.ndarray
        SI2OP values for each layer (cm²), shape (n_layers,)
    """
    n_layers = temp.size

    # Temperature interpolation indices (atlas7v.for lines 9077-9080)
    # N = MAX(MIN(5, INT(T/2000) - 4), 1)
    nt = np.clip((temp / 2000.0).astype(int) - 4, 1, 5)
    dt = (tlog - _SI2OP_TLG[nt - 1]) / (_SI2OP_TLG[nt] - _SI2OP_TLG[nt - 1])

    # Frequency interpolation (atlas7v.for lines 9083-9093)
    n = 0
    for i in range(7):
        if freq > _SI2OP_FREQSI[i]:
            n = i + 1
            break
    else:
        n = 8

    # Adjust index based on Fortran logic
    d = (
        (freqlg - _SI2OP_FLOG[n - 1]) / (_SI2OP_FLOG[n] - _SI2OP_FLOG[n - 1])
        if n > 0 and n < 9
        else 0.0
    )

    # Map n to Peach table index
    if n > 2:
        n = 2 * n - 2
    n = min(n, 13)

    d1 = 1.0 - d

    # Interpolate in frequency (atlas7v.for lines 9092-9093)
    if n < 14:
        x = _SI2OP_PEACH[n] * d + _SI2OP_PEACH[n - 1] * d1 if n > 0 else _SI2OP_PEACH[0]
    else:
        x = _SI2OP_PEACH[13]

    # Interpolate in temperature and compute final value (atlas7v.for lines 9094-9095)
    result = np.zeros(n_layers, dtype=np.float64)
    for j in range(n_layers):
        nj = nt[j] - 1  # 0-indexed
        if nj < 5:
            val = x[nj] * (1.0 - dt[j]) + x[nj + 1] * dt[j]
        else:
            val = x[5]
        result[j] = np.exp(val) * 6.0

    return result


# =============================================================================
# MOLECULAR OPACITY FUNCTIONS: CHOP, OHOP, H2COLLOP
# =============================================================================
# These contribute to COOLOP for cool stars (T < 9000K)
# ACOOL = AC1 + AMG1 + AAL1 + ASI1 + AFE1 + CHOP*XNFPCH + OHOP*XNFPOH + AH2COLL

# CH Partition function (atlas7v.for line 8348-8355)
_CH_PARTITION = np.array(
    [
        203.741,
        249.643,
        299.341,
        353.477,
        412.607,
        477.237,
        547.817,
        624.786,
        708.543,
        799.463,
        897.912,
        1004.227,
        1118.738,
        1241.761,
        1373.588,
        1514.481,
        1664.677,
        1824.394,
        1993.801,
        2173.050,
        2362.234,
        2561.424,
        2770.674,
        2989.930,
        3219.204,
        3458.378,
        3707.355,
        3966.005,
        4234.155,
        4511.604,
        4798.135,
        5093.554,
        5397.593,
        5709.948,
        6030.401,
        6358.646,
        6694.379,
        7037.313,
        7387.147,
        7743.579,
        8106.313,
    ],
    dtype=np.float64,
)

# OH Partition function (atlas7v.for line 8665-8672)
_OH_PARTITION = np.array(
    [
        145.979,
        178.033,
        211.618,
        247.053,
        284.584,
        324.398,
        366.639,
        411.425,
        458.854,
        509.012,
        561.976,
        617.823,
        676.626,
        738.448,
        803.363,
        871.437,
        942.735,
        1017.330,
        1095.284,
        1176.654,
        1261.510,
        1349.898,
        1441.875,
        1537.483,
        1636.753,
        1739.733,
        1846.434,
        1956.883,
        2071.080,
        2189.029,
        2310.724,
        2436.155,
        2565.283,
        2698.103,
        2834.571,
        2974.627,
        3118.242,
        3265.366,
        3415.912,
        3569.837,
        3727.077,
    ],
    dtype=np.float64,
)

# CH cross-section table (atlas7v.for lines 8138-8347)
# Shape: (105, 15) - 105 energy bins (0.1-10.5 eV), 15 temperature points
# Stored as log10(cross-section * partition_function)
# Temperature grid: 2000K to 9000K in 500K steps (15 points)
_CH_CROSSSECT = np.array(
    [
        # Energy 0.1-1.0 eV (rows 0-9) - these are essentially zero (-38)
        [-38.000] * 15,  # 0.1 eV
        [-38.000] * 15,  # 0.1 eV (duplicate in Fortran data)
        [
            -32.727,
            -31.151,
            -30.133,
            -29.432,
            -28.925,
            -28.547,
            -28.257,
            -28.030,
            -27.848,
            -27.701,
            -27.580,
            -27.479,
            -27.395,
            -27.322,
            -27.261,
        ],  # 0.2
        [
            -31.588,
            -30.011,
            -28.993,
            -28.290,
            -27.784,
            -27.405,
            -27.115,
            -26.887,
            -26.705,
            -26.558,
            -26.437,
            -26.336,
            -26.251,
            -26.179,
            -26.117,
        ],  # 0.3
        [
            -30.407,
            -28.830,
            -27.811,
            -27.108,
            -26.601,
            -26.223,
            -25.932,
            -25.705,
            -25.523,
            -25.376,
            -25.255,
            -25.154,
            -25.069,
            -24.997,
            -24.935,
        ],  # 0.4
        [
            -29.513,
            -27.937,
            -26.920,
            -26.218,
            -25.712,
            -25.334,
            -25.043,
            -24.816,
            -24.635,
            -24.487,
            -24.366,
            -24.266,
            -24.181,
            -24.109,
            -24.047,
        ],  # 0.5
        [
            -28.910,
            -27.341,
            -26.327,
            -25.628,
            -25.123,
            -24.746,
            -24.457,
            -24.230,
            -24.049,
            -23.902,
            -23.782,
            -23.681,
            -23.597,
            -23.525,
            -23.464,
        ],  # 0.6
        [
            -28.517,
            -26.961,
            -25.955,
            -25.261,
            -24.760,
            -24.385,
            -24.098,
            -23.873,
            -23.694,
            -23.548,
            -23.429,
            -23.329,
            -23.245,
            -23.174,
            -23.113,
        ],  # 0.7
        [
            -28.213,
            -26.675,
            -25.680,
            -24.993,
            -24.497,
            -24.127,
            -23.843,
            -23.620,
            -23.443,
            -23.299,
            -23.181,
            -23.082,
            -22.999,
            -22.929,
            -22.869,
        ],  # 0.8
        [
            -27.942,
            -26.427,
            -25.446,
            -24.769,
            -24.280,
            -23.915,
            -23.635,
            -23.416,
            -23.241,
            -23.100,
            -22.983,
            -22.887,
            -22.805,
            -22.736,
            -22.677,
        ],  # 0.9
        [
            -27.706,
            -26.210,
            -25.241,
            -24.572,
            -24.088,
            -23.728,
            -23.451,
            -23.235,
            -23.063,
            -22.923,
            -22.808,
            -22.713,
            -22.633,
            -22.565,
            -22.507,
        ],  # 1.0
        # Energy 1.1-2.0 eV (rows 10-19)
        [
            -27.475,
            -26.000,
            -25.043,
            -24.382,
            -23.905,
            -23.548,
            -23.275,
            -23.062,
            -22.891,
            -22.753,
            -22.640,
            -22.546,
            -22.467,
            -22.400,
            -22.343,
        ],  # 1.1
        [
            -27.221,
            -25.783,
            -24.844,
            -24.193,
            -23.723,
            -23.372,
            -23.102,
            -22.892,
            -22.724,
            -22.588,
            -22.476,
            -22.384,
            -22.306,
            -22.240,
            -22.184,
        ],  # 1.2
        [
            -26.863,
            -25.506,
            -24.607,
            -23.979,
            -23.523,
            -23.182,
            -22.919,
            -22.714,
            -22.550,
            -22.417,
            -22.309,
            -22.218,
            -22.142,
            -22.078,
            -22.023,
        ],  # 1.3
        [
            -26.685,
            -25.347,
            -24.457,
            -23.835,
            -23.382,
            -23.044,
            -22.784,
            -22.580,
            -22.418,
            -22.286,
            -22.178,
            -22.089,
            -22.014,
            -21.950,
            -21.896,
        ],  # 1.4
        [
            -26.085,
            -24.903,
            -24.105,
            -23.538,
            -23.120,
            -22.805,
            -22.561,
            -22.370,
            -22.217,
            -22.093,
            -21.991,
            -21.906,
            -21.835,
            -21.775,
            -21.723,
        ],  # 1.5
        [
            -25.902,
            -24.727,
            -23.936,
            -23.376,
            -22.964,
            -22.654,
            -22.415,
            -22.227,
            -22.076,
            -21.955,
            -21.855,
            -21.772,
            -21.702,
            -21.644,
            -21.593,
        ],  # 1.6
        [
            -25.215,
            -24.196,
            -23.510,
            -23.019,
            -22.655,
            -22.378,
            -22.163,
            -21.992,
            -21.855,
            -21.744,
            -21.653,
            -21.577,
            -21.513,
            -21.459,
            -21.412,
        ],  # 1.7
        [
            -24.914,
            -23.937,
            -23.284,
            -22.820,
            -22.475,
            -22.212,
            -22.007,
            -21.845,
            -21.715,
            -21.609,
            -21.522,
            -21.449,
            -21.388,
            -21.336,
            -21.292,
        ],  # 1.8
        [
            -24.519,
            -23.637,
            -23.039,
            -22.606,
            -22.281,
            -22.030,
            -21.834,
            -21.678,
            -21.552,
            -21.450,
            -21.365,
            -21.295,
            -21.236,
            -21.185,
            -21.142,
        ],  # 1.9
        [
            -24.086,
            -23.222,
            -22.650,
            -22.246,
            -21.948,
            -21.722,
            -21.546,
            -21.407,
            -21.296,
            -21.205,
            -21.131,
            -21.070,
            -21.018,
            -20.974,
            -20.937,
        ],  # 2.0
        # Energy 2.1-3.0 eV (rows 20-29)
        [
            -23.850,
            -23.018,
            -22.472,
            -22.088,
            -21.805,
            -21.590,
            -21.422,
            -21.289,
            -21.182,
            -21.095,
            -21.024,
            -20.964,
            -20.914,
            -20.872,
            -20.835,
        ],  # 2.1
        [
            -23.136,
            -22.445,
            -21.994,
            -21.676,
            -21.440,
            -21.259,
            -21.117,
            -21.004,
            -20.912,
            -20.837,
            -20.775,
            -20.723,
            -20.679,
            -20.642,
            -20.611,
        ],  # 2.2
        [
            -23.199,
            -22.433,
            -21.927,
            -21.573,
            -21.314,
            -21.119,
            -20.969,
            -20.851,
            -20.758,
            -20.682,
            -20.621,
            -20.571,
            -20.529,
            -20.493,
            -20.463,
        ],  # 2.3
        [
            -22.696,
            -22.020,
            -21.585,
            -21.286,
            -21.071,
            -20.912,
            -20.791,
            -20.697,
            -20.622,
            -20.563,
            -20.514,
            -20.475,
            -20.442,
            -20.414,
            -20.391,
        ],  # 2.4
        [
            -22.119,
            -21.557,
            -21.194,
            -20.943,
            -20.761,
            -20.624,
            -20.518,
            -20.434,
            -20.367,
            -20.313,
            -20.268,
            -20.231,
            -20.201,
            -20.175,
            -20.153,
        ],  # 2.5
        [
            -21.855,
            -21.300,
            -20.931,
            -20.673,
            -20.485,
            -20.344,
            -20.235,
            -20.151,
            -20.084,
            -20.031,
            -19.988,
            -19.953,
            -19.924,
            -19.900,
            -19.880,
        ],  # 2.6
        [
            -21.126,
            -20.673,
            -20.382,
            -20.184,
            -20.044,
            -19.943,
            -19.868,
            -19.811,
            -19.769,
            -19.736,
            -19.710,
            -19.690,
            -19.674,
            -19.662,
            -19.652,
        ],  # 2.7
        [
            -20.502,
            -20.150,
            -19.922,
            -19.766,
            -19.657,
            -19.578,
            -19.520,
            -19.478,
            -19.446,
            -19.422,
            -19.404,
            -19.390,
            -19.379,
            -19.371,
            -19.365,
        ],  # 2.8
        [
            -20.030,
            -19.724,
            -19.530,
            -19.399,
            -19.309,
            -19.245,
            -19.199,
            -19.166,
            -19.142,
            -19.125,
            -19.112,
            -19.103,
            -19.096,
            -19.091,
            -19.088,
        ],  # 2.9
        [
            -19.640,
            -19.364,
            -19.189,
            -19.074,
            -18.996,
            -18.943,
            -18.906,
            -18.881,
            -18.863,
            -18.852,
            -18.844,
            -18.839,
            -18.837,
            -18.836,
            -18.836,
        ],  # 3.0
        # Energy 3.1-4.0 eV (rows 30-39)
        [
            -19.333,
            -19.092,
            -18.939,
            -18.838,
            -18.770,
            -18.725,
            -18.695,
            -18.675,
            -18.662,
            -18.655,
            -18.651,
            -18.649,
            -18.649,
            -18.651,
            -18.653,
        ],  # 3.1
        [
            -19.070,
            -18.880,
            -18.756,
            -18.674,
            -18.621,
            -18.585,
            -18.562,
            -18.548,
            -18.540,
            -18.536,
            -18.536,
            -18.537,
            -18.539,
            -18.542,
            -18.546,
        ],  # 3.2
        [
            -18.851,
            -18.708,
            -18.617,
            -18.558,
            -18.521,
            -18.498,
            -18.484,
            -18.477,
            -18.475,
            -18.476,
            -18.478,
            -18.482,
            -18.487,
            -18.493,
            -18.498,
        ],  # 3.3
        [
            -18.709,
            -18.599,
            -18.533,
            -18.494,
            -18.471,
            -18.459,
            -18.454,
            -18.454,
            -18.457,
            -18.462,
            -18.469,
            -18.476,
            -18.483,
            -18.490,
            -18.498,
        ],  # 3.4
        [
            -18.656,
            -18.572,
            -18.524,
            -18.497,
            -18.485,
            -18.480,
            -18.482,
            -18.486,
            -18.493,
            -18.501,
            -18.510,
            -18.519,
            -18.527,
            -18.536,
            -18.544,
        ],  # 3.5
        [
            -18.670,
            -18.613,
            -18.582,
            -18.566,
            -18.561,
            -18.562,
            -18.568,
            -18.575,
            -18.583,
            -18.592,
            -18.601,
            -18.610,
            -18.619,
            -18.627,
            -18.635,
        ],  # 3.6
        [
            -18.728,
            -18.700,
            -18.687,
            -18.683,
            -18.685,
            -18.691,
            -18.698,
            -18.706,
            -18.715,
            -18.723,
            -18.731,
            -18.739,
            -18.745,
            -18.752,
            -18.758,
        ],  # 3.7
        [
            -18.839,
            -18.835,
            -18.836,
            -18.842,
            -18.849,
            -18.857,
            -18.865,
            -18.872,
            -18.878,
            -18.883,
            -18.888,
            -18.892,
            -18.895,
            -18.898,
            -18.900,
        ],  # 3.8
        [
            -19.034,
            -19.041,
            -19.049,
            -19.057,
            -19.064,
            -19.069,
            -19.071,
            -19.071,
            -19.070,
            -19.068,
            -19.065,
            -19.061,
            -19.058,
            -19.054,
            -19.051,
        ],  # 3.9
        [
            -19.372,
            -19.378,
            -19.382,
            -19.380,
            -19.372,
            -19.359,
            -19.341,
            -19.321,
            -19.300,
            -19.280,
            -19.261,
            -19.243,
            -19.227,
            -19.212,
            -19.199,
        ],  # 4.0
        # Energy 4.1-5.0 eV (rows 40-49)
        [
            -19.780,
            -19.777,
            -19.763,
            -19.732,
            -19.686,
            -19.631,
            -19.573,
            -19.517,
            -19.465,
            -19.419,
            -19.379,
            -19.344,
            -19.314,
            -19.288,
            -19.265,
        ],  # 4.1
        [
            -20.151,
            -20.133,
            -20.087,
            -20.009,
            -19.911,
            -19.810,
            -19.715,
            -19.631,
            -19.559,
            -19.497,
            -19.446,
            -19.402,
            -19.365,
            -19.333,
            -19.306,
        ],  # 4.2
        [
            -20.525,
            -20.454,
            -20.312,
            -20.138,
            -19.970,
            -19.825,
            -19.705,
            -19.607,
            -19.528,
            -19.464,
            -19.411,
            -19.367,
            -19.330,
            -19.300,
            -19.274,
        ],  # 4.3
        [
            -20.869,
            -20.655,
            -20.366,
            -20.104,
            -19.894,
            -19.731,
            -19.604,
            -19.505,
            -19.426,
            -19.363,
            -19.312,
            -19.271,
            -19.236,
            -19.208,
            -19.184,
        ],  # 4.4
        [
            -21.179,
            -20.768,
            -20.380,
            -20.081,
            -19.856,
            -19.686,
            -19.556,
            -19.454,
            -19.375,
            -19.311,
            -19.260,
            -19.218,
            -19.184,
            -19.155,
            -19.131,
        ],  # 4.5
        [
            -21.167,
            -20.601,
            -20.206,
            -19.925,
            -19.719,
            -19.565,
            -19.447,
            -19.355,
            -19.283,
            -19.226,
            -19.180,
            -19.143,
            -19.112,
            -19.087,
            -19.066,
        ],  # 4.6
        [
            -20.918,
            -20.348,
            -19.976,
            -19.720,
            -19.536,
            -19.401,
            -19.299,
            -19.220,
            -19.159,
            -19.112,
            -19.073,
            -19.043,
            -19.018,
            -18.998,
            -18.981,
        ],  # 4.7
        [
            -20.753,
            -20.204,
            -19.847,
            -19.602,
            -19.427,
            -19.299,
            -19.203,
            -19.129,
            -19.072,
            -19.028,
            -18.993,
            -18.965,
            -18.942,
            -18.924,
            -18.909,
        ],  # 4.8
        [
            -20.456,
            -19.987,
            -19.677,
            -19.460,
            -19.302,
            -19.186,
            -19.098,
            -19.030,
            -18.978,
            -18.937,
            -18.904,
            -18.878,
            -18.857,
            -18.841,
            -18.827,
        ],  # 4.9
        [
            -20.154,
            -19.734,
            -19.461,
            -19.272,
            -19.136,
            -19.035,
            -18.960,
            -18.902,
            -18.858,
            -18.824,
            -18.797,
            -18.775,
            -18.759,
            -18.745,
            -18.735,
        ],  # 5.0
        # Energy 5.1-6.0 eV (rows 50-59)
        [
            -19.941,
            -19.544,
            -19.288,
            -19.114,
            -18.992,
            -18.903,
            -18.837,
            -18.788,
            -18.751,
            -18.723,
            -18.701,
            -18.684,
            -18.671,
            -18.661,
            -18.654,
        ],  # 5.1
        [
            -19.657,
            -19.321,
            -19.104,
            -18.956,
            -18.853,
            -18.779,
            -18.724,
            -18.684,
            -18.655,
            -18.632,
            -18.615,
            -18.602,
            -18.592,
            -18.585,
            -18.579,
        ],  # 5.2
        [
            -19.388,
            -19.109,
            -18.930,
            -18.810,
            -18.725,
            -18.664,
            -18.620,
            -18.586,
            -18.562,
            -18.543,
            -18.529,
            -18.518,
            -18.510,
            -18.503,
            -18.498,
        ],  # 5.3
        [
            -19.201,
            -18.953,
            -18.794,
            -18.686,
            -18.611,
            -18.556,
            -18.515,
            -18.485,
            -18.462,
            -18.446,
            -18.433,
            -18.423,
            -18.416,
            -18.410,
            -18.406,
        ],  # 5.4
        [
            -18.923,
            -18.719,
            -18.588,
            -18.500,
            -18.439,
            -18.396,
            -18.365,
            -18.344,
            -18.328,
            -18.318,
            -18.311,
            -18.307,
            -18.304,
            -18.303,
            -18.302,
        ],  # 5.5
        [
            -18.614,
            -18.458,
            -18.361,
            -18.298,
            -18.258,
            -18.232,
            -18.216,
            -18.206,
            -18.202,
            -18.201,
            -18.202,
            -18.205,
            -18.208,
            -18.213,
            -18.218,
        ],  # 5.6
        [
            -18.419,
            -18.295,
            -18.222,
            -18.178,
            -18.153,
            -18.139,
            -18.132,
            -18.131,
            -18.133,
            -18.138,
            -18.143,
            -18.150,
            -18.157,
            -18.164,
            -18.172,
        ],  # 5.7
        [
            -18.296,
            -18.201,
            -18.148,
            -18.118,
            -18.101,
            -18.094,
            -18.091,
            -18.093,
            -18.096,
            -18.101,
            -18.107,
            -18.113,
            -18.120,
            -18.126,
            -18.132,
        ],  # 5.8
        [
            -18.021,
            -17.992,
            -17.977,
            -17.970,
            -17.967,
            -17.968,
            -17.970,
            -17.974,
            -17.978,
            -17.983,
            -17.989,
            -17.994,
            -18.000,
            -18.005,
            -18.011,
        ],  # 5.9
        [
            -17.694,
            -17.686,
            -17.686,
            -17.691,
            -17.698,
            -17.708,
            -17.718,
            -17.729,
            -17.740,
            -17.750,
            -17.761,
            -17.771,
            -17.781,
            -17.790,
            -17.798,
        ],  # 6.0
        # Energy 6.1-7.0 eV (rows 60-69)
        [
            -17.374,
            -17.384,
            -17.400,
            -17.420,
            -17.440,
            -17.462,
            -17.483,
            -17.503,
            -17.523,
            -17.541,
            -17.558,
            -17.575,
            -17.590,
            -17.603,
            -17.616,
        ],  # 6.1
        [
            -17.169,
            -17.199,
            -17.230,
            -17.262,
            -17.293,
            -17.323,
            -17.351,
            -17.378,
            -17.404,
            -17.427,
            -17.449,
            -17.469,
            -17.488,
            -17.505,
            -17.520,
        ],  # 6.2
        [
            -17.151,
            -17.184,
            -17.217,
            -17.250,
            -17.282,
            -17.313,
            -17.342,
            -17.369,
            -17.395,
            -17.418,
            -17.440,
            -17.461,
            -17.480,
            -17.497,
            -17.513,
        ],  # 6.3
        [
            -17.230,
            -17.260,
            -17.290,
            -17.320,
            -17.348,
            -17.375,
            -17.401,
            -17.425,
            -17.448,
            -17.469,
            -17.489,
            -17.508,
            -17.525,
            -17.541,
            -17.556,
        ],  # 6.4
        [
            -17.379,
            -17.403,
            -17.425,
            -17.446,
            -17.467,
            -17.486,
            -17.505,
            -17.524,
            -17.541,
            -17.558,
            -17.574,
            -17.588,
            -17.602,
            -17.615,
            -17.627,
        ],  # 6.5
        [
            -17.596,
            -17.604,
            -17.609,
            -17.612,
            -17.616,
            -17.622,
            -17.628,
            -17.636,
            -17.644,
            -17.652,
            -17.661,
            -17.670,
            -17.679,
            -17.687,
            -17.695,
        ],  # 6.6
        [
            -17.846,
            -17.823,
            -17.795,
            -17.770,
            -17.750,
            -17.735,
            -17.725,
            -17.719,
            -17.716,
            -17.715,
            -17.716,
            -17.719,
            -17.722,
            -17.726,
            -17.730,
        ],  # 6.7
        [
            -18.089,
            -18.015,
            -17.942,
            -17.882,
            -17.836,
            -17.802,
            -17.777,
            -17.760,
            -17.748,
            -17.740,
            -17.736,
            -17.734,
            -17.733,
            -17.734,
            -17.736,
        ],  # 6.8
        [
            -18.299,
            -18.156,
            -18.038,
            -17.947,
            -17.881,
            -17.833,
            -17.798,
            -17.774,
            -17.757,
            -17.745,
            -17.738,
            -17.733,
            -17.730,
            -17.729,
            -17.729,
        ],  # 6.9
        [
            -18.441,
            -18.243,
            -18.096,
            -17.991,
            -17.915,
            -17.860,
            -17.821,
            -17.792,
            -17.772,
            -17.757,
            -17.746,
            -17.738,
            -17.733,
            -17.730,
            -17.728,
        ],  # 7.0
        # Energy 7.1-8.0 eV (rows 70-79)
        [
            -18.474,
            -18.262,
            -18.111,
            -18.004,
            -17.926,
            -17.869,
            -17.826,
            -17.795,
            -17.771,
            -17.753,
            -17.740,
            -17.730,
            -17.722,
            -17.717,
            -17.713,
        ],  # 7.1
        [
            -18.387,
            -18.191,
            -18.053,
            -17.952,
            -17.878,
            -17.823,
            -17.782,
            -17.752,
            -17.729,
            -17.711,
            -17.698,
            -17.689,
            -17.681,
            -17.676,
            -17.672,
        ],  # 7.2
        [
            -18.161,
            -17.990,
            -17.874,
            -17.793,
            -17.736,
            -17.696,
            -17.668,
            -17.648,
            -17.634,
            -17.625,
            -17.619,
            -17.616,
            -17.614,
            -17.614,
            -17.615,
        ],  # 7.3
        [
            -17.908,
            -17.774,
            -17.690,
            -17.637,
            -17.604,
            -17.583,
            -17.572,
            -17.567,
            -17.566,
            -17.568,
            -17.571,
            -17.576,
            -17.581,
            -17.587,
            -17.593,
        ],  # 7.4
        [
            -17.681,
            -17.589,
            -17.540,
            -17.515,
            -17.506,
            -17.505,
            -17.511,
            -17.520,
            -17.530,
            -17.542,
            -17.554,
            -17.566,
            -17.578,
            -17.589,
            -17.600,
        ],  # 7.5
        [
            -17.647,
            -17.606,
            -17.584,
            -17.575,
            -17.573,
            -17.576,
            -17.582,
            -17.589,
            -17.597,
            -17.605,
            -17.614,
            -17.623,
            -17.631,
            -17.639,
            -17.646,
        ],  # 7.6
        [
            -17.300,
            -17.291,
            -17.291,
            -17.297,
            -17.307,
            -17.319,
            -17.333,
            -17.347,
            -17.361,
            -17.375,
            -17.389,
            -17.402,
            -17.415,
            -17.427,
            -17.438,
        ],  # 7.7
        [
            -16.786,
            -16.802,
            -16.825,
            -16.853,
            -16.883,
            -16.914,
            -16.944,
            -16.974,
            -17.003,
            -17.030,
            -17.055,
            -17.079,
            -17.101,
            -17.122,
            -17.141,
        ],  # 7.8
        [
            -16.489,
            -16.533,
            -16.579,
            -16.625,
            -16.670,
            -16.713,
            -16.754,
            -16.793,
            -16.830,
            -16.864,
            -16.896,
            -16.925,
            -16.952,
            -16.977,
            -17.000,
        ],  # 7.9
        [
            -16.694,
            -16.724,
            -16.756,
            -16.789,
            -16.823,
            -16.856,
            -16.888,
            -16.919,
            -16.949,
            -16.976,
            -17.002,
            -17.026,
            -17.048,
            -17.069,
            -17.088,
        ],  # 8.0
        # Energy 8.1-9.0 eV (rows 80-89)
        [
            -16.935,
            -16.951,
            -16.971,
            -16.993,
            -17.016,
            -17.040,
            -17.064,
            -17.088,
            -17.111,
            -17.132,
            -17.153,
            -17.172,
            -17.190,
            -17.206,
            -17.222,
        ],  # 8.1
        [
            -17.200,
            -17.208,
            -17.220,
            -17.235,
            -17.251,
            -17.269,
            -17.286,
            -17.304,
            -17.322,
            -17.338,
            -17.354,
            -17.369,
            -17.384,
            -17.397,
            -17.409,
        ],  # 8.2
        [
            -17.597,
            -17.591,
            -17.589,
            -17.590,
            -17.594,
            -17.600,
            -17.608,
            -17.617,
            -17.626,
            -17.635,
            -17.645,
            -17.654,
            -17.662,
            -17.671,
            -17.679,
        ],  # 8.3
        [
            -18.166,
            -18.134,
            -18.107,
            -18.085,
            -18.068,
            -18.056,
            -18.047,
            -18.041,
            -18.038,
            -18.036,
            -18.035,
            -18.035,
            -18.036,
            -18.038,
            -18.039,
        ],  # 8.4
        [
            -19.000,
            -18.917,
            -18.838,
            -18.770,
            -18.714,
            -18.669,
            -18.632,
            -18.603,
            -18.579,
            -18.560,
            -18.545,
            -18.532,
            -18.522,
            -18.514,
            -18.507,
        ],  # 8.5
        [
            -20.313,
            -19.982,
            -19.754,
            -19.592,
            -19.472,
            -19.380,
            -19.309,
            -19.253,
            -19.208,
            -19.172,
            -19.143,
            -19.119,
            -19.099,
            -19.083,
            -19.069,
        ],  # 8.6
        [
            -19.751,
            -19.611,
            -19.520,
            -19.461,
            -19.423,
            -19.398,
            -19.382,
            -19.372,
            -19.366,
            -19.364,
            -19.363,
            -19.364,
            -19.366,
            -19.368,
            -19.371,
        ],  # 8.7
        [
            -19.581,
            -19.431,
            -19.337,
            -19.277,
            -19.240,
            -19.218,
            -19.207,
            -19.202,
            -19.203,
            -19.207,
            -19.212,
            -19.220,
            -19.228,
            -19.236,
            -19.245,
        ],  # 8.8
        [
            -19.685,
            -19.506,
            -19.389,
            -19.311,
            -19.258,
            -19.222,
            -19.199,
            -19.184,
            -19.175,
            -19.170,
            -19.168,
            -19.169,
            -19.171,
            -19.174,
            -19.177,
        ],  # 8.9
        [
            -19.977,
            -19.756,
            -19.606,
            -19.501,
            -19.425,
            -19.370,
            -19.330,
            -19.300,
            -19.278,
            -19.262,
            -19.250,
            -19.241,
            -19.235,
            -19.230,
            -19.227,
        ],  # 9.0
        # Energy 9.1-10.0 eV (rows 90-99)
        [
            -20.445,
            -20.158,
            -19.958,
            -19.815,
            -19.711,
            -19.633,
            -19.574,
            -19.528,
            -19.493,
            -19.465,
            -19.442,
            -19.425,
            -19.410,
            -19.398,
            -19.389,
        ],  # 9.1
        [
            -20.980,
            -20.625,
            -20.391,
            -20.229,
            -20.110,
            -20.020,
            -19.949,
            -19.892,
            -19.846,
            -19.807,
            -19.775,
            -19.748,
            -19.724,
            -19.704,
            -19.687,
        ],  # 9.2
        [
            -21.404,
            -21.023,
            -20.771,
            -20.594,
            -20.461,
            -20.358,
            -20.274,
            -20.205,
            -20.148,
            -20.099,
            -20.058,
            -20.022,
            -19.991,
            -19.965,
            -19.942,
        ],  # 9.3
        [
            -21.309,
            -20.970,
            -20.753,
            -20.603,
            -20.495,
            -20.412,
            -20.348,
            -20.295,
            -20.252,
            -20.215,
            -20.185,
            -20.158,
            -20.135,
            -20.115,
            -20.098,
        ],  # 9.4
        [
            -21.221,
            -20.906,
            -20.707,
            -20.574,
            -20.480,
            -20.412,
            -20.361,
            -20.322,
            -20.292,
            -20.268,
            -20.249,
            -20.233,
            -20.221,
            -20.210,
            -20.201,
        ],  # 9.5
        [
            -21.441,
            -21.097,
            -20.878,
            -20.728,
            -20.623,
            -20.546,
            -20.489,
            -20.446,
            -20.413,
            -20.387,
            -20.368,
            -20.352,
            -20.340,
            -20.330,
            -20.322,
        ],  # 9.6
        [
            -21.668,
            -21.305,
            -21.071,
            -20.911,
            -20.797,
            -20.713,
            -20.650,
            -20.602,
            -20.565,
            -20.536,
            -20.514,
            -20.496,
            -20.481,
            -20.470,
            -20.460,
        ],  # 9.7
        [
            -21.926,
            -21.556,
            -21.316,
            -21.150,
            -21.031,
            -20.942,
            -20.874,
            -20.822,
            -20.782,
            -20.750,
            -20.724,
            -20.704,
            -20.687,
            -20.674,
            -20.663,
        ],  # 9.8
        [
            -22.319,
            -21.937,
            -21.686,
            -21.510,
            -21.380,
            -21.282,
            -21.206,
            -21.147,
            -21.099,
            -21.061,
            -21.031,
            -21.006,
            -20.985,
            -20.968,
            -20.954,
        ],  # 9.9
        [
            -22.969,
            -22.561,
            -22.288,
            -22.092,
            -21.945,
            -21.832,
            -21.743,
            -21.672,
            -21.616,
            -21.570,
            -21.533,
            -21.503,
            -21.477,
            -21.457,
            -21.439,
        ],  # 10.0
        # Energy 10.1-10.5 eV (rows 100-104)
        [
            -24.001,
            -23.527,
            -23.199,
            -22.957,
            -22.772,
            -22.629,
            -22.516,
            -22.427,
            -22.355,
            -22.297,
            -22.250,
            -22.212,
            -22.180,
            -22.153,
            -22.131,
        ],  # 10.1
        [
            -24.233,
            -23.774,
            -23.477,
            -23.273,
            -23.128,
            -23.022,
            -22.943,
            -22.883,
            -22.837,
            -22.802,
            -22.774,
            -22.752,
            -22.735,
            -22.721,
            -22.710,
        ],  # 10.2
        [
            -24.550,
            -23.913,
            -23.521,
            -23.266,
            -23.094,
            -22.976,
            -22.893,
            -22.836,
            -22.796,
            -22.768,
            -22.750,
            -22.737,
            -22.730,
            -22.726,
            -22.725,
        ],  # 10.3
        [
            -24.301,
            -23.665,
            -23.274,
            -23.019,
            -22.848,
            -22.730,
            -22.648,
            -22.591,
            -22.552,
            -22.525,
            -22.507,
            -22.495,
            -22.489,
            -22.485,
            -22.485,
        ],  # 10.4
        [
            -24.519,
            -23.883,
            -23.491,
            -23.237,
            -23.065,
            -22.948,
            -22.866,
            -22.809,
            -22.770,
            -22.743,
            -22.724,
            -22.713,
            -22.706,
            -22.703,
            -22.702,
        ],  # 10.5
    ],
    dtype=np.float64,
)


def _chop_opacity(freq: float, temp: np.ndarray) -> np.ndarray:
    """CH molecular opacity (atlas7v.for lines 8120-8384).

    Returns cross-section * partition function for each layer.
    Only active for T < 9000K and energy 2.0-10.5 eV.

    Parameters
    ----------
    freq : float
        Frequency (Hz)
    temp : np.ndarray
        Temperature array (K), shape (n_layers,)

    Returns
    -------
    np.ndarray
        CHOP values for each layer (cm²), shape (n_layers,)
    """
    n_layers = temp.size
    result = np.zeros(n_layers, dtype=np.float64)

    # Convert frequency to energy in eV
    waveno = freq / 2.99792458e10  # cm^-1
    evolt = waveno / 8065.479  # eV

    # Energy index (0.1 eV bins starting at 0)
    n = int(evolt * 10)
    if n < 20 or n >= 105:  # Energy range 2.0-10.5 eV
        return result

    en = float(n) * 0.1

    # Interpolate cross-section in energy (index is n-2 for array starting at 0.2 eV)
    # Fortran data starts at 0.1 eV but first real values are at 0.2 eV
    idx = n - 2  # Adjust for 0-based indexing and 0.2 eV start
    if idx < 0 or idx >= 104:
        return result

    # Cross-section at each temperature (interpolated in energy)
    crosscht = np.zeros(15, dtype=np.float64)
    for it in range(15):
        crosscht[it] = (
            _CH_CROSSSECT[idx, it]
            + (_CH_CROSSSECT[idx + 1, it] - _CH_CROSSSECT[idx, it]) * (evolt - en) / 0.1
        )

    # For each layer, interpolate in temperature
    for j in range(n_layers):
        t_j = temp[j]
        if t_j >= 9000.0:
            continue

        # Partition function interpolation (200K grid starting at 1000K)
        it_part = int((t_j - 1000.0) / 200.0)
        it_part = max(0, min(it_part, 39))
        tn_part = float(it_part) * 200.0 + 1000.0
        part = (
            _CH_PARTITION[it_part]
            + (_CH_PARTITION[it_part + 1] - _CH_PARTITION[it_part])
            * (t_j - tn_part)
            / 200.0
        )

        # Cross-section interpolation (500K grid starting at 2000K)
        it_cross = int((t_j - 2000.0) / 500.0)
        it_cross = max(0, min(it_cross, 13))
        tn_cross = float(it_cross) * 500.0 + 2000.0

        log_xsect = (
            crosscht[it_cross]
            + (crosscht[it_cross + 1] - crosscht[it_cross]) * (t_j - tn_cross) / 500.0
        )

        # Convert from log10 to linear
        result[j] = np.exp(log_xsect * 2.30258509299405) * part

    return result


# OH cross-section table (atlas7v.for lines 8405-8664)
# Shape: (130, 15) - 130 energy bins (2.1-15.0 eV), 15 temperature points
# Temperature grid: 2000K to 9000K in 500K steps (15 points)
# Stored as log10(cross-section * partition_function)
_OH_CROSSSECT = np.array(
    [
        # Energy 2.1-3.0 eV (rows 0-9)
        [
            -30.855,
            -29.121,
            -27.976,
            -27.166,
            -26.566,
            -26.106,
            -25.742,
            -25.448,
            -25.207,
            -25.006,
            -24.836,
            -24.691,
            -24.566,
            -24.457,
            -24.363,
        ],
        [
            -30.494,
            -28.760,
            -27.615,
            -26.806,
            -26.206,
            -25.745,
            -25.381,
            -25.088,
            -24.846,
            -24.645,
            -24.475,
            -24.330,
            -24.205,
            -24.097,
            -24.002,
        ],
        [
            -30.157,
            -28.425,
            -27.280,
            -26.472,
            -25.872,
            -25.411,
            -25.048,
            -24.754,
            -24.513,
            -24.312,
            -24.142,
            -23.997,
            -23.872,
            -23.764,
            -23.669,
        ],
        [
            -29.848,
            -28.117,
            -26.974,
            -26.165,
            -25.566,
            -25.105,
            -24.742,
            -24.448,
            -24.207,
            -24.006,
            -23.836,
            -23.692,
            -23.567,
            -23.458,
            -23.364,
        ],
        [
            -29.567,
            -27.837,
            -26.693,
            -25.885,
            -25.286,
            -24.826,
            -24.462,
            -24.169,
            -23.928,
            -23.727,
            -23.557,
            -23.412,
            -23.287,
            -23.179,
            -23.084,
        ],
        [
            -29.307,
            -27.578,
            -26.436,
            -25.628,
            -25.029,
            -24.569,
            -24.205,
            -23.912,
            -23.671,
            -23.470,
            -23.300,
            -23.155,
            -23.031,
            -22.922,
            -22.828,
        ],
        [
            -29.068,
            -27.341,
            -26.199,
            -25.391,
            -24.792,
            -24.332,
            -23.969,
            -23.676,
            -23.435,
            -23.234,
            -23.064,
            -22.920,
            -22.795,
            -22.687,
            -22.592,
        ],
        [
            -28.820,
            -27.115,
            -25.978,
            -25.172,
            -24.574,
            -24.115,
            -23.752,
            -23.459,
            -23.218,
            -23.017,
            -22.848,
            -22.703,
            -22.579,
            -22.470,
            -22.376,
        ],
        [
            -28.540,
            -26.891,
            -25.768,
            -24.968,
            -24.372,
            -23.914,
            -23.552,
            -23.259,
            -23.019,
            -22.818,
            -22.649,
            -22.504,
            -22.380,
            -22.272,
            -22.177,
        ],
        [
            -28.275,
            -26.681,
            -25.574,
            -24.779,
            -24.186,
            -23.729,
            -23.368,
            -23.076,
            -22.836,
            -22.636,
            -22.467,
            -22.322,
            -22.198,
            -22.090,
            -21.996,
        ],
        # Energy 3.1-4.0 eV (rows 10-19)
        [
            -27.993,
            -26.470,
            -25.388,
            -24.602,
            -24.014,
            -23.560,
            -23.200,
            -22.909,
            -22.669,
            -22.470,
            -22.301,
            -22.157,
            -22.033,
            -21.925,
            -21.831,
        ],
        [
            -27.698,
            -26.252,
            -25.204,
            -24.433,
            -23.851,
            -23.401,
            -23.043,
            -22.754,
            -22.515,
            -22.316,
            -22.148,
            -22.005,
            -21.881,
            -21.773,
            -21.679,
        ],
        [
            -27.398,
            -26.026,
            -25.019,
            -24.267,
            -23.696,
            -23.251,
            -22.896,
            -22.609,
            -22.372,
            -22.174,
            -22.007,
            -21.864,
            -21.741,
            -21.634,
            -21.540,
        ],
        [
            -27.100,
            -25.791,
            -24.828,
            -24.102,
            -23.543,
            -23.106,
            -22.756,
            -22.472,
            -22.238,
            -22.041,
            -21.875,
            -21.733,
            -21.611,
            -21.504,
            -21.411,
        ],
        [
            -26.807,
            -25.549,
            -24.631,
            -23.933,
            -23.391,
            -22.964,
            -22.621,
            -22.341,
            -22.109,
            -21.915,
            -21.751,
            -21.610,
            -21.488,
            -21.383,
            -21.290,
        ],
        [
            -26.531,
            -25.310,
            -24.431,
            -23.761,
            -23.238,
            -22.823,
            -22.488,
            -22.214,
            -21.986,
            -21.795,
            -21.633,
            -21.494,
            -21.374,
            -21.269,
            -21.178,
        ],
        [
            -26.239,
            -25.066,
            -24.225,
            -23.585,
            -23.082,
            -22.681,
            -22.356,
            -22.089,
            -21.866,
            -21.679,
            -21.520,
            -21.383,
            -21.265,
            -21.162,
            -21.072,
        ],
        [
            -25.945,
            -24.824,
            -24.017,
            -23.405,
            -22.923,
            -22.538,
            -22.223,
            -21.964,
            -21.748,
            -21.565,
            -21.410,
            -21.276,
            -21.160,
            -21.059,
            -20.970,
        ],
        [
            -25.663,
            -24.587,
            -23.810,
            -23.222,
            -22.761,
            -22.391,
            -22.088,
            -21.838,
            -21.629,
            -21.452,
            -21.300,
            -21.170,
            -21.057,
            -20.958,
            -20.872,
        ],
        [
            -25.372,
            -24.350,
            -23.603,
            -23.038,
            -22.596,
            -22.241,
            -21.950,
            -21.710,
            -21.508,
            -21.337,
            -21.190,
            -21.064,
            -20.954,
            -20.858,
            -20.774,
        ],
        # Energy 4.1-5.0 eV (rows 20-29)
        [
            -25.076,
            -24.111,
            -23.396,
            -22.853,
            -22.429,
            -22.088,
            -21.809,
            -21.578,
            -21.384,
            -21.220,
            -21.078,
            -20.957,
            -20.851,
            -20.758,
            -20.676,
        ],
        [
            -24.779,
            -23.870,
            -23.189,
            -22.669,
            -22.261,
            -21.934,
            -21.667,
            -21.445,
            -21.259,
            -21.101,
            -20.965,
            -20.848,
            -20.746,
            -20.656,
            -20.578,
        ],
        [
            -24.486,
            -23.629,
            -22.983,
            -22.486,
            -22.095,
            -21.781,
            -21.524,
            -21.311,
            -21.132,
            -20.980,
            -20.850,
            -20.737,
            -20.639,
            -20.553,
            -20.478,
        ],
        [
            -24.183,
            -23.382,
            -22.774,
            -22.302,
            -21.928,
            -21.627,
            -21.381,
            -21.177,
            -21.005,
            -20.859,
            -20.734,
            -20.625,
            -20.531,
            -20.449,
            -20.376,
        ],
        [
            -23.867,
            -23.127,
            -22.561,
            -22.116,
            -21.761,
            -21.474,
            -21.238,
            -21.043,
            -20.878,
            -20.738,
            -20.617,
            -20.513,
            -20.423,
            -20.344,
            -20.274,
        ],
        [
            -23.538,
            -22.862,
            -22.340,
            -21.926,
            -21.592,
            -21.320,
            -21.096,
            -20.909,
            -20.751,
            -20.617,
            -20.502,
            -20.402,
            -20.315,
            -20.239,
            -20.172,
        ],
        [
            -23.234,
            -22.604,
            -22.120,
            -21.734,
            -21.422,
            -21.166,
            -20.953,
            -20.776,
            -20.625,
            -20.497,
            -20.387,
            -20.291,
            -20.208,
            -20.135,
            -20.071,
        ],
        [
            -22.934,
            -22.347,
            -21.898,
            -21.541,
            -21.250,
            -21.010,
            -20.811,
            -20.643,
            -20.500,
            -20.378,
            -20.273,
            -20.182,
            -20.102,
            -20.033,
            -19.971,
        ],
        [
            -22.637,
            -22.092,
            -21.676,
            -21.345,
            -21.075,
            -20.853,
            -20.666,
            -20.508,
            -20.374,
            -20.259,
            -20.159,
            -20.073,
            -19.997,
            -19.931,
            -19.872,
        ],
        [
            -22.337,
            -21.835,
            -21.452,
            -21.147,
            -20.899,
            -20.693,
            -20.520,
            -20.373,
            -20.247,
            -20.139,
            -20.046,
            -19.964,
            -19.892,
            -19.830,
            -19.774,
        ],
        # Energy 5.1-6.0 eV (rows 30-39)
        [
            -22.049,
            -21.584,
            -21.230,
            -20.950,
            -20.721,
            -20.531,
            -20.372,
            -20.236,
            -20.119,
            -20.019,
            -19.931,
            -19.855,
            -19.788,
            -19.729,
            -19.676,
        ],
        [
            -21.768,
            -21.337,
            -21.011,
            -20.754,
            -20.544,
            -20.370,
            -20.223,
            -20.098,
            -19.991,
            -19.898,
            -19.817,
            -19.746,
            -19.683,
            -19.628,
            -19.579,
        ],
        [
            -21.494,
            -21.096,
            -20.796,
            -20.559,
            -20.367,
            -20.208,
            -20.074,
            -19.960,
            -19.861,
            -19.776,
            -19.701,
            -19.636,
            -19.578,
            -19.527,
            -19.482,
        ],
        [
            -21.233,
            -20.861,
            -20.585,
            -20.368,
            -20.193,
            -20.048,
            -19.926,
            -19.821,
            -19.732,
            -19.654,
            -19.586,
            -19.526,
            -19.473,
            -19.426,
            -19.384,
        ],
        [
            -20.983,
            -20.635,
            -20.380,
            -20.181,
            -20.021,
            -19.889,
            -19.778,
            -19.683,
            -19.602,
            -19.531,
            -19.469,
            -19.415,
            -19.367,
            -19.324,
            -19.286,
        ],
        [
            -20.743,
            -20.418,
            -20.182,
            -19.999,
            -19.853,
            -19.733,
            -19.633,
            -19.547,
            -19.474,
            -19.410,
            -19.354,
            -19.305,
            -19.261,
            -19.223,
            -19.189,
        ],
        [
            -20.515,
            -20.210,
            -19.991,
            -19.824,
            -19.690,
            -19.581,
            -19.490,
            -19.413,
            -19.347,
            -19.290,
            -19.240,
            -19.196,
            -19.157,
            -19.122,
            -19.092,
        ],
        [
            -20.297,
            -20.011,
            -19.808,
            -19.654,
            -19.532,
            -19.434,
            -19.352,
            -19.282,
            -19.223,
            -19.172,
            -19.127,
            -19.088,
            -19.054,
            -19.023,
            -18.996,
        ],
        [
            -20.090,
            -19.822,
            -19.633,
            -19.491,
            -19.381,
            -19.291,
            -19.218,
            -19.156,
            -19.103,
            -19.057,
            -19.018,
            -18.983,
            -18.952,
            -18.925,
            -18.901,
        ],
        [
            -19.893,
            -19.642,
            -19.467,
            -19.337,
            -19.236,
            -19.155,
            -19.089,
            -19.034,
            -18.987,
            -18.946,
            -18.912,
            -18.881,
            -18.854,
            -18.831,
            -18.810,
        ],
        # Energy 6.1-7.0 eV (rows 40-49)
        [
            -19.705,
            -19.472,
            -19.309,
            -19.190,
            -19.098,
            -19.025,
            -18.966,
            -18.917,
            -18.876,
            -18.840,
            -18.810,
            -18.783,
            -18.760,
            -18.739,
            -18.721,
        ],
        [
            -19.527,
            -19.310,
            -19.161,
            -19.051,
            -18.968,
            -18.903,
            -18.851,
            -18.807,
            -18.771,
            -18.740,
            -18.713,
            -18.690,
            -18.670,
            -18.653,
            -18.637,
        ],
        [
            -19.357,
            -19.159,
            -19.022,
            -18.922,
            -18.847,
            -18.789,
            -18.743,
            -18.704,
            -18.673,
            -18.646,
            -18.623,
            -18.603,
            -18.586,
            -18.571,
            -18.558,
        ],
        [
            -19.195,
            -19.016,
            -18.892,
            -18.803,
            -18.736,
            -18.684,
            -18.643,
            -18.610,
            -18.583,
            -18.560,
            -18.540,
            -18.523,
            -18.509,
            -18.496,
            -18.485,
        ],
        [
            -19.042,
            -18.883,
            -18.772,
            -18.693,
            -18.634,
            -18.589,
            -18.553,
            -18.525,
            -18.501,
            -18.481,
            -18.465,
            -18.451,
            -18.438,
            -18.428,
            -18.419,
        ],
        [
            -18.894,
            -18.758,
            -18.662,
            -18.593,
            -18.542,
            -18.503,
            -18.473,
            -18.448,
            -18.428,
            -18.412,
            -18.398,
            -18.386,
            -18.376,
            -18.367,
            -18.359,
        ],
        [
            -18.752,
            -18.639,
            -18.559,
            -18.501,
            -18.458,
            -18.426,
            -18.400,
            -18.380,
            -18.363,
            -18.350,
            -18.338,
            -18.328,
            -18.320,
            -18.313,
            -18.306,
        ],
        [
            -18.611,
            -18.523,
            -18.460,
            -18.415,
            -18.381,
            -18.355,
            -18.334,
            -18.318,
            -18.304,
            -18.293,
            -18.284,
            -18.276,
            -18.269,
            -18.263,
            -18.258,
        ],
        [
            -18.471,
            -18.408,
            -18.362,
            -18.329,
            -18.304,
            -18.285,
            -18.269,
            -18.257,
            -18.247,
            -18.238,
            -18.231,
            -18.224,
            -18.219,
            -18.214,
            -18.210,
        ],
        [
            -18.330,
            -18.290,
            -18.261,
            -18.239,
            -18.223,
            -18.211,
            -18.201,
            -18.192,
            -18.185,
            -18.179,
            -18.174,
            -18.169,
            -18.165,
            -18.162,
            -18.159,
        ],
        # Energy 7.1-8.0 eV (rows 50-59)
        [
            -18.190,
            -18.168,
            -18.154,
            -18.143,
            -18.135,
            -18.129,
            -18.124,
            -18.120,
            -18.116,
            -18.112,
            -18.109,
            -18.106,
            -18.104,
            -18.102,
            -18.100,
        ],
        [
            -18.055,
            -18.047,
            -18.043,
            -18.042,
            -18.040,
            -18.039,
            -18.039,
            -18.038,
            -18.037,
            -18.036,
            -18.035,
            -18.034,
            -18.033,
            -18.033,
            -18.032,
        ],
        [
            -17.929,
            -17.931,
            -17.935,
            -17.939,
            -17.943,
            -17.946,
            -17.948,
            -17.950,
            -17.952,
            -17.953,
            -17.955,
            -17.956,
            -17.957,
            -17.958,
            -17.959,
        ],
        [
            -17.818,
            -17.826,
            -17.834,
            -17.842,
            -17.849,
            -17.855,
            -17.860,
            -17.865,
            -17.869,
            -17.872,
            -17.875,
            -17.878,
            -17.881,
            -17.883,
            -17.886,
        ],
        [
            -17.724,
            -17.736,
            -17.747,
            -17.758,
            -17.767,
            -17.775,
            -17.782,
            -17.788,
            -17.793,
            -17.798,
            -17.803,
            -17.807,
            -17.811,
            -17.815,
            -17.819,
        ],
        [
            -17.651,
            -17.665,
            -17.678,
            -17.690,
            -17.701,
            -17.710,
            -17.718,
            -17.725,
            -17.732,
            -17.738,
            -17.744,
            -17.749,
            -17.755,
            -17.760,
            -17.765,
        ],
        [
            -17.601,
            -17.615,
            -17.629,
            -17.642,
            -17.653,
            -17.663,
            -17.672,
            -17.680,
            -17.688,
            -17.695,
            -17.701,
            -17.708,
            -17.714,
            -17.720,
            -17.726,
        ],
        [
            -17.572,
            -17.587,
            -17.602,
            -17.614,
            -17.626,
            -17.636,
            -17.645,
            -17.654,
            -17.662,
            -17.670,
            -17.677,
            -17.684,
            -17.691,
            -17.698,
            -17.704,
        ],
        [
            -17.565,
            -17.581,
            -17.595,
            -17.607,
            -17.619,
            -17.629,
            -17.638,
            -17.647,
            -17.656,
            -17.664,
            -17.671,
            -17.679,
            -17.686,
            -17.693,
            -17.700,
        ],
        [
            -17.580,
            -17.594,
            -17.608,
            -17.620,
            -17.630,
            -17.640,
            -17.650,
            -17.658,
            -17.667,
            -17.675,
            -17.682,
            -17.690,
            -17.697,
            -17.704,
            -17.711,
        ],
        # Energy 8.1-9.0 eV (rows 60-69)
        [
            -17.613,
            -17.626,
            -17.639,
            -17.649,
            -17.659,
            -17.669,
            -17.677,
            -17.686,
            -17.694,
            -17.701,
            -17.709,
            -17.716,
            -17.723,
            -17.730,
            -17.737,
        ],
        [
            -17.663,
            -17.675,
            -17.685,
            -17.695,
            -17.703,
            -17.711,
            -17.719,
            -17.727,
            -17.734,
            -17.741,
            -17.748,
            -17.755,
            -17.761,
            -17.768,
            -17.774,
        ],
        [
            -17.728,
            -17.737,
            -17.745,
            -17.752,
            -17.759,
            -17.766,
            -17.772,
            -17.778,
            -17.785,
            -17.791,
            -17.797,
            -17.803,
            -17.808,
            -17.814,
            -17.820,
        ],
        [
            -17.803,
            -17.809,
            -17.814,
            -17.818,
            -17.823,
            -17.828,
            -17.832,
            -17.837,
            -17.842,
            -17.847,
            -17.852,
            -17.856,
            -17.861,
            -17.866,
            -17.871,
        ],
        [
            -17.884,
            -17.886,
            -17.888,
            -17.889,
            -17.891,
            -17.893,
            -17.896,
            -17.899,
            -17.902,
            -17.905,
            -17.908,
            -17.912,
            -17.915,
            -17.919,
            -17.922,
        ],
        [
            -17.966,
            -17.964,
            -17.961,
            -17.959,
            -17.958,
            -17.958,
            -17.958,
            -17.959,
            -17.960,
            -17.961,
            -17.963,
            -17.964,
            -17.966,
            -17.968,
            -17.970,
        ],
        [
            -18.040,
            -18.034,
            -18.028,
            -18.023,
            -18.019,
            -18.016,
            -18.013,
            -18.012,
            -18.010,
            -18.010,
            -18.009,
            -18.009,
            -18.009,
            -18.009,
            -18.010,
        ],
        [
            -18.096,
            -18.087,
            -18.078,
            -18.071,
            -18.065,
            -18.059,
            -18.055,
            -18.051,
            -18.047,
            -18.045,
            -18.042,
            -18.040,
            -18.039,
            -18.037,
            -18.036,
        ],
        [
            -18.125,
            -18.115,
            -18.105,
            -18.097,
            -18.089,
            -18.082,
            -18.076,
            -18.070,
            -18.065,
            -18.061,
            -18.057,
            -18.053,
            -18.051,
            -18.048,
            -18.046,
        ],
        [
            -18.120,
            -18.112,
            -18.103,
            -18.095,
            -18.087,
            -18.079,
            -18.072,
            -18.066,
            -18.060,
            -18.055,
            -18.050,
            -18.046,
            -18.042,
            -18.039,
            -18.036,
        ],
        # Energy 9.1-10.0 eV (rows 70-79)
        [
            -18.083,
            -18.078,
            -18.071,
            -18.064,
            -18.057,
            -18.050,
            -18.044,
            -18.037,
            -18.032,
            -18.026,
            -18.022,
            -18.017,
            -18.014,
            -18.010,
            -18.007,
        ],
        [
            -18.025,
            -18.022,
            -18.017,
            -18.012,
            -18.006,
            -18.000,
            -17.994,
            -17.989,
            -17.984,
            -17.979,
            -17.975,
            -17.971,
            -17.968,
            -17.965,
            -17.963,
        ],
        [
            -17.957,
            -17.955,
            -17.952,
            -17.948,
            -17.943,
            -17.938,
            -17.934,
            -17.929,
            -17.925,
            -17.922,
            -17.918,
            -17.916,
            -17.913,
            -17.911,
            -17.910,
        ],
        [
            -17.890,
            -17.889,
            -17.886,
            -17.882,
            -17.879,
            -17.875,
            -17.871,
            -17.867,
            -17.864,
            -17.862,
            -17.860,
            -17.858,
            -17.857,
            -17.856,
            -17.855,
        ],
        [
            -17.831,
            -17.829,
            -17.826,
            -17.822,
            -17.819,
            -17.815,
            -17.812,
            -17.810,
            -17.807,
            -17.806,
            -17.804,
            -17.803,
            -17.803,
            -17.803,
            -17.803,
        ],
        [
            -17.786,
            -17.782,
            -17.777,
            -17.773,
            -17.769,
            -17.766,
            -17.763,
            -17.761,
            -17.759,
            -17.758,
            -17.757,
            -17.757,
            -17.757,
            -17.758,
            -17.759,
        ],
        [
            -17.753,
            -17.747,
            -17.741,
            -17.735,
            -17.731,
            -17.727,
            -17.724,
            -17.722,
            -17.721,
            -17.720,
            -17.720,
            -17.720,
            -17.721,
            -17.722,
            -17.724,
        ],
        [
            -17.733,
            -17.724,
            -17.716,
            -17.709,
            -17.703,
            -17.699,
            -17.696,
            -17.694,
            -17.693,
            -17.692,
            -17.692,
            -17.693,
            -17.694,
            -17.695,
            -17.697,
        ],
        [
            -17.723,
            -17.711,
            -17.700,
            -17.691,
            -17.685,
            -17.680,
            -17.676,
            -17.674,
            -17.673,
            -17.672,
            -17.673,
            -17.673,
            -17.675,
            -17.676,
            -17.678,
        ],
        [
            -17.718,
            -17.702,
            -17.689,
            -17.679,
            -17.672,
            -17.667,
            -17.663,
            -17.660,
            -17.659,
            -17.659,
            -17.659,
            -17.660,
            -17.661,
            -17.663,
            -17.665,
        ],
        # Energy 10.1-11.0 eV (rows 80-89)
        [
            -17.713,
            -17.695,
            -17.681,
            -17.670,
            -17.662,
            -17.656,
            -17.653,
            -17.650,
            -17.649,
            -17.649,
            -17.649,
            -17.650,
            -17.651,
            -17.653,
            -17.655,
        ],
        [
            -17.705,
            -17.686,
            -17.671,
            -17.660,
            -17.652,
            -17.647,
            -17.643,
            -17.641,
            -17.640,
            -17.640,
            -17.640,
            -17.641,
            -17.643,
            -17.645,
            -17.647,
        ],
        [
            -17.690,
            -17.671,
            -17.657,
            -17.647,
            -17.640,
            -17.635,
            -17.632,
            -17.630,
            -17.630,
            -17.630,
            -17.631,
            -17.632,
            -17.634,
            -17.636,
            -17.639,
        ],
        [
            -17.667,
            -17.649,
            -17.637,
            -17.629,
            -17.623,
            -17.619,
            -17.618,
            -17.617,
            -17.617,
            -17.618,
            -17.619,
            -17.621,
            -17.623,
            -17.626,
            -17.628,
        ],
        [
            -17.635,
            -17.621,
            -17.611,
            -17.605,
            -17.601,
            -17.600,
            -17.599,
            -17.599,
            -17.601,
            -17.602,
            -17.604,
            -17.607,
            -17.609,
            -17.612,
            -17.615,
        ],
        [
            -17.596,
            -17.585,
            -17.579,
            -17.576,
            -17.575,
            -17.575,
            -17.576,
            -17.578,
            -17.580,
            -17.582,
            -17.585,
            -17.588,
            -17.591,
            -17.595,
            -17.598,
        ],
        [
            -17.550,
            -17.544,
            -17.542,
            -17.542,
            -17.544,
            -17.546,
            -17.548,
            -17.552,
            -17.555,
            -17.558,
            -17.562,
            -17.566,
            -17.570,
            -17.573,
            -17.577,
        ],
        [
            -17.501,
            -17.500,
            -17.501,
            -17.504,
            -17.508,
            -17.513,
            -17.517,
            -17.521,
            -17.526,
            -17.530,
            -17.535,
            -17.539,
            -17.544,
            -17.548,
            -17.553,
        ],
        [
            -17.449,
            -17.452,
            -17.457,
            -17.463,
            -17.470,
            -17.476,
            -17.482,
            -17.488,
            -17.493,
            -17.499,
            -17.504,
            -17.509,
            -17.514,
            -17.519,
            -17.524,
        ],
        [
            -17.396,
            -17.403,
            -17.412,
            -17.420,
            -17.429,
            -17.437,
            -17.444,
            -17.451,
            -17.458,
            -17.464,
            -17.470,
            -17.476,
            -17.481,
            -17.487,
            -17.492,
        ],
        # Energy 11.1-12.0 eV (rows 90-99)
        [
            -17.344,
            -17.355,
            -17.366,
            -17.377,
            -17.387,
            -17.396,
            -17.405,
            -17.413,
            -17.420,
            -17.427,
            -17.434,
            -17.440,
            -17.446,
            -17.452,
            -17.458,
        ],
        [
            -17.295,
            -17.307,
            -17.321,
            -17.333,
            -17.345,
            -17.355,
            -17.365,
            -17.373,
            -17.382,
            -17.389,
            -17.397,
            -17.404,
            -17.410,
            -17.417,
            -17.423,
        ],
        [
            -17.249,
            -17.264,
            -17.278,
            -17.292,
            -17.304,
            -17.316,
            -17.326,
            -17.335,
            -17.344,
            -17.352,
            -17.360,
            -17.368,
            -17.375,
            -17.382,
            -17.389,
        ],
        [
            -17.209,
            -17.225,
            -17.241,
            -17.255,
            -17.268,
            -17.280,
            -17.291,
            -17.301,
            -17.310,
            -17.319,
            -17.327,
            -17.335,
            -17.343,
            -17.350,
            -17.357,
        ],
        [
            -17.177,
            -17.194,
            -17.210,
            -17.225,
            -17.239,
            -17.251,
            -17.262,
            -17.272,
            -17.282,
            -17.291,
            -17.300,
            -17.308,
            -17.316,
            -17.324,
            -17.331,
        ],
        [
            -17.154,
            -17.172,
            -17.189,
            -17.204,
            -17.218,
            -17.230,
            -17.242,
            -17.252,
            -17.262,
            -17.272,
            -17.280,
            -17.289,
            -17.298,
            -17.306,
            -17.314,
        ],
        [
            -17.144,
            -17.162,
            -17.179,
            -17.194,
            -17.208,
            -17.220,
            -17.232,
            -17.242,
            -17.253,
            -17.262,
            -17.271,
            -17.280,
            -17.289,
            -17.297,
            -17.306,
        ],
        [
            -17.146,
            -17.164,
            -17.181,
            -17.196,
            -17.210,
            -17.222,
            -17.234,
            -17.245,
            -17.255,
            -17.265,
            -17.274,
            -17.283,
            -17.292,
            -17.301,
            -17.309,
        ],
        [
            -17.163,
            -17.180,
            -17.197,
            -17.212,
            -17.225,
            -17.237,
            -17.249,
            -17.260,
            -17.270,
            -17.280,
            -17.289,
            -17.298,
            -17.307,
            -17.316,
            -17.325,
        ],
        [
            -17.193,
            -17.211,
            -17.227,
            -17.241,
            -17.254,
            -17.266,
            -17.277,
            -17.288,
            -17.298,
            -17.308,
            -17.317,
            -17.327,
            -17.336,
            -17.345,
            -17.353,
        ],
        # Energy 12.1-13.0 eV (rows 100-109)
        [
            -17.239,
            -17.256,
            -17.271,
            -17.284,
            -17.297,
            -17.309,
            -17.320,
            -17.330,
            -17.340,
            -17.350,
            -17.359,
            -17.369,
            -17.378,
            -17.387,
            -17.395,
        ],
        [
            -17.299,
            -17.315,
            -17.329,
            -17.342,
            -17.354,
            -17.365,
            -17.376,
            -17.386,
            -17.396,
            -17.405,
            -17.415,
            -17.424,
            -17.433,
            -17.442,
            -17.451,
        ],
        [
            -17.373,
            -17.388,
            -17.402,
            -17.414,
            -17.425,
            -17.436,
            -17.446,
            -17.456,
            -17.466,
            -17.475,
            -17.484,
            -17.493,
            -17.502,
            -17.511,
            -17.520,
        ],
        [
            -17.462,
            -17.476,
            -17.489,
            -17.500,
            -17.511,
            -17.521,
            -17.531,
            -17.541,
            -17.550,
            -17.559,
            -17.569,
            -17.578,
            -17.587,
            -17.595,
            -17.604,
        ],
        [
            -17.567,
            -17.581,
            -17.592,
            -17.603,
            -17.613,
            -17.623,
            -17.632,
            -17.641,
            -17.651,
            -17.660,
            -17.669,
            -17.678,
            -17.686,
            -17.695,
            -17.704,
        ],
        [
            -17.689,
            -17.701,
            -17.712,
            -17.722,
            -17.732,
            -17.741,
            -17.750,
            -17.759,
            -17.768,
            -17.777,
            -17.786,
            -17.795,
            -17.803,
            -17.812,
            -17.821,
        ],
        [
            -17.829,
            -17.840,
            -17.851,
            -17.860,
            -17.869,
            -17.878,
            -17.887,
            -17.896,
            -17.904,
            -17.913,
            -17.922,
            -17.930,
            -17.939,
            -17.948,
            -17.956,
        ],
        [
            -17.988,
            -18.000,
            -18.010,
            -18.019,
            -18.028,
            -18.036,
            -18.045,
            -18.053,
            -18.062,
            -18.070,
            -18.079,
            -18.087,
            -18.096,
            -18.104,
            -18.112,
        ],
        [
            -18.171,
            -18.183,
            -18.192,
            -18.201,
            -18.210,
            -18.218,
            -18.227,
            -18.235,
            -18.243,
            -18.252,
            -18.260,
            -18.268,
            -18.277,
            -18.285,
            -18.293,
        ],
        [
            -18.381,
            -18.393,
            -18.403,
            -18.413,
            -18.422,
            -18.430,
            -18.438,
            -18.447,
            -18.455,
            -18.463,
            -18.471,
            -18.479,
            -18.487,
            -18.495,
            -18.503,
        ],
        # Energy 13.1-14.0 eV (rows 110-119)
        [
            -18.625,
            -18.638,
            -18.650,
            -18.660,
            -18.669,
            -18.678,
            -18.687,
            -18.695,
            -18.703,
            -18.711,
            -18.719,
            -18.726,
            -18.734,
            -18.742,
            -18.750,
        ],
        [
            -18.912,
            -18.929,
            -18.943,
            -18.955,
            -18.966,
            -18.975,
            -18.984,
            -18.993,
            -19.001,
            -19.008,
            -19.016,
            -19.023,
            -19.031,
            -19.038,
            -19.045,
        ],
        [
            -19.260,
            -19.283,
            -19.303,
            -19.320,
            -19.333,
            -19.345,
            -19.355,
            -19.364,
            -19.372,
            -19.380,
            -19.387,
            -19.394,
            -19.400,
            -19.407,
            -19.413,
        ],
        [
            -19.704,
            -19.740,
            -19.771,
            -19.796,
            -19.816,
            -19.832,
            -19.845,
            -19.855,
            -19.863,
            -19.870,
            -19.876,
            -19.882,
            -19.887,
            -19.892,
            -19.897,
        ],
        [
            -20.339,
            -20.386,
            -20.424,
            -20.454,
            -20.476,
            -20.492,
            -20.502,
            -20.509,
            -20.513,
            -20.516,
            -20.518,
            -20.520,
            -20.521,
            -20.523,
            -20.524,
        ],
        [
            -21.052,
            -21.075,
            -21.093,
            -21.105,
            -21.114,
            -21.120,
            -21.123,
            -21.125,
            -21.126,
            -21.127,
            -21.128,
            -21.130,
            -21.131,
            -21.133,
            -21.135,
        ],
        [
            -21.174,
            -21.203,
            -21.230,
            -21.255,
            -21.278,
            -21.299,
            -21.320,
            -21.339,
            -21.357,
            -21.375,
            -21.392,
            -21.408,
            -21.424,
            -21.439,
            -21.454,
        ],
        [
            -21.285,
            -21.317,
            -21.346,
            -21.372,
            -21.395,
            -21.416,
            -21.435,
            -21.452,
            -21.468,
            -21.483,
            -21.497,
            -21.511,
            -21.524,
            -21.536,
            -21.548,
        ],
        [
            -21.396,
            -21.429,
            -21.459,
            -21.486,
            -21.511,
            -21.532,
            -21.551,
            -21.569,
            -21.585,
            -21.600,
            -21.614,
            -21.627,
            -21.640,
            -21.652,
            -21.663,
        ],
        [
            -21.516,
            -21.549,
            -21.580,
            -21.609,
            -21.635,
            -21.658,
            -21.678,
            -21.696,
            -21.713,
            -21.728,
            -21.742,
            -21.755,
            -21.767,
            -21.779,
            -21.790,
        ],
        # Energy 14.1-15.0 eV (rows 120-129)
        [
            -21.651,
            -21.681,
            -21.711,
            -21.738,
            -21.763,
            -21.785,
            -21.804,
            -21.821,
            -21.837,
            -21.851,
            -21.864,
            -21.876,
            -21.887,
            -21.898,
            -21.908,
        ],
        [
            -21.810,
            -21.831,
            -21.853,
            -21.874,
            -21.893,
            -21.910,
            -21.925,
            -21.938,
            -21.950,
            -21.961,
            -21.971,
            -21.980,
            -21.989,
            -21.998,
            -22.006,
        ],
        [
            -22.009,
            -22.016,
            -22.026,
            -22.037,
            -22.048,
            -22.058,
            -22.066,
            -22.074,
            -22.081,
            -22.088,
            -22.094,
            -22.099,
            -22.105,
            -22.111,
            -22.117,
        ],
        [
            -22.353,
            -22.317,
            -22.296,
            -22.284,
            -22.276,
            -22.270,
            -22.266,
            -22.262,
            -22.260,
            -22.258,
            -22.257,
            -22.257,
            -22.257,
            -22.258,
            -22.259,
        ],
        [
            -22.705,
            -22.609,
            -22.552,
            -22.515,
            -22.488,
            -22.468,
            -22.451,
            -22.438,
            -22.427,
            -22.418,
            -22.410,
            -22.405,
            -22.400,
            -22.397,
            -22.395,
        ],
        [
            -22.889,
            -22.791,
            -22.731,
            -22.690,
            -22.659,
            -22.634,
            -22.612,
            -22.594,
            -22.579,
            -22.566,
            -22.555,
            -22.546,
            -22.539,
            -22.533,
            -22.528,
        ],
        [
            -23.211,
            -23.109,
            -23.041,
            -22.989,
            -22.945,
            -22.906,
            -22.872,
            -22.842,
            -22.816,
            -22.793,
            -22.774,
            -22.757,
            -22.743,
            -22.732,
            -22.722,
        ],
        [
            -25.312,
            -24.669,
            -24.250,
            -23.959,
            -23.746,
            -23.587,
            -23.463,
            -23.366,
            -23.288,
            -23.225,
            -23.173,
            -23.131,
            -23.095,
            -23.066,
            -23.041,
        ],
        [
            -25.394,
            -24.752,
            -24.333,
            -24.041,
            -23.829,
            -23.669,
            -23.546,
            -23.449,
            -23.371,
            -23.308,
            -23.256,
            -23.214,
            -23.178,
            -23.149,
            -23.124,
        ],
        [
            -25.430,
            -24.787,
            -24.369,
            -24.077,
            -23.865,
            -23.705,
            -23.582,
            -23.484,
            -23.407,
            -23.344,
            -23.292,
            -23.249,
            -23.214,
            -23.185,
            -23.160,
        ],
    ],
    dtype=np.float64,
)


def _ohop_opacity(freq: float, temp: np.ndarray) -> np.ndarray:
    """OH molecular opacity (atlas7v.for lines 8385-8701).

    Returns cross-section * partition function for each layer.
    Only active for T < 9000K and energy 2.1-15.0 eV.

    Parameters
    ----------
    freq : float
        Frequency (Hz)
    temp : np.ndarray
        Temperature array (K), shape (n_layers,)

    Returns
    -------
    np.ndarray
        OHOP values for each layer (cm²), shape (n_layers,)
    """
    n_layers = temp.size
    result = np.zeros(n_layers, dtype=np.float64)

    # Convert frequency to energy in eV
    waveno = freq / 2.99792458e10  # cm^-1
    evolt = waveno / 8065.479  # eV

    # Energy index (0.1 eV bins starting at 2.1 eV)
    n = int(evolt * 10) - 20  # Shifted by 2.0 eV
    if n <= 0 or n >= 130:  # Energy range 2.1-15.0 eV
        return result

    en = float(n) * 0.1 + 2.0

    # Interpolate cross-section in energy (atlas7v.for lines 8683-8685)
    idx = n - 1  # 0-based index
    if idx < 0 or idx >= 129:
        return result

    crossoht = np.zeros(15, dtype=np.float64)
    for it in range(15):
        crossoht[it] = (
            _OH_CROSSSECT[idx, it]
            + (_OH_CROSSSECT[idx + 1, it] - _OH_CROSSSECT[idx, it]) * (evolt - en) / 0.1
        )

    # For each layer, interpolate in temperature
    for j in range(n_layers):
        t_j = temp[j]
        if t_j >= 9000.0:
            continue

        # Partition function interpolation (200K grid starting at 1000K)
        it_part = int((t_j - 1000.0) / 200.0)
        it_part = max(0, min(it_part, 39))
        tn_part = float(it_part) * 200.0 + 1000.0
        part = (
            _OH_PARTITION[it_part]
            + (_OH_PARTITION[it_part + 1] - _OH_PARTITION[it_part])
            * (t_j - tn_part)
            / 200.0
        )

        # Cross-section interpolation (500K grid starting at 2000K)
        it_cross = int((t_j - 2000.0) / 500.0)
        it_cross = max(0, min(it_cross, 13))
        tn_cross = float(it_cross) * 500.0 + 2000.0

        log_xsect = (
            crossoht[it_cross]
            + (crossoht[it_cross + 1] - crossoht[it_cross]) * (t_j - tn_cross) / 500.0
        )

        # Convert from log10 to linear
        result[j] = np.exp(log_xsect * 2.30258509299405) * part

    return result


# H2 collision-induced absorption tables (atlas7v.for lines 8733-8912)
# H2-H2 and H2-He tables: 7 temperature points × 81 wavenumber points
# Temperature grid: 1000K to 7000K in 1000K steps
# Wavenumber grid: 0 to 20000 cm^-1 in 250 cm^-1 steps

# Complete H2-H2 collision-induced absorption table (atlas7v.for lines 8733-8822)
# 81 wavenumber bins (0-20000 cm^-1 in 250 cm^-1 steps) × 7 temperature points (1000-7000K)
# Values are log10(absorption coefficient in cm^5)
_H2_COLL_H2H2 = np.array(
    [
        # H2H21: rows 0-8 (wavenumber 0-2000 cm^-1)
        [-46.000, -46.000, -46.000, -46.000, -46.000, -46.000, -46.000],
        [-45.350, -45.350, -45.350, -45.350, -45.350, -45.350, -45.350],
        [-44.850, -44.850, -44.850, -44.850, -44.850, -45.850, -45.850],
        [-44.375, -44.465, -44.497, -44.504, -44.502, -44.657, -44.656],
        [-44.161, -44.216, -44.249, -44.255, -44.245, -44.231, -44.227],
        [-44.160, -44.081, -44.081, -44.076, -44.063, -44.047, -44.042],
        [-44.249, -44.017, -43.966, -43.940, -43.918, -43.898, -43.891],
        [-44.450, -44.020, -43.900, -43.844, -43.806, -43.776, -43.764],
        [-44.712, -44.080, -43.881, -43.785, -43.726, -43.682, -43.662],
        # H2H22: rows 9-17 (wavenumber 2250-4250 cm^-1)
        [-45.016, -44.186, -43.902, -43.763, -43.677, -43.616, -43.586],
        [-45.308, -44.319, -43.958, -43.773, -43.659, -43.579, -43.537],
        [-45.452, -44.442, -44.034, -43.810, -43.669, -43.570, -43.514],
        [-45.306, -44.500, -44.100, -43.858, -43.697, -43.580, -43.511],
        [-45.081, -44.452, -44.111, -43.887, -43.724, -43.598, -43.518],
        [-44.801, -44.302, -44.049, -43.876, -43.734, -43.608, -43.522],
        [-44.494, -44.104, -43.945, -43.832, -43.720, -43.603, -43.516],
        [-44.177, -43.936, -43.849, -43.783, -43.704, -43.596, -43.511],
        [-44.042, -43.865, -43.807, -43.767, -43.712, -43.611, -43.527],
        # H2H23: rows 18-26 (wavenumber 4500-6500 cm^-1)
        [-44.148, -43.922, -43.846, -43.806, -43.763, -43.662, -43.578],
        [-44.293, -44.042, -43.936, -43.884, -43.843, -43.742, -43.653],
        [-44.444, -44.179, -44.052, -43.984, -43.937, -43.832, -43.739],
        [-44.594, -44.311, -44.173, -44.091, -44.033, -43.924, -43.827],
        [-44.818, -44.448, -44.292, -44.196, -44.124, -44.012, -43.910],
        [-45.097, -44.600, -44.414, -44.300, -44.210, -44.095, -43.989],
        [-45.437, -44.782, -44.548, -44.409, -44.294, -44.177, -44.068],
        [-45.771, -44.992, -44.702, -44.533, -44.391, -44.269, -44.154],
        [-46.088, -45.218, -44.873, -44.672, -44.503, -44.374, -44.251],
        # H2H24: rows 27-35 (wavenumber 6750-8750 cm^-1)
        [-46.371, -45.438, -45.046, -44.813, -44.621, -44.483, -44.351],
        [-46.554, -45.632, -45.209, -44.949, -44.738, -44.590, -44.448],
        [-46.593, -45.788, -45.352, -45.074, -44.848, -44.692, -44.542],
        [-46.513, -45.887, -45.463, -45.181, -44.950, -44.786, -44.627],
        [-46.391, -45.917, -45.542, -45.271, -45.041, -44.873, -44.707],
        [-46.197, -45.896, -45.601, -45.350, -45.124, -44.952, -44.781],
        [-46.086, -45.911, -45.664, -45.423, -45.198, -45.023, -44.848],
        [-46.127, -45.958, -45.723, -45.487, -45.265, -45.089, -44.913],
        [-46.077, -45.963, -45.755, -45.534, -45.322, -45.149, -44.973],
        # H2H25: rows 36-44 (wavenumber 9000-11000 cm^-1)
        [-46.057, -45.947, -45.770, -45.571, -45.371, -45.204, -45.030],
        [-46.122, -45.959, -45.792, -45.610, -45.422, -45.260, -45.088],
        [-46.302, -46.023, -45.840, -45.662, -45.480, -45.322, -45.149],
        [-46.560, -46.146, -45.928, -45.741, -45.557, -45.394, -45.218],
        [-46.891, -46.327, -46.058, -45.844, -45.648, -45.477, -45.292],
        [-47.245, -46.558, -46.226, -45.967, -45.753, -45.568, -45.372],
        [-47.527, -46.793, -46.408, -46.110, -45.871, -45.668, -45.457],
        [-47.729, -47.001, -46.589, -46.254, -45.992, -45.771, -45.544],
        [-47.829, -47.161, -46.750, -46.391, -46.111, -45.872, -45.630],
        # H2H26: rows 45-53 (wavenumber 11250-13250 cm^-1)
        [-47.825, -47.265, -46.879, -46.547, -46.239, -45.980, -45.719],
        [-47.740, -47.317, -46.979, -46.658, -46.345, -46.075, -45.803],
        [-47.635, -47.340, -47.055, -46.755, -46.444, -46.166, -45.882],
        [-47.593, -47.358, -47.122, -46.844, -46.536, -46.252, -45.961],
        [-47.488, -47.375, -47.178, -46.921, -46.621, -46.334, -46.036],
        [-47.517, -47.387, -47.213, -46.982, -46.696, -46.412, -46.109],
        [-47.511, -47.385, -47.234, -47.031, -46.765, -46.485, -46.180],
        [-47.601, -47.428, -47.274, -47.084, -46.834, -46.558, -46.251],
        [-47.740, -47.509, -47.339, -47.150, -46.906, -46.632, -46.322],
        # H2H27: rows 54-62 (wavenumber 13500-15500 cm^-1)
        [-48.007, -47.632, -47.429, -47.233, -46.988, -46.710, -46.395],
        [-48.371, -47.825, -47.563, -47.341, -47.081, -46.794, -46.469],
        [-48.778, -48.074, -47.739, -47.476, -47.189, -46.884, -46.547],
        [-49.170, -48.341, -47.936, -47.625, -47.304, -46.977, -46.625],
        [-49.531, -48.604, -48.136, -47.780, -47.424, -47.074, -46.704],
        [-49.869, -48.850, -48.328, -47.932, -47.543, -47.170, -46.784],
        [-50.189, -49.080, -48.510, -48.078, -47.660, -47.264, -46.863],
        [-50.496, -49.299, -48.682, -48.218, -47.774, -47.358, -46.940],
        [-50.797, -49.508, -48.847, -48.353, -47.885, -47.449, -47.018],
        # H2H28: rows 63-71 (wavenumber 15750-17750 cm^-1)
        [-51.088, -49.711, -49.008, -48.484, -47.993, -47.540, -47.094],
        [-51.374, -49.907, -49.163, -48.613, -48.100, -47.629, -47.170],
        [-51.655, -50.102, -49.317, -48.740, -48.205, -47.717, -47.246],
        [-51.931, -50.293, -49.468, -48.865, -48.309, -47.804, -47.321],
        [-52.205, -50.481, -49.617, -48.989, -48.413, -47.891, -47.396],
        [-52.475, -50.670, -49.767, -49.112, -48.516, -47.978, -47.470],
        [-52.742, -50.855, -49.915, -49.235, -48.619, -48.064, -47.545],
        [-53.010, -51.038, -50.062, -49.358, -48.721, -48.150, -47.619],
        [-53.277, -51.221, -50.209, -49.481, -48.824, -48.236, -47.692],
        # H2H29: rows 72-80 (wavenumber 18000-20000 cm^-1)
        [-53.545, -51.399, -50.353, -49.602, -48.925, -48.321, -47.765],
        [-53.812, -51.575, -50.496, -49.722, -49.026, -48.405, -47.839],
        [-54.080, -51.748, -50.634, -49.840, -49.125, -48.489, -47.911],
        [-54.347, -51.918, -50.769, -49.954, -49.222, -48.571, -47.984],
        [-54.615, -52.086, -50.900, -50.065, -49.317, -48.653, -48.055],
        [-54.882, -52.253, -51.029, -50.174, -49.411, -48.733, -48.125],
        [-55.150, -52.419, -51.158, -50.282, -49.506, -48.813, -48.196],
        [-55.417, -52.584, -51.288, -50.399, -49.642, -48.903, -48.268],
        [-55.685, -52.778, -51.420, -50.527, -49.732, -48.981, -48.338],
    ],
    dtype=np.float64,
)

# Complete H2-He collision-induced absorption table (atlas7v.for lines 8823-8912)
# 81 wavenumber bins × 7 temperature points
_H2_COLL_H2HE = np.array(
    [
        # H2HE1: rows 0-8 (wavenumber 0-2000 cm^-1)
        [-46.000, -46.000, -46.000, -46.000, -46.000, -46.000, -46.000],
        [-44.288, -44.288, -44.288, -44.288, -44.288, -44.288, -44.288],
        [-44.288, -44.142, -44.045, -43.997, -43.949, -44.900, -43.852],
        [-44.362, -44.090, -43.978, -43.901, -43.833, -43.939, -43.716],
        [-44.461, -44.114, -43.954, -43.863, -43.786, -43.717, -43.654],
        [-44.601, -44.195, -43.973, -43.875, -43.791, -43.715, -43.646],
        [-44.777, -44.292, -44.012, -43.905, -43.813, -43.732, -43.658],
        [-45.000, -44.402, -44.061, -43.946, -43.844, -43.756, -43.678],
        [-45.268, -44.530, -44.122, -43.996, -43.883, -43.786, -43.703],
        # H2HE2: rows 9-17 (wavenumber 2250-4250 cm^-1)
        [-45.562, -44.680, -44.199, -44.059, -43.932, -43.823, -43.733],
        [-45.841, -44.841, -44.289, -44.128, -43.983, -43.862, -43.766],
        [-46.012, -44.969, -44.371, -44.182, -44.017, -43.891, -43.789],
        [-45.931, -44.975, -44.394, -44.173, -43.999, -43.872, -43.779],
        [-45.621, -44.790, -44.293, -44.062, -43.905, -43.793, -43.726],
        [-45.151, -44.469, -44.084, -43.871, -43.755, -43.705, -43.666],
        [-44.620, -44.131, -43.871, -43.715, -43.640, -43.644, -43.628],
        [-44.166, -43.892, -43.748, -43.674, -43.639, -43.628, -43.625],
        [-44.023, -43.837, -43.743, -43.710, -43.691, -43.663, -43.660],
        # H2HE3: rows 18-26 (wavenumber 4500-6500 cm^-1)
        [-44.190, -43.942, -43.830, -43.782, -43.755, -43.735, -43.719],
        [-44.446, -44.120, -43.967, -43.884, -43.839, -43.807, -43.776],
        [-44.689, -44.312, -44.120, -44.011, -43.932, -43.872, -43.826],
        [-44.904, -44.491, -44.269, -44.134, -44.022, -43.941, -43.881],
        [-45.133, -44.656, -44.407, -44.244, -44.115, -44.016, -43.941],
        [-45.398, -44.824, -44.543, -44.359, -44.217, -44.098, -44.006],
        [-45.701, -45.010, -44.686, -44.481, -44.322, -44.186, -44.076],
        [-46.024, -45.221, -44.843, -44.610, -44.431, -44.275, -44.147],
        [-46.350, -45.449, -45.015, -44.747, -44.542, -44.366, -44.219],
        # H2HE4: rows 27-35 (wavenumber 6750-8750 cm^-1)
        [-46.736, -45.674, -45.189, -44.887, -44.657, -44.458, -44.294],
        [-46.993, -45.865, -45.347, -45.023, -44.771, -44.551, -44.367],
        [-47.031, -45.981, -45.469, -45.141, -44.878, -44.640, -44.437],
        [-46.787, -46.008, -45.553, -45.244, -44.979, -44.727, -44.506],
        [-46.496, -45.969, -45.618, -45.343, -45.085, -44.820, -44.579],
        [-46.310, -45.953, -45.689, -45.449, -45.198, -44.919, -44.656],
        [-46.295, -46.001, -45.787, -45.572, -45.321, -45.021, -44.732],
        [-46.434, -46.122, -45.919, -45.717, -45.453, -45.123, -44.804],
        [-46.671, -46.306, -46.085, -45.896, -45.588, -45.224, -44.873],
        # H2HE5: rows 36-44 (wavenumber 9000-11000 cm^-1)
        [-46.964, -46.539, -46.284, -46.068, -45.723, -45.320, -44.937],
        [-47.295, -46.807, -46.501, -46.241, -45.858, -45.412, -44.998],
        [-47.662, -47.097, -46.723, -46.415, -45.996, -45.500, -45.056],
        [-48.050, -47.399, -46.949, -46.583, -46.135, -45.587, -45.111],
        [-48.416, -47.683, -47.169, -46.749, -46.274, -45.671, -45.165],
        [-48.678, -47.892, -47.359, -46.907, -46.412, -45.752, -45.215],
        [-48.720, -47.963, -47.494, -47.044, -46.551, -45.828, -45.263],
        [-48.583, -47.912, -47.566, -47.160, -46.689, -45.901, -45.309],
        [-48.380, -47.807, -47.574, -47.236, -46.828, -45.972, -45.354],
        # H2HE6: rows 45-53 (wavenumber 11250-13250 cm^-1)
        [-48.164, -47.692, -47.543, -47.281, -46.953, -46.041, -45.397],
        [-47.988, -47.603, -47.513, -47.300, -47.028, -46.106, -45.438],
        [-47.874, -47.562, -47.506, -47.326, -47.085, -46.171, -45.479],
        [-47.846, -47.571, -47.518, -47.361, -47.141, -46.235, -45.519],
        [-47.827, -47.577, -47.536, -47.397, -47.194, -46.298, -45.558],
        [-47.841, -47.583, -47.548, -47.416, -47.234, -46.357, -45.596],
        [-47.949, -47.631, -47.550, -47.411, -47.253, -46.412, -45.632],
        [-48.168, -47.763, -47.580, -47.428, -47.282, -46.467, -45.668],
        [-48.442, -47.955, -47.682, -47.516, -47.360, -46.528, -45.704],
        # H2HE7: rows 54-62 (wavenumber 13500-15500 cm^-1)
        [-48.685, -48.145, -47.839, -47.654, -47.473, -46.593, -45.741],
        [-48.859, -48.310, -47.990, -47.778, -47.575, -46.655, -45.777],
        [-48.989, -48.445, -48.118, -47.878, -47.660, -46.714, -45.813],
        [-49.121, -48.560, -48.250, -47.981, -47.749, -46.773, -45.847],
        [-49.277, -48.667, -48.390, -48.094, -47.842, -46.831, -45.881],
        [-49.469, -48.778, -48.525, -48.202, -47.933, -46.888, -45.916],
        [-49.697, -48.907, -48.650, -48.303, -48.019, -46.943, -45.949],
        [-49.939, -49.059, -48.774, -48.403, -48.104, -46.996, -45.982],
        [-50.225, -49.227, -48.898, -48.504, -48.190, -47.049, -46.015],
        # H2HE8: rows 63-71 (wavenumber 15750-17750 cm^-1)
        [-50.537, -49.406, -49.016, -48.603, -48.273, -47.101, -46.048],
        [-50.831, -49.598, -49.130, -48.697, -48.354, -47.152, -46.080],
        [-50.981, -49.807, -49.239, -48.791, -48.435, -47.202, -46.112],
        [-51.106, -50.006, -49.345, -48.882, -48.514, -47.251, -46.145],
        [-51.231, -50.131, -49.445, -48.972, -48.591, -47.299, -46.176],
        [-51.356, -50.256, -49.540, -49.060, -48.667, -47.347, -46.208],
        [-51.481, -50.381, -49.629, -49.143, -48.741, -47.392, -46.239],
        [-51.606, -50.506, -49.711, -49.225, -48.813, -47.437, -46.271],
        [-51.731, -50.631, -49.787, -49.303, -48.885, -47.481, -46.302],
        # H2HE9: rows 72-80 (wavenumber 18000-20000 cm^-1)
        [-51.856, -50.756, -49.858, -49.377, -48.955, -47.523, -46.333],
        [-51.981, -50.881, -49.929, -49.449, -49.023, -47.566, -46.364],
        [-52.106, -51.006, -50.000, -49.517, -49.089, -47.607, -46.395],
        [-52.231, -51.131, -50.069, -49.581, -49.154, -47.647, -46.425],
        [-52.356, -51.256, -50.133, -49.642, -49.217, -47.687, -46.456],
        [-52.481, -51.381, -50.204, -49.699, -49.278, -47.726, -46.486],
        [-52.606, -51.506, -50.275, -49.752, -49.337, -47.765, -46.517],
        [-52.731, -51.631, -50.347, -49.803, -49.396, -47.802, -46.548],
        [-52.856, -51.756, -50.418, -49.850, -49.450, -47.839, -46.578],
    ],
    dtype=np.float64,
)


def _h2_collision_opacity(
    freq: float,
    temp: np.ndarray,
    xnfph1: np.ndarray,
    bhyd1: np.ndarray,
    xnfhe1: np.ndarray,
    rho: np.ndarray,
    tkev: np.ndarray,
    tlog: np.ndarray,
    stim: np.ndarray,
) -> np.ndarray:
    """H2 collision-induced absorption (atlas7v.for lines 8702-8951).

    Computes H2-H2 and H2-He collision-induced dipole absorption.
    Based on Borysow, Jorgensen, and Zheng (1997) A&A 324, 185-195.

    Only active for wavenumber < 20000 cm^-1.

    Parameters
    ----------
    freq : float
        Frequency (Hz)
    temp : np.ndarray
        Temperature array (K), shape (n_layers,)
    xnfph1 : np.ndarray
        Ground-state hydrogen population (atoms/cm³)
    bhyd1 : np.ndarray
        Hydrogen partition function ground state
    xnfhe1 : np.ndarray
        Helium I population (atoms/cm³)
    rho : np.ndarray
        Mass density (g/cm³)
    tkev : np.ndarray
        Temperature in eV
    tlog : np.ndarray
        Log of temperature
    stim : np.ndarray
        Stimulated emission factor

    Returns
    -------
    np.ndarray
        H2 collision-induced opacity (cm²/g), shape (n_layers,)
    """
    n_layers = temp.size
    result = np.zeros(n_layers, dtype=np.float64)

    waveno = freq / 2.99792458e10  # cm^-1
    if waveno > 20000.0:
        return result

    # Compute H2 number density using equilibrium formula
    # XNH2 = (XNFPH1 * 2 * BHYD1)^2 * exp(...) / RHO
    # From atlas7v.for lines 8916-8923
    poly_t = (
        1.63660e-3
        + (
            -4.93992e-7
            + (
                1.11822e-10
                + (-1.49567e-14 + (1.06206e-18 - 3.08720e-23 * temp) * temp) * temp
            )
            * temp
        )
        * temp
    ) * temp

    exp_term = 4.478 / tkev - 46.4584 + poly_t - 1.5 * tlog
    exp_term = np.clip(exp_term, -100, 100)

    xnh2 = (xnfph1 * 2.0 * bhyd1) ** 2 * np.exp(exp_term)

    # Wavenumber interpolation (atlas7v.for lines 8932-8938)
    # NU = wavenumber bin index (0-79), DELNU = fractional interpolation weight
    nu = int(waveno / 250.0)
    nu = min(79, nu)
    delnu = (waveno - 250.0 * nu) / 250.0

    # Interpolate tables in wavenumber first (atlas7v.for line 8937-8938)
    # H2H2NU(IT) = H2H2(IT,NU+1)*DELNU + H2H2(IT,NU+2)*(1-DELNU)
    # Note: Fortran is 1-indexed, Python is 0-indexed
    # Also note: Fortran table is (7,81) = (temp, waveno), Python is (81,7) = (waveno, temp)
    h2h2_nu = np.zeros(7, dtype=np.float64)
    h2he_nu = np.zeros(7, dtype=np.float64)

    for it in range(7):
        # Interpolate between wavenumber bins nu and nu+1
        idx1 = min(nu, 80)
        idx2 = min(nu + 1, 80)
        h2h2_nu[it] = _H2_COLL_H2H2[idx1, it] * delnu + _H2_COLL_H2H2[idx2, it] * (
            1.0 - delnu
        )
        h2he_nu[it] = _H2_COLL_H2HE[idx1, it] * delnu + _H2_COLL_H2HE[idx2, it] * (
            1.0 - delnu
        )

    # For each layer, interpolate in temperature (atlas7v.for lines 8940-8948)
    for j in range(n_layers):
        t_j = temp[j]

        # Temperature bin index (1000K grid, 1-indexed in Fortran -> 0-indexed in Python)
        it = int(t_j / 1000.0)
        it = max(1, min(6, it))  # Clamp to valid range (Fortran: 1-6)

        # Fractional temperature interpolation weight
        delt = (t_j - 1000.0 * it) / 1000.0
        delt = max(0.0, min(1.0, delt))

        # Interpolate in temperature (atlas7v.for lines 8944-8945)
        # XH2H2 = H2H2NU(IT)*DELT + H2H2NU(IT+1)*(1-DELT)
        # Fortran IT is 1-6, maps to Python indices it-1 and it
        xh2h2 = h2h2_nu[it - 1] * delt + h2h2_nu[it] * (1.0 - delt)
        xh2he = h2he_nu[it - 1] * delt + h2he_nu[it] * (1.0 - delt)

        # Final opacity (atlas7v.for lines 8947-8948)
        # AH2COLL = (10^XH2HE * XNFHE + 10^XH2H2 * XNH2) * XNH2 / RHO * STIM
        result[j] = (
            (10.0**xh2he * xnfhe1[j] + 10.0**xh2h2 * xnh2[j])
            * xnh2[j]
            / rho[j]
            * stim[j]
        )

    return result


# =============================================================================
# HYDROGEN PARTITION FUNCTION AND GROUND-STATE POPULATION
# =============================================================================
# To compute SIGH (hydrogen Rayleigh scattering) correctly, we need the
# ground-state population XNFPH, not total neutral hydrogen XNFH.
# Fortran computes XNFPH via POPS(1.01D0,11,XNFPH) but fort.10 only stores XNFH.
# We compute the ground-state fraction using the Boltzmann partition function.

# Hydrogen energy levels and statistical weights from Fortran atlas7v.for
# DATA EHYD/0.D0,82259.105D0,97492.302D0,102823.893D0,105291.651D0,106632.160D0/
# DATA GHYD/2.,8.,18.,32.,50.,72./
# Energy in cm^-1, convert to eV using 1 eV = 8065.479 cm^-1
H_ENERGY_CM = np.array([0.0, 82259.105, 97492.302, 102823.893, 105291.651, 106632.160])
H_ENERGY_EV = H_ENERGY_CM / 8065.479  # Convert to eV
H_STAT_WEIGHT = np.array([2.0, 8.0, 18.0, 32.0, 50.0, 72.0])  # g_n = 2n²

# Maximum principal quantum number for partition function sum
# Fortran GHYD/EHYD arrays have 6 levels (n=1 to n=6)
H_MAX_LEVEL = 6


def compute_hydrogen_partition_function(temperature: np.ndarray) -> np.ndarray:
    """Compute the hydrogen partition function U(T).

    Uses Fortran's EHYD and GHYD tables from atlas7v.for:
        U(T) = Σ g_n * exp(-E_n / kT)

    where g_n = 2n² (statistical weight) and E_n are from EHYD table.

    Parameters
    ----------
    temperature : np.ndarray
        Temperature in Kelvin (shape: (n_layers,))

    Returns
    -------
    partition_func : np.ndarray
        Hydrogen partition function U(T) (shape: (n_layers,))
    """
    # Boltzmann factor kT in eV
    kt_ev = KBOLTZ_EV * temperature  # eV

    # Initialize partition function
    partition_func = np.zeros_like(temperature, dtype=np.float64)

    # Sum over all levels (matching Fortran's 6 levels)
    for i in range(H_MAX_LEVEL):
        g_n = H_STAT_WEIGHT[i]
        e_n = H_ENERGY_EV[i]  # Already in eV

        # Boltzmann factor: exp(-E_n / kT)
        # Avoid overflow for low temperatures
        with np.errstate(over="ignore", invalid="ignore"):
            boltz = np.exp(-e_n / kt_ev)
            boltz = np.where(np.isfinite(boltz), boltz, 0.0)

        partition_func += g_n * boltz

    return partition_func


def compute_ground_state_hydrogen(
    xnf_h: np.ndarray, temperature: np.ndarray
) -> np.ndarray:
    """Compute ground-state hydrogen population from total neutral hydrogen.

    Fortran's PFSAHA with MODE=11 returns ionization_fraction / partition_function,
    while MODE=12 returns ionization_fraction.

    So: XNFPH(J,1) = XNFH(J) / U(T)

    where U(T) is the hydrogen partition function.

    This replicates what Fortran's POPS(1.01D0,11,XNFPH) computes.

    Parameters
    ----------
    xnf_h : np.ndarray
        Total neutral hydrogen number density (atoms/cm³), shape (n_layers,)
        This is what POPS(1.00D0,12,XNFH) returns.
    temperature : np.ndarray
        Temperature in Kelvin, shape (n_layers,)

    Returns
    -------
    xnfph : np.ndarray
        Ground-state hydrogen number density (atoms/cm³), shape (n_layers,)
        This is what POPS(1.01D0,11,XNFPH)(:,1) returns.
    """
    partition_func = compute_hydrogen_partition_function(temperature)
    # CRITICAL: Divide by partition function (not multiply by ground-state fraction!)
    # XNFPH = XNFH / U(T), matching Fortran's PFSAHA MODE=11 vs MODE=12
    return xnf_h / partition_func


# Default IFOP values from Fortran atlas7v.for line 2822:
# DATA IFOP/1,1,1,1,1,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0/
# Index:     1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
# IFOP(4)=1: HRAYOP (H Rayleigh)  - ENABLED by default
# IFOP(8)=0: HERAOP (He Rayleigh) - DISABLED by default
# IFOP(9)=1: COOLOP (C1,Mg1,Al1,Si1,Fe1 + molecules) - ENABLED by default
# IFOP(10)=0: LUKEOP (N1,O1,Mg2,Si2,Ca2) - DISABLED by default
# IFOP(11)=0: HOTOP (hot star opacities) - DISABLED by default
# IFOP(13)=0: H2RAOP (H2 Rayleigh) - DISABLED by default
DEFAULT_IFOP = [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]


def get_ifop_from_physics(
    atmosphere: "AtmosphereModel", match_fortran: bool = True
) -> list[int]:
    """
    Determine IFOP flags based on physical conditions of the atmosphere.

    IFOP flags control which opacity sources are included:
    - IFOP(1-3): Various bound-free opacities
    - IFOP(4): HRAYOP - Hydrogen Rayleigh scattering
    - IFOP(5-7): More bound-free opacities
    - IFOP(8): HERAOP - Helium Rayleigh scattering
    - IFOP(9-12): Various opacities
    - IFOP(13): H2RAOP - Molecular hydrogen Rayleigh scattering
    - IFOP(14-20): Specialized/experimental opacities

    Parameters
    ----------
    atmosphere : AtmosphereModel
        The atmosphere model with temperature and other properties
    match_fortran : bool
        If True (default), use Fortran defaults exactly (for validation).
        If False, use physics-based decisions that may differ from Fortran.

    Returns
    -------
    list[int]
        20-element IFOP array (0=off, 1=on) for each opacity source
    """
    # CRITICAL FIX: Start from Fortran defaults for matching
    # Previously started with [1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,0,0] which had
    # IFOP(8)=1 and IFOP(13)=1 enabled, but Fortran defaults have them disabled!
    # This caused Python to compute extra scattering (SIGHE, SIGH2) that Fortran doesn't.
    ifop = list(DEFAULT_IFOP)  # Copy to avoid modifying the constant

    if match_fortran:
        # Use exact Fortran defaults - do not modify anything
        logger.info(f"Using Fortran default IFOP: {ifop}")
        return ifop

    # Physics-based mode: enable additional opacity sources based on conditions
    # NOTE: This mode will NOT match Fortran exactly but may be more physically accurate
    t_char = float(np.median(atmosphere.temperature))

    # IFOP(4) = HRAYOP (H Rayleigh): Keep as default (enabled)
    # ifop[3] = 1  # Already set in DEFAULT_IFOP

    # IFOP(8) = HERAOP (He Rayleigh): Enable for physics accuracy
    # Fortran default is OFF, but He Rayleigh can be significant
    ifop[7] = 1
    logger.info(f"HERAOP enabled (physics mode): He Rayleigh scattering included")

    # IFOP(13) = H2RAOP (H2 Rayleigh): ON for cool stars where H2 forms
    # H2 becomes significant below ~5000-6000K due to molecular equilibrium
    if t_char < 6000.0:
        ifop[12] = 1
        logger.info(
            f"H2RAOP enabled: T_char={t_char:.0f}K < 6000K (cool star, H2 present)"
        )
    else:
        ifop[12] = 0
        logger.info(
            f"H2RAOP disabled: T_char={t_char:.0f}K >= 6000K (hot star, H2 dissociated)"
        )

    # IFOP(10) = LUKEOP (N1, O1, Mg2, Si2, Ca2): Enable for intermediate/hot stars
    # These opacities are important in the UV for stars with T > 5000K
    if t_char > 4500.0:
        ifop[9] = 1
        logger.info(
            f"LUKEOP enabled: T_char={t_char:.0f}K > 4500K (UV metal opacities)"
        )
    else:
        ifop[9] = 0
        logger.info(f"LUKEOP disabled: T_char={t_char:.0f}K <= 4500K (cool star)")

    # IFOP(11) = HOTOP (hot star free-free + bound-free): Enable for hot stars
    # These opacities are important for T > 8000K where metals are highly ionized
    if t_char > 8000.0:
        ifop[10] = 1
        logger.info(f"HOTOP enabled: T_char={t_char:.0f}K > 8000K (hot star opacities)")
    else:
        ifop[10] = 0
        logger.info(f"HOTOP disabled: T_char={t_char:.0f}K <= 8000K")

    logger.info(f"Physics-based IFOP: {ifop}")
    return ifop


def compute_kapp_continuum(
    atmosphere: "AtmosphereModel",
    freq: np.ndarray,
    atlas_tables: dict[str, np.ndarray],
    ifop: list[int] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ACONT and SIGMAC using full KAPP logic.

    Parameters
    ----------
    atmosphere:
        The atmosphere model with populations (xnfph, xnf_he1, xnf_he2, etc.)
    freq:
        Frequency array in Hz (shape: (nfreq,))
    atlas_tables:
        Dictionary of B-tables (bhyd, bhe1, bhe2, bc1, bmg1, bal1, bsi1, bfe1, etc.)
        Shape: (n_layers, n_levels)
    ifop:
        Opacity flags controlling which opacity sources are included (1-indexed).
        Default: Fortran defaults [1,1,1,1,1,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0]
        IFOP(4)=1: HRAYOP (H Rayleigh)
        IFOP(8)=0: HERAOP (He Rayleigh) - DISABLED by default in Fortran
        IFOP(13)=0: H2RAOP (H2 Rayleigh) - DISABLED by default

    Returns
    -------
    acont:
        Continuum absorption coefficient (shape: (n_layers, nfreq)) in cm²/g
    sigmac:
        Continuum scattering coefficient (shape: (n_layers, nfreq)) in cm²/g
    scont:
        Continuum source function (shape: (n_layers, nfreq)) in erg/s/cm²/Hz/ster
    """
    # Auto-detect IFOP from physics if not provided (Fortran-independent!)
    if ifop is None:
        ifop = get_ifop_from_physics(atmosphere)
    n_layers = atmosphere.layers
    nfreq = freq.size

    logger.info(f"Computing KAPP continuum: {n_layers} layers × {nfreq} frequencies")

    # Initialize arrays
    acont = np.zeros((n_layers, nfreq), dtype=np.float64)
    sigmac = np.zeros((n_layers, nfreq), dtype=np.float64)
    scont = np.zeros((n_layers, nfreq), dtype=np.float64)

    # Compute Planck functions for all frequencies
    temp = np.asarray(atmosphere.temperature, dtype=np.float64)
    pop = getattr(atmosphere, "population_per_ion", None)
    has_pop_grid = (
        pop is not None
        and isinstance(pop, np.ndarray)
        and pop.ndim == 3
        and pop.shape[0] == n_layers
    )
    xnfphe_mode11 = None
    if has_pop_grid and pop.shape[1] > 2 and pop.shape[2] > 1:
        # Fortran XNFPHE from POPS(...,11): He I / He II / He III populations.
        xnfphe_mode11 = np.column_stack([pop[:, 0, 1], pop[:, 1, 1], pop[:, 2, 1]])
    bnu_all = np.zeros((n_layers, nfreq), dtype=np.float64)
    for i, f in enumerate(freq):
        bnu_all[:, i] = _planck_nu(f, temp)

    # Compute frequency-dependent quantities
    wavelength_nm = C_LIGHT_NM / np.maximum(freq, 1e-30)
    # hkt is per layer (shape: (n_layers,))
    hkt = H_PLANCK / (K_BOLTZ * temp)
    # hckt = hkt * c (cm²/s) - used in HE1OP, HE2OP, etc. (atlas7v.for line 81: HCKT(J)=HKT(J)*2.99792458D10)
    hckt = hkt * C_LIGHT_CM
    # ehvkt and stim are per layer and frequency (shape: (n_layers, nfreq))
    ehvkt = np.exp(-H_PLANCK * freq[None, :] / (K_BOLTZ * temp[:, None]))
    stim = 1.0 - ehvkt
    waveno = freq / C_LIGHT_CM

    # Initialize component arrays
    ahyd = np.zeros((n_layers, nfreq), dtype=np.float64)
    ahmin = np.zeros((n_layers, nfreq), dtype=np.float64)
    ah2p = np.zeros((n_layers, nfreq), dtype=np.float64)
    ahe1 = np.zeros((n_layers, nfreq), dtype=np.float64)
    ahe2 = np.zeros((n_layers, nfreq), dtype=np.float64)
    ahemin = np.zeros((n_layers, nfreq), dtype=np.float64)
    ac1 = np.zeros((n_layers, nfreq), dtype=np.float64)
    amg1 = np.zeros((n_layers, nfreq), dtype=np.float64)
    aal1 = np.zeros((n_layers, nfreq), dtype=np.float64)
    asi1 = np.zeros((n_layers, nfreq), dtype=np.float64)
    afe1 = np.zeros((n_layers, nfreq), dtype=np.float64)
    acool = np.zeros((n_layers, nfreq), dtype=np.float64)
    aluke = np.zeros((n_layers, nfreq), dtype=np.float64)
    ahot = np.zeros((n_layers, nfreq), dtype=np.float64)
    axcont = np.zeros((n_layers, nfreq), dtype=np.float64)

    shyd = np.zeros((n_layers, nfreq), dtype=np.float64)
    shmin = np.zeros((n_layers, nfreq), dtype=np.float64)
    she1 = np.zeros((n_layers, nfreq), dtype=np.float64)
    she2 = np.zeros((n_layers, nfreq), dtype=np.float64)
    sc1 = np.zeros((n_layers, nfreq), dtype=np.float64)
    smg1 = np.zeros((n_layers, nfreq), dtype=np.float64)
    sal1 = np.zeros((n_layers, nfreq), dtype=np.float64)
    ssi1 = np.zeros((n_layers, nfreq), dtype=np.float64)
    sfe1 = np.zeros((n_layers, nfreq), dtype=np.float64)
    sxcont = np.zeros((n_layers, nfreq), dtype=np.float64)

    # Initialize scattering arrays (will be computed below)
    sigh = np.zeros((n_layers, nfreq), dtype=np.float64)
    sighe = np.zeros((n_layers, nfreq), dtype=np.float64)
    sigel = np.zeros((n_layers, nfreq), dtype=np.float64)
    sigh2 = np.zeros((n_layers, nfreq), dtype=np.float64)
    sigx = np.zeros((n_layers, nfreq), dtype=np.float64)

    rho = np.maximum(np.asarray(atmosphere.mass_density, dtype=np.float64), 1e-30)

    # Define xne at function scope (Fortran has it in COMMON, always accessible)
    if atmosphere.electron_density is not None:
        xne = np.asarray(atmosphere.electron_density, dtype=np.float64)
    else:
        xne = np.zeros(n_layers, dtype=np.float64)

    # HOP: Hydrogen opacity (atlas7v.for line 4596)
    if atmosphere.xnfph is not None:
        logger.info("Computing HOP (hydrogen opacity)...")
        xnfph = np.asarray(atmosphere.xnfph, dtype=np.float64)
        bhyd = atlas_tables.get("bhyd", np.ones((n_layers, 8), dtype=np.float64))

        xne = np.asarray(atmosphere.electron_density, dtype=np.float64)

        for j in range(nfreq):
            f = freq[j]
            freq3 = 2.815e29 / (f * f * f)
            wno = waveno[j]
            bnu_j = bnu_all[:, j]
            ehvkt_j = ehvkt[:, j]
            stim_j = stim[:, j]

            # H continuum computation (matching atlas7v.for HOP)
            # CRITICAL FIX: Fortran uses HCKT = HKT * C_LIGHT_CM in hydrogen opacity
            # From atlas7v_1.for line 1933: HCKT(J)=HKT(J)*2.99792458D10
            # And line 4045 uses: 109677.576D0*HCKT(J)
            hckt = hkt * C_LIGHT_CM
            # N=16 to infinity
            h = (
                freq3
                * 2.0
                / 2.0
                / (RYDBERG_CM * hckt)
                * (
                    np.exp(-np.maximum(109250.336, 109678.764 - wno) * hckt)
                    - np.exp(-109678.764 * hckt)
                )
                * stim_j
            )

            s = h * bnu_j

            # N=1 to 15 (add bound-free contributions)
            # For N=1-6, use BHYD departure coefficients
            # For N=7-15, use standard formula

            # N=15
            if wno >= 487.456:
                x = xkarsas(f, 1.0, 15, 15)
                a = x * 450.0 * np.exp(-109191.313 * hckt) * stim_j
                h = h + a
                s = s + a * bnu_j

            # N=14
            if wno >= 559.579:
                x = xkarsas(f, 1.0, 14, 14)
                a = x * 392.0 * np.exp(-109119.188 * hckt) * stim_j
                h = h + a
                s = s + a * bnu_j

            # N=13
            if wno >= 648.980:
                x = xkarsas(f, 1.0, 13, 13)
                a = x * 338.0 * np.exp(-109029.789 * hckt) * stim_j
                h = h + a
                s = s + a * bnu_j

            # N=12
            if wno >= 761.649:
                x = xkarsas(f, 1.0, 12, 12)
                a = x * 288.0 * np.exp(-108917.117 * hckt) * stim_j
                h = h + a
                s = s + a * bnu_j

            # N=11
            if wno >= 906.426:
                x = xkarsas(f, 1.0, 11, 11)
                a = x * 242.0 * np.exp(-108772.336 * hckt) * stim_j
                h = h + a
                s = s + a * bnu_j

            # N=10
            if wno >= 1096.776:
                x = xkarsas(f, 1.0, 10, 10)
                a = x * 200.0 * np.exp(-108581.992 * hckt) * stim_j
                h = h + a
                s = s + a * bnu_j

            # N=9
            if wno >= 1354.044:
                x = xkarsas(f, 1.0, 9, 9)
                a = x * 162.0 * np.exp(-108324.719 * hckt) * stim_j
                h = h + a
                s = s + a * bnu_j

            # N=8
            if wno >= 1713.713:
                x = xkarsas(f, 1.0, 8, 8)
                a = x * 128.0 * np.exp(-107965.051 * hckt) * stim_j
                h = h + a
                s = s + a * bnu_j

            # N=7
            if wno >= 2238.320:
                x = xkarsas(f, 1.0, 7, 7)
                a = x * 98.0 * np.exp(-107440.444 * hckt) * stim_j
                h = h + a
                s = s + a * bnu_j

            # N=6 (uses BHYD departure coefficient)
            if wno >= 3046.604:
                x = xkarsas(f, 1.0, 6, 6)
                bhyd_6 = (
                    bhyd[:, 5]
                    if bhyd.shape[1] > 5
                    else np.ones(n_layers, dtype=np.float64)
                )
                a = x * 72.0 * np.exp(-106632.160 * hckt) * (bhyd_6 - ehvkt_j)
                h = h + a
                s = s + a * bnu_j * stim_j / np.maximum(bhyd_6 - ehvkt_j, 1e-40)

            # N=5
            if wno >= 4387.113:
                x = xkarsas(f, 1.0, 5, 5)
                bhyd_5 = (
                    bhyd[:, 4]
                    if bhyd.shape[1] > 4
                    else np.ones(n_layers, dtype=np.float64)
                )
                a = x * 50.0 * np.exp(-105291.651 * hckt) * (bhyd_5 - ehvkt_j)
                h = h + a
                s = s + a * bnu_j * stim_j / np.maximum(bhyd_5 - ehvkt_j, 1e-40)

            # N=4
            if wno >= 6854.871:
                x = xkarsas(f, 1.0, 4, 4)
                bhyd_4 = (
                    bhyd[:, 3]
                    if bhyd.shape[1] > 3
                    else np.ones(n_layers, dtype=np.float64)
                )
                a = x * 32.0 * np.exp(-102823.893 * hckt) * (bhyd_4 - ehvkt_j)
                h = h + a
                s = s + a * bnu_j * stim_j / np.maximum(bhyd_4 - ehvkt_j, 1e-40)

            # N=3
            if wno >= 12186.462:
                x = xkarsas(f, 1.0, 3, 3)
                bhyd_3 = (
                    bhyd[:, 2]
                    if bhyd.shape[1] > 2
                    else np.ones(n_layers, dtype=np.float64)
                )
                a = x * 18.0 * np.exp(-97492.302 * hckt) * (bhyd_3 - ehvkt_j)
                h = h + a
                s = s + a * bnu_j * stim_j / np.maximum(bhyd_3 - ehvkt_j, 1e-40)

            # N=2
            if wno >= 27419.659:
                x = xkarsas(f, 1.0, 2, 2)
                bhyd_2 = (
                    bhyd[:, 1]
                    if bhyd.shape[1] > 1
                    else np.ones(n_layers, dtype=np.float64)
                )
                a = x * 8.0 * np.exp(-82259.105 * hckt) * (bhyd_2 - ehvkt_j)
                h = h + a
                s = s + a * bnu_j * stim_j / np.maximum(bhyd_2 - ehvkt_j, 1e-40)

            # N=1
            if wno >= 109678.764:
                x = xkarsas(f, 1.0, 1, 1)
                bhyd_1 = (
                    bhyd[:, 0]
                    if bhyd.shape[1] > 0
                    else np.ones(n_layers, dtype=np.float64)
                )
                a = x * 2.0 * 1.0 * (bhyd_1 - ehvkt_j)
                h = h + a
                s = s + a * bnu_j * stim_j / np.maximum(bhyd_1 - ehvkt_j, 1e-40)

            # Multiply by populations and normalize by density
            # H=H*XNFPH(J,1)/RHO(J)  (atlas7v.for line 4706)
            if xnfph.shape[1] > 0:
                xnfph1 = xnfph[:, 0]
                h = h * xnfph1 / rho
                s = s * xnfph1 / rho
            else:
                h = h / rho
                s = s / rho

            # Free-free contribution (atlas7v.for line 4709-4711)
            # A=3.6919E8/SQRT(T(J))*COULFF(J,1)/FREQ*XNE(J)/FREQ*XNFPH(J,2)/FREQ*STIM(J)/RHO(J)
            freqlg = np.log(f)
            tlog_arr = np.log(np.maximum(temp, 1e-10))
            coulff_arr = np.array(
                [
                    _coulff(j_idx, 1, f, freqlg, temp, tlog_arr)
                    for j_idx in range(n_layers)
                ]
            )

            if xnfph.shape[1] > 1:
                xnfph2 = xnfph[:, 1]
                a_ff = (
                    3.6919e8
                    / np.sqrt(temp)
                    * coulff_arr
                    / f
                    * xne
                    / f
                    * xnfph2
                    / f
                    * stim_j
                    / rho
                )
            else:
                a_ff = (
                    3.6919e8
                    / np.sqrt(temp)
                    * coulff_arr
                    / f
                    * xne
                    / f
                    / f
                    * stim_j
                    / rho
                )

            h = h + a_ff
            s = s + a_ff * bnu_j

            ahyd[:, j] = h
            shyd[:, j] = np.where(h > 0, s / h, bnu_j)

    # H2PLOP: H2+ opacity (atlas7v.for line 5189-5211)
    if atmosphere.xnfph is not None:
        logger.info("Computing H2PLOP (H2+ opacity)...")
        xnfph_arr = np.asarray(atmosphere.xnfph, dtype=np.float64)
        bhyd = atlas_tables.get("bhyd", np.ones((n_layers, 8), dtype=np.float64))
        tkev = np.asarray(atmosphere.temperature, dtype=np.float64) * KBOLTZ_EV

        for j in range(nfreq):
            f = freq[j]
            if f > 3.28805e15:
                continue
            wno = waveno[j]
            freqlg = np.log(f)
            freq15 = f / 1.0e15

            # FR = polynomial in FREQLG (atlas7v.for line 5200-5201)
            fr = (
                -3.0233e3
                + (
                    3.7797e2
                    + (-1.82496e1 + (3.9207e-1 - 3.1672e-3 * freqlg) * freqlg) * freqlg
                )
                * freqlg
            )

            # ES = polynomial in FREQ15 (atlas7v.for line 5203-5204)
            es = (
                -7.342e-3
                + (
                    -2.409e0
                    + (
                        1.028e0
                        + (-4.230e-1 + (1.224e-1 - 1.351e-2 * freq15) * freq15) * freq15
                    )
                    * freq15
                )
                * freq15
            )

            # AH2P = EXP(-ES/TKEV + FR + LOG(XNFPH(J,1))) * 2. * BHYD(J,1) * XNFPH(J,2) / RHO(J) * STIM(J)
            if xnfph_arr.shape[1] >= 2:
                xnfph1 = xnfph_arr[:, 0]
                xnfph2 = xnfph_arr[:, 1]
                bhyd1 = (
                    bhyd[:, 0]
                    if bhyd.shape[1] > 0
                    else np.ones(n_layers, dtype=np.float64)
                )
                stim_j = stim[:, j]

                ah2p_val = (
                    np.exp(-es / tkev + fr + np.log(np.maximum(xnfph1, 1e-40)))
                    * 2.0
                    * bhyd1
                    * xnfph2
                    / rho
                    * stim_j
                )
                ah2p[:, j] = ah2p_val

    # HE1OP: Helium I opacity (atlas7v.for line 5499-5704)
    if atmosphere.xnf_he1 is not None:
        logger.info("Computing HE1OP (Helium I opacity)...")
        if xnfphe_mode11 is not None:
            xnfphe = xnfphe_mode11
        else:
            xnfphe = np.asarray(atmosphere.xnf_he1, dtype=np.float64)
            if xnfphe.ndim == 1:
                xnfphe = xnfphe[:, np.newaxis]  # fallback only

        # XNFHE is POPS(...,12): mode-12 helium populations used in HE1 free-free term.
        if not hasattr(atmosphere, "xnf_he2") or atmosphere.xnf_he2 is None:
            raise ValueError(
                "Atmosphere model missing He II populations required by KAPP"
            )
        he1_mode12 = np.asarray(atmosphere.xnf_he1, dtype=np.float64)
        he2_mode12 = np.asarray(atmosphere.xnf_he2, dtype=np.float64)
        if he1_mode12.ndim > 1:
            he1_mode12 = he1_mode12[:, 0]
        if he2_mode12.ndim > 1:
            he2_mode12 = he2_mode12[:, 0]
        xnfhe = np.column_stack([he1_mode12, he2_mode12])

        bhe1 = atlas_tables.get("bhe1", np.ones((n_layers, 29), dtype=np.float64))
        bhe2 = atlas_tables.get("bhe2", np.ones((n_layers, 6), dtype=np.float64))

        for j in range(nfreq):
            f = freq[j]
            wno = waveno[j]
            freqlg = np.log(f)
            freq3 = 2.815e29 / (f * f * f)
            bnu_j = bnu_all[:, j]
            ehvkt_j = ehvkt[:, j]
            stim_j = stim[:, j]
            wl_nm = C_LIGHT_NM / f

            # N=6 to infinity (atlas7v.for line 5513-5516)
            # CRITICAL FIX: Fortran uses HCKT (cm²/s), not HKT (cm)!
            # HCKT = HKT * C_LIGHT_CM (atlas7v.for line 81)
            rydberg_he = 109722.267
            h = (
                freq3
                * 4.0
                / 2.0
                / (rydberg_he * hckt)
                * (
                    np.exp(-np.maximum(195262.919, 198310.76 - wno) * hckt)
                    - np.exp(-198310.76 * hckt)
                )
                * stim_j
                * (bhe2[:, 0] if bhe2.shape[1] > 0 else np.ones(n_layers))
            )
            s = h * bnu_j

            # Add bound-free contributions (N=5 down to N=1) - all 29 levels
            # BHE1 indices: 29 (5P 1P) down to 1 (1S 1S), BHE2 index: 1 (He II ground state)
            bhe2_1 = bhe2[:, 0] if bhe2.shape[1] > 0 else np.ones(n_layers)

            # N=5 levels (BHE1 indices 29 down to 20)
            if wno >= 4368.190 and bhe1.shape[1] > 28:  # 5P 1P
                x = freq3 / 3125.0
                a = (
                    x
                    * 3.0
                    * np.exp(-193942.57 * hckt)
                    * (bhe1[:, 28] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 28] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 4388.260 and bhe1.shape[1] > 27:  # 5G 1G
                x = freq3 / 3125.0
                a = (
                    x
                    * 9.0
                    * np.exp(-193922.5 * hckt)
                    * (bhe1[:, 27] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 27] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 4388.260 and bhe1.shape[1] > 26:  # 5G 3G
                x = freq3 / 3125.0
                a = (
                    x
                    * 27.0
                    * np.exp(-193922.5 * hckt)
                    * (bhe1[:, 26] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 26] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 4389.390 and bhe1.shape[1] > 25:  # 5F 1F
                x = freq3 / 3125.0
                a = (
                    x
                    * 7.0
                    * np.exp(-193921.37 * hckt)
                    * (bhe1[:, 25] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 25] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 4389.450 and bhe1.shape[1] > 24:  # 5F 3F
                x = freq3 / 3125.0
                a = (
                    x
                    * 15.0
                    * np.exp(-193921.31 * hckt)
                    * (bhe1[:, 24] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 24] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 4392.369 and bhe1.shape[1] > 23:  # 5D 1D
                x = freq3 / 3125.0
                a = (
                    x
                    * 5.0
                    * np.exp(-193918.391 * hckt)
                    * (bhe1[:, 23] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 23] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 4393.515 and bhe1.shape[1] > 22:  # 5D 3D
                x = freq3 / 3125.0
                a = (
                    x
                    * 15.0
                    * np.exp(-193917.245 * hckt)
                    * (bhe1[:, 22] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 22] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 4509.980 and bhe1.shape[1] > 21:  # 5P 3P
                x = freq3 / 3125.0
                a = (
                    x
                    * 9.0
                    * np.exp(-193800.78 * hckt)
                    * (bhe1[:, 21] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 21] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 4647.133 and bhe1.shape[1] > 20:  # 5S 1S
                x = freq3 / 3125.0
                a = (
                    x
                    * 1.0
                    * np.exp(-193663.627 * hckt)
                    * (bhe1[:, 20] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 20] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 4963.671 and bhe1.shape[1] > 19:  # 5S 3S
                x = freq3 / 3125.0
                a = (
                    x
                    * 3.0
                    * np.exp(-193347.089 * hckt)
                    * (bhe1[:, 19] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 19] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # N=4 levels (BHE1 indices 19 down to 12)
            if wno >= 6817.943 and bhe1.shape[1] > 18:  # 4P 1P
                x = freq3 / 1024.0
                a = (
                    x
                    * 3.0
                    * np.exp(-191492.817 * hckt)
                    * (bhe1[:, 18] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 18] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 6858.680 and bhe1.shape[1] > 17:  # 4F 1F
                x = freq3 / 1024.0
                a = (
                    x
                    * 7.0
                    * np.exp(-191452.08 * hckt)
                    * (bhe1[:, 17] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 17] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 6858.960 and bhe1.shape[1] > 16:  # 4F 3F
                x = freq3 / 1024.0
                a = (
                    x
                    * 21.0
                    * np.exp(-191451.80 * hckt)
                    * (bhe1[:, 16] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 16] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 6864.201 and bhe1.shape[1] > 15:  # 4D 1D
                x = freq3 / 1024.0
                a = (
                    x
                    * 5.0
                    * np.exp(-191446.559 * hckt)
                    * (bhe1[:, 15] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 15] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 6866.172 and bhe1.shape[1] > 14:  # 4D 3D
                x = freq3 / 1024.0
                a = (
                    x
                    * 15.0
                    * np.exp(-191444.588 * hckt)
                    * (bhe1[:, 14] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 14] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 7093.620 and bhe1.shape[1] > 13:  # 4P 3P
                x = freq3 / 1024.0
                a = (
                    x
                    * 9.0
                    * np.exp(-191217.14 * hckt)
                    * (bhe1[:, 13] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 13] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 7370.429 and bhe1.shape[1] > 12:  # 4S 1S
                x = freq3 / 1024.0
                a = (
                    x
                    * 1.0
                    * np.exp(-190940.331 * hckt)
                    * (bhe1[:, 12] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 12] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 8012.550 and bhe1.shape[1] > 11:  # 4S 3S
                x = freq3 / 1024.0
                a = (
                    x
                    * 3.0
                    * np.exp(-190298.210 * hckt)
                    * (bhe1[:, 11] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 11] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # N=3 levels (BHE1 indices 11 down to 6)
            if wno >= 12101.289 and bhe1.shape[1] > 10:  # 3P 1P
                x = np.exp(58.81 - 2.89 * freqlg)
                a = (
                    x
                    * 3.0
                    * np.exp(-186209.471 * hckt)
                    * (bhe1[:, 10] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 10] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 12205.695 and bhe1.shape[1] > 9:  # 3D 1D
                x = np.exp(85.20 - 3.69 * freqlg)
                a = (
                    x
                    * 5.0
                    * np.exp(-186105.065 * hckt)
                    * (bhe1[:, 9] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 9] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 12209.106 and bhe1.shape[1] > 8:  # 3D 3D
                x = np.exp(85.20 - 3.69 * freqlg)
                a = (
                    x
                    * 15.0
                    * np.exp(-186101.654 * hckt)
                    * (bhe1[:, 8] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 8] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 12746.066 and bhe1.shape[1] > 7:  # 3P 3P
                x = np.exp(49.30 - 2.60 * freqlg)
                a = (
                    x
                    * 9.0
                    * np.exp(-185564.694 * hckt)
                    * (bhe1[:, 7] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 7] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 13445.824 and bhe1.shape[1] > 6:  # 3S 1S
                x = np.exp(23.85 - 1.86 * freqlg)
                a = (
                    x
                    * 1.0
                    * np.exp(-184864.936 * hckt)
                    * (bhe1[:, 6] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 6] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 15073.868 and bhe1.shape[1] > 5:  # 3S 3S
                x = np.exp(12.69 - 1.54 * freqlg)
                a = (
                    x
                    * 3.0
                    * np.exp(-183236.892 * hckt)
                    * (bhe1[:, 5] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 5] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # N=2 levels (BHE1 indices 5 down to 2)
            if wno >= 27175.760 and bhe1.shape[1] > 4:  # 2P 1P
                x = np.exp(81.35 - 3.5 * freqlg)
                a = (
                    x
                    * 3.0
                    * np.exp(-171135.000 * hckt)
                    * (bhe1[:, 4] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 4] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 29223.753 and bhe1.shape[1] > 3:  # 2P 3P
                x = np.exp(61.21 - 2.9 * freqlg)
                a = (
                    x
                    * 9.0
                    * np.exp(-169087.007 * hckt)
                    * (bhe1[:, 3] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 3] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 32033.214 and bhe1.shape[1] > 2:  # 2S 1S
                x = np.exp(26.83 - 1.91 * freqlg)
                a = (
                    x
                    * 1.0
                    * np.exp(-166277.546 * hckt)
                    * (bhe1[:, 2] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 2] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            if wno >= 38454.691 and bhe1.shape[1] > 1:  # 2S 3S
                x = np.exp(-390.026 + (21.035 - 0.318 * freqlg) * freqlg)
                a = (
                    x
                    * 3.0
                    * np.exp(-159856.069 * hckt)
                    * (bhe1[:, 1] - bhe2_1 * ehvkt_j)
                )
                h = h + a
                denom = bhe1[:, 1] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # N=1 level (BHE1 index 1)
            if wno >= 198310.760 and bhe1.shape[1] > 0:  # 1S 1S
                x = np.exp(33.32 - 2.0 * freqlg)
                a = x * 1.0 * 1.0 * (bhe1[:, 0] - bhe2_1 * ehvkt_j)
                h = h + a
                denom = bhe1[:, 0] / np.maximum(bhe2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # Multiply by populations and normalize (atlas7v.for line 5691-5692)
            if xnfphe.shape[1] > 0:
                xnfphe1 = xnfphe[:, 0]
                h = h * xnfphe1 / rho
                s = s * xnfphe1 / rho

            # Free-free contribution (atlas7v.for line 5694-5697)
            freqlg_arr = np.full(n_layers, freqlg)
            tlog_arr = np.log(np.maximum(temp, 1e-10))
            coulff_arr = np.array(
                [
                    _coulff(j_idx, 1, f, freqlg, temp, tlog_arr)
                    for j_idx in range(n_layers)
                ]
            )

            if xnfhe.shape[1] > 1:
                xnfhe2 = xnfhe[:, 1]
                a_ff = (
                    3.619e8
                    / np.sqrt(temp)
                    * coulff_arr
                    / f
                    * xne
                    / f
                    * xnfhe2
                    / f
                    * stim_j
                    / rho
                )
            else:
                a_ff = (
                    3.619e8
                    / np.sqrt(temp)
                    * coulff_arr
                    / f
                    * xne
                    / f
                    / f
                    * stim_j
                    / rho
                )

            h = h + a_ff
            s = s + a_ff * bnu_j

            ahe1[:, j] = h
            she1[:, j] = np.where(h > 0, s / h, bnu_j)

    # HE2OP: Helium II opacity (atlas7v.for line 5705-5793)
    if atmosphere.xnf_he2 is not None:
        logger.info("Computing HE2OP (Helium II opacity)...")
        if xnfphe_mode11 is not None:
            xnfphe = xnfphe_mode11
        else:
            xnfphe = np.asarray(atmosphere.xnf_he2, dtype=np.float64)
            if xnfphe.ndim == 1:
                xnfphe = np.column_stack([np.zeros(n_layers), xnfphe, np.zeros(n_layers)])
            elif xnfphe.shape[1] == 1:
                xnfphe = np.column_stack([np.zeros(n_layers), xnfphe[:, 0], np.zeros(n_layers)])
            elif xnfphe.shape[1] == 2:
                xnfphe = np.column_stack([np.zeros(n_layers), xnfphe[:, 0], xnfphe[:, 1]])
        xnfphe3 = (
            xnfphe[:, 2]
            if xnfphe.shape[1] > 2
            else np.zeros(n_layers, dtype=np.float64)
        )

        bhe2 = atlas_tables.get("bhe2", np.ones((n_layers, 6), dtype=np.float64))

        for j in range(nfreq):
            f = freq[j]
            wno = waveno[j]
            freq3 = 2.815e29 / (f * f * f)
            bnu_j = bnu_all[:, j]
            ehvkt_j = ehvkt[:, j]
            stim_j = stim[:, j]

            # XNFPRHO = XNFPHE(J,2)/RHO(J) (atlas7v.for line 5719)
            xnfprho = (
                xnfphe[:, 1] if xnfphe.shape[1] > 1 else np.zeros(n_layers)
            ) / rho

            # N=10 to infinity (atlas7v.for line 5720-5723)
            rydberg_he2 = 438889.068
            h = (
                freq3
                * 16.0
                * 2.0
                / 2.0
                / (rydberg_he2 * hckt)
                * (
                    np.exp(-np.maximum(434519.959, 438908.85 - wno) * hckt)
                    - np.exp(-438908.85 * hckt)
                )
                * stim_j
                * xnfprho
            )
            s = h * bnu_j

            # Add bound-free contributions (N=9 down to N=1)
            if wno >= 5418.390:  # N=9
                x = freq3 / 59049.0 * 16.0
                a = x * 162.0 * np.exp(-433490.46 * hckt) * stim_j * xnfprho
                h = h + a
                s = s + a * bnu_j

            if wno >= 6857.660:  # N=8
                x = freq3 * 16.0 / 32768.0
                a = x * 128.0 * np.exp(-432051.19 * hckt) * stim_j * xnfprho
                h = h + a
                s = s + a * bnu_j

            if wno >= 8956.950:  # N=7
                x = freq3 * 16.0 / 16807.0
                a = x * 98.0 * np.exp(-429951.90 * hckt) * stim_j * xnfprho
                h = h + a
                s = s + a * bnu_j

            if wno >= 12191.437:  # N=6
                bhe2_6 = bhe2[:, 5] if bhe2.shape[1] > 5 else np.ones(n_layers)
                x = freq3 * 16.0 / 7776.0 * (1.0986 + (-2.704e13 + 1.229e27 / f) / f)
                a = x * 72.0 * np.exp(-426717.413 * hckt) * (bhe2_6 - ehvkt_j) * xnfprho
                h = h + a
                s = s + a * bnu_j * stim_j / np.maximum(bhe2_6 - ehvkt_j, 1e-40)

            if wno >= 17555.715:  # N=5
                bhe2_5 = bhe2[:, 4] if bhe2.shape[1] > 4 else np.ones(n_layers)
                x = freq3 * 16.0 / 3125.0 * (1.102 + (-3.909e13 + 2.371e27 / f) / f)
                a = x * 50.0 * np.exp(-421353.135 * hckt) * (bhe2_5 - ehvkt_j) * xnfprho
                h = h + a
                s = s + a * bnu_j * stim_j / np.maximum(bhe2_5 - ehvkt_j, 1e-40)

            if wno >= 27430.925:  # N=4
                bhe2_4 = bhe2[:, 3] if bhe2.shape[1] > 3 else np.ones(n_layers)
                x = freq3 * 16.0 / 1024.0 * (1.101 + (-5.765e13 + 4.593e27 / f) / f)
                a = x * 32.0 * np.exp(-411477.925 * hckt) * (bhe2_4 - ehvkt_j) * xnfprho
                h = h + a
                s = s + a * bnu_j * stim_j / np.maximum(bhe2_4 - ehvkt_j, 1e-40)

            if wno >= 48766.491:  # N=3
                bhe2_3 = bhe2[:, 2] if bhe2.shape[1] > 2 else np.ones(n_layers)
                x = freq3 * 16.0 / 243.0 * (1.101 + (-9.863e13 + 1.035e28 / f) / f)
                a = x * 18.0 * np.exp(-390142.359 * hckt) * (bhe2_3 - ehvkt_j) * xnfprho
                h = h + a
                s = s + a * bnu_j * stim_j / np.maximum(bhe2_3 - ehvkt_j, 1e-40)

            if wno >= 109726.529:  # N=2
                bhe2_2 = bhe2[:, 1] if bhe2.shape[1] > 1 else np.ones(n_layers)
                x = freq3 * 16.0 / 32.0 * (1.105 + (-2.375e14 + 4.077e28 / f) / f)
                a = x * 8.0 * np.exp(-329182.321 * hckt) * (bhe2_2 - ehvkt_j) * xnfprho
                h = h + a
                s = s + a * bnu_j * stim_j / np.maximum(bhe2_2 - ehvkt_j, 1e-40)

            if wno >= 438908.850:  # N=1
                bhe2_1 = bhe2[:, 0] if bhe2.shape[1] > 0 else np.ones(n_layers)
                x = freq3 * 16.0 / 1.0 * (0.9916 + (2.719e13 - 2.268e30 / f) / f)
                a = x * 2.0 * 1.0 * (bhe2_1 - ehvkt_j) * xnfprho
                h = h + a
                s = s + a * bnu_j * stim_j / np.maximum(bhe2_1 - ehvkt_j, 1e-40)

            # Free-free contribution (atlas7v.for line 5783-5786)
            freqlg_arr = np.full(n_layers, freqlg)
            tlog_arr = np.log(np.maximum(temp, 1e-10))
            coulff_arr = np.array(
                [
                    _coulff(j_idx, 2, f, freqlg, temp, tlog_arr)
                    for j_idx in range(n_layers)
                ]
            )

            a_ff = (
                3.6919e8
                * 4.0
                / np.sqrt(temp)
                * coulff_arr
                / f
                * xne
                / f
                * xnfphe3
                / f
                * stim_j
                / rho
            )
            h = h + a_ff
            s = s + a_ff * bnu_j

            ahe2[:, j] = h
            she2[:, j] = np.where(h > 0, s / h, bnu_j)

    # HEMIOP: He- opacity (atlas7v.for line 7296-7318)
    # Fortran evidence:
    #   AHEMIN(J)=(A*T(J)+B+C/T(J))/1.D15*XNE(J)/1.D15*XNFPHE(J,1)/1.D15/RHO(J)
    # where A, B, C are frequency-dependent polynomials in 1/FREQ.
    if ifop[6] == 1 and atmosphere.xnf_he1 is not None and atmosphere.electron_density is not None:
        logger.info("Computing HEMIOP (He- opacity)...")
        xnfphe = np.asarray(atmosphere.xnf_he1, dtype=np.float64)
        if xnfphe.ndim == 1:
            xnfphe = xnfphe[:, np.newaxis]
        xnfphe1 = xnfphe[:, 0] if xnfphe.shape[1] > 0 else np.ones(n_layers)
        xne = np.asarray(atmosphere.electron_density, dtype=np.float64)

        for j in range(nfreq):
            f = freq[j]
            a_coeff = 3.397e-01 + (-5.216e14 + 7.039e30 / f) / f
            b_coeff = -4.116e03 + (1.067e19 + 8.135e34 / f) / f
            c_coeff = 5.081e08 + (-8.724e22 - 5.659e37 / f) / f
            ahemin[:, j] = (
                (a_coeff * temp + b_coeff + c_coeff / temp)
                / 1.0e15
                * xne
                / 1.0e15
                * xnfphe1
                / 1.0e15
                / rho
            )

    # TODO: Complete HE1OP implementation (all 29 levels)
    # HMINOP: H- opacity (atlas7v.for line 5212-5316)
    if atmosphere.xnfph is not None and atmosphere.electron_density is not None:
        logger.info("Computing HMINOP (H- opacity)...")
        xnfph_arr = np.asarray(atmosphere.xnfph, dtype=np.float64)
        xne = np.asarray(atmosphere.electron_density, dtype=np.float64)
        bhyd = atlas_tables.get("bhyd", np.ones((n_layers, 8), dtype=np.float64))
        bmin = atlas_tables.get("bmin", np.ones((n_layers,), dtype=np.float64))
        tkev = temp * KBOLTZ_EV

        # Pre-compute XHMIN (atlas7v.for line 5298-5299) - per layer, not per frequency
        # Fortran uses XNFPH(J,1) from POPS (mode=11). Prefer the explicit XNFPH array.
        if xnfph_arr.shape[1] > 0:
            xnfph1 = xnfph_arr[:, 0]
        elif atmosphere.xnf_h is not None:
            xnfph1 = compute_ground_state_hydrogen(
                np.asarray(atmosphere.xnf_h, dtype=np.float64), temp
            )
        else:
            xnfph1 = np.ones(n_layers)
        bhyd1 = bhyd[:, 0] if bhyd.shape[1] > 0 else np.ones(n_layers)
        xhmin = (
            np.exp(0.754209 / tkev)
            / (2.0 * 2.4148e15 * temp * np.sqrt(temp))
            * bmin
            * bhyd1
            * xnfph1
            * xne
        )

        # Pre-compute THETA = 5040/T (atlas7v.for line 5296) - per layer
        theta = 5040.0 / temp

        # Pre-compute WFFLOG = log(91.134/WAVEK) (atlas7v.for line 5284)
        wfflog = np.log(91.134 / HMINOP_WAVEK)

        # Pre-compute FFLOG (atlas7v.for line 5290-5291) - once for all frequencies
        nwavek = HMINOP_WAVEK.size
        nthetaff = HMINOP_THETAFF.size
        # FF is (11, 22) array: first 11 columns from FFBEG, last 11 from FFEND
        ff_full = np.zeros((nthetaff, 22), dtype=np.float64)
        for it in range(nthetaff):
            for iw in range(22):
                # Fortran tables are column-major; index as [iw, it] to match.
                if iw < 11:
                    ff_full[it, iw] = HMINOP_FFBEG[iw, it]
                else:
                    ff_full[it, iw] = HMINOP_FFEND[iw - 11, it]

        # Pre-compute FFLOG = log(FF/THETAFF * 5040 * K_BOLTZ)
        fflog = np.zeros((22, nthetaff), dtype=np.float64)
        for iw in range(22):
            for it in range(nthetaff):
                ff_val = ff_full[it, iw]
                fflog[iw, it] = np.log(ff_val / HMINOP_THETAFF[it] * 5040.0 * K_BOLTZ)

        for j in range(nfreq):
            f = freq[j]
            wno = waveno[j]
            bnu_j = bnu_all[:, j]
            ehvkt_j = ehvkt[:, j]
            stim_j = stim[:, j]

            # WAVE = 2.99792458e17 / FREQ (atlas7v.for line 5300)
            wave = 2.99792458e17 / f  # wavelength in nm
            # WAVELOG matches Fortran exactly - log of wavelength in nm
            # The WFFLOG array uses log(91.134/WAVEK) where WAVEK = 91.134/wavelength_nm
            # So WFFLOG = log(91.134 / (91.134/wave_nm)) = log(wave_nm)
            wavelog = np.log(wave)

            # Interpolate FFLOG to get FFTT for each THETA (atlas7v.for line 5302-5304)
            fftheta = np.zeros((n_layers,), dtype=np.float64)
            for layer_idx in range(n_layers):
                # For each THETA, interpolate FFLOG over wavelength
                fftt_for_theta = np.zeros((nthetaff,), dtype=np.float64)
                for it in range(nthetaff):
                    # Interpolate FFLOG over WFFLOG (wavelength dimension)
                    fftt_val = _linter(wfflog, fflog[:, it], np.array([wavelog]))[0]
                    fftt_for_theta[it] = np.exp(fftt_val)

                # Interpolate FFTT over THETA (atlas7v.for line 5308)
                fftheta[layer_idx] = _linter(
                    HMINOP_THETAFF, fftt_for_theta, np.array([theta[layer_idx]])
                )[0]

            # HMINBF from MAP1 (atlas7v.for line 5306)
            hminbf = 0.0
            if f > 1.82365e14:
                hminbf = _map1_simple(HMINOP_WBF, HMINOP_BF, wave)

            # Compute H- opacity (atlas7v.for line 5309-5313)
            # HMINFF = FFTETA * XNFPH(J,1) * 2. * BHYD(J,1) * XNE(J) / RHO(J) * 1e-26
            hminff = _SCALE_HMINFF * fftheta * xnfph1 * 2.0 * bhyd1 * xne / rho * 1e-26

            # H = HMINBF * 1e-18 * (1. - EHVKT(J)/BMIN(J)) * XHMIN(J) / RHO(J)
            h_bf = (
                hminbf * 1e-18 * (1.0 - ehvkt_j / np.maximum(bmin, 1e-40)) * xhmin / rho
            )

            ahmin[:, j] = h_bf + hminff

            # Source function (atlas7v.for line 5313-5314)
            # SHMIN = (H * BNU(J) * STIM(J) / (BMIN(J) - EHVKT(J)) + HMINFF * BNU(J)) / AHMIN(J)
            bmin_expanded = np.broadcast_to(bmin, (n_layers,))
            denom = bmin_expanded - ehvkt_j
            h_bf_src = h_bf * bnu_j * stim_j / np.maximum(denom, 1e-40)
            shmin[:, j] = np.where(
                ahmin[:, j] > 0, (h_bf_src + hminff * bnu_j) / ahmin[:, j], bnu_j
            )

    # Scattering subroutines
    # ELECOP: Electron scattering (atlas7v.for line 7806-7817) - Simple!
    if atmosphere.electron_density is not None:
        logger.info("Computing ELECOP (electron scattering)...")
        xne = np.asarray(atmosphere.electron_density, dtype=np.float64)
        for j in range(nfreq):
            # SIGEL = 0.6653e-24 * XNE / RHO (atlas7v.for line 7815)
            sigel[:, j] = _SCALE_ELECOP * 0.6653e-24 * xne / rho

    # HRAYOP: Hydrogen Rayleigh scattering (atlas7v.for line 5332-5482)
    # CRITICAL FIX: Fortran uses XNFPH(J,1) which is GROUND-STATE hydrogen population,
    # computed by POPS(1.01D0,11,XNFPH). But fort.10 only stores XNFH (total neutral H).
    # We must compute ground-state population from total H using partition function.
    xnfph1 = None
    if atmosphere.xnf_h is not None:
        logger.info("Computing HRAYOP (hydrogen Rayleigh scattering)...")
        xnf_h_total = np.asarray(atmosphere.xnf_h, dtype=np.float64)
        # Compute ground-state hydrogen from total neutral hydrogen
        xnfph1 = compute_ground_state_hydrogen(xnf_h_total, temp)
        logger.info(
            f"  Ground-state H fraction at layer 0: {xnfph1[0]/xnf_h_total[0]:.4f}"
        )
        logger.info(
            f"  XNFH[0] (total): {xnf_h_total[0]:.6e}, XNFPH[0] (ground): {xnfph1[0]:.6e}"
        )
    elif atmosphere.xnfph is not None:
        # Fallback: use xnfph if available (legacy behavior)
        logger.info("Computing HRAYOP using legacy xnfph (may be inaccurate)...")
        xnfph_arr = np.asarray(atmosphere.xnfph, dtype=np.float64)
        xnfph1 = xnfph_arr[:, 0] if xnfph_arr.shape[1] > 0 else np.ones(n_layers)

    if xnfph1 is not None:
        bhyd = atlas_tables.get("bhyd", np.ones((n_layers, 8), dtype=np.float64))
        bhyd1 = bhyd[:, 0] if bhyd.shape[1] > 0 else np.ones(n_layers)

        freq_lyman = 3.288051e15  # Lyman limit frequency
        freq_step = 3.288051e13  # Step size for GAVRILAM

        for j in range(nfreq):
            f = freq[j]
            g = 0.0

            # Compute G from Gavrila tables (atlas7v.for line 5421-5477)
            if f < freq_lyman * 0.01:  # FREQ < 3.288051e13
                # Linear extrapolation below table (atlas7v.for line 5422-5424)
                g = HRAYOP_GAVRILAM[0] * (f / freq_step) ** 2
            elif f <= freq_lyman * 0.74:  # FREQ <= 0.74 * Lyman
                # Interpolate in GAVRILAM (atlas7v.for line 5426-5431)
                # Fortran: I=FREQ/3.288051D13, I=MIN(I+1,74)
                #          G=GAVRILAM(I-1)+(GAVRILAM(I)-GAVRILAM(I-1))/3.288051E13*(FREQ-(I-1)*3.288051D13)
                i = int(f / freq_step)
                i = min(i + 1, 74)
                i = max(1, i)
                if i >= len(HRAYOP_GAVRILAM):
                    i = len(HRAYOP_GAVRILAM) - 1
                if i > 1:
                    # CRITICAL FIX: Fortran 1-based indexing to Python 0-based:
                    # Fortran I=31 uses GAVRILAM(30) and GAVRILAM(31)
                    # Python must use HRAYOP_GAVRILAM[29] and HRAYOP_GAVRILAM[30]
                    # So use [i-2] and [i-1] for GAVRILAM(I-1) and GAVRILAM(I)
                    g = HRAYOP_GAVRILAM[i - 2] + (
                        HRAYOP_GAVRILAM[i - 1] - HRAYOP_GAVRILAM[i - 2]
                    ) / freq_step * (f - (i - 1) * freq_step)
                else:
                    g = HRAYOP_GAVRILAM[0]
            elif f < freq_lyman * 0.755:  # FREQ < 0.755 * Lyman
                g = 15.57  # Constant (atlas7v.for line 5433-5435)
            elif f <= freq_lyman * 0.885:  # FREQ <= 0.885 * Lyman
                # Interpolate in GAVRILAMAB (atlas7v.for line 5437-5444)
                # Fortran: I=(FREQ-.755D0*3.288051D15)/1.644026D13
                #          I=I+1
                #          I=MIN(I+1,27)
                #          G=GAVRILAMAB(I-1)+(GAVRILAMAB(I)-GAVRILAMAB(I-1))/1.644026D13*
                #            (FREQ-(.755D0*3.288051D15+((I-1)-1)*1.664026D13))
                step_ab = 1.644026e13
                i = int((f - freq_lyman * 0.755) / step_ab)
                i = i + 1  # First increment (matches Fortran I=I+1)
                i = min(i + 1, 27)  # Second increment (matches Fortran I=MIN(I+1,27))
                i = max(1, i)
                if i >= len(HRAYOP_GAVRILAMAB):
                    i = len(HRAYOP_GAVRILAMAB) - 1
                if i > 1:
                    # CRITICAL FIX: Fortran 1-based indexing to Python 0-based
                    # Fortran uses GAVRILAMAB(I-1) and GAVRILAMAB(I)
                    # Python uses [i-2] and [i-1]
                    # Note: Fortran uses 1.664026D13 in freq offset (line 5442), might be typo but match exactly
                    freq_base = freq_lyman * 0.755
                    freq_offset_step = (
                        1.664026e13  # From Fortran line 5442 (different from step_ab!)
                    )
                    freq1 = freq_base + ((i - 1) - 1) * freq_offset_step
                    g = HRAYOP_GAVRILAMAB[i - 2] + (
                        HRAYOP_GAVRILAMAB[i - 1] - HRAYOP_GAVRILAMAB[i - 2]
                    ) / step_ab * (f - freq1)
                else:
                    g = HRAYOP_GAVRILAMAB[0]
            elif f < freq_lyman * 0.890:  # FREQ < 0.890 * Lyman
                g = 8.0  # Constant (atlas7v.for line 5446-5448)
            elif f <= freq_lyman * 0.936:  # FREQ <= 0.936 * Lyman
                # Interpolate in GAVRILAMBC (atlas7v.for line 5450-5457)
                # Fortran: I=(FREQ-.890D0*3.28851D15)/0.657610D13
                #          I=I+1
                #          I=MIN(I+1,24)
                #          G=GAVRILAMBC(I-1)+(GAVRILAMBC(I)-GAVRILAMBC(I-1))/0.657610D13*
                #            (FREQ-(.890D0*3.288051D15+((I-1)-1)*0.657610D13))
                step_bc = 0.657610e13
                i = int((f - freq_lyman * 0.890) / step_bc)
                i = i + 1  # First increment (matches Fortran I=I+1)
                i = min(i + 1, 24)  # Second increment (matches Fortran I=MIN(I+1,24))
                i = max(1, i)
                if i >= len(HRAYOP_GAVRILAMBC):
                    i = len(HRAYOP_GAVRILAMBC) - 1
                if i > 1:
                    # CRITICAL FIX: Fortran 1-based indexing to Python 0-based
                    # Fortran uses GAVRILAMBC(I-1) and GAVRILAMBC(I)
                    # Python uses [i-2] and [i-1]
                    freq_base = freq_lyman * 0.890
                    freq1 = freq_base + ((i - 1) - 1) * step_bc
                    g = HRAYOP_GAVRILAMBC[i - 2] + (
                        HRAYOP_GAVRILAMBC[i - 1] - HRAYOP_GAVRILAMBC[i - 2]
                    ) / step_bc * (f - freq1)
                else:
                    g = HRAYOP_GAVRILAMBC[0]
            elif f < freq_lyman * 0.938:  # FREQ < 0.938 * Lyman
                g = 9.0  # Constant (atlas7v.for line 5459-5461)
            elif f <= freq_lyman * 0.959:  # FREQ <= 0.959 * Lyman
                # Interpolate in GAVRILAMCD (atlas7v.for line 5463-5470)
                # Fortran: I=(FREQ-.938D0*3.288051D15)/0.3288051D13
                #          I=I+1
                #          I=MIN(I+1,22)
                #          G=GAVRILAMCD(I-1)+(GAVRILAMCD(I)-GAVRILAMCD(I-1))/0.3288051D13*
                #            (FREQ-(.938D0*3.288051D15+((I-1)-1)*0.3288051D13))
                step_cd = 0.3288051e13
                i = int((f - freq_lyman * 0.938) / step_cd)
                i = i + 1  # First increment (matches Fortran I=I+1)
                i = min(i + 1, 22)  # Second increment (matches Fortran I=MIN(I+1,22))
                i = max(1, i)
                if i >= len(HRAYOP_GAVRILAMCD):
                    i = len(HRAYOP_GAVRILAMCD) - 1
                if i > 1:
                    # CRITICAL FIX: Fortran 1-based indexing to Python 0-based
                    # Fortran uses GAVRILAMCD(I-1) and GAVRILAMCD(I)
                    # Python uses [i-2] and [i-1]
                    freq_base = freq_lyman * 0.938
                    freq1 = freq_base + ((i - 1) - 1) * step_cd
                    g = HRAYOP_GAVRILAMCD[i - 2] + (
                        HRAYOP_GAVRILAMCD[i - 1] - HRAYOP_GAVRILAMCD[i - 2]
                    ) / step_cd * (f - freq1)
                else:
                    g = HRAYOP_GAVRILAMCD[0]
            elif f <= freq_lyman:  # FREQ <= 1.000 * Lyman
                g = HRAYOP_GAVRILALYMANCONT[0]  # Constant (atlas7v.for line 5472-5474)
            else:  # FREQ > Lyman
                # Use MAP1 interpolation in GAVRILALYMANCONT (atlas7v.for line 5476-5477)
                freqlg_normalized = f / freq_lyman
                g = _map1_simple(
                    HRAYOP_FGAVRILALYMANCONT, HRAYOP_GAVRILALYMANCONT, freqlg_normalized
                )

            # XSECT = 6.65e-25 * G^2 (atlas7v.for line 5478)
            xsect = 6.65e-25 * g**2

            # SIGH = XSECT * XNFPH(J,1) * 2. * BHYD(J,1) / RHO(J) (atlas7v.for line 5480)
            sigh[:, j] = _SCALE_HRAYOP * xsect * xnfph1 * 2.0 * bhyd1 / rho

    # HERAOP: Helium Rayleigh scattering (atlas7v.for line 5818-5832)
    # CRITICAL: Fortran only calls HERAOP if IFOP(8) == 1 (atlas7v.for line 4046)
    # Fortran's default is IFOP(8) = 0 (atlas7v.for line 2822: DATA IFOP/...,0,0,.../)
    # This means HERAOP is DISABLED by default in Fortran!
    if (
        ifop[7] == 1 and atmosphere.xnf_he1 is not None
    ):  # IFOP(8) in Fortran = ifop[7] in Python (0-indexed)
        logger.info("Computing HERAOP (helium Rayleigh scattering)...")
        xnfphe = np.asarray(atmosphere.xnf_he1, dtype=np.float64)
        if xnfphe.ndim == 1:
            xnfphe = xnfphe[:, np.newaxis]  # Make it 2D
        bhe1 = atlas_tables.get("bhe1", np.ones((n_layers, 29), dtype=np.float64))

        for j in range(nfreq):
            f = freq[j]
            # WAVE = 2.99792458e18 / min(FREQ, 5.15e15) (atlas7v.for line 5826)
            wave = 2.99792458e18 / min(f, 5.15e15)
            ww = wave**2
            # SIG = 5.484e-14 / WW / WW * (1. + (2.44e5 + 5.94e10 / (WW - 2.90e5)) / WW)^2 (atlas7v.for line 5828)
            sig = (
                5.484e-14
                / (ww * ww)
                * (1.0 + (2.44e5 + 5.94e10 / max(ww - 2.90e5, 1e-10)) / ww) ** 2
            )
            xnfphe1 = xnfphe[:, 0] if xnfphe.shape[1] > 0 else np.ones(n_layers)
            bhe1_1 = bhe1[:, 0] if bhe1.shape[1] > 0 else np.ones(n_layers)
            sighe[:, j] = sig * xnfphe1 / rho * bhe1_1
    else:
        logger.info("Skipping HERAOP (helium Rayleigh scattering) - IFOP(8)=0")

    # H2RAOP: H2 Rayleigh scattering (atlas7v.for line 6823-6853)
    # CRITICAL: Fortran only calls H2RAOP if IFOP(13) == 1
    if ifop[12] == 1 and xnfph1 is not None:
        logger.info("Computing H2RAOP (H2 Rayleigh scattering)...")
        bhyd1 = bhyd[:, 0] if bhyd.shape[1] > 0 else np.ones(n_layers)

        # Compute XNH2 (H2 number density per gram) from XNFPH using equilibrium
        # Formula from atlas7v.for H2RAOP subroutine (lines 9744-9747):
        # XNH2(J) = (XNFPH(J,1)*2.*BHYD(J,1))**2 * EXP(4.478D0/TKEV(J) -
        #   4.64584D1 + poly(T) - 1.5*TLOG(J)) / RHO(J)
        # The /RHO(J) is outside the EXP() on line 9747 continuation line 3.
        tkev_arr = KBOLTZ_EV * temp
        tlog_arr = np.log(temp)

        # Polynomial in T: (1.63660e-3 + (-4.93992e-7 + (1.11822e-10 + (-1.49567e-14 +
        #                  (1.06206e-18 - 3.08720e-23*T)*T)*T)*T)*T)*T
        poly_T = (
            1.63660e-3
            + (
                -4.93992e-7
                + (
                    1.11822e-10
                    + (-1.49567e-14 + (1.06206e-18 - 3.08720e-23 * temp) * temp) * temp
                )
                * temp
            )
            * temp
        ) * temp

        exp_term = 4.478 / tkev_arr - 4.64584e1 + poly_T - 1.5 * tlog_arr

        # Avoid overflow
        exp_term = np.clip(exp_term, -100, 100)

        xnh2 = (xnfph1 * 2.0 * bhyd1) ** 2 * np.exp(exp_term) / rho

        for j in range(nfreq):
            f = freq[j]
            # Wave in Angstrom, capped at frequency 2.922e15 Hz
            wave = 2.99792458e18 / min(f, 2.922e15)
            ww = wave**2

            # Cross-section formula (atlas7v.for line 6847)
            sig = (8.14e-13 + 1.28e-6 / ww + 1.61 / (ww * ww)) / (ww * ww)

            sigh2[:, j] = _SCALE_H2RAOP * sig * xnh2

        logger.info(f"  SIGH2[0] at first freq: {sigh2[0, 0]:.6e}")
    else:
        logger.info("Skipping H2RAOP (H2 Rayleigh scattering) - IFOP(13)=0 or no XNFPH")

    # XSOP: Dummy scattering (atlas7v.for line 8083-8091) - does nothing
    # sigx remains zeros

    # Metal opacities
    # C1OP: Carbon I opacity (atlas7v.for line 5859-6033)
    if hasattr(atmosphere, "xnfpc") and atmosphere.xnfpc is not None:
        logger.info("Computing C1OP (Carbon I opacity)...")
        xnfpc = np.asarray(atmosphere.xnfpc, dtype=np.float64)
        bc1 = atlas_tables.get("bc1", np.ones((n_layers, 14), dtype=np.float64))
        bc2 = atlas_tables.get("bc2", np.ones((n_layers, 6), dtype=np.float64))
        ryd = 109732.298  # Carbon Rydberg constant

        for j in range(nfreq):
            f = freq[j]
            wno = waveno[j]
            bnu_j = bnu_all[:, j]
            ehvkt_j = ehvkt[:, j]
            stim_j = stim[:, j]

            h = 1e-30 * np.ones(n_layers)
            s = np.zeros(n_layers)

            # Bound-free contributions (atlas7v.for line 5873-6025)
            # PP 1S (BC1 index 13)
            if wno >= 16886.790:
                x = 0.0  # Placeholder - would need full cross-section
                bc1_13 = bc1[:, 12] if bc1.shape[1] > 12 else np.ones(n_layers)
                bc2_1 = bc2[:, 0] if bc2.shape[1] > 0 else np.ones(n_layers)
                a = x * 1.0 * np.exp(-73975.91 * hckt) * (bc1_13 - bc2_1 * ehvkt_j)
                h = h + a
                denom = bc1_13 / np.maximum(bc2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # PP 1D (BC1 index 12)
            if wno >= 18251.980:
                x = 0.0
                bc1_12 = bc1[:, 11] if bc1.shape[1] > 11 else np.ones(n_layers)
                bc2_1 = bc2[:, 0] if bc2.shape[1] > 0 else np.ones(n_layers)
                a = x * 5.0 * np.exp(-72610.72 * hckt) * (bc1_12 - bc2_1 * ehvkt_j)
                h = h + a
                denom = bc1_12 / np.maximum(bc2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # PP 3P (BC1 index 11)
            if wno >= 19487.800:
                x = 0.0
                bc1_11 = bc1[:, 10] if bc1.shape[1] > 10 else np.ones(n_layers)
                bc2_1 = bc2[:, 0] if bc2.shape[1] > 0 else np.ones(n_layers)
                a = x * 9.0 * np.exp(-71374.90 * hckt) * (bc1_11 - bc2_1 * ehvkt_j)
                h = h + a
                denom = bc1_11 / np.maximum(bc2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # PP 3S (BC1 index 10)
            if wno >= 20118.750:
                x = 0.0
                bc1_10 = bc1[:, 9] if bc1.shape[1] > 9 else np.ones(n_layers)
                bc2_1 = bc2[:, 0] if bc2.shape[1] > 0 else np.ones(n_layers)
                a = x * 3.0 * np.exp(-70743.95 * hckt) * (bc1_10 - bc2_1 * ehvkt_j)
                h = h + a
                denom = bc1_10 / np.maximum(bc2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # PP 3D (BC1 index 9)
            if wno >= 21140.700:
                x = 0.0
                bc1_9 = bc1[:, 8] if bc1.shape[1] > 8 else np.ones(n_layers)
                bc2_1 = bc2[:, 0] if bc2.shape[1] > 0 else np.ones(n_layers)
                a = x * 15.0 * np.exp(-69722.00 * hckt) * (bc1_9 - bc2_1 * ehvkt_j)
                h = h + a
                denom = bc1_9 / np.maximum(bc2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # PP 1P (BC1 index 8) - has cross-section formula
            if wno >= 22006.370:
                x = 2.1e-18 * (22006.370 / wno) ** 1.5
                bc1_8 = bc1[:, 7] if bc1.shape[1] > 7 else np.ones(n_layers)
                bc2_1 = bc2[:, 0] if bc2.shape[1] > 0 else np.ones(n_layers)
                a = x * 3.0 * np.exp(-68856.33 * hckt) * (bc1_8 - bc2_1 * ehvkt_j)
                h = h + a
                denom = bc1_8 / np.maximum(bc2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # PS 1P (BC1 index 6)
            if wno >= 28880.880:
                x = 1.54e-18 * (28880.880 / wno) ** 1.2
                bc1_6 = bc1[:, 5] if bc1.shape[1] > 5 else np.ones(n_layers)
                bc2_1 = bc2[:, 0] if bc2.shape[1] > 0 else np.ones(n_layers)
                a = x * 3.0 * np.exp(-61981.82 * hckt) * (bc1_6 - bc2_1 * ehvkt_j)
                h = h + a
                denom = bc1_6 / np.maximum(bc2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # PS 3P (BC1 index 5)
            if wno >= 30489.700:
                x = 0.2e-18 * (30489.700 / wno) ** 1.2
                bc1_5 = bc1[:, 4] if bc1.shape[1] > 4 else np.ones(n_layers)
                bc2_1 = bc2[:, 0] if bc2.shape[1] > 0 else np.ones(n_layers)
                a = x * 9.0 * np.exp(-60373.00 * hckt) * (bc1_5 - bc2_1 * ehvkt_j)
                h = h + a
                denom = bc1_5 / np.maximum(bc2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # P2 1S (BC1 index 3) - complex formula with resonances
            if wno >= 69172.400:
                x = 10.0 ** (-16.80 - (wno - 69172.400) / 3.0 / ryd)
                eps = (wno - 97700.0) * 2.0 / 2743.0
                a_val = 68e-18
                b_val = 118e-18
                x = x + (a_val * eps + b_val) / (eps**2 + 1.0)
                x = x / 3.0
                bc1_3 = bc1[:, 2] if bc1.shape[1] > 2 else np.ones(n_layers)
                bc2_1 = bc2[:, 0] if bc2.shape[1] > 0 else np.ones(n_layers)
                a = x * 1.0 * np.exp(-21648.02 * hckt) * (bc1_3 - bc2_1 * ehvkt_j)
                h = h + a
                denom = bc1_3 / np.maximum(bc2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                # Second contribution (atlas7v.for line 5944-5948)
                if wno >= 69235.820:
                    a = a * 2.0
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # P2 1D (BC1 index 2) - complex formula
            if wno >= 80627.760:
                x = 10.0 ** (-16.80 - (wno - 80627.760) / 3.0 / ryd)
                eps1 = (wno - 93917.0) * 2.0 / 9230.0
                a1 = 22e-18
                b1 = 26e-18
                x = x + (a1 * eps1 + b1) / (eps1**2 + 1.0)
                eps2 = (wno - 111130.0) * 2.0 / 2743.0
                a2 = -10.5e-18
                b2 = 46e-18
                x = x + (a2 * eps2 + b2) / (eps2**2 + 1.0)
                x = x / 3.0
                bc1_2 = bc1[:, 1] if bc1.shape[1] > 1 else np.ones(n_layers)
                bc2_1 = bc2[:, 0] if bc2.shape[1] > 0 else np.ones(n_layers)
                a = x * 5.0 * np.exp(-10192.66 * hckt) * (bc1_2 - bc2_1 * ehvkt_j)
                h = h + a
                denom = bc1_2 / np.maximum(bc2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                if wno >= 80691.180:
                    a = a * 2.0
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # P2 3P (BC1 index 1) - complex with multiple contributions
            if wno >= 90777.000:
                x = 10.0 ** (-16.80 - (wno - 90777.000) / 3.0 / ryd)
                x = x / 3.0
                bc1_1 = bc1[:, 0] if bc1.shape[1] > 0 else np.ones(n_layers)
                bc2_1 = bc2[:, 0] if bc2.shape[1] > 0 else np.ones(n_layers)

                if wno >= 90777.000:
                    a = x * 5.0 * np.exp(-43.42 * hckt) * (bc1_1 - bc2_1 * ehvkt_j)
                    h = h + a
                    denom = bc1_1 / np.maximum(bc2_1, 1e-40) - ehvkt_j
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                if wno >= 90804.000:
                    a = x * 3.0 * np.exp(-16.42 * hckt) * (bc1_1 - bc2_1 * ehvkt_j)
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                if wno >= 90820.420:
                    a = x * 1.0 * 1.0 * (bc1_1 - bc2_1 * ehvkt_j)
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                if wno >= 90840.420:
                    x = x * 2.0
                    a = x * 5.0 * np.exp(-43.42 * hckt) * (bc1_1 - bc2_1 * ehvkt_j)
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                if wno >= 90867.420:
                    a = x * 3.0 * np.exp(-16.42 * hckt) * (bc1_1 - bc2_1 * ehvkt_j)
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                if wno >= 90883.840:
                    a = x * 1.0 * 1.0 * (bc1_1 - bc2_1 * ehvkt_j)
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # P3 5S (BC1 index 4)
            if wno >= 100121.000:
                x = 1e-18 * (100121.000 / wno) ** 3
                bc1_4 = bc1[:, 3] if bc1.shape[1] > 3 else np.ones(n_layers)
                bc2_1 = bc2[:, 0] if bc2.shape[1] > 0 else np.ones(n_layers)
                a = x * 5.0 * np.exp(-33735.20 * hckt) * (bc1_4 - bc2_1 * ehvkt_j)
                h = h + a
                denom = bc1_4 / np.maximum(bc2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # Normalize and store (atlas7v.for line 6027-6030)
            xnfpc1 = xnfpc[:, 0] if xnfpc.shape[1] > 0 else np.ones(n_layers)
            h = h * xnfpc1 / rho
            s = s * xnfpc1 / rho

            ac1[:, j] = h
            sc1[:, j] = np.where(h > 0, s / h, bnu_j)

    # MG1OP: Magnesium I opacity (atlas7v.for line 6187-6261)
    if hasattr(atmosphere, "xnfpmg") and atmosphere.xnfpmg is not None:
        logger.info("Computing MG1OP (Magnesium I opacity)...")
        xnfpmg = np.asarray(atmosphere.xnfpmg, dtype=np.float64)
        bmg1 = atlas_tables.get("bmg1", np.ones((n_layers, 11), dtype=np.float64))
        bmg2 = atlas_tables.get("bmg2", np.ones((n_layers, 6), dtype=np.float64))

        for j in range(nfreq):
            f = freq[j]
            wno = waveno[j]
            bnu_j = bnu_all[:, j]
            ehvkt_j = ehvkt[:, j]
            stim_j = stim[:, j]

            h = 1e-30 * np.ones(n_layers)
            s = np.zeros(n_layers)

            # 3D 3D (BMG1 index 8)
            if wno >= 13713.986:
                x = 25e-18 * (13713.986 / wno) ** 2.7
                bmg1_8 = bmg1[:, 7] if bmg1.shape[1] > 7 else np.ones(n_layers)
                bmg2_1 = bmg2[:, 0] if bmg2.shape[1] > 0 else np.ones(n_layers)
                a = x * 15.0 * np.exp(-47957.034 * hckt) * (bmg1_8 - bmg2_1 * ehvkt_j)
                h = h + a
                denom = bmg1_8 / np.maximum(bmg2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # 4P 3P (BMG1 index 7)
            if wno >= 13823.223:
                x = 33.8e-18 * (13823.223 / wno) ** 2.8
                bmg1_7 = bmg1[:, 6] if bmg1.shape[1] > 6 else np.ones(n_layers)
                bmg2_1 = bmg2[:, 0] if bmg2.shape[1] > 0 else np.ones(n_layers)
                a = x * 9.0 * np.exp(-47847.797 * hckt) * (bmg1_7 - bmg2_1 * ehvkt_j)
                h = h + a
                denom = bmg1_7 / np.maximum(bmg2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # 3D 1D (BMG1 index 6)
            if wno >= 15267.955:
                x = 45e-18 * (15267.955 / wno) ** 2.7
                bmg1_6 = bmg1[:, 5] if bmg1.shape[1] > 5 else np.ones(n_layers)
                bmg2_1 = bmg2[:, 0] if bmg2.shape[1] > 0 else np.ones(n_layers)
                a = x * 5.0 * np.exp(-46403.065 * hckt) * (bmg1_6 - bmg2_1 * ehvkt_j)
                h = h + a
                denom = bmg1_6 / np.maximum(bmg2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # 4S 1S (BMG1 index 5)
            if wno >= 18167.687:
                x = 0.43e-18 * (18167.687 / wno) ** 2.6
                bmg1_5 = bmg1[:, 4] if bmg1.shape[1] > 4 else np.ones(n_layers)
                bmg2_1 = bmg2[:, 0] if bmg2.shape[1] > 0 else np.ones(n_layers)
                a = x * 1.0 * np.exp(-43503.333 * hckt) * (bmg1_5 - bmg2_1 * ehvkt_j)
                h = h + a
                denom = bmg1_5 / np.maximum(bmg2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # 4S 3S (BMG1 index 4)
            if wno >= 20473.617:
                x = 2.1e-18 * (20473.617 / wno) ** 2.6
                bmg1_4 = bmg1[:, 3] if bmg1.shape[1] > 3 else np.ones(n_layers)
                bmg2_1 = bmg2[:, 0] if bmg2.shape[1] > 0 else np.ones(n_layers)
                a = x * 3.0 * np.exp(-41197.043 * hckt) * (bmg1_4 - bmg2_1 * ehvkt_j)
                h = h + a
                denom = bmg1_4 / np.maximum(bmg2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # 3P 1P (BMG1 index 3)
            if wno >= 26619.756:
                x = (
                    16e-18 * (26619.756 / wno) ** 2.1
                    - 7.8e-18 * (26619.756 / wno) ** 9.5
                )
                bmg1_3 = bmg1[:, 2] if bmg1.shape[1] > 2 else np.ones(n_layers)
                bmg2_1 = bmg2[:, 0] if bmg2.shape[1] > 0 else np.ones(n_layers)
                a = x * 3.0 * np.exp(-35051.264 * hckt) * (bmg1_3 - bmg2_1 * ehvkt_j)
                h = h + a
                denom = bmg1_3 / np.maximum(bmg2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # 3P 3P (BMG1 index 2) - multiple contributions
            if wno >= 39759.842:
                x = 20e-18 * (39759.842 / wno) ** 2.7
                x = np.maximum(x, 40e-18 * (39759.842 / wno) ** 14)
                bmg1_2 = bmg1[:, 1] if bmg1.shape[1] > 1 else np.ones(n_layers)
                bmg2_1 = bmg2[:, 0] if bmg2.shape[1] > 0 else np.ones(n_layers)

                a = x * 5.0 * np.exp(-21911.178 * hckt) * (bmg1_2 - bmg2_1 * ehvkt_j)
                h = h + a
                denom = bmg1_2 / np.maximum(bmg2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                if wno >= 39800.556:
                    a = (
                        x
                        * 3.0
                        * np.exp(-21870.464 * hckt)
                        * (bmg1_2 - bmg2_1 * ehvkt_j)
                    )
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                if wno >= 39820.615:
                    a = (
                        x
                        * 1.0
                        * np.exp(-21850.405 * hckt)
                        * (bmg1_2 - bmg2_1 * ehvkt_j)
                    )
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # 3S 1S (BMG1 index 1)
            if wno >= 61671.020:
                x = 1.1e-18 * (61671.020 / wno) ** 10
                bmg1_1 = bmg1[:, 0] if bmg1.shape[1] > 0 else np.ones(n_layers)
                bmg2_1 = bmg2[:, 0] if bmg2.shape[1] > 0 else np.ones(n_layers)
                a = x * 1.0 * 1.0 * (bmg1_1 - bmg2_1 * ehvkt_j)
                h = h + a
                denom = bmg1_1 / np.maximum(bmg2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # Normalize and store (atlas7v.for line 6258-6260)
            # Handle both 1D (n_layers,) and 2D (n_layers, n_ions) arrays
            xnfpmg1 = (
                xnfpmg
                if xnfpmg.ndim == 1
                else (xnfpmg[:, 0] if xnfpmg.shape[1] > 0 else np.ones(n_layers))
            )
            h = h * xnfpmg1 / rho
            s = s * xnfpmg1 / rho

            amg1[:, j] = h
            smg1[:, j] = np.where(h > 0, s / h, bnu_j)

    # FE1OP: Iron I opacity (atlas7v.for line 6623-6665) - simpler structure
    if hasattr(atmosphere, "xnfpfe") and atmosphere.xnfpfe is not None:
        logger.info("Computing FE1OP (Iron I opacity)...")
        xnfpfe = np.asarray(atmosphere.xnfpfe, dtype=np.float64)
        bfe1 = atlas_tables.get("bfe1", np.ones((n_layers, 15), dtype=np.float64))
        bsi1 = atlas_tables.get("bsi1", np.ones((n_layers, 11), dtype=np.float64))

        # FE1OP uses arrays for transitions (atlas7v.for line 6635-6650)
        fe1_g = np.array(
            [
                25.0,
                35.0,
                21.0,
                15.0,
                9.0,
                35.0,
                33.0,
                21.0,
                27.0,
                49.0,
                9.0,
                21.0,
                27.0,
                9.0,
                9.0,
                25.0,
                33.0,
                15.0,
                35.0,
                3.0,
                5.0,
                11.0,
                15.0,
                13.0,
                15.0,
                9.0,
                21.0,
                15.0,
                21.0,
                25.0,
                35.0,
                9.0,
                5.0,
                45.0,
                27.0,
                21.0,
                15.0,
                21.0,
                15.0,
                25.0,
                21.0,
                35.0,
                5.0,
                15.0,
                45.0,
                35.0,
                55.0,
                25.0,
            ],
            dtype=np.float64,
        )

        fe1_e = np.array(
            [
                500.0,
                7500.0,
                12500.0,
                17500.0,
                19000.0,
                19500.0,
                19500.0,
                21000.0,
                22000.0,
                23000.0,
                23000.0,
                24000.0,
                24000.0,
                24500.0,
                24500.0,
                26000.0,
                26500.0,
                26500.0,
                27000.0,
                27500.0,
                28500.0,
                29000.0,
                29500.0,
                29500.0,
                29500.0,
                30000.0,
                31500.0,
                31500.0,
                33500.0,
                33500.0,
                34000.0,
                34500.0,
                34500.0,
                35000.0,
                35500.0,
                37000.0,
                37000.0,
                37000.0,
                38500.0,
                40000.0,
                40000.0,
                41000.0,
                41000.0,
                43000.0,
                43000.0,
                43000.0,
                43000.0,
                44000.0,
            ],
            dtype=np.float64,
        )

        fe1_wno = np.array(
            [
                63500.0,
                58500.0,
                53500.0,
                59500.0,
                45000.0,
                44500.0,
                44500.0,
                43000.0,
                58000.0,
                41000.0,
                54000.0,
                40000.0,
                40000.0,
                57500.0,
                55500.0,
                38000.0,
                57500.0,
                57500.0,
                37000.0,
                54500.0,
                53500.0,
                55000.0,
                34500.0,
                34500.0,
                34500.0,
                34000.0,
                32500.0,
                32500.0,
                32500.0,
                32500.0,
                32000.0,
                29500.0,
                29500.0,
                31000.0,
                30500.0,
                29000.0,
                27000.0,
                54000.0,
                27500.0,
                24000.0,
                47000.0,
                23000.0,
                44000.0,
                42000.0,
                42000.0,
                21000.0,
                42000.0,
                42000.0,
            ],
            dtype=np.float64,
        )

        # FE1OP processes all frequencies - individual transitions are checked inside
        for j in range(nfreq):
            wno = waveno[j]
            if wno < 21000.0:  # Skip if below Fe I first edge
                continue

            bnu_j = bnu_all[:, j]
            ehvkt_j = ehvkt[:, j]
            stim_j = stim[:, j]

            # BFUDGE = BSI1(J,1) (atlas7v.for line 6652)
            bfudge = bsi1[:, 0] if bsi1.shape[1] > 0 else np.ones(n_layers)

            h = np.zeros(n_layers)

            # Sum contributions from all transitions (atlas7v.for line 6655-6660)
            for i in range(len(fe1_wno)):
                if fe1_wno[i] <= wno:
                    xsect = 3e-18 / (
                        1.0 + ((fe1_wno[i] + 3000.0 - wno) / fe1_wno[i] / 0.1) ** 4
                    )
                    h = h + xsect * fe1_g[i] * np.exp(-fe1_e[i] * hckt)

            # Normalize and store (atlas7v.for line 6661-6663)
            # Handle both 1D (n_layers,) and 2D (n_layers, n_ions) arrays
            xnfpfe1 = (
                xnfpfe
                if xnfpfe.ndim == 1
                else (xnfpfe[:, 0] if xnfpfe.shape[1] > 0 else np.ones(n_layers))
            )
            h = h * stim_j * xnfpfe1 / rho

            afe1[:, j] = h
            sfe1[:, j] = bnu_j * stim_j / np.maximum(bfudge - ehvkt_j, 1e-40)

    # AL1OP: Aluminum I opacity (atlas7v.for line 7716-7792)
    if hasattr(atmosphere, "xnfpal") and atmosphere.xnfpal is not None:
        logger.info("Computing AL1OP (Aluminum I opacity)...")
        xnfpal = np.asarray(atmosphere.xnfpal, dtype=np.float64)
        bal1 = atlas_tables.get("bal1", np.ones((n_layers, 9), dtype=np.float64))
        bal2 = atlas_tables.get("bal2")
        # BAL2 is exp(-E_ion * HCKT) where E_ion = 48278.37 cm^-1 for Al I ionization limit
        if bal2 is None:
            # Compute BAL2 in LTE: departure coefficient at ionization limit
            bal2 = np.exp(-48278.37 * hckt)[:, np.newaxis]  # (n_layers, 1)

        for j in range(nfreq):
            wno = waveno[j]
            bnu_j = bnu_all[:, j]
            ehvkt_j = ehvkt[:, j]
            stim_j = stim[:, j]

            h = 1e-30 * np.ones(n_layers)
            s = np.zeros(n_layers)

            # Get BAL2(J,1) for this layer
            bal2_1 = bal2[:, 0] if bal2.shape[1] > 0 else np.ones(n_layers)

            # 4F 2F (BAL1 index 9)
            if wno >= 6958.993:
                x = 0.0  # X=0 in Fortran
                bal1_9 = bal1[:, 8] if bal1.shape[1] > 8 else np.ones(n_layers)
                a = x * 14.0 * np.exp(-41319.377 * hckt) * (bal1_9 - bal2_1 * ehvkt_j)
                h = h + a
                denom = bal1_9 / np.maximum(bal2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # 5P 2P (BAL1 index 8)
            if wno >= 8002.467:
                x = 50e-18 * (8002.467 / wno) ** 3
                bal1_8 = bal1[:, 7] if bal1.shape[1] > 7 else np.ones(n_layers)
                a = x * 6.0 * np.exp(-40275.903 * hckt) * (bal1_8 - bal2_1 * ehvkt_j)
                h = h + a
                denom = bal1_8 / np.maximum(bal2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # 4D 2D (BAL1 index 7)
            if wno >= 9346.231:
                x = 50e-18 * (9346.231 / wno) ** 3
                bal1_7 = bal1[:, 6] if bal1.shape[1] > 6 else np.ones(n_layers)
                a = x * 10.0 * np.exp(-38932.139 * hckt) * (bal1_7 - bal2_1 * ehvkt_j)
                h = h + a
                denom = bal1_7 / np.maximum(bal2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # 5S 2S (BAL1 index 6)
            if wno >= 10588.957:
                x = 56.7e-18 * (10588.957 / wno) ** 1.9
                bal1_6 = bal1[:, 5] if bal1.shape[1] > 5 else np.ones(n_layers)
                a = x * 2.0 * np.exp(-37689.413 * hckt) * (bal1_6 - bal2_1 * ehvkt_j)
                h = h + a
                denom = bal1_6 / np.maximum(bal2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # 4P 2P (BAL1 index 5)
            if wno >= 15318.007:
                x = 14.5e-18 * 15318.007 / wno
                bal1_5 = bal1[:, 4] if bal1.shape[1] > 4 else np.ones(n_layers)
                a = x * 6.0 * np.exp(-32960.363 * hckt) * (bal1_5 - bal2_1 * ehvkt_j)
                h = h + a
                denom = bal1_5 / np.maximum(bal2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # 3D 2D (BAL1 index 4)
            if wno >= 15842.129:
                x = 47e-18 * (15842.129 / wno) ** 1.83
                bal1_4 = bal1[:, 3] if bal1.shape[1] > 3 else np.ones(n_layers)
                a = x * 10.0 * np.exp(-32436.241 * hckt) * (bal1_4 - bal2_1 * ehvkt_j)
                h = h + a
                denom = bal1_4 / np.maximum(bal2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # 4S 2S (BAL1 index 2)
            if wno >= 22930.614:
                x = 10e-18 * (22930.614 / wno) ** 2
                bal1_2 = bal1[:, 1] if bal1.shape[1] > 1 else np.ones(n_layers)
                a = x * 2.0 * np.exp(-25347.756 * hckt) * (bal1_2 - bal2_1 * ehvkt_j)
                h = h + a
                denom = bal1_2 / np.maximum(bal2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # 3P 2P (BAL1 index 1) - ground state edge at 48278.37 cm^-1
            if wno >= 48166.309:
                x = 65e-18 * (48166.309 / wno) ** 5
                bal1_1 = bal1[:, 0] if bal1.shape[1] > 0 else np.ones(n_layers)
                a = x * 4.0 * np.exp(-112.061 * hckt) * (bal1_1 - bal2_1 * ehvkt_j)
                h = h + a
                denom = bal1_1 / np.maximum(bal2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                if wno >= 48278.370:
                    a = x * 2.0 * 1.0 * (bal1_1 - bal2_1 * ehvkt_j)
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # P2 4P (BAL1 index 3)
            if wno >= 55903.260:
                x = 10e-18 * (55903.260 / wno) ** 2
                bal1_3 = bal1[:, 2] if bal1.shape[1] > 2 else np.ones(n_layers)
                a = x * 12.0 * np.exp(-29097.11 * hckt) * (bal1_3 - bal2_1 * ehvkt_j)
                h = h + a
                denom = bal1_3 / np.maximum(bal2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # Normalize and store (atlas7v.for line 7789-7790)
            # Handle both 1D (n_layers,) and 2D (n_layers, n_ions) arrays
            xnfpal1 = (
                xnfpal
                if xnfpal.ndim == 1
                else (xnfpal[:, 0] if xnfpal.shape[1] > 0 else np.ones(n_layers))
            )
            h = h * xnfpal1 / rho
            s = s * xnfpal1 / rho

            aal1[:, j] = h
            sal1[:, j] = np.where(h > 0, s / h, bnu_j)

    # SI1OP: Silicon I opacity (atlas7v.for line 7793-7948)
    if hasattr(atmosphere, "xnfpsi") and atmosphere.xnfpsi is not None:
        logger.info("Computing SI1OP (Silicon I opacity)...")
        xnfpsi = np.asarray(atmosphere.xnfpsi, dtype=np.float64)
        bsi1 = atlas_tables.get("bsi1", np.ones((n_layers, 11), dtype=np.float64))
        bsi2 = atlas_tables.get("bsi2", np.ones((n_layers, 10), dtype=np.float64))

        for j in range(nfreq):
            wno = waveno[j]
            bnu_j = bnu_all[:, j]
            ehvkt_j = ehvkt[:, j]
            stim_j = stim[:, j]

            h = 1e-30 * np.ones(n_layers)
            s = np.zeros(n_layers)

            # Get BSI2(J,1) for this layer
            bsi2_1 = bsi2[:, 0] if bsi2.shape[1] > 0 else np.ones(n_layers)

            # PP 3P (BSI1 index 11)
            if wno >= 16810.969:
                x = 0.0  # X=0 in Fortran
                bsi1_11 = bsi1[:, 10] if bsi1.shape[1] > 10 else np.ones(n_layers)
                a = x * 9.0 * np.exp(-49128.131 * hckt) * (bsi1_11 - bsi2_1 * ehvkt_j)
                h = h + a
                denom = bsi1_11 / np.maximum(bsi2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # PP 3D (BSI1 index 10)
            if wno >= 17777.641:
                x = 18e-18 * (17777.641 / wno) ** 3
                bsi1_10 = bsi1[:, 9] if bsi1.shape[1] > 9 else np.ones(n_layers)
                a = x * 15.0 * np.exp(-48161.459 * hckt) * (bsi1_10 - bsi2_1 * ehvkt_j)
                h = h + a
                denom = bsi1_10 / np.maximum(bsi2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # PD 1D (BSI1 index 9)
            if wno >= 18587.546:
                x = 0.0  # X=0 in Fortran
                bsi1_9 = bsi1[:, 8] if bsi1.shape[1] > 8 else np.ones(n_layers)
                a = x * 5.0 * np.exp(-47351.554 * hckt) * (bsi1_9 - bsi2_1 * ehvkt_j)
                h = h + a
                denom = bsi1_9 / np.maximum(bsi2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # PP 1P (BSI1 index 8)
            if wno >= 18655.039:
                x = 0.0  # X=0 in Fortran
                bsi1_8 = bsi1[:, 7] if bsi1.shape[1] > 7 else np.ones(n_layers)
                a = x * 3.0 * np.exp(-47284.061 * hckt) * (bsi1_8 - bsi2_1 * ehvkt_j)
                h = h + a
                denom = bsi1_8 / np.maximum(bsi2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # PS 1P (BSI1 index 6)
            if wno >= 24947.216:
                x = 4.09e-18 * (24947.216 / wno) ** 2
                bsi1_6 = bsi1[:, 5] if bsi1.shape[1] > 5 else np.ones(n_layers)
                a = x * 3.0 * np.exp(-40991.884 * hckt) * (bsi1_6 - bsi2_1 * ehvkt_j)
                h = h + a
                denom = bsi1_6 / np.maximum(bsi2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # PS 3P (BSI1 index 5)
            if wno >= 26079.180:
                x = 1.25e-18 * (26079.180 / wno) ** 2
                bsi1_5 = bsi1[:, 4] if bsi1.shape[1] > 4 else np.ones(n_layers)
                a = x * 9.0 * np.exp(-39859.920 * hckt) * (bsi1_5 - bsi2_1 * ehvkt_j)
                h = h + a
                denom = bsi1_5 / np.maximum(bsi2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # P2 1S (BSI1 index 3) with resonance
            if wno >= 50353.180:
                eps = (wno - 70000.0) * 2.0 / 6500.0
                reson1 = (97e-18 * eps + 94e-18) / (eps**2 + 1.0)
                x = 37e-18 * (50353.180 / wno) ** 2.40 + reson1
                bsi1_3 = bsi1[:, 2] if bsi1.shape[1] > 2 else np.ones(n_layers)
                bolt = 1.0 * np.exp(-15394.370 * hckt) * (bsi1_3 - bsi2_1 * ehvkt_j)
                a = x * bolt / 3.0
                h = h + a
                denom = bsi1_3 / np.maximum(bsi2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                # Second limit at 50640.630
                if wno >= 50640.630:
                    x = 37e-18 * (50640.630 / wno) ** 2.40 + reson1
                    a = x * bolt * 2.0 / 3.0
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # P2 1D (BSI1 index 2) with resonance
            if wno >= 59448.700:
                eps = (wno - 78600.0) * 2.0 / 13000.0
                reson1 = (-10e-18 * eps + 77e-18) / (eps**2 + 1.0)
                x = 24.5e-18 * (59448.700 / wno) ** 1.85 + reson1
                bsi1_2 = bsi1[:, 1] if bsi1.shape[1] > 1 else np.ones(n_layers)
                bolt = 5.0 * np.exp(-6298.850 * hckt) * (bsi1_2 - bsi2_1 * ehvkt_j)
                a = x * bolt / 3.0
                h = h + a
                denom = bsi1_2 / np.maximum(bsi2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                # Second limit at 59736.150
                if wno >= 59736.150:
                    x = 24.5e-18 * (59736.150 / wno) ** 1.85 + reson1
                    a = x * bolt * 2.0 / 3.0
                    h = h + a
                    bsi1_1 = bsi1[:, 0] if bsi1.shape[1] > 0 else np.ones(n_layers)
                    denom = bsi1_1 / np.maximum(bsi2_1, 1e-40) - ehvkt_j
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # P3 3D (BSI1 index 7)
            if wno >= 63446.510:
                x = 18e-18 * (63446.510 / wno) ** 3
                bsi1_7 = bsi1[:, 6] if bsi1.shape[1] > 6 else np.ones(n_layers)
                a = x * 15.0 * np.exp(-45303.310 * hckt) * (bsi1_7 - bsi2_1 * ehvkt_j)
                h = h + a
                denom = bsi1_7 / np.maximum(bsi2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # P2 3P (BSI1 index 1) - ground state edge with multiple contributions
            if wno >= 65524.393:
                x = 72e-18 * (65524.393 / wno) ** 1.90
                if wno > 74000.0:
                    x = 93e-18 * (65524.393 / wno) ** 4.00
                x = x / 3.0
                bsi1_1 = bsi1[:, 0] if bsi1.shape[1] > 0 else np.ones(n_layers)
                a = x * 5.0 * np.exp(-223.157 * hckt) * (bsi1_1 - bsi2_1 * ehvkt_j)
                h = h + a
                denom = bsi1_1 / np.maximum(bsi2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                if wno >= 65670.435:
                    a = x * 3.0 * np.exp(-77.115 * hckt) * (bsi1_1 - bsi2_1 * ehvkt_j)
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                if wno >= 65747.550:
                    a = x * 1.0 * 1.0 * (bsi1_1 - bsi2_1 * ehvkt_j)
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                # Second fine-structure component
                if wno >= 65811.843:
                    x = x * 2.0
                    a = x * 5.0 * np.exp(-223.157 * hckt) * (bsi1_1 - bsi2_1 * ehvkt_j)
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                if wno >= 65957.885:
                    a = x * 3.0 * np.exp(-77.115 * hckt) * (bsi1_1 - bsi2_1 * ehvkt_j)
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

                if wno >= 66035.000:
                    a = x * 1.0 * 1.0 * (bsi1_1 - bsi2_1 * ehvkt_j)
                    h = h + a
                    s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # P3 5S (BSI1 index 4)
            if wno >= 75423.767:
                x = 15e-18 * (75423.767 / wno) ** 3
                bsi1_4 = bsi1[:, 3] if bsi1.shape[1] > 3 else np.ones(n_layers)
                a = x * 5.0 * np.exp(-33326.053 * hckt) * (bsi1_4 - bsi2_1 * ehvkt_j)
                h = h + a
                denom = bsi1_4 / np.maximum(bsi2_1, 1e-40) - ehvkt_j
                s = s + a * bnu_j * stim_j / np.maximum(denom, 1e-40)

            # Normalize and store (atlas7v.for line 7944-7946)
            # Handle both 1D (n_layers,) and 2D (n_layers, n_ions) arrays
            xnfpsi1 = (
                xnfpsi
                if xnfpsi.ndim == 1
                else (xnfpsi[:, 0] if xnfpsi.shape[1] > 0 else np.ones(n_layers))
            )
            h = h * xnfpsi1 / rho
            s = s * xnfpsi1 / rho

            asi1[:, j] = h
            ssi1[:, j] = np.where(h > 0, s / h, bnu_j)

    # LUKEOP: Lukewarm star opacity (atlas7v.for line 8952-8977)
    # Computes: N1OP, O1OP, MG2OP, SI2OP, CA2OP
    # Only computed if IFOP(10) = 1
    if ifop[9] == 1:  # IFOP(10) in Fortran = ifop[9] in Python (0-indexed)
        logger.info("Computing LUKEOP (N1, O1, Mg2, Si2, Ca2 opacity)...")
        if has_pop_grid and pop.shape[1] > 1 and pop.shape[2] > 19:
            # POPS grid is stored as [layer, ion_stage(0-based), element(0-based)].
            # Map Fortran quantities: XNFPN, XNFPO, XNFPMG(II), XNFPSI(II), XNFPCA(II).
            xnfpn = pop[:, 0, 6]
            xnfpo = pop[:, 0, 7]
            xnfpmg2 = pop[:, 1, 11]
            xnfpsi2 = pop[:, 1, 13]
            xnfpca2 = pop[:, 1, 19]
        else:
            logger.warning(
                "LUKEOP enabled but population_per_ion is unavailable/incomplete; using zero LUKEOP to avoid non-Fortran placeholder opacity."
            )
            xnfpn = np.zeros(n_layers, dtype=np.float64)
            xnfpo = np.zeros(n_layers, dtype=np.float64)
            xnfpmg2 = np.zeros(n_layers, dtype=np.float64)
            xnfpsi2 = np.zeros(n_layers, dtype=np.float64)
            xnfpca2 = np.zeros(n_layers, dtype=np.float64)

        tkev = KBOLTZ_EV * temp  # eV (matches Fortran TKEV(J) = 8.6171D-5 * T(J))

        for j in range(nfreq):
            f = freq[j]
            freqlg = np.log(f)
            stim_j = stim[:, j]

            # N1OP: Nitrogen I opacity (atlas7v.for line 8978-9005)
            # Uses SEATON cross-sections at 3 edges: 853Å, 1020Å, 1130Å
            c1130 = 6.0 * np.exp(-3.575 / tkev)  # Level population factor
            c1020 = 10.0 * np.exp(-2.384 / tkev)

            x853 = 0.0
            x1020 = 0.0
            x1130 = 0.0

            if f >= 3.517915e15:  # 853 Å edge
                x853 = _seaton(3.517915e15, 1.142e-17, 2.0, 4.29, f)
            if f >= 2.941534e15:  # 1020 Å edge
                x1020 = _seaton(2.941534e15, 4.41e-18, 1.5, 3.85, f)
            if f >= 2.653317e15:  # 1130 Å edge
                x1130 = _seaton(2.653317e15, 4.2e-18, 1.5, 4.34, f)

            n1op = x853 * 4.0 + x1020 * c1020 + x1130 * c1130  # (n_layers,)

            # O1OP: Oxygen I opacity (atlas7v.for line 9006-9019)
            x911 = 0.0
            if f >= 3.28805e15:  # 911 Å edge
                x911 = _seaton(3.28805e15, 2.94e-18, 1.0, 2.66, f)
            o1op = x911 * 9.0  # scalar, broadcast to layers

            # MG2OP: Magnesium II opacity (atlas7v.for line 9020-9042)
            c1169 = 6.0 * np.exp(-4.43 / tkev)
            x824 = 0.0
            x1169 = 0.0
            if f >= 3.635492e15:  # 824 Å edge
                x824 = _seaton(3.635492e15, 1.40e-19, 4.0, 6.7, f)
            if f >= 2.564306e15:  # 1169 Å edge
                x1169 = 5.11e-19 * (2.564306e15 / f) ** 3
            mg2op = x824 * 2.0 + x1169 * c1169  # (n_layers,)

            # SI2OP: Silicon II opacity (atlas7v.for line 9043-9097)
            # Uses Peach tables with temperature/frequency interpolation
            si2op = _si2op_vectorized(f, freqlg, temp, np.log(temp))

            # CA2OP: Calcium II opacity (atlas7v.for line 9098-9122)
            c1218 = 10.0 * np.exp(-1.697 / tkev)
            c1420 = 6.0 * np.exp(-3.142 / tkev)
            x1044 = 0.0
            x1218 = 0.0
            x1420 = 0.0
            if f >= 2.870454e15:  # 1044 Å edge
                x1044 = 5.4e-20 * (2.870454e15 / f) ** 3
            if f >= 2.460127e15:  # 1218 Å edge
                x1218 = 1.64e-17 * np.sqrt(2.460127e15 / f)
            if f >= 2.110779e15:  # 1420 Å edge
                x1420 = _seaton(2.110779e15, 4.13e-18, 3.0, 0.69, f)
            ca2op = x1044 * 2.0 + x1218 * c1218 + x1420 * c1420  # (n_layers,)

            # Match Fortran LUKEOP weighting by ion populations.
            aluke[:, j] = (
                n1op * xnfpn
                + o1op * xnfpo
                + mg2op * xnfpmg2
                + si2op * xnfpsi2
                + ca2op * xnfpca2
            ) * stim_j / rho
    else:
        logger.info("Skipping LUKEOP - IFOP(10)=0")

    # HOTOP: Hot star opacity (atlas7v.for line 9124-9251)
    # Free-free from C, N, O, Ne, Mg, Si, S, Fe ionization stages I-V
    # Only computed if IFOP(11) = 1
    if ifop[10] == 1:  # IFOP(11) in Fortran = ifop[10] in Python (0-indexed)
        logger.info("Computing HOTOP (hot star opacity)...")
        xne = np.asarray(atmosphere.electron_density, dtype=np.float64)
        tlog_arr = np.log(np.maximum(temp, 1e-10))
        tkev = KBOLTZ_EV * temp

        # Build HOTOP population vectors matching Fortran POPS calls:
        # XNFP(1:4)=C I-IV, XNFP(5:9)=N I-V, XNFP(10:15)=O I-VI, XNFP(16:21)=Ne I-VI.
        hotop_xnfp = np.zeros((n_layers, 21), dtype=np.float64)
        xnf_sumqq = np.zeros((n_layers, 5), dtype=np.float64)
        if has_pop_grid and pop.shape[1] > 5 and pop.shape[2] > 25:
            hotop_xnfp[:, 0:4] = pop[:, 0:4, 5]
            hotop_xnfp[:, 4:9] = pop[:, 0:5, 6]
            hotop_xnfp[:, 9:15] = pop[:, 0:6, 7]
            hotop_xnfp[:, 15:21] = pop[:, 0:6, 9]

            # XNFSUMQQ = sum_elements[ IZ^2 * XNF(IZ+1) ], IZ=1..5 (Fortran line 9281)
            for elem_idx in (5, 6, 7, 9, 11, 13, 15, 25):  # C,N,O,Ne,Mg,Si,S,Fe
                for iz in range(1, 6):
                    xnf_sumqq[:, iz - 1] += (iz * iz) * pop[:, iz, elem_idx]
        else:
            logger.warning(
                "HOTOP enabled but population_per_ion is unavailable/incomplete; using zero HOTOP populations."
            )

        sqrt_temp = np.sqrt(np.maximum(temp, 1e-30))
        exp_hot = np.exp(
            -HOTOP_TRANSITIONS[:, 5][np.newaxis, :] / np.maximum(tkev[:, np.newaxis], 1e-30)
        )
        hot_id_idx = np.clip(HOTOP_TRANSITIONS[:, 6].astype(np.int64) - 1, 0, 20)
        chunk = 4096

        for i0 in range(0, nfreq, chunk):
            i1 = min(i0 + chunk, nfreq)
            f_chunk = freq[i0:i1]
            stim_chunk = stim[:, i0:i1]
            freqlg_chunk = np.log(f_chunk)

            # FREE = sum_q COULFF(q) * XNFSUMQQ(q), q=1..5 (atlas7v.for line 9286-9288)
            free = np.zeros((n_layers, f_chunk.size), dtype=np.float64)
            for q in range(1, 6):
                free += _coulff_grid(q, freqlg_chunk, tlog_arr) * xnf_sumqq[:, q - 1][:, np.newaxis]

            ahot_chunk = (
                free
                * (3.6919e8 / (f_chunk[np.newaxis, :] ** 3))
                * (xne[:, np.newaxis] / sqrt_temp[:, np.newaxis])
            )

            # Bound-free additions from HOTOP transition table (atlas7v.for line 9291-9302)
            for k in range(HOTOP_TRANSITIONS.shape[0]):
                freq0, xsect0, alpha0, power0, mult0, _, _ = HOTOP_TRANSITIONS[k]
                use = f_chunk >= freq0
                if not np.any(use):
                    continue
                ratio = freq0 / f_chunk[use]
                xsect = xsect0 * (
                    alpha0 + ratio - alpha0 * ratio
                ) * np.sqrt(ratio ** int(power0))
                xx = (
                    xsect[np.newaxis, :]
                    * hotop_xnfp[:, hot_id_idx[k]][:, np.newaxis]
                    * mult0
                )
                threshold = ahot_chunk[:, use] / 100.0
                ahot_chunk[:, use] += np.where(
                    xx > threshold,
                    xx * exp_hot[:, k][:, np.newaxis],
                    0.0,
                )

            ahot[:, i0:i1] = ahot_chunk * stim_chunk / rho[:, np.newaxis]
    else:
        logger.info("Skipping HOTOP - IFOP(11)=0")

    # Molecular opacities for COOLOP (atlas7v.for lines 7302-7310)
    # ACOOL = AC1 + AMG1 + AAL1 + ASI1 + AFE1 + CHOP*XNFPCH + OHOP*XNFPOH + AH2COLL
    # C1, Mg1, Al1, Si1, Fe1 are already computed above
    # Now compute CHOP, OHOP, H2COLL for cool stars (T < 9000K)
    acool_mol = np.zeros((n_layers, nfreq), dtype=np.float64)

    # Check if we have cool temperatures (molecular opacities only matter for T < 9000K)
    t_min = temp.min()
    if t_min < 9000.0 and ifop[8] == 1:  # IFOP(9) = COOLOP enabled
        logger.info("Computing molecular opacities (CHOP, OHOP, H2COLL) for COOLOP...")

        # Get molecular populations if available
        # For now, use placeholder scaling - full implementation would need XNFPCH, XNFPOH
        # These populations come from molecular equilibrium (NMOLEC)
        xnfpch = getattr(atmosphere, "xnfpch", None)
        xnfpoh = getattr(atmosphere, "xnfpoh", None)

        # Use hydrogen and helium populations for H2 collision-induced
        tkev_arr = KBOLTZ_EV * temp
        tlog_arr = np.log(temp)

        for j in range(nfreq):
            f = freq[j]
            stim_j = stim[:, j]

            # CHOP: CH molecular opacity (scaled by CH population)
            if xnfpch is not None:
                chop_xsect = _chop_opacity(f, temp)
                acool_mol[:, j] += chop_xsect * xnfpch / rho * stim_j

            # OHOP: OH molecular opacity (scaled by OH population)
            if xnfpoh is not None:
                ohop_xsect = _ohop_opacity(f, temp)
                acool_mol[:, j] += ohop_xsect * xnfpoh / rho * stim_j

            # H2COLL: H2 collision-induced absorption
            # This is computed from H2 equilibrium, not a stored population
            if xnfph1 is not None:
                xnfhe1_arr = (
                    np.asarray(atmosphere.xnf_he1, dtype=np.float64)
                    if atmosphere.xnf_he1 is not None
                    else np.zeros(n_layers)
                )
                h2coll = _h2_collision_opacity(
                    f,
                    temp,
                    xnfph1,
                    bhyd[:, 0] if bhyd.shape[1] > 0 else np.ones(n_layers),
                    xnfhe1_arr,
                    rho,
                    tkev_arr,
                    tlog_arr,
                    stim_j,
                )
                acool_mol[:, j] += h2coll

        logger.info(
            f"  Molecular opacity range: [{acool_mol.min():.6e}, {acool_mol.max():.6e}]"
        )

    # Sum ACONT (atlas7v.for line 4571-4573)
    # a_base includes ALUKE, AHOT (if enabled)
    # Fortran KAPP (atlas7v.for line 6280) does NOT include ACOOL in ACONT.
    a_base = ah2p + ahemin + aluke + ahot
    acont = (
        a_base + ahyd + ahmin + axcont + ahe1 + ahe2 + ac1 + amg1 + aal1 + asi1 + afe1
    )

    # Compute SCONT (atlas7v.for line 4575-4579)
    scont = bnu_all.copy()
    mask = acont > 0
    numerator = (
        a_base * bnu_all
        + ahyd * shyd
        + ahmin * shmin
        + axcont * sxcont
        + ahe1 * she1
        + ahe2 * she2
        + ac1 * sc1
        + amg1 * smg1
        + aal1 * sal1
        + asi1 * ssi1
        + afe1 * sfe1
    )
    scont[mask] = numerator[mask] / acont[mask]

    # Sum SIGMAC (atlas7v.for line 4584)
    sigmac = sigh + sighe + sigel + sigh2 + sigx

    logger.info(
        f"KAPP continuum computed: ACONT range [{acont.min():.6e}, {acont.max():.6e}]"
    )

    return acont, sigmac, scont
