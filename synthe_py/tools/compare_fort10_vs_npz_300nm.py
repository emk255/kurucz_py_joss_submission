#!/usr/bin/env python3
"""
Compare fort.10 ACONT/SIGMAC with Python npz at 300nm (and optionally other wavelengths).

Useful for investigating the ~30% JOSH discrepancy at scattering-dominated wavelengths:
  ALPHA = (SIGMAC + SIGMAL) / ABTOT
  If ALPHA differs (0.5 Python vs 0.003 Fortran at deep layers), ACONT/SIGMAC may differ.

Usage:
  # First run Fortran with KEEP_WORKDIR=1 to preserve fort.10:
  #   KEEP_WORKDIR=1 ./run_fortran_atm.sh samples/at12_aaaaa_t02500g-1.0.atm /tmp/out.spec
  # Then find the workdir (printed at end) and run:
  python synthe_py/tools/compare_fort10_vs_npz_300nm.py \\
    --fort10 /path/to/workdir/fort.10 \\
    --python-npz results/validation_100/python_npz/at12_aaaaa_t02500g-1.0.npz \\
    --wavelength 300

  # Or use convert_fort10 output if you have it:
  python synthe_py/tools/compare_fort10_vs_npz_300nm.py \\
    --fort10-npz /path/to/fort10_converted.npz \\
    --python-npz results/validation_100/python_npz/at12_aaaaa_t02500g-1.0.npz \\
    --wavelength 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from synthe_py.io.atmosphere import load_cached


def interpolate_acont_sigmac_at_wl(
    wledge: np.ndarray,
    cont_abs_coeff: np.ndarray,
    cont_scat_coeff: np.ndarray,
    half_edge: np.ndarray | None,
    delta_edge: np.ndarray | None,
    wavelength_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate ACONT and SIGMAC at a given wavelength from edge-based coefficients."""
    wledge_abs = np.abs(np.asarray(wledge, dtype=np.float64))
    n_layers = cont_abs_coeff.shape[0]
    n_edges = wledge_abs.size

    if half_edge is None:
        half_edge = 0.5 * (wledge_abs[:-1] + wledge_abs[1:])
    if delta_edge is None:
        delta_edge = wledge_abs[1:] - wledge_abs[:-1]

    edge_idx = np.searchsorted(wledge_abs, wavelength_nm, side="right") - 1
    edge_idx = np.clip(edge_idx, 0, n_edges - 2)

    wl_left = wledge_abs[edge_idx]
    wl_right = wledge_abs[edge_idx + 1]
    half = half_edge[edge_idx]
    delta = delta_edge[edge_idx]

    c1 = (wavelength_nm - half) * (wavelength_nm - wl_right) / delta
    c2 = (wl_left - wavelength_nm) * (wavelength_nm - wl_right) * 2.0 / delta
    c3 = (wavelength_nm - wl_left) * (wavelength_nm - half) / delta

    log_abs = (
        cont_abs_coeff[:, edge_idx, 0] * c1
        + cont_abs_coeff[:, edge_idx, 1] * c2
        + cont_abs_coeff[:, edge_idx, 2] * c3
    )
    log_scat = (
        cont_scat_coeff[:, edge_idx, 0] * c1
        + cont_scat_coeff[:, edge_idx, 1] * c2
        + cont_scat_coeff[:, edge_idx, 2] * c3
    )

    acont = 10.0**np.clip(log_abs, -50, 50)
    sigmac = 10.0**np.clip(log_scat, -50, 50)
    return acont, sigmac


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare fort.10 ACONT/SIGMAC with Python npz at a wavelength (default 300nm)"
    )
    parser.add_argument(
        "--fort10",
        type=Path,
        default=None,
        help="Path to fort.10 binary (from xnfpelsyn)",
    )
    parser.add_argument(
        "--fort10-npz",
        type=Path,
        default=None,
        help="Path to npz from convert_fort10 (alternative to --fort10)",
    )
    parser.add_argument(
        "--python-npz",
        type=Path,
        required=True,
        help="Path to Python validation npz (e.g. results/validation_100/python_npz/at12_aaaaa_t02500g-1.0.npz)",
    )
    parser.add_argument(
        "--wavelength",
        type=float,
        default=300.0,
        help="Wavelength in nm (default: 300)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=5,
        help="Number of layers to print in detail (default: 5)",
    )
    args = parser.parse_args()

    if args.fort10 is None and args.fort10_npz is None:
        print(
            "ERROR: Provide either --fort10 (path to fort.10) or --fort10-npz (path to convert_fort10 output)"
        )
        print(
            "  To get fort.10: KEEP_WORKDIR=1 ./run_fortran_atm.sh <atm> <out.spec>"
        )
        print(
            "  Then run: python synthe_py/tools/convert_fort10.py <workdir>/fort.10 fort10.npz"
        )
        return 1

    py_npz = Path(args.python_npz)
    if not py_npz.exists():
        print(f"ERROR: Python npz not found: {py_npz}")
        return 1

    # Load Python npz
    atm_py = load_cached(py_npz)
    if (
        atm_py.continuum_abs_coeff is None
        or atm_py.continuum_scat_coeff is None
        or atm_py.continuum_wledge is None
    ):
        print("ERROR: Python npz missing continuum coefficients (cont_abs_coeff, cont_scat_coeff, wledge)")
        return 1

    wl = args.wavelength

    # Get Fortran values
    if args.fort10_npz is not None:
        fort_npz = Path(args.fort10_npz)
        if not fort_npz.exists():
            print(f"ERROR: Fortran npz not found: {fort_npz}")
            return 1
        atm_ft = load_cached(fort_npz)
    else:
        # Convert fort.10 to npz in a temp file
        from synthe_py.tools.convert_fort10 import convert_fort10

        fort10_path = Path(args.fort10)
        if not fort10_path.exists():
            print(f"ERROR: fort.10 not found: {fort10_path}")
            return 1
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            tmp_npz = Path(f.name)
        try:
            convert_fort10(fort10_path, tmp_npz)
            atm_ft = load_cached(tmp_npz)
        finally:
            tmp_npz.unlink(missing_ok=True)

    if (
        atm_ft.continuum_abs_coeff is None
        or atm_ft.continuum_scat_coeff is None
        or atm_ft.continuum_wledge is None
    ):
        print("ERROR: Fortran npz missing continuum coefficients")
        return 1

    # Interpolate at wavelength
    wledge_py = np.asarray(atm_py.continuum_wledge, dtype=np.float64)
    wledge_ft = np.asarray(atm_ft.continuum_wledge, dtype=np.float64)

    acont_py, sigmac_py = interpolate_acont_sigmac_at_wl(
        wledge_py,
        atm_py.continuum_abs_coeff,
        atm_py.continuum_scat_coeff,
        getattr(atm_py, "continuum_half_edge", None),
        getattr(atm_py, "continuum_delta_edge", None),
        wl,
    )
    acont_ft, sigmac_ft = interpolate_acont_sigmac_at_wl(
        wledge_ft,
        atm_ft.continuum_abs_coeff,
        atm_ft.continuum_scat_coeff,
        getattr(atm_ft, "continuum_half_edge", None),
        getattr(atm_ft, "continuum_delta_edge", None),
        wl,
    )

    # Extend to same length if needed
    n = min(acont_py.size, acont_ft.size)
    acont_py = acont_py[:n]
    sigmac_py = sigmac_py[:n]
    acont_ft = acont_ft[:n]
    sigmac_ft = sigmac_ft[:n]

    # ALPHA (continuum only, no lines): alpha = SIGMAC / (ACONT + SIGMAC)
    abtot_ft = acont_ft + sigmac_ft
    abtot_py = acont_py + sigmac_py
    alpha_ft = np.where(abtot_ft > 1e-50, sigmac_ft / abtot_ft, 0.0)
    alpha_py = np.where(abtot_py > 1e-50, sigmac_py / abtot_py, 0.0)

    print("=" * 80)
    print(f"ACONT / SIGMAC / ALPHA COMPARISON at {wl:.2f} nm")
    print("=" * 80)
    print(f"Fortran source: {args.fort10 or args.fort10_npz}")
    print(f"Python npz:     {py_npz}")
    print(f"Layers: {n}")
    print()
    print(
        f"{'Layer':<6} {'ACONT_Ft':<14} {'ACONT_Py':<14} {'ACONT_rel%':<12} "
        f"{'SIGMAC_Ft':<14} {'SIGMAC_Py':<14} {'SIGMAC_rel%':<12} "
        f"{'ALPHA_Ft':<10} {'ALPHA_Py':<10}"
    )
    print("-" * 120)

    n_show = min(args.layers, n)
    for j in range(n_show):
        ac_rel = (
            100.0 * (acont_py[j] - acont_ft[j]) / max(acont_ft[j], 1e-50)
            if acont_ft[j] > 1e-50
            else 0.0
        )
        sc_rel = (
            100.0 * (sigmac_py[j] - sigmac_ft[j]) / max(sigmac_ft[j], 1e-50)
            if sigmac_ft[j] > 1e-50
            else 0.0
        )
        print(
            f"{j:<6} {acont_ft[j]:<14.6e} {acont_py[j]:<14.6e} {ac_rel:<12.2f} "
            f"{sigmac_ft[j]:<14.6e} {sigmac_py[j]:<14.6e} {sc_rel:<12.2f} "
            f"{alpha_ft[j]:<10.6f} {alpha_py[j]:<10.6f}"
        )

    if n > n_show:
        print("...")
        j = n - 1
        ac_rel = (
            100.0 * (acont_py[j] - acont_ft[j]) / max(acont_ft[j], 1e-50)
            if acont_ft[j] > 1e-50
            else 0.0
        )
        sc_rel = (
            100.0 * (sigmac_py[j] - sigmac_ft[j]) / max(sigmac_ft[j], 1e-50)
            if sigmac_ft[j] > 1e-50
            else 0.0
        )
        print(
            f"{j:<6} {acont_ft[j]:<14.6e} {acont_py[j]:<14.6e} {ac_rel:<12.2f} "
            f"{sigmac_ft[j]:<14.6e} {sigmac_py[j]:<14.6e} {sc_rel:<12.2f} "
            f"{alpha_ft[j]:<10.6f} {alpha_py[j]:<10.6f}"
        )

    # Summary
    ac_rel_all = np.where(
        acont_ft > 1e-50,
        100.0 * np.abs(acont_py - acont_ft) / acont_ft,
        0.0,
    )
    sc_rel_all = np.where(
        sigmac_ft > 1e-50,
        100.0 * np.abs(sigmac_py - sigmac_ft) / sigmac_ft,
        0.0,
    )
    alpha_diff = np.abs(alpha_py - alpha_ft)

    print()
    print("Summary:")
    print(f"  ACONT  max rel err: {np.max(ac_rel_all):.2f}%")
    print(f"  SIGMAC max rel err: {np.max(sc_rel_all):.2f}%")
    print(f"  ALPHA  max abs diff: {np.max(alpha_diff):.6f}")
    print(f"  ALPHA  at layer 0: Fortran={alpha_ft[0]:.6f} Python={alpha_py[0]:.6f}")
    print(
        f"  ALPHA  at layer {n-1}: Fortran={alpha_ft[-1]:.6f} Python={alpha_py[-1]:.6f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
