#!/usr/bin/env python3
"""Compare Python and Fortran spectra and compute statistics."""

import numpy as np
from pathlib import Path
import sys


def load_spectrum(filepath: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load spectrum file with wavelength, flux, continuum columns."""
    wavelengths = []
    fluxes = []
    continua = []

    with open(filepath, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    wl = float(parts[0])
                    flux = float(parts[1])
                    cont = float(parts[2])
                    wavelengths.append(wl)
                    fluxes.append(flux)
                    continua.append(cont)
                except ValueError:
                    continue

    return np.array(wavelengths), np.array(fluxes), np.array(continua)


def compare_spectra(python_file: Path, fortran_file: Path, wl_range: tuple = None):
    """Compare two spectra and print statistics."""

    print(f"Loading Python spectrum: {python_file}")
    py_wl, py_flux, py_cont = load_spectrum(python_file)
    print(
        f"  {len(py_wl)} points, wavelength range: {py_wl.min():.2f} - {py_wl.max():.2f} nm"
    )

    print(f"Loading Fortran spectrum: {fortran_file}")
    ft_wl, ft_flux, ft_cont = load_spectrum(fortran_file)
    print(
        f"  {len(ft_wl)} points, wavelength range: {ft_wl.min():.2f} - {ft_wl.max():.2f} nm"
    )

    # Find common wavelength range
    wl_min = max(py_wl.min(), ft_wl.min())
    wl_max = min(py_wl.max(), ft_wl.max())

    if wl_range:
        wl_min = max(wl_min, wl_range[0])
        wl_max = min(wl_max, wl_range[1])

    print(f"\nComparing in range: {wl_min:.2f} - {wl_max:.2f} nm")

    # Filter to common range
    py_mask = (py_wl >= wl_min) & (py_wl <= wl_max)
    ft_mask = (ft_wl >= wl_min) & (ft_wl <= wl_max)

    py_wl_common = py_wl[py_mask]
    py_flux_common = py_flux[py_mask]
    py_cont_common = py_cont[py_mask]

    ft_wl_common = ft_wl[ft_mask]
    ft_flux_common = ft_flux[ft_mask]
    ft_cont_common = ft_cont[ft_mask]

    # Interpolate Fortran to Python wavelengths
    ft_flux_interp = np.interp(py_wl_common, ft_wl_common, ft_flux_common)
    ft_cont_interp = np.interp(py_wl_common, ft_wl_common, ft_cont_common)

    n_points = len(py_wl_common)
    print(f"Comparing {n_points} wavelength points")

    # Relative differences in percent (avoid division by zero)
    flux_rel = np.where(
        np.abs(ft_flux_interp) > 1e-30,
        100 * (py_flux_common - ft_flux_interp) / ft_flux_interp,
        0,
    )
    cont_rel = np.where(
        np.abs(ft_cont_interp) > 1e-30,
        100 * (py_cont_common - ft_cont_interp) / ft_cont_interp,
        0,
    )

    # Normalized flux (flux / continuum)
    py_norm = py_flux_common / np.maximum(py_cont_common, 1e-30)
    ft_norm = ft_flux_interp / np.maximum(ft_cont_interp, 1e-30)
    norm_diff = py_norm - ft_norm

    # Compute summary statistics
    flux_rms = np.sqrt(np.mean(flux_rel**2))
    cont_rms = np.sqrt(np.mean(cont_rel**2))

    print("\n" + "=" * 60)
    print("SPECTRUM COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n{'Column':<20} {'Mean':<12} {'Median':<12} {'RMS':<12}")
    print("-" * 56)
    print(
        f"{'Flux':<20} {np.mean(flux_rel):+.2f}%{'':<5} {np.median(flux_rel):+.2f}%{'':<5} {flux_rms:.2f}%"
    )
    print(
        f"{'Continuum':<20} {np.mean(cont_rel):+.2f}%{'':<5} {np.median(cont_rel):+.2f}%{'':<5} {cont_rms:.2f}%"
    )
    print(
        f"{'Normalized (F/C)':<20} {np.mean(norm_diff):+.4f}{'':<3} {np.median(norm_diff):+.4f}{'':<3} {np.sqrt(np.mean(norm_diff**2)):.4f}"
    )
    print()

    # Status check
    cont_status = "✅" if cont_rms < 1.0 else "❌"
    flux_status = "✅" if flux_rms < 1.0 else "❌"
    print(f"Sub-percent accuracy: Continuum {cont_status}  Flux {flux_status}")
    print("=" * 60)

    return {
        "flux_mean_rel": np.mean(flux_rel),
        "flux_median_rel": np.median(flux_rel),
        "flux_rms_rel": flux_rms,
        "cont_mean_rel": np.mean(cont_rel),
        "cont_median_rel": np.median(cont_rel),
        "cont_rms_rel": cont_rms,
        "norm_rms": np.sqrt(np.mean(norm_diff**2)),
    }


if __name__ == "__main__":
    python_file = Path("synthe_py/out/test_fixed_3750.spec")
    fortran_file = Path("grids/at12_aaaaa/spec/at12_aaaaa_t03750g3.50.spec")

    if len(sys.argv) > 2:
        python_file = Path(sys.argv[1])
        fortran_file = Path(sys.argv[2])

    if not python_file.exists():
        print(f"Error: Python spectrum not found: {python_file}")
        sys.exit(1)
    if not fortran_file.exists():
        print(f"Error: Fortran spectrum not found: {fortran_file}")
        sys.exit(1)

    compare_spectra(python_file, fortran_file)
