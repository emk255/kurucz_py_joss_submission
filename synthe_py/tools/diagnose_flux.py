#!/usr/bin/env python3
"""Diagnostic script to compare Python vs Fortran intermediate values step-by-step."""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from synthe_py.io.atmosphere import load_cached
from synthe_py.engine.opacity import run_synthesis
from synthe_py.config import (
    SynthesisConfig,
    WavelengthGrid,
    LineDataConfig,
    AtmosphereInput,
    OutputConfig,
)


def main():
    # Load diagnostics if available
    diag_path = Path("synthe_py/out/diagnostics_debug.npz")
    if diag_path.exists():
        print("Loading diagnostics...")
        diag = np.load(diag_path)

        print("\n=== DIAGNOSTIC VALUES (First wavelength point) ===")
        print(f"Wavelength: {diag['wavelength'][0]:.8f} nm")
        print(
            f"\nContinuum absorption (first layer): {diag['continuum_absorption'][0, 0]:.6E}"
        )
        print(
            f"Continuum scattering (first layer): {diag['continuum_scattering'][0, 0]:.6E}"
        )
        print(f"Line opacity (first layer): {diag['line_opacity'][0, 0]:.6E}")
        print(f"Line scattering (first layer): {diag['line_scattering'][0, 0]:.6E}")
        print(f"Line source (first layer): {diag['line_source'][0, 0]:.6E}")
        print(f"\nFlux total: {diag['flux_total'][0]:.6E}")
        print(f"Flux continuum: {diag['flux_continuum'][0]:.6E}")
        print(
            f"Ratio (flux/cont): {diag['flux_total'][0] / diag['flux_continuum'][0]:.6f}"
        )

        # Check unit conversion
        wavelength_cm = diag["wavelength"][0] * 1e-7  # nm to cm
        C_LIGHT_CM = 2.99792458e10  # cm/s
        conversion_python = 4.0 * np.pi * C_LIGHT_CM / (wavelength_cm**2) * 1e-8
        conversion_fortran = 2.99792458e17 / (diag["wavelength"][0] ** 2)

        print(f"\n=== UNIT CONVERSION COMPARISON ===")
        print(f"Python conversion factor: {conversion_python:.6E}")
        print(f"Fortran conversion factor: {conversion_fortran:.6E}")
        print(f"Ratio (Python/Fortran): {conversion_python / conversion_fortran:.6f}")
        print(f"Expected ratio: {4.0 * np.pi:.6f}")

        # What would flux be with Fortran conversion?
        # Need to get flux before conversion - check if we can compute it
        print(f"\n=== FLUX BEFORE CONVERSION (estimated) ===")
        flux_before_python = diag["flux_total"][0] / conversion_python
        flux_before_fortran_est = diag["flux_total"][0] / conversion_fortran
        print(f"Flux before conversion (Python): {flux_before_python:.6E}")
        print(f"Flux before conversion (Fortran est): {flux_before_fortran_est:.6E}")

    else:
        print(f"Diagnostics file not found: {diag_path}")
        print("Run synthesis with --diagnostics flag first")


if __name__ == "__main__":
    main()
