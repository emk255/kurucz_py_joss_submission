"""Command-line entry point for the Python SYNTHE workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from . import config
from .engine.opacity import run_synthesis
from .io import persist
from .utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Python SYNTHE reimplementation")
    parser.add_argument(
        "model", type=Path, help="Path to the model atmosphere file (.atm or .npz)"
    )
    parser.add_argument("atomic", type=Path, help="Atomic line catalog (e.g. gfallvac)")
    parser.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Explicit path to .npz atmosphere file (overrides automatic lookup for .atm files)",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("spectrum.spec"),
        help="Destination path for the synthesized spectrum",
    )
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=None,
        help="Optional path for detailed synthesis diagnostics (.npz)",
    )
    parser.add_argument(
        "--wl-start", type=float, default=300.0, help="Start wavelength (nm)"
    )
    parser.add_argument(
        "--wl-end", type=float, default=1800.0, help="End wavelength (nm)"
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=300_000.0,
        help="Resolving power lambda/dlambda",
    )
    parser.add_argument(
        "--microturb",
        type=float,
        default=0.0,
        help="Microturbulent velocity in km/s",
    )
    parser.add_argument(
        "--no-vacuum",
        action="store_true",
        help="Treat wavelengths as air instead of vacuum",
    )
    parser.add_argument(
        "--cutoff", type=float, default=1e-3, help="Opacity cutoff factor"
    )
    parser.add_argument(
        "--linout", type=int, default=30, help="Line output control flag"
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Optional directory for cached line data",
    )
    parser.add_argument(
        "--allow-tfort-runtime",
        action="store_true",
        help="Allow using tfort.* files as runtime line input (compatibility/debug mode only).",
    )
    parser.add_argument(
        "--fort20",
        type=Path,
        default=None,
        help="Optional fort.20 (line core) NPZ (deprecated, not used)",
    )
    parser.add_argument(
        "--fort29",
        type=Path,
        default=None,
        help="Optional fort.29 (ASYNTH) NPZ (deprecated, not used - wavelength grid built from config)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of parallel workers for radiative transfer (default: auto-detect, use 1 for sequential)",
    )
    parser.add_argument(
        "--no-helium-wings",
        action="store_true",
        help="Disable detailed helium wing profiles (faster but less accurate)",
    )
    parser.add_argument(
        "--skip-hydrogen-wings",
        action="store_true",
        help="Skip hydrogen wing computation (much faster, continuum only)",
    )
    parser.add_argument(
        "--no-line-filter",
        action="store_true",
        help="Disable wavelength filtering of the line catalog (matches full SYNTHE runs but slower)",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Subsample wavelength grid by taking every Nth point (e.g., 2 = half points, 10 = 10x faster)",
    )
    parser.add_argument(
        "--nlte", action="store_true", help="Enable NLTE line source handling"
    )
    parser.add_argument(
        "--scat-iterations",
        type=int,
        default=8,
        help="Maximum scattering iterations per frequency",
    )
    parser.add_argument(
        "--scat-tol",
        type=float,
        default=1e-3,
        help="Relative tolerance for scattering iteration convergence",
    )
    parser.add_argument(
        "--rhoxj",
        type=float,
        default=0.0,
        help="Scattering scale height RHOXJ (cm^-2). Use 0 for LTE core.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity level",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debug output for JOSH solver (verbose, use only for debugging)",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    diagnostics_path = args.diagnostics
    if diagnostics_path is None and args.spec is not None:
        diagnostics_path = args.spec.with_suffix(".npz")

    # Parse wavelength range if provided
    cfg = config.SynthesisConfig.from_cli(
        spec_path=args.spec,
        diagnostics_path=diagnostics_path,
        atmosphere_path=args.model,
        atomic_catalog=args.atomic,
        wl_start=args.wl_start,
        wl_end=args.wl_end,
        resolution=args.resolution,
        velocity_microturb=args.microturb,
        vacuum=not args.no_vacuum,
        cutoff=args.cutoff,
        linout=args.linout,
        nlte=args.nlte,
        scattering_iterations=args.scat_iterations,
        scattering_tolerance=args.scat_tol,
        fort20=args.fort20,
        fort29=args.fort29,
        rhoxj_scale=args.rhoxj,
        enable_helium_wings=not args.no_helium_wings,
        skip_hydrogen_wings=args.skip_hydrogen_wings,
        line_filter=not args.no_line_filter,
        wavelength_subsample=args.subsample,
        npz_path=args.npz,
        n_workers=args.n_workers,
        debug=args.debug,
        allow_tfort_runtime=args.allow_tfort_runtime,
    )
    if args.cache:
        cfg.line_data.cache_directory = args.cache
    cfg.log_level = args.log_level

    configure_logging(cfg.log_level)
    persist.ensure_cache_dirs(cfg)
    run_synthesis(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover - main guard
    raise SystemExit(main())
