"""Configuration models for the Python SYNTHE pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class WavelengthGrid:
    """Defines the spectral sampling to synthesise."""

    start: float
    end: float
    resolution: float
    velocity_microturb: float = 0.0
    vacuum: bool = True


@dataclass
class LineDataConfig:
    """Controls which line lists are included in the synthesis.

    Note: Populations are computed from Saha-Boltzmann equations (no fort.10 dependency).
    Line opacity is computed from first principles using the atomic catalog.
    """

    atomic_catalog: Path
    molecular_catalogs: List[Path] = field(default_factory=list)
    include_predicted: bool = False
    cache_directory: Optional[Path] = None
    allow_tfort_runtime: bool = False


@dataclass
class AtmosphereInput:
    """Describes the input model atmosphere."""

    model_path: Path
    format: str = "atlas12"  # could support atlas9, phoenix, etc.
    npz_path: Optional[Path] = None  # Optional explicit path to .npz file


@dataclass
class OutputConfig:
    """Specifies the artefacts produced by the pipeline."""

    spec_path: Path
    diagnostics_path: Optional[Path] = None


@dataclass
class SynthesisConfig:
    """Global settings for a SYNTHE run."""

    wavelength_grid: WavelengthGrid
    line_data: LineDataConfig
    atmosphere: AtmosphereInput
    output: OutputConfig
    cutoff: float = 1e-3
    linout: int = 30
    nlte: bool = False
    scattering_iterations: int = 8
    scattering_tolerance: float = 1e-3
    rhoxj_scale: float = 0.0
    log_level: str = "INFO"
    enable_helium_wings: bool = True
    skip_hydrogen_wings: bool = False
    line_filter: bool = True
    wavelength_subsample: int = 1
    n_workers: Optional[int] = (
        None  # Number of parallel workers for radiative transfer (None = auto, 1 = sequential)
    )

    @classmethod
    def from_cli(
        cls,
        spec_path: Path,
        diagnostics_path: Optional[Path],
        atmosphere_path: Path,
        atomic_catalog: Path,
        wl_start: float,
        wl_end: float,
        resolution: float,
        velocity_microturb: float = 0.0,
        vacuum: bool = True,
        cutoff: float = 1e-3,
        linout: int = 30,
        nlte: bool = False,
        scattering_iterations: int = 8,
        scattering_tolerance: float = 1e-3,
        rhoxj_scale: float = 0.0,
        enable_helium_wings: bool = True,
        skip_hydrogen_wings: bool = False,
        line_filter: bool = True,
        wavelength_subsample: int = 1,
        npz_path: Optional[Path] = None,
        n_workers: Optional[int] = None,
        allow_tfort_runtime: bool = False,
    ) -> "SynthesisConfig":
        """Helper for the default CLI entry point."""

        return cls(
            wavelength_grid=WavelengthGrid(
                start=wl_start,
                end=wl_end,
                resolution=resolution,
                velocity_microturb=velocity_microturb,
                vacuum=vacuum,
            ),
            line_data=LineDataConfig(
                atomic_catalog=atomic_catalog,
                allow_tfort_runtime=allow_tfort_runtime,
            ),
            atmosphere=AtmosphereInput(model_path=atmosphere_path, npz_path=npz_path),
            output=OutputConfig(spec_path=spec_path, diagnostics_path=diagnostics_path),
            cutoff=cutoff,
            linout=linout,
            nlte=nlte,
            scattering_iterations=scattering_iterations,
            scattering_tolerance=scattering_tolerance,
            rhoxj_scale=rhoxj_scale,
            enable_helium_wings=enable_helium_wings,
            skip_hydrogen_wings=skip_hydrogen_wings,
            line_filter=line_filter,
            wavelength_subsample=wavelength_subsample,
            n_workers=n_workers,
        )


DEFAULT_WAVELENGTH: Tuple[float, float, float] = (300.0, 1800.0, 300_000.0)
"""Default wavelength grid (start, end, resolving power)."""
