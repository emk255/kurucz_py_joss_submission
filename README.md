# Kurucz Stellar Synthesis Pipeline

This repository contains both the original **Kurucz Fortran code** for stellar atmosphere synthesis and a **Python reimplementation** (`synthe_py`) that aims to achieve sub-percent accuracy with the Fortran implementation.

## Overview

This project stems from the existing Kurucz Fortran synthesis code (located in `src/` and related directories) and provides a modern Python reimplementation that computes stellar spectra from first principles, matching Fortran's physics exactly. The Python implementation is designed to be:

- **Self-contained**: No reliance on Fortran-generated intermediate files
- **Accurate**: Sub-percent (<1%) agreement with Fortran results
- **Verifiable**: Direct comparison tools for validation

## Directory Structure

```
kurucz/
├── src/                    # Original Kurucz Fortran source code
│   ├── atlas7v.for        # ATLAS atmosphere code
│   ├── xnfpelsyn.for      # Population/continuum computation
│   ├── spectrv.for         # Spectrum synthesis
│   ├── synthe.for          # Line synthesis
│   └── Makefile            # Fortran build configuration
│
├── bin/                    # Compiled Fortran executables
│   ├── xnfpelsyn.exe
│   ├── spectrv.exe
│   ├── synthe.exe
│   └── ...
│
├── synthe/                 # Fortran synthesis workflow
│   ├── synthe.sh           # Main Fortran synthesis script
│   ├── synthe_debug.sh     # Debug version with output capture
│   ├── stmp_*/             # Temporary working directories
│   └── Lines_v5_PL/        # Line list data files
│
├── synthe_py/              # Python reimplementation
│   ├── __main__.py         # CLI entry point
│   ├── cli.py              # Command-line interface
│   ├── engine/             # Core synthesis engine
│   │   ├── opacity.py      # Opacity computation
│   │   ├── radiative.py    # Radiative transfer
│   │   └── transport.py    # Transport solver
│   ├── io/                 # Input/output handling
│   │   ├── atmosphere.py   # Atmosphere model loading
│   │   ├── lines/          # Line list parsing
│   │   └── export.py       # Spectrum export
│   ├── physics/            # Physics modules
│   │   ├── kapp.py         # Continuum opacity (KAPP)
│   │   └── ...             # Other physics routines
│   ├── tools/              # Utility tools
│   │   ├── convert_atm_to_npz.py  # Convert .atm to NPZ format
│   │   ├── compare_spectra.py      # Compare Python vs Fortran
│   │   └── ...             # Other tools
│   └── data/               # Precomputed data tables (.npz)
│
├── grids/                  # Atmosphere model grids
│   └── at12_aaaaa/         # Example grid
│       ├── *.atm           # Atmosphere files
│       └── spec/           # Fortran-generated spectra
│
├── lines/                  # Line list data files
│   ├── gfallvac.latest     # Atomic line list
│   ├── molecules.dat       # Molecular data
│   ├── continua.dat        # Continuum edges
│   └── he1tables.dat       # Helium tables
│
├── compile_fortran.sh      # Compile Fortran code
├── run_fortran_with_debug.sh  # Run Fortran with debug output
├── plot.py                 # Visualization tool
└── tests/                  # Test files
```

## Requirements

### Fortran Compilation

- **Intel Fortran** (`ifort`) or **GNU Fortran** (`gfortran`)
- For Intel Fortran: Intel oneAPI HPC Toolkit
- For GNU Fortran: `brew install gcc` (macOS) or system package manager

### Python

- Python 3.8+
- Required packages: `numpy`, `scipy`, `numba`, `matplotlib`
- See `synthe_py/` for full dependency list

## Fortran Workflow

### 1. Compile Fortran Code

```bash
./compile_fortran.sh
```

This script:

- Detects available Fortran compiler (Intel or GNU)
- Compiles `atlas7v.for` and related source files
- Produces executables in `bin/`

### 2. Run Fortran Synthesis

```bash
./run_fortran_with_debug.sh <model_name>
```

Example:

```bash
./run_fortran_with_debug.sh at12_aaaaa
```

This script:

- Runs the complete Fortran synthesis pipeline
- Captures debug output to `fortran_debug_<model>.log`
- Generates spectra in `grids/<model>/spec/`

The Fortran workflow processes `.atm` atmosphere files through:

1. `xnfpelsyn.exe` - Computes populations and continuum opacities
2. `synthe.exe` - Synthesizes spectral lines
3. `spectrv.exe` - Performs radiative transfer

## Python Workflow

### 1. Convert Atmosphere File to NPZ Format

Convert a `.atm` file to the NPZ format used by the Python pipeline:

```bash
python synthe_py/tools/convert_atm_to_npz.py \
    grids/at12_aaaaa/atm/at12_aaaaa_t03750g3.50.atm \
    output.npz
```

This tool automatically finds required data files (`atlas_tables.npz`, `continua.dat`, `molecules.dat`) in standard locations. It:

- Parses the `.atm` file
- Computes populations using exact POPS/PFSAHA implementation
- Computes continuum opacities using KAPP
- Generates Doppler broadening coefficients
- Outputs a complete NPZ file ready for synthesis

### 2. Run Python Synthesis

Use the CLI pipeline to synthesize a spectrum. You can use either a `.atm` file directly (with `--npz` to specify the NPZ output) or a pre-converted `.npz` file:

```bash
python -m synthe_py \
    grids/at12_aaaaa/atm/at12_aaaaa_t03750g3.50.atm \
    lines/gfallvac.latest \
    --npz output.npz \
    --wl-range 300:1800 \
    --spec spectrum.spec
```

Or with a pre-converted NPZ file:

```bash
python -m synthe_py \
    output.npz \
    lines/gfallvac.latest \
    --wl-range 300:1800 \
    --spec spectrum.spec
```

Key arguments:

- First argument - Atmosphere model (`.atm` or `.npz` file)
- Second argument - Atomic line list (e.g., `lines/gfallvac.latest`)
- `--npz` - Output NPZ file (only needed when using `.atm` input)
- `--spec` - Output spectrum file
- `--wl-range` - Wavelength range as `start:end` (nm)

## Comparison and Evaluation

### Compare Spectra

Compare Python and Fortran spectra:

```bash
python synthe_py/tools/compare_spectra.py \
    synthe_py/out/test_fixed_3750.spec \
    grids/at12_aaaaa/spec/at12_aaaaa_t03750g3.50.spec
```

This tool:

- Loads both spectrum files
- Interpolates to common wavelength grid
- Computes relative differences (mean, median, RMS)
- Reports sub-percent accuracy status

### Visualize Spectra

Use `plot.py` to visualize and compare spectra:

```bash
python plot.py
```

Edit `plot.py` to set paths to your spectrum files. The script creates side-by-side plots of normalized spectra.

## Accuracy Goals

The Python implementation targets **sub-percent (<1%) accuracy** with Fortran for:

- Continuum flux at all wavelengths
- Total flux (including spectral lines)
- All stellar temperature regimes (2500K - 10000K+)

See `SYNTHESIS_ACCURACY_PROGRESS.md` for detailed progress tracking and bug fixes.

## Key Features

### Python Implementation (`synthe_py`)

- **Exact POPS/PFSAHA**: Population computation matching Fortran exactly
- **Full KAPP**: Continuum opacity computation with all sources (H, He, metals, molecules)
- **NMOLEC**: Molecular equilibrium for cool stars
- **Line Synthesis**: Complete line opacity and radiative transfer
- **No Fortran Dependencies**: Computes everything from first principles

### Fortran Code (`src/`)

- Original Kurucz synthesis code
- Reference implementation for validation
- Used for generating ground-truth spectra

## Notes

- The Python implementation is designed to match Fortran physics exactly, not to improve upon it
- All physics constants and algorithms are matched to Fortran values
- Debug output from both implementations can be compared for detailed validation
- The NPZ format stores precomputed populations, opacities, and coefficients for efficient synthesis

## License

This repository contains Kurucz Fortran code (see original Kurucz distribution for licensing) and a Python reimplementation. Please refer to individual source files for licensing information.
