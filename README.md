# Kurucz Validation Workflow

This repo runs a practical Fortran-vs-Python validation loop on `samples/*.atm`.

## Current Workflow

1. Compile Fortran once.
2. For each atmosphere file:
   - Fortran run -> `results/validation_100/fortran_specs/<name>.spec`
   - Python run (`convert_atm_to_npz.py` + `synthe_py.cli`) ->
     - `results/validation_100/python_npz/<name>.npz`
     - `results/validation_100/python_specs/<name>.spec`
3. Inspect differences with `plot.py`.

Default validation settings:

- Wavelength: `300` to `1800` nm
- Resolution: `300000`

## Directory Structure (Relevant)

```text
kurucz/
├── bin/                         # Fortran executables
├── lines/                       # continua.dat, molecules.dat, gfallvac.latest, ...
├── samples/                     # Input atmospheres (*.atm)
├── synthe/Lines_v5_PL/          # Prebuilt Fortran tfort.* bundle
├── synthe_py/                   # Python pipeline
├── requirements.txt             # Python dependencies
├── compile_fortran.sh
├── run_fortran_atm.sh
├── run_validation_100.sh
├── plot.py
└── results/validation_100/
    ├── fortran_specs/
    ├── python_npz/
    ├── python_specs/
    └── logs/
```

## 1) Python Setup

Create a virtual environment and install dependencies:

```bash
cd /path/to/kurucz
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Key dependencies: `numpy`, `scipy`, `numba`, `matplotlib`.

## 2) Build Fortran

```bash
cd /path/to/kurucz
./compile_fortran.sh
```

## 3) Run Validation

Run all samples (both Fortran and Python):

```bash
cd /Users/ElliotKim/Desktop/Research/kurucz
./run_validation_100.sh -n all
```

Run one sample:

```bash
cd /Users/ElliotKim/Desktop/Research/kurucz
./run_validation_100.sh -n 1 --atm at12_aaaaa_t02500g-1.0.atm
```

Run only Python synthesis (skip Fortran):

```bash
./run_validation_100.sh -n all --python-only
./run_validation_100.sh -n 1 --atm at12_aaaaa_t02500g-1.0.atm --python-only
```

Run only Fortran synthesis (skip Python):

```bash
./run_validation_100.sh -n all --fortran-only
./run_validation_100.sh -n 1 --atm at12_aaaaa_t02500g-1.0.atm --fortran-only
```

Resume behavior is automatic:

- skips Fortran if output spec already exists
- skips Python if output spec already exists

## 4) Python-Only Pipeline

One-command wrapper (only parameters: `.atm`, `wl_start`, `wl_end`):

```bash
cd /Users/ElliotKim/Desktop/Research/kurucz
./run_python_pipeline.sh samples/at12_aaaaa_t02500g-1.0.atm 300 1800
```

Equivalent explicit commands:

```bash
cd /Users/ElliotKim/Desktop/Research/kurucz
python3 synthe_py/tools/convert_atm_to_npz.py \
  samples/at12_aaaaa_t02500g-1.0.atm \
  results/validation_100/python_npz/at12_aaaaa_t02500g-1.0.npz \
  --atlas-tables synthe_py/data/atlas_tables.npz

python3 -m synthe_py.cli \
  samples/at12_aaaaa_t02500g-1.0.atm \
  lines/gfallvac.latest \
  --npz results/validation_100/python_npz/at12_aaaaa_t02500g-1.0.npz \
  --spec results/validation_100/python_specs/at12_aaaaa_t02500g-1.0.spec \
  --wl-start 300 --wl-end 1800 \
  --resolution 300000 \
  --n-workers "$(python3 -c 'import os; print(os.cpu_count() or 1)')" \
  --cache synthe_py/out/line_cache \
  --log-level INFO
```

Diagnostics are opt-in only. If `--diagnostics` is not passed, no diagnostics NPZ is written by `synthe_py.cli`.

## Caching (Python Line Data)

Python synthesis uses two line-data cache layers:

- Parsed cache: stores parsed `gfallvac.latest` arrays (base parse cache).
- Compiled cache: stores runtime-ready compiled line arrays.

Cache location is controlled by `--cache` in `synthe_py.cli` (the validation script uses `synthe_py/out/line_cache`).

What affects cache reuse:

- Parsed cache key: source file fingerprint + cache logic version.
- Compiled cache key: source fingerprint plus `wlbeg`, `wlend`, `resolution`, and `line_filter`.

So if you change wavelength range or resolution, compiled cache entries are regenerated for those settings.

To disable caches explicitly:

```bash
PY_DISABLE_PARSED_CACHE=1 PY_DISABLE_COMPILED_CACHE=1 python3 -m synthe_py.cli ...
```

## 5) Plot Spectra

Default (uses built-in defaults in `plot.py`):

```bash
cd /Users/ElliotKim/Desktop/Research/kurucz
python3 plot.py
```

By atmosphere name:

```bash
python3 plot.py --atmosphere at12_aaaaa_t04250g2.50.spec
```

Explicit file paths:

```bash
python3 plot.py \
  --python-spec results/validation_100/python_specs/at12_aaaaa_t04250g2.50.spec \
  --fortran-spec results/validation_100/fortran_specs/at12_aaaaa_t04250g2.50.spec \
  --wl-start 300 --wl-end 1800
```

## 6) Compare Spectra (Text Metrics)

Use `synthe_py/tools/compare_spectra.py` to compute numeric agreement metrics.

Basic comparison:

```bash
cd /Users/ElliotKim/Desktop/Research/kurucz
python3 synthe_py/tools/compare_spectra.py \
  results/validation_100/python_specs/at12_aaaaa_t04250g2.50.spec \
  results/validation_100/fortran_specs/at12_aaaaa_t04250g2.50.spec
```

Restricted range with top outliers:

```bash
python3 synthe_py/tools/compare_spectra.py \
  results/validation_100/python_specs/at12_aaaaa_t04250g2.50.spec \
  results/validation_100/fortran_specs/at12_aaaaa_t04250g2.50.spec \
  --range 300 1800 \
  --top 20
```

What it reports:

- Flux mean / median / RMS relative difference (%)
- Continuum mean / median / RMS relative difference (%)
- Normalized flux RMS (`F/C`)
- Optional top-N fractional flux outlier wavelengths

## Notes

- Current Fortran validation path uses prebuilt `synthe/Lines_v5_PL/tfort.*` files.
- Legacy debug scripts are not required for this current flow.
