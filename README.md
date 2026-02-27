# Kurucz Python Synthesis

Python implementation of Kurucz-style spectrum synthesis. Reads `.atm` atmosphere files and produces spectra via `run_python_pipeline.sh`.

## Workflow

1. **Prerequisites**: `synthe_py/data/fortran_data.npz`, `synthe_py/data/atlas_tables.npz`, `lines/gfallvac.latest`
2. Run synthesis: `./run_python_pipeline.sh <atm_file> <wl_start> <wl_end>`
3. Plot: `python3 synthe_py/tools/plot.py --spec <stem>_<wl>_<wl>.spec`

Default wavelength: 300–1800 nm, resolution 300000.

## Directory Structure

```text
kurucz_python/
├── lines/                       # gfallvac.latest, continua.dat, molecules.dat, ...
├── samples/                     # Input atmospheres (*.atm)
├── synthe_py/
│   ├── data/                    # atlas_tables.npz, fortran_data.npz, ...
│   ├── out/line_cache/          # Parsed & compiled line caches (populated on first run)
│   ├── tools/                   # convert_atm_to_npz, plot.py, extract_*, ...
│   └── ...
├── results/
│   ├── npz/                     # Converted atmosphere NPZ files
│   ├── spec/                    # Output spectra (*.spec)
│   ├── logs/                    # Pipeline logs
│   └── plots/                   # Saved plot images (from synthe_py/tools/plot.py)
├── requirements.txt
└── run_python_pipeline.sh
```

## 1) Python Setup

```bash
cd /path/to/kurucz_python
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Key dependencies: `numpy`, `scipy`, `numba`, `matplotlib`.

## 2) Prerequisites

Ensure these exist before running the pipeline:

The pipeline script checks only `atlas_tables.npz` and `gfallvac.latest`; missing `fortran_data.npz`, `pfiron_data.npz`, or `kapp_tables.npz` will cause `convert_atm_to_npz` to fail at runtime.

## 3) Run Synthesis

One-command wrapper (parameters: `.atm` file, `wl_start`, `wl_end`):

```bash
cd /path/to/kurucz_python
./run_python_pipeline.sh samples/at12_aaaaa_t02500g-1.0.atm 300 1800
```

Output:

- `results/npz/at12_aaaaa_t02500g-1.0.npz`
- `results/spec/at12_aaaaa_t02500g-1.0_300_1800.spec`
- `results/logs/at12_aaaaa_t02500g-1.0_300_1800.log`

Equivalent explicit commands:

```bash
python3 synthe_py/tools/convert_atm_to_npz.py \
  samples/at12_aaaaa_t02500g-1.0.atm \
  results/npz/at12_aaaaa_t02500g-1.0.npz \
  --atlas-tables synthe_py/data/atlas_tables.npz

python3 -m synthe_py.cli \
  samples/at12_aaaaa_t02500g-1.0.atm \
  lines/gfallvac.latest \
  --npz results/npz/at12_aaaaa_t02500g-1.0.npz \
  --spec results/spec/at12_aaaaa_t02500g-1.0_300_1800.spec \
  --wl-start 300 --wl-end 1800 \
  --resolution 300000 \
  --n-workers "$(python3 -c 'import os; print(os.cpu_count() or 1)')" \
  --cache synthe_py/out/line_cache \
  --log-level INFO
```

## 4) Caching (Line Data)

Python synthesis uses two cache layers in `synthe_py/out/line_cache`:

- **Parsed cache**: parsed `gfallvac.latest` arrays
- **Compiled cache**: runtime-ready compiled line arrays (keyed by wavelength range + resolution)

Cache is populated on first run. To disable:

```bash
PY_DISABLE_PARSED_CACHE=1 PY_DISABLE_COMPILED_CACHE=1 python3 -m synthe_py.cli ...
```

## 5) Plot Spectra

Plot normalized flux vs wavelength (default: `at12_aaaaa_t02500g-1.0_300_1800.spec`):

```bash
python3 synthe_py/tools/plot.py
```

By spectrum filename (under `results/spec/`):

```bash
python3 synthe_py/tools/plot.py --spec at12_aaaaa_t02500g-1.0_300_1800.spec
```

Wavelength range and save path:

```bash
python3 synthe_py/tools/plot.py --spec at12_aaaaa_t02500g-1.0_300_1800.spec \
  --wl-start 400 --wl-end 700 --save spectrum.png --no-show
```

## 6) Testing (Validation vs Fortran Ground Truth)

To validate the Python synthesis against Fortran reference spectra, you need the test samples and ground-truth Fortran specs. These are **not included in the repository** (to keep the codebase minimal) but are available from Google Drive.

### 1. Download validation data

1. Download the validation bundle from: **[Google Drive: kurucz_py_joss_submission_data](https://drive.google.com/drive/folders/1E90whMPn-Dsf0H6eFAb60XFA2gtxPzEl?usp=drive_link)**

2. The bundle contains:
   - `samples/` — atmosphere files (`.atm`) to run through the pipeline
   - `fortran_specs/` — reference spectra computed with the original Fortran SYNTHE

### 2. Set up directories

```bash
cd /path/to/kurucz_python

# Replace the repo's samples/ with the downloaded samples
rm -rf samples
# Extract or copy the downloaded samples/ here (e.g. unzip, or cp -r from Drive sync)

# Create fortran_specs/ and place the downloaded Fortran spectra
mkdir -p fortran_specs
# Copy the downloaded fortran_specs/*.spec files into fortran_specs/
```

### 3. Run synthesis for all samples

```bash
for atm in samples/*.atm; do
  ./run_python_pipeline.sh "$atm" 300 1800
done
```

### 4. Compare against ground truth

For each sample, run `compare_spectra.py` to compute flux/continuum agreement metrics:

```bash
for atm in samples/*.atm; do
  stem=$(basename "$atm" .atm)
  echo "=== $stem ==="
  python3 synthe_py/tools/compare_spectra.py \
    results/spec/${stem}_300_1800.spec \
    fortran_specs/${stem}.spec \
    --range 300 1800 --top 5
done
```

Or compare a single pair:

```bash
python3 synthe_py/tools/compare_spectra.py \
  results/spec/at12_aaaaa_t02500g-1.0_300_1800.spec \
  fortran_specs/at12_aaaaa_t02500g-1.0.spec \
  --range 300 1800 --top 20
```

The tool reports flux/continuum mean/median/RMS relative difference (%), normalized flux RMS, and top outlier wavelengths.