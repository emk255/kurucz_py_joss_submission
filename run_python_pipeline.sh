#!/bin/bash
# Minimal Python-only synthesis wrapper.
# Parameters (and only parameters): ATM file, wl_start, wl_end.
#
# Usage:
#   ./run_python_pipeline.sh <atm_file> <wl_start> <wl_end>
#
# Example:
#   ./run_python_pipeline.sh samples/at12_aaaaa_t02500g-1.0.atm 300 1800

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <atm_file> <wl_start> <wl_end>"
    exit 1
fi

ATM_FILE="$1"
WL_START="$2"
WL_END="$3"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

if [ ! -f "${ATM_FILE}" ]; then
    echo "ERROR: Atmosphere file not found: ${ATM_FILE}"
    exit 1
fi
if [ ! -f "${REPO_ROOT}/lines/gfallvac.latest" ]; then
    echo "ERROR: Missing line list: ${REPO_ROOT}/lines/gfallvac.latest"
    exit 1
fi
if [ ! -f "${REPO_ROOT}/synthe_py/data/atlas_tables.npz" ]; then
    echo "ERROR: Missing atlas tables: ${REPO_ROOT}/synthe_py/data/atlas_tables.npz"
    exit 1
fi

RESOLUTION="300000"
N_WORKERS="$(python3 -c 'import os; print(os.cpu_count() or 1)')"

sanitize_float() {
    # Keep filenames shell-safe and deterministic.
    printf "%s" "$1" | sed 's/\./p/g'
}

ATM_STEM="$(basename "${ATM_FILE%.atm}")"
WL_START_TAG="$(sanitize_float "${WL_START}")"
WL_END_TAG="$(sanitize_float "${WL_END}")"

OUT_ROOT="${REPO_ROOT}/results"
OUT_NPZ_DIR="${OUT_ROOT}/npz"
OUT_SPEC_DIR="${OUT_ROOT}/spec"
OUT_LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${OUT_NPZ_DIR}" "${OUT_SPEC_DIR}" "${OUT_LOG_DIR}"

NPZ_OUT="${OUT_NPZ_DIR}/${ATM_STEM}.npz"
SPEC_OUT="${OUT_SPEC_DIR}/${ATM_STEM}_${WL_START_TAG}_${WL_END_TAG}.spec"
LOG_OUT="${OUT_LOG_DIR}/${ATM_STEM}_${WL_START_TAG}_${WL_END_TAG}.log"

{
    echo "[1/2] convert_atm_to_npz.py"
    python3 synthe_py/tools/convert_atm_to_npz.py \
        "${ATM_FILE}" \
        "${NPZ_OUT}" \
        --atlas-tables "${REPO_ROOT}/synthe_py/data/atlas_tables.npz"

    echo "[2/2] python synthesis"
    python3 -m synthe_py.cli \
        "${ATM_FILE}" \
        "${REPO_ROOT}/lines/gfallvac.latest" \
        --npz "${NPZ_OUT}" \
        --spec "${SPEC_OUT}" \
        --wl-start "${WL_START}" \
        --wl-end "${WL_END}" \
        --resolution "${RESOLUTION}" \
        --n-workers "${N_WORKERS}" \
        --cache "${REPO_ROOT}/synthe_py/out/line_cache" \
        --log-level INFO
} > "${LOG_OUT}" 2>&1

echo "Done."
echo "NPZ:  ${NPZ_OUT}"
echo "SPEC: ${SPEC_OUT}"
echo "LOG:  ${LOG_OUT}"


