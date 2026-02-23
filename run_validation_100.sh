#!/bin/bash
# Simple runner:
# - Runs Fortran on .atm from samples/
# - Runs Python conversion + synthesis on same .atm
# - Stores outputs in a structured folder
#
# Defaults are fixed:
#   wavelength = 300 to 1800 nm
#   resolution = 300000
#
# Usage:
#   ./run_validation_100.sh -n all
#   ./run_validation_100.sh -n 1 --atm at12_feh+0.25_afe+0.6_t02500g-1.0.atm
#   ./run_validation_100.sh -n all --python-only
#   ./run_validation_100.sh -n all --fortran-only

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

SAMPLES_DIR="${REPO_ROOT}/samples"
RESULT_ROOT="${REPO_ROOT}/results/validation_100"
FORTRAN_SPECS="${RESULT_ROOT}/fortran_specs"
PYTHON_NPZ="${RESULT_ROOT}/python_npz"
PYTHON_SPECS="${RESULT_ROOT}/python_specs"
LOG_DIR="${RESULT_ROOT}/logs"

WL_START=300
WL_END=1800
RESOLUTION=300000
N_WORKERS="$(python3 -c 'import os; print(os.cpu_count() or 1)')"

MODE="all"  # all | 1
ATM_ONE=""
RUN_MODE="both"  # both | python | fortran

usage() {
    cat <<EOF
Usage:
  $0 -n all
  $0 -n 1 --atm <atm file name or path>
  $0 -n all --python-only
  $0 -n all --fortran-only

Options:
  -n <all|1>       Run all samples or exactly one atmosphere
  --atm <value>    Required when -n 1 (filename in samples/ or full path)
  --python-only    Run only Python synthesis (skip Fortran)
  --fortran-only   Run only Fortran synthesis (skip Python)
  -h, --help       Show this help
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        -n)
            MODE="$2"
            shift 2
            ;;
        --atm)
            ATM_ONE="$2"
            shift 2
            ;;
        --python-only)
            RUN_MODE="python"
            shift
            ;;
        --fortran-only)
            RUN_MODE="fortran"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if [ "${MODE}" != "all" ] && [ "${MODE}" != "1" ]; then
    echo "ERROR: -n must be 'all' or '1'"
    exit 1
fi

if [ ! -d "${SAMPLES_DIR}" ]; then
    echo "ERROR: Missing samples directory: ${SAMPLES_DIR}"
    exit 1
fi
if [ "${RUN_MODE}" = "python" ] || [ "${RUN_MODE}" = "both" ]; then
    if [ ! -f "${REPO_ROOT}/lines/gfallvac.latest" ]; then
        echo "ERROR: Missing line list: ${REPO_ROOT}/lines/gfallvac.latest"
        exit 1
    fi
    if [ ! -f "${REPO_ROOT}/synthe_py/data/atlas_tables.npz" ]; then
        echo "ERROR: Missing atlas tables: ${REPO_ROOT}/synthe_py/data/atlas_tables.npz"
        exit 1
    fi
fi
if [ "${RUN_MODE}" = "fortran" ] || [ "${RUN_MODE}" = "both" ]; then
    if [ ! -x "${REPO_ROOT}/run_fortran_atm.sh" ]; then
        echo "ERROR: Missing or non-executable runner: ${REPO_ROOT}/run_fortran_atm.sh"
        echo "Run: chmod +x run_fortran_atm.sh"
        exit 1
    fi
fi

mkdir -p "${FORTRAN_SPECS}" "${PYTHON_NPZ}" "${PYTHON_SPECS}" "${LOG_DIR}"

declare -a ATM_LIST=()
if [ "${MODE}" = "all" ]; then
    while IFS= read -r atm_file; do
        ATM_LIST+=("${atm_file}")
    done < <(find "${SAMPLES_DIR}" -maxdepth 1 -type f -name "*.atm" | sort)
    if [ "${#ATM_LIST[@]}" -eq 0 ]; then
        echo "ERROR: No .atm files found in ${SAMPLES_DIR}"
        exit 1
    fi
else
    if [ -z "${ATM_ONE}" ]; then
        echo "ERROR: --atm is required when -n 1"
        exit 1
    fi
    if [ -f "${ATM_ONE}" ]; then
        ATM_LIST=("${ATM_ONE}")
    elif [ -f "${SAMPLES_DIR}/${ATM_ONE}" ]; then
        ATM_LIST=("${SAMPLES_DIR}/${ATM_ONE}")
    else
        echo "ERROR: Atmosphere not found: ${ATM_ONE}"
        exit 1
    fi
fi

echo "Running ${#ATM_LIST[@]} atmosphere(s) [mode: ${RUN_MODE}]"
echo "Fixed settings: wl=${WL_START}:${WL_END}, resolution=${RESOLUTION}"
echo "Resume mode: existing outputs are skipped"

for atm_path in "${ATM_LIST[@]}"; do
    stem="$(basename "${atm_path%.atm}")"
    ft_spec="${FORTRAN_SPECS}/${stem}.spec"
    py_npz="${PYTHON_NPZ}/${stem}.npz"
    py_spec="${PYTHON_SPECS}/${stem}.spec"
    ft_log="${LOG_DIR}/${stem}_fortran.log"
    py_log="${LOG_DIR}/${stem}_python.log"

    echo "=== ${stem} ==="

    if [ "${RUN_MODE}" = "fortran" ] || [ "${RUN_MODE}" = "both" ]; then
        if [ -f "${ft_spec}" ]; then
            echo "FORTRAN SKIP: ${stem} (exists: ${ft_spec})"
        else
            if ! ./run_fortran_atm.sh "${atm_path}" "${ft_spec}" > "${ft_log}" 2>&1; then
                echo "FORTRAN FAIL: ${stem} (see ${ft_log})"
            fi
        fi
    fi

    if [ "${RUN_MODE}" = "python" ] || [ "${RUN_MODE}" = "both" ]; then
        if [ -f "${py_spec}" ]; then
            echo "PYTHON SKIP: ${stem} (exists: ${py_spec})"
        else
            if ! {
                python3 synthe_py/tools/convert_atm_to_npz.py \
                    "${atm_path}" \
                    "${py_npz}" \
                    --atlas-tables "${REPO_ROOT}/synthe_py/data/atlas_tables.npz" && \
                python3 -m synthe_py.cli \
                    "${atm_path}" \
                    "${REPO_ROOT}/lines/gfallvac.latest" \
                    --npz "${py_npz}" \
                    --spec "${py_spec}" \
                    --wl-start "${WL_START}" \
                    --wl-end "${WL_END}" \
                    --resolution "${RESOLUTION}" \
                    --n-workers "${N_WORKERS}" \
                    --cache "${REPO_ROOT}/synthe_py/out/line_cache" \
                    --log-level INFO
            } > "${py_log}" 2>&1; then
                echo "PYTHON FAIL: ${stem} (see ${py_log})"
            fi
        fi
    fi

done

echo ""
echo "Done."
echo "Fortran specs: ${FORTRAN_SPECS}"
echo "Python npz:    ${PYTHON_NPZ}"
echo "Python specs:  ${PYTHON_SPECS}"
echo "Logs:          ${LOG_DIR}"

