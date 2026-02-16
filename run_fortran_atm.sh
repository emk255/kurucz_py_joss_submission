#!/bin/bash
# Run Fortran synthesis for one explicit atmosphere file and write one .spec output.
#
# Usage:
#   ./run_fortran_atm.sh <atm_path> <output_spec> [line_dir]
#
# Example:
#   ./run_fortran_atm.sh samples/my_model.atm results/fortran_specs/my_model.spec

set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <atm_path> <output_spec> [line_dir]"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINDIR="${REPO_ROOT}/bin"
ATM_PATH="$1"
OUT_SPEC="$2"
LINE_DIR="${3:-${REPO_ROOT}/synthe/Lines_v5_PL}"
SPECTRV_INPUT="${REPO_ROOT}/infiles/spectrv_std.input"

if [ ! -f "${ATM_PATH}" ]; then
    echo "ERROR: Atmosphere file not found: ${ATM_PATH}"
    exit 1
fi

# Resolve to absolute paths before entering the temporary working directory.
ATM_PATH="$(cd "$(dirname "${ATM_PATH}")" && pwd)/$(basename "${ATM_PATH}")"
if [[ "${OUT_SPEC}" = /* ]]; then
    OUT_SPEC="${OUT_SPEC}"
else
    OUT_SPEC="${REPO_ROOT}/${OUT_SPEC}"
fi
if [[ "${LINE_DIR}" != /* ]]; then
    LINE_DIR="${REPO_ROOT}/${LINE_DIR}"
fi

for exe in at12tosyn.exe xnfpelsyn.exe synthe.exe spectrv.exe syntoascanga.exe; do
    if [ ! -x "${BINDIR}/${exe}" ]; then
        echo "ERROR: Missing executable: ${BINDIR}/${exe}"
        echo "Run ./compile_fortran.sh first."
        exit 1
    fi
done

for req in molecules.dat continua.dat he1tables.dat; do
    if [ ! -f "${REPO_ROOT}/lines/${req}" ]; then
        echo "ERROR: Missing required input file: ${REPO_ROOT}/lines/${req}"
        exit 1
    fi
done

if [ ! -f "${SPECTRV_INPUT}" ]; then
    echo "ERROR: Missing spectrv input file: ${SPECTRV_INPUT}"
    exit 1
fi

for req in tfort.12 tfort.14 tfort.19 tfort.20 tfort.93; do
    if [ ! -f "${LINE_DIR}/${req}" ]; then
        echo "ERROR: Missing line bundle file: ${LINE_DIR}/${req}"
        exit 1
    fi
done

mkdir -p "$(dirname "${OUT_SPEC}")"

ATM_BASENAME="$(basename "${ATM_PATH}")"
WORKDIR="$(mktemp -d "${REPO_ROOT}/synthe/stmp_batch_${ATM_BASENAME%.atm}.XXXXXX")"
cleanup() {
    if [ "${KEEP_WORKDIR:-0}" != "1" ]; then
        rm -rf "${WORKDIR}"
    else
        echo "KEEP_WORKDIR=1, preserving ${WORKDIR}"
    fi
}
trap cleanup EXIT

cp "${ATM_PATH}" "${WORKDIR}/${ATM_BASENAME}"
cd "${WORKDIR}"
mkdir -p logs

"${BINDIR}/at12tosyn.exe" "${ATM_PATH}" "${ATM_BASENAME}"

cp "${REPO_ROOT}/lines/molecules.dat" fort.2
cp "${REPO_ROOT}/lines/continua.dat" fort.17
"${BINDIR}/xnfpelsyn.exe" < "${ATM_BASENAME}" > /dev/null

cp "${LINE_DIR}/tfort.12" fort.12
cp "${LINE_DIR}/tfort.14" fort.14
cp "${LINE_DIR}/tfort.19" fort.19
cp "${LINE_DIR}/tfort.20" fort.20
cp "${LINE_DIR}/tfort.93" fort.93
cp "${REPO_ROOT}/lines/he1tables.dat" fort.18

"${BINDIR}/synthe.exe" > /dev/null
ln -s "${ATM_BASENAME}" fort.5
ln -s "${SPECTRV_INPUT}" fort.25
"${BINDIR}/spectrv.exe" > /dev/null

mv fort.7 fort.1
rm -f lineinfo.dat headinfo.dat
"${BINDIR}/syntoascanga.exe" > /dev/null

mv specfile.dat "${OUT_SPEC}"
echo "Wrote Fortran spectrum: ${OUT_SPEC}"

