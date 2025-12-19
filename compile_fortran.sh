#!/bin/bash
# Helper script to compile Fortran code
# Tries to find and use the appropriate compiler

printf '=%.0s' {1..80}
echo ""
echo "COMPILING FORTRAN CODE WITH DEBUG STATEMENTS"
printf '=%.0s' {1..80}
echo ""

# Check for Intel Fortran
if command -v ifort &> /dev/null; then
    echo "✓ Found ifort (Intel Fortran)"
    FC="ifort"
elif [ -f /opt/intel/oneapi/setvars.sh ]; then
    echo "Found Intel oneAPI, sourcing environment..."
    source /opt/intel/oneapi/setvars.sh
    if command -v ifort &> /dev/null; then
        echo "✓ ifort now available"
        FC="ifort"
    else
        echo "✗ ifort still not found after sourcing"
    fi
fi

# Check for GNU Fortran
if [ -z "$FC" ] && command -v gfortran &> /dev/null; then
    echo "✓ Found gfortran (GNU Fortran)"
    echo "  Makefile will use the gfortran-safe flag set automatically."
    FC="gfortran"
fi

if [ -z "$FC" ]; then
    echo ""
    echo "✗ ERROR: No Fortran compiler found!"
    echo ""
    echo "Please install one of:"
    echo "  1. Intel Fortran (ifort) - from Intel oneAPI"
    echo "  2. GNU Fortran (gfortran) - install via: brew install gcc"
    echo ""
    echo "Or if Intel Fortran is installed elsewhere, source its environment:"
    echo "  source /opt/intel/oneapi/setvars.sh"
    echo ""
    exit 1
fi

echo ""
echo "Using compiler: $FC"
echo ""

# Makefile now adapts to gfortran (fixed-form + static locals)
if [ "$FC" = "gfortran" ]; then
    echo "Using gfortran-specific flags from src/Makefile (fixed-form, no automatic)."
fi

# Compile
cd src

# Check if atlas7v.for exists
if [ ! -f atlas7v.for ]; then
    echo "✗ ERROR: atlas7v.for not found in src/ directory!"
    exit 1
fi

echo "Cleaning previous build..."
make clean 2>/dev/null || rm -f *.o *.exe

echo ""
echo "Compiling atlas7v.for and dependencies..."
echo "  Note: atlas7v.for is compiled first as atlas7lib.o"
echo "  This object file is then linked with xnfpelsyn.exe and spectrv.exe"
echo ""

make FC="$FC" 2>&1 | tee ../compile.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "✓ Compilation successful!"
    echo ""
    echo "Verifying executables were rebuilt:"
    ls -lh ../bin/xnfpelsyn.exe ../bin/synthe.exe ../bin/spectrv.exe 2>/dev/null | awk '{print "  "$9": "$6" "$7" "$8}'
    echo ""
    echo "Next steps:"
    echo "  1. Run: ./run_fortran_with_debug.sh at12_aaaaa"
    echo "  2. Check: grep 'DEBUG' fortran_debug_at12_aaaaa.log"
    echo "  3. Compare: python compare_debug_outputs.py --fortran fortran_debug_at12_aaaaa.log --python python_debug.log"
else
    echo ""
    echo "✗ Compilation failed!"
    echo "  Check compile.log for details"
    exit 1
fi

