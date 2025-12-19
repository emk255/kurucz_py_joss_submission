#!/bin/bash
# Script to run Fortran SYNTHE pipeline with debug output captured
# Usage: ./run_fortran_with_debug.sh at12_aaaaa

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_name>"
    echo "Example: $0 at12_aaaaa"
    exit 1
fi

MODEL=$1
DEBUG_LOG="fortran_debug_${MODEL}.log"

echo "Running Fortran SYNTHE pipeline with debug output..."
echo "Model: $MODEL"
echo "Debug log will be saved to: $DEBUG_LOG"
echo ""

# Run synthe_debug.sh (modified version that doesn't redirect to /dev/null)
# This will capture both stdout and stderr, including DEBUG lines
# Note: synthe_debug.sh is in synthe/ directory
cd synthe
if [ -f synthe_debug.sh ]; then
    echo "Using synthe_debug.sh (captures DEBUG output)"
    ./synthe_debug.sh "$MODEL" 2>&1 | tee "../$DEBUG_LOG"
else
    echo "WARNING: synthe_debug.sh not found, using synthe.sh"
    echo "Note: Some output may be redirected to /dev/null"
    ./synthe.sh "$MODEL" 2>&1 | tee "../$DEBUG_LOG"
fi
cd ..

echo ""
echo "Debug output saved to: $DEBUG_LOG"
echo ""
echo "To compare with Python debug log, run:"
echo "  python compare_debug_outputs.py --fortran $DEBUG_LOG --python python_debug.log"

