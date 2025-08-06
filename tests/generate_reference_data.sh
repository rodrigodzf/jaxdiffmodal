#!/bin/bash
#####################################################################
#                                                                   #
#                Generate All Reference Data                        #
#                                                                   #
#####################################################################
#
# This script generates all reference data files used by the Python test suite.
# It runs the MATLAB reference generation script.
#
# Usage:
#   cd tests
#   ./generate_reference_data.sh
#
# Requirements:
#   - MATLAB installation
#   - VKGong submodule initialized (git submodule update --init)
#
# Output:
#   All reference data files are saved to tests/reference_data/
#

set -e  # Exit on any error

echo "=========================================="
echo "Generating All Reference Data"
echo "=========================================="
echo

# Check if we're in the tests directory
if [ ! -f "generate_all_reference_data.m" ]; then
    echo "Error: Please run this script from the tests/ directory"
    echo "Usage: cd tests && ./generate_reference_data.sh"
    exit 1
fi

# Check if MATLAB is available
if ! command -v matlab &> /dev/null; then
    echo "Error: MATLAB is not available in PATH"
    echo "Please ensure MATLAB is installed and available in PATH"
    exit 1
fi

# Check if VKGong submodule is initialized
if [ ! -d "../third_party/VKGong/matlab" ]; then
    echo "Error: VKGong submodule not found"
    echo "Please initialize the submodule:"
    echo "  git submodule update --init --recursive"
    exit 1
fi

echo "Running MATLAB reference data generation..."
echo "This may take several minutes..."
echo

# Run MATLAB script
matlab -batch "generate_all_reference_data"

echo
echo "=========================================="
echo "Reference Data Generation Complete"
echo "=========================================="
echo
echo "Generated files should be in tests/reference_data/"
echo "Run Python tests to verify everything works correctly:"
echo "  python -m pytest tests/ -v"
echo