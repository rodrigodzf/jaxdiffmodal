#!/bin/bash

INPUT_FILES=("benchmark_input_010.mat" "benchmark_input_050.mat" "benchmark_input_100.mat")
NUM_ITERATIONS=50
USE_SINGLE_PRECISION=true
USE_TM=true  # Set to true or false to control the use_tm flag
SCRIPT_DIR=$(dirname "$(realpath "$0")")
CPP_EXECUTABLE="$SCRIPT_DIR/cpp/benchmark_sequence"

# Check if the C++ executable exists
if [ ! -f "$CPP_EXECUTABLE" ]; then
    echo "Error: C++ executable not found at $CPP_EXECUTABLE"
    echo "Please build the C++ benchmark first."
    exit 1
fi

for INPUT_FILE in "${INPUT_FILES[@]}"; do
    echo "Running C++ benchmark with $INPUT_FILE..."
    echo "Settings: single precision = $USE_SINGLE_PRECISION, use tension modulation = $USE_TM"
    INPUT_FILE_ABS="$SCRIPT_DIR/$INPUT_FILE"  # Create absolute path to input file
    
    "$CPP_EXECUTABLE" --input "$INPUT_FILE_ABS" --iterations "$NUM_ITERATIONS" --single "$USE_SINGLE_PRECISION" --use_tm "$USE_TM"
    
    echo "C++ benchmark completed for $INPUT_FILE"
    echo "----------------------------------------"
done

echo "All C++ benchmarks completed" 