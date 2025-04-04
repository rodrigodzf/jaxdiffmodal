#!/bin/bash

INPUT_FILES=("benchmark_input_010.mat" "benchmark_input_050.mat" "benchmark_input_100.mat")
NUM_ITERATIONS=50
USE_SINGLE_PRECISION=true
USE_TM=true
SCRIPT_DIR=$(dirname "$(realpath "$0")")

for INPUT_FILE in "${INPUT_FILES[@]}"; do
    echo "Running benchmark with $INPUT_FILE..."
    INPUT_FILE_ABS="$SCRIPT_DIR/$INPUT_FILE"  # Create absolute path to input file
    
    matlab -nodisplay -nosplash -nodesktop -r "cd('$SCRIPT_DIR'); sv_matlab('$INPUT_FILE_ABS', $NUM_ITERATIONS, $USE_SINGLE_PRECISION, $USE_TM); exit;"
    
    echo "MATLAB benchmark completed for $INPUT_FILE"
    echo "----------------------------------------"
done

echo "All MATLAB benchmarks completed"