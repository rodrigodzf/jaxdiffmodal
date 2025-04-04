#!/bin/bash

INPUT_FILES=("benchmark_input_010.mat" "benchmark_input_050.mat" "benchmark_input_100.mat")
NUM_ITERATIONS=50
SCRIPT_DIR=$(dirname "$(realpath "$0")")
USE_TM=True
SINGLE=True

for INPUT_FILE in "${INPUT_FILES[@]}"; do
    echo "Running Python benchmark with $INPUT_FILE..."
    INPUT_FILE_ABS="$SCRIPT_DIR/$INPUT_FILE"  # Create absolute path to input file
    
    # Conditional command construction based on USE_TM value
    python "$SCRIPT_DIR/sv_python.py" --input "$INPUT_FILE_ABS" --iterations "$NUM_ITERATIONS" --single $SINGLE --use_tm $USE_TM
    
    echo "Python benchmark completed for $INPUT_FILE"
    echo "----------------------------------------"
done

echo "All Python benchmarks completed" 