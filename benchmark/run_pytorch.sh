#!/bin/bash

INPUT_FILES=("benchmark_input_010.mat" "benchmark_input_050.mat" "benchmark_input_100.mat")
NUM_ITERATIONS=50
USE_SINGLE_PRECISION=true
USE_GPU=true
USE_JIT=true
USE_TM=true  # Set to true or false for tension modulation
SCRIPT_DIR=$(dirname "$(realpath "$0")")

for INPUT_FILE in "${INPUT_FILES[@]}"; do
    echo "Running PyTorch benchmark with $INPUT_FILE..."
    INPUT_FILE_ABS="$SCRIPT_DIR/$INPUT_FILE"  # Create absolute path to input file
    
    # Prepare flags based on variables
    SINGLE_FLAG=""
    if [ "$USE_SINGLE_PRECISION" = true ]; then
        SINGLE_FLAG="--single"
    fi
    
    GPU_FLAG=""
    if [ "$USE_GPU" = true ]; then
        GPU_FLAG="--gpu"
    fi
    
    MODE="jit"
    if [ "$USE_JIT" = true ]; then
        MODE="jit"
    fi
    
    TM_FLAG=""
    if [ "$USE_TM" = true ]; then
        TM_FLAG="--use_tm"
    fi
    
    # Run the benchmark with configured flags
    python "$SCRIPT_DIR/sv_pytorch.py" --input "$INPUT_FILE_ABS" --iterations "$NUM_ITERATIONS" $SINGLE_FLAG $GPU_FLAG -m $MODE $TM_FLAG
    
    echo "PyTorch benchmark completed for $INPUT_FILE"
    echo "----------------------------------------"
done

echo "All PyTorch benchmarks completed" 