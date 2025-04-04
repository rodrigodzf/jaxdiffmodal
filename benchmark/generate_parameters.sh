#!/bin/bash

MODES=(10 50 100)

for MODE in "${MODES[@]}"; do
    echo "Generating parameters for n_modes = $MODE..."
    python generate_parameters.py --n_modes $MODE
done

echo "All parameter files generated successfully"
