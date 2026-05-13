# jaxdiffmodal

[![arXiv](https://img.shields.io/badge/arXiv-2505.05940-b31b1b.svg)](https://arxiv.org/abs/2505.05940)

Fast, differentiable, and GPU-accelerated simulation framework for modelling the dynamics of strings, membranes, and plates using modal methods implemented in [JAX](https://github.com/google/jax).

## Features

- Differentiable implementation using JAX
- Simulates linear and nonlinear models:
  - Tension-modulated string (Kirchhoff–Carrier)
  - Tension-modulated membrane (Berger model)
  - von Kármán nonlinear plate
- Fast GPU-accelerated time integration
- Designed for real-time synthesis, inverse modelling, and dataset generation
- Includes example notebooks to reproduce results from the paper

## Installation

It is recommended to use the [`uv`](https://github.com/astral-sh/uv) package manager to install the environment and dependencies.

```bash
uv sync
```

Or if you want to install the development dependencies:

```bash
uv sync --extra dev --extra benchmark
```

Or using Python's native virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Testing

The project includes comprehensive tests to validate the Python implementation against MATLAB reference code.

### Running Tests

Prerequisites: The tests use `pytest` and require `scipy` for eigenvalue computations.

Run all tests from the project root directory:
```bash
python -m pytest tests/ -v
```

Run specific test modules:
```bash
# Test K and M matrix assembly with MATLAB comparison
python -m pytest tests/test_K_M_matlab_comparison.py -v

# Test integral functions (int1, int2, int4)
python -m pytest tests/test_intx_matlab_comparison.py -v

# Test matrix integration functions (i_mat)
python -m pytest tests/test_imat_matlab_comparison.py -v
```

### Generating MATLAB Reference Data

To regenerate MATLAB reference data (requires MATLAB installation and the `VKGong` submodule):
```bash
git submodule update --init
cd tests
matlab -batch test_K_M_matlab_reference
matlab -batch test_intx_matlab_reference
matlab -batch test_imat_matlab_reference
```

### Test Coverage

- **K, M matrix assembly**: Perfect match with MATLAB (1e-8 tolerance)
- **Airy stress coefficients**: Validated using MATLAB eigenvalues/eigenvectors as input (1e-6 tolerance)
- **Integral functions**: Comprehensive parameter space validation
- **Basic functionality**: Matrix properties, dimensions, and mathematical consistency

## Examples

The `docs/examples` directory includes:

- Synthetic and real-world inverse modelling experiments for strings and plates
- Optimisation of nonlinear parameters and coupling tensors
- Scripts to reproduce figures from the paper

The `benchmark` directory includes comparisons against:

- An optimised C++ implementation using Eigen and BLAS
- A JIT-compiled PyTorch implementation (GPU)
- A MATLAB baseline

> We plan to add more benchmarks, examples, and real-time synthesis demos in future updates.

## Acknowledgements

- Mode processing adapted from [VKPlate](https://github.com/Nemus-Project/VKPlate)
- Plate mode computation using [magpie-python](https://github.com/Nemus-Project/magpie-python)
- Coupling coefficient implementation based on [VKGong](https://github.com/rodrigodzf/VKGong)

