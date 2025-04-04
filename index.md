

# jaxdiffmodal

Fast, differentiable, and GPU-accelerated simulation framework for
modelling the dynamics of strings, membranes, and plates using modal
methods implemented in [JAX](https://github.com/google/jax).

## Features

- Differentiable implementation using JAX
- Simulates linear and nonlinear models:
  - Tension-modulated string (Kirchhoff–Carrier)
  - Tension-modulated membrane (Berger model)
  - von Kármán nonlinear plate
- Fast GPU-accelerated time integration
- Designed for real-time synthesis, inverse modelling, and dataset
  generation
- Includes example notebooks to reproduce results from the paper

## Installation for Development

It is recommended to use the [`uv`](https://github.com/astral-sh/uv)
package manager to install the environment and dependencies.

``` bash
uv sync --all-extras
```

otherwise you can create a virtual environment and install the
dependencies manually:

``` bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Examples

The `nbs/examples` directory includes:

- Synthetic and real-world inverse modelling experiments for strings and
  plates
- Optimisation of nonlinear parameters and coupling tensors
- Scripts to reproduce figures from the paper

The `benchmark` directory includes comparisons against: - An optimised
C++ implementation using Eigen and BLAS - A JIT-compiled PyTorch
implementation (GPU) - A MATLAB baseline

> We plan to add more benchmarks, examples, and real-time synthesis
> demos in future updates.

## Acknowledgements

- Mode processing adapted from
  [VKPlate](https://github.com/Nemus-Project/VKPlate)
- Plate mode computation using
  [magpie-python](https://github.com/Nemus-Project/magpie-python)
- Coupling coefficient implementation based on
  [VKGong](https://github.com/rodrigodzf/VKGong)
