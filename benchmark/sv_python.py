#!/usr/bin/env python3
import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.io as sio

from vkplatejax.sv import make_vk_nl_fn, solve_sv_vk_jax_scan, make_tm_nl_fn


def run_benchmark(input_file, num_iterations=50, use_tm=False):
    """
    Run the benchmark using the JAX implementation.

    Args:
        input_file: Path to the input .mat file
        num_iterations: Number of benchmark iterations to run

    Returns:
        out_pos_python: Output at the readout position
        times_python: Array of execution times
    """
    print(f"Running benchmark with {input_file}")
    print(f"Number of iterations: {num_iterations}")
    print(f"Using {'VK' if not use_tm else 'TM'} nonlinearity")

    # Load the input data
    data = sio.loadmat(input_file)

    # Extract variables
    modal_excitation_normalised = jnp.array(data["modal_excitation_normalised"].T)
    B = jnp.array(data["B"]).flatten()
    C = jnp.array(data["C"]).flatten()
    A_inv = jnp.array(data["A_inv"]).flatten()
    modal_gains_at_readout = jnp.array(data["modal_gains_at_readout"]).flatten()
    if use_tm:
        lambda_mu = jnp.array(data["lambda_mu"]).flatten()
        tau_with_norms = jnp.array(data["tau_with_norms"]).flatten()
    else:
        H = jnp.array(data["H"])

    # Get dimensions
    T = modal_excitation_normalised.shape[0]
    n_modes = modal_excitation_normalised.shape[1]

    print(f"Benchmark parameters: n_modes = {n_modes}, T = {T}")

    # Create the nonlinear function
    if use_tm:
        nl_fn = jax.jit(make_tm_nl_fn(lambda_mu, tau_with_norms))
    else:
        nl_fn = jax.jit(make_vk_nl_fn(H))

    # Compile the function once to avoid including compilation time in the benchmark
    _, modal_sol = solve_sv_vk_jax_scan(
        A_inv,
        B,
        C,
        modal_excitation_normalised,
        g=A_inv,
        nl_fn=nl_fn,
    )

    # Initialize timing array
    times_python = np.zeros(num_iterations)

    # Run the benchmark multiple times
    for iter in range(num_iterations):
        start_time = time.time()

        # Run the computation and ensure it's complete
        result = solve_sv_vk_jax_scan(
            A_inv,
            B,
            C,
            modal_excitation_normalised,
            g=A_inv,
            nl_fn=nl_fn,
        )[1]
        result.block_until_ready()  # This ensures GPU computation is complete

        end_time = time.time()
        times_python[iter] = end_time - start_time

        print(f"Iteration {iter + 1}: {times_python[iter]:.4f} seconds")

    # Run one more time to get the final result
    _, modal_sol = solve_sv_vk_jax_scan(
        A_inv,
        B,
        C,
        modal_excitation_normalised,
        g=A_inv,
        nl_fn=nl_fn,
    )

    # Calculate output at a single position
    out_pos_python = np.array(modal_sol @ modal_gains_at_readout)

    # Calculate and display statistics
    mean_time = np.mean(times_python)
    std_time = np.std(times_python)
    min_time = np.min(times_python)
    max_time = np.max(times_python)

    print("\nBenchmark Statistics:")
    print(f"Mean execution time: {mean_time:.4f} seconds")
    print(f"Standard deviation: {std_time:.4f} seconds")
    print(f"Minimum time: {min_time:.4f} seconds")
    print(f"Maximum time: {max_time:.4f} seconds")

    return out_pos_python, times_python


def main():
    parser = argparse.ArgumentParser(
        description="Python benchmark for the VK plate model"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="benchmark_input_010.mat",
        help="Input file path (default: benchmark_input_010.mat)",
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=50,
        help="Number of benchmark iterations (default: 50)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path (default: sv_python_output_XXX.mat where XXX is the number of modes)",
    )

    parser.add_argument(
        "-s",
        "--single",
        type=bool,
        default=False,
        help="Use single precision (default: False)",
    )

    parser.add_argument(
        "--use_tm",
        type=bool,
        default=False,
        help="Use tension modulation (default: False)",
    )
    args = parser.parse_args()

    use_single_precision = args.single
    if use_single_precision:
        jax.config.update("jax_enable_x64", False)
    else:
        jax.config.update("jax_enable_x64", True)

    # Determine output file path
    if args.output:
        output_file = args.output
    else:
        # Extract the number of modes from the input filename (assuming format benchmark_input_XXX.mat)
        filename = os.path.basename(args.input)
        n_modes_str = filename.split("_")[2].split(".")[0]
        output_file = f"sv_python_output_{n_modes_str}_{'single' if use_single_precision else 'double'}_{'tm' if args.use_tm else 'vk'}.mat"

    print(f"Output will be saved to: {output_file}")

    # Run the benchmark
    out_pos_python, times_python = run_benchmark(
        args.input,
        args.iterations,
        args.use_tm,
    )

    # Save the results
    sio.savemat(
        output_file,
        {
            "out_pos_python": out_pos_python,
            "times_python": times_python,
        },
    )

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
