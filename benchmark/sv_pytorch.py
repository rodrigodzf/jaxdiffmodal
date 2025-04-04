#!/usr/bin/env python3
import argparse
import os
import time

import einops
import numpy as np
import scipy.io as sio
import torch


# Define a non-JIT version that can be used with torch.compile or directly
def solve_sv_vk_pytorch_raw(
    A_inv,
    B,
    C,
    modal_excitation,
    H=None,
    use_tm=False,
    lambda_mu=None,
    tau_with_norms=None,
):
    """
    Solve the state-variable form of the von Karman plate model using PyTorch.
    Plain function without JIT for flexibility.

    Args:
        A_inv, B, C: Model parameters
        modal_excitation: Excitation input
        H: Tensor for VK nonlinearity (used when use_tm=False)
        use_tm: Whether to use tension modulation nonlinearity
        lambda_mu: Tension modulation parameter (used when use_tm=True)
        tau_with_norms: Tension modulation parameter (used when use_tm=True)
    """
    device = A_inv.device
    T, n_modes = modal_excitation.shape

    # Initialize state variables
    q = torch.zeros(n_modes, device=device, dtype=A_inv.dtype)
    q_prev = torch.zeros(n_modes, device=device, dtype=A_inv.dtype)
    modal_sol = torch.zeros((T, n_modes), device=device, dtype=A_inv.dtype)

    # Main loop
    for i in range(T):
        # Nonlinear term calculation - depends on the method
        if use_tm:
            # Tension modulation nonlinearity
            q_squared = q * q
            tmp_scalar = torch.dot(tau_with_norms, q_squared)
            nl = lambda_mu * q * tmp_scalar
        else:
            # Original von Karman nonlinearity
            t0 = torch.matmul(H, q)
            t2 = torch.matmul(t0, q)
            nl = torch.matmul(t0.T, t2)

        # State variable update
        q_next = B * q + C * q_prev - A_inv * nl + modal_excitation[i]
        q_prev = q
        q = q_next
        modal_sol[i] = q

    return q, modal_sol


# Create a JIT-compiled version for comparison
@torch.jit.script
def solve_sv_vk_pytorch(
    A_inv: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    modal_excitation: torch.Tensor,
    H: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Solve the state-variable form of the von Karman plate model using PyTorch.
    JIT-compiled for performance.
    """
    device = A_inv.device
    T, n_modes = modal_excitation.shape

    # Initialize state variables
    q = torch.zeros(n_modes, device=device, dtype=A_inv.dtype)
    q_prev = torch.zeros(n_modes, device=device, dtype=A_inv.dtype)
    modal_sol = torch.zeros((T, n_modes), device=device, dtype=A_inv.dtype)

    # Main loop
    for i in range(T):
        # Nonlinear term calculation
        t0 = torch.matmul(H, q)
        t2 = torch.matmul(t0, q)
        nl = torch.matmul(t0.T, t2)

        # State variable update
        q_next = B * q + C * q_prev - A_inv * nl + modal_excitation[i]
        q_prev = q
        q = q_next
        modal_sol[i] = q

    return q, modal_sol


# Add a JIT-compiled version for tension modulation
@torch.jit.script
def solve_sv_tm_pytorch(
    A_inv: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    modal_excitation: torch.Tensor,
    lambda_mu: torch.Tensor,
    tau_with_norms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Solve the state-variable form with tension modulation using PyTorch.
    JIT-compiled for performance.
    """
    device = A_inv.device
    T, n_modes = modal_excitation.shape

    # Initialize state variables
    q = torch.zeros(n_modes, device=device, dtype=A_inv.dtype)
    q_prev = torch.zeros(n_modes, device=device, dtype=A_inv.dtype)
    modal_sol = torch.zeros((T, n_modes), device=device, dtype=A_inv.dtype)

    # Main loop
    for i in range(T):
        # Tension modulation nonlinearity
        q_squared = q * q
        tmp_scalar = torch.dot(tau_with_norms, q_squared)
        nl = lambda_mu * q * tmp_scalar

        # State variable update
        q_next = B * q + C * q_prev - A_inv * nl + modal_excitation[i]
        q_prev = q
        q = q_next
        modal_sol[i] = q

    return q, modal_sol


# Vectorized implementation that might be faster in some cases
def solve_sv_vk_pytorch_vectorized(
    A_inv,
    B,
    C,
    modal_excitation,
    H=None,
    use_tm=False,
    lambda_mu=None,
    tau_with_norms=None,
):
    """
    Solve the state-variable form using a more vectorized approach.
    This can be significantly faster for some problems.
    """
    device = A_inv.device
    T, n_modes = modal_excitation.shape

    # Pre-allocate output array
    modal_sol = torch.zeros((T, n_modes), device=device, dtype=A_inv.dtype)

    # Initial conditions
    q = torch.zeros(n_modes, device=device, dtype=A_inv.dtype)
    q_prev = torch.zeros(n_modes, device=device, dtype=A_inv.dtype)

    # Manual unrolling of first iterations to establish q and q_prev
    if T > 0:
        if use_tm:
            # Tension modulation nonlinearity
            q_squared = q * q
            tmp_scalar = torch.dot(tau_with_norms, q_squared)
            nl = lambda_mu * q * tmp_scalar
        else:
            # Original von Karman nonlinearity
            t0 = torch.matmul(H, q)
            t2 = torch.matmul(t0, q)
            nl = torch.matmul(t0.T, t2)

        q_next = B * q + C * q_prev - A_inv * nl + modal_excitation[0]
        modal_sol[0] = q_next
        q_prev, q = q, q_next

    # Main loop - use pytorch operations as much as possible
    for i in range(1, T):
        if use_tm:
            # Tension modulation nonlinearity
            q_squared = q * q
            tmp_scalar = torch.dot(tau_with_norms, q_squared)
            nl = lambda_mu * q * tmp_scalar
        else:
            # Original von Karman nonlinearity
            t0 = torch.matmul(H, q)
            t2 = torch.matmul(t0, q)
            nl = torch.matmul(t0.T, t2)

        q_next = B * q + C * q_prev - A_inv * nl + modal_excitation[i]
        modal_sol[i] = q_next
        q_prev, q = q, q_next

    return q, modal_sol


def run_benchmark(
    input_file, num_iterations=50, use_gpu=False, mode="jit", use_tm=False
):
    """
    Run the benchmark using the PyTorch implementation.

    Args:
        input_file: Path to the input .mat file
        num_iterations: Number of benchmark iterations to run
        use_gpu: Whether to use GPU acceleration
        mode: Optimization mode to use ('jit', 'compile', 'raw', 'vectorized')
        use_tm: Whether to use tension modulation nonlinearity

    Returns:
        out_pos_pytorch: Output at the readout position
        times_pytorch: Array of execution times
    """
    print(f"Running benchmark with {input_file}")
    print(f"Number of iterations: {num_iterations}")
    print(f"Using GPU: {use_gpu}")
    print(f"Optimization mode: {mode}")
    print(f"Using {'tension modulation' if use_tm else 'von Karman'} nonlinearity")

    # Set device
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    if use_gpu and not torch.cuda.is_available():
        print("Warning: GPU requested but not available. Using CPU instead.")

    # Load the input data
    data = sio.loadmat(input_file)

    # Get the current default dtype
    dtype = torch.get_default_dtype()
    print(f"Using precision: {dtype}")

    # Extract variables and convert to PyTorch tensors with the correct dtype
    modal_excitation_normalised = torch.tensor(
        data["modal_excitation_normalised"].T,
        device=device,
        dtype=dtype,
    )

    B = torch.tensor(
        data["B"].flatten(),
        device=device,
        dtype=dtype,
    )
    C = torch.tensor(
        data["C"].flatten(),
        device=device,
        dtype=dtype,
    )
    A_inv = torch.tensor(
        data["A_inv"].flatten(),
        device=device,
        dtype=dtype,
    )
    modal_gains_at_readout = torch.tensor(
        data["modal_gains_at_readout"].flatten(),
        device=device,
        dtype=dtype,
    )

    # Load nonlinearity-specific parameters
    if use_tm:
        lambda_mu = torch.tensor(
            data["lambda_mu"].flatten(),
            device=device,
            dtype=dtype,
        )
        tau_with_norms = torch.tensor(
            data["tau_with_norms"].flatten(),
            device=device,
            dtype=dtype,
        )
        H = None  # Not needed for TM
    else:
        H = torch.tensor(
            data["H"],
            device=device,
            dtype=dtype,
        )
        lambda_mu = None
        tau_with_norms = None

    # Get dimensions
    T = modal_excitation_normalised.shape[0]
    n_modes = modal_excitation_normalised.shape[1]

    print(f"Benchmark parameters: n_modes = {n_modes}, T = {T}")

    # Select the solver function based on the mode and nonlinearity type
    if mode == "jit":
        if use_tm:
            solver_fn = solve_sv_tm_pytorch
        else:
            solver_fn = solve_sv_vk_pytorch
    elif mode == "compile":
        try:
            print("Compiling solver function with torch.compile()...")
            solver_fn = torch.compile(
                solve_sv_vk_pytorch_raw, mode="reduce-overhead", fullgraph=True
            )
            print("Compilation configuration complete")
        except Exception as e:
            print(f"Warning: Failed to use torch.compile: {e}")
            print("Falling back to raw version")
            solver_fn = solve_sv_vk_pytorch_raw
    elif mode == "vectorized":
        solver_fn = solve_sv_vk_pytorch_vectorized
    else:  # raw mode
        solver_fn = solve_sv_vk_pytorch_raw

    # Apply optimizations for GPU if applicable
    if device.type == "cuda":
        # Set higher precision for matmul operations on GPU
        torch.backends.cuda.matmul.allow_tf32 = (
            False  # Keep full precision for benchmarking
        )

        # Tune the cuBLAS workspace size for potentially better performance
        if hasattr(torch.backends.cuda, "cufft_plan_cache"):
            torch.backends.cuda.cufft_plan_cache.max_size = 4096

        # Prime the GPU with some computation to ensure it's at full speed
        torch.cuda.empty_cache()
        _ = torch.ones(1000, 1000, device=device) @ torch.ones(
            1000, 1000, device=device
        )
        torch.cuda.synchronize()

    print("Warming up with initial run...")
    # Run once to ensure compilation is complete before benchmarking
    if mode == "jit":
        if use_tm:
            _ = solver_fn(
                A_inv,
                B,
                C,
                modal_excitation_normalised,
                lambda_mu,
                tau_with_norms,
            )
        else:
            _ = solver_fn(
                A_inv,
                B,
                C,
                modal_excitation_normalised,
                H,
            )
    else:
        _ = solver_fn(
            A_inv,
            B,
            C,
            modal_excitation_normalised,
            H=H,
            use_tm=use_tm,
            lambda_mu=lambda_mu,
            tau_with_norms=tau_with_norms,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    print("Warmup complete")

    # Initialize timing array
    times_pytorch = np.zeros(num_iterations)

    # Run the benchmark multiple times
    for iter in range(num_iterations):
        # Synchronize before starting timer if using GPU
        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()

        # Run the computation with the appropriate arguments based on mode and nonlinearity
        if mode == "jit":
            if use_tm:
                _, result = solver_fn(
                    A_inv,
                    B,
                    C,
                    modal_excitation_normalised,
                    lambda_mu,
                    tau_with_norms,
                )
            else:
                _, result = solver_fn(
                    A_inv,
                    B,
                    C,
                    modal_excitation_normalised,
                    H,
                )
        else:
            _, result = solver_fn(
                A_inv,
                B,
                C,
                modal_excitation_normalised,
                H=H,
                use_tm=use_tm,
                lambda_mu=lambda_mu,
                tau_with_norms=tau_with_norms,
            )

        # Synchronize after computation if using GPU
        if device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()
        times_pytorch[iter] = end_time - start_time

        print(f"Iteration {iter + 1}: {times_pytorch[iter]:.4f} seconds")

    # Run one more time to get the final result
    if mode == "jit":
        if use_tm:
            _, modal_sol = solver_fn(
                A_inv,
                B,
                C,
                modal_excitation_normalised,
                lambda_mu,
                tau_with_norms,
            )
        else:
            _, modal_sol = solver_fn(
                A_inv,
                B,
                C,
                modal_excitation_normalised,
                H,
            )
    else:
        _, modal_sol = solver_fn(
            A_inv,
            B,
            C,
            modal_excitation_normalised,
            H=H,
            use_tm=use_tm,
            lambda_mu=lambda_mu,
            tau_with_norms=tau_with_norms,
        )

    # Calculate output at a single position
    out_pos_pytorch = (modal_sol @ modal_gains_at_readout).cpu().numpy()

    # Calculate and display statistics
    mean_time = np.mean(times_pytorch)
    std_time = np.std(times_pytorch)
    min_time = np.min(times_pytorch)
    max_time = np.max(times_pytorch)

    print("\nBenchmark Statistics:")
    print(f"Mean execution time: {mean_time:.4f} seconds")
    print(f"Standard deviation: {std_time:.4f} seconds")
    print(f"Minimum time: {min_time:.4f} seconds")
    print(f"Maximum time: {max_time:.4f} seconds")

    return out_pos_pytorch, times_pytorch


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch benchmark for the VK plate model"
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
        help="Output file path (default: sv_pytorch_output_XXX.mat where XXX is the number of modes)",
    )
    parser.add_argument(
        "-s",
        "--single",
        action="store_true",
        help="Use single precision (default: False)",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        action="store_true",
        help="Use GPU acceleration (default: False)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["jit", "compile", "raw", "vectorized"],
        default="jit",
        help="Optimization mode to use (default: jit)",
    )
    parser.add_argument(
        "--use_tm",
        action="store_true",
        help="Use tension modulation nonlinearity (default: False)",
    )

    args = parser.parse_args()

    # Set precision
    if args.single:
        print("Using single precision")
        torch.set_default_dtype(torch.float32)
    else:
        print("Using double precision")
        torch.set_default_dtype(torch.float64)

    # Determine output file path
    if args.output:
        output_file = args.output
    else:
        # Extract the number of modes from the input filename (assuming format benchmark_input_XXX.mat)
        filename = os.path.basename(args.input)
        n_modes_str = filename.split("_")[2].split(".")[0]
        device_str = "gpu" if args.gpu and torch.cuda.is_available() else "cpu"
        nl_str = "tm" if args.use_tm else "vk"
        output_file = f"sv_pytorch_output_{n_modes_str}_{'single' if args.single else 'double'}_{device_str}_{args.mode}_{nl_str}.mat"

    print(f"Output will be saved to: {output_file}")

    # Run the benchmark
    out_pos_pytorch, times_pytorch = run_benchmark(
        args.input, args.iterations, args.gpu, args.mode, args.use_tm
    )

    # Save the results
    sio.savemat(
        output_file,
        {
            "out_pos_pytorch": out_pos_pytorch,
            "times_pytorch": times_pytorch,
            "use_tm": args.use_tm,
        },
    )

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
