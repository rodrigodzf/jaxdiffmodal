import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import soundfile as sf
from IPython.display import Audio
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat

from vkplatejax.excitations import create_1d_raised_cosine
from vkplatejax.ftm import (
    PlateParameters,
    damping_term_simple,
    stiffness_term,
)
from vkplatejax.num_utils import (
    compute_coupling_matrix_numerical,
    multiresolution_eigendecomposition,
)
from vkplatejax.sv import (
    A_inv_vector,
    B_vector,
    C_vector,
    make_vk_nl_fn,
    solve_sv_vk_jax_scan,
)


def generate_parameters(
    n_modes=100,
    sampling_rate=44100,
    h=0.004,  # grid spacing in the lowest resolution
    nx=50,  # number of grid points in the x direction in the lowest resolution
    ny=75,  # number of grid points in the y direction in the lowest resolution
    levels=2,  # number of grid refinements to perform
    excitation_duration=1.0,
    excitation_amplitude=0.3,
    output_file=None,
):
    """Generate parameters for benchmarking.

    Args:
        n_modes: Number of modes to compute
        sampling_rate: Sampling rate in Hz
        h: Grid spacing in the lowest resolution
        nx: Number of grid points in the x direction in the lowest resolution
        ny: Number of grid points in the y direction in the lowest resolution
        levels: Number of grid refinements to perform
        excitation_duration: Duration of the excitation in seconds
        excitation_amplitude: Amplitude of the excitation
        output_file: Output file name. If None, a default name will be generated.

    Returns:
        Path to the generated parameter file
    """
    sampling_period = 1 / sampling_rate

    if output_file is None:
        output_file = f"benchmark_input_{n_modes:03d}.mat"

    params = PlateParameters(
        E=2e12,
        nu=0.3,
        rho=7850,
        h=5e-4,
        l1=0.2,
        l2=0.3,
        Ts0=0,
    )

    # boundary conditions for the transverse modes
    bcs_phi = np.array(
        [
            [1e15, 0],
            [1e15, 0],
            [1e15, 0],
            [1e15, 0],
        ]
    )

    # boundary conditions for the in-plane modes
    bcs_psi = np.array(
        [
            [1e15, 1e15],
            [1e15, 1e15],
            [1e15, 1e15],
            [1e15, 1e15],
        ]
    )

    psi, zeta_mu, nx_final, ny_final, h_final, psi_norms = (
        multiresolution_eigendecomposition(
            params,
            n_modes,
            bcs_psi,
            h,
            nx,
            ny,
            levels=levels,
        )
    )

    phi, lambda_mu, nx_final, ny_final, h_final, phi_norms = (
        multiresolution_eigendecomposition(
            params,
            n_modes,
            bcs_phi,
            h,
            nx,
            ny,
            levels=levels,
        )
    )

    H = compute_coupling_matrix_numerical(
        psi,
        phi,
        h_final,
        nx_final,
        ny_final,
    )
    e = params.E / (2 * params.rho)
    H = H * np.sqrt(e)

    # we assume always the lambda mu comes from the decomposition of
    # the laplacian and the numerical lambda mu comes from the decomposition of
    # the biharmonic operator
    lambda_mu = np.sqrt(lambda_mu)

    # this is used only for the tension modulated case
    tau_with_norms = (
        (params.E * params.h)
        / (2 * params.l1 * params.l2 * (1 - params.nu**2))
        * lambda_mu
        / phi_norms
    )

    omega_mu_squared = stiffness_term(params, lambda_mu)
    c = damping_term_simple(np.sqrt(omega_mu_squared))
    print(f"omega_mu = {np.sqrt(omega_mu_squared[:5])}")

    A_inv = A_inv_vector(sampling_period, c * 2)
    B = B_vector(sampling_period, omega_mu_squared) * A_inv
    C = C_vector(sampling_period, c * 2) * A_inv

    force_position = (0.05, 0.05)
    readout_position = (0.1, 0.1)

    # generate a 1d raised cosine excitation
    rc = create_1d_raised_cosine(
        duration=excitation_duration,
        start_time=0.010,
        end_time=0.012,
        amplitude=excitation_amplitude,
        sample_rate=sampling_rate,
    )

    phi_reshaped = np.reshape(
        phi,
        shape=(ny_final + 1, nx_final + 1, n_modes),
        order="F",
    )

    mode_gains_at_pos = phi_reshaped[
        int(force_position[1] * ny_final),
        int(force_position[0] * nx_final),
        :,
    ]

    mode_gains_at_readout = phi_reshaped[
        int(readout_position[1] * ny_final),
        int(readout_position[0] * nx_final),
        :,
    ]
    # the modal excitation needs to be scaled by A_inv and divided by the density
    mode_gains_at_pos_normalised = (mode_gains_at_pos / params.density) * A_inv
    modal_excitation_normalised = rc[:, None] * mode_gains_at_pos_normalised

    print(f"Saving {output_file}")
    savemat(
        output_file,
        {
            "density": params.density,
            "Ts0": params.Ts0,
            "E": params.E,
            "nu": params.nu,
            "l1": params.l1,
            "l2": params.l2,
            "h": params.h,
            "modal_gains_at_pos": mode_gains_at_pos.reshape(-1, 1),
            "modal_gains_at_readout": mode_gains_at_readout.reshape(-1, 1),
            "sampling_rate": sampling_rate,
            "modal_excitation_normalised": modal_excitation_normalised.T,
            "H": H,
            "A_inv": A_inv.reshape(-1, 1),
            "B": B.reshape(-1, 1),
            "C": C.reshape(-1, 1),
            "lambda_mu": lambda_mu,
            "tau_with_norms": tau_with_norms,
        },
    )

    return output_file


def main():
    """Parse command line arguments and generate parameters."""
    parser = argparse.ArgumentParser(
        description="Generate parameters for benchmarking the VK plate model."
    )

    parser.add_argument(
        "--n_modes",
        type=int,
        default=100,
        help="Number of modes to compute (default: 100)",
    )

    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=44100,
        help="Sampling rate in Hz (default: 44100)",
    )

    parser.add_argument(
        "--h",
        type=float,
        default=0.004,
        help="Grid spacing in the lowest resolution (default: 0.004)",
    )

    parser.add_argument(
        "--nx",
        type=int,
        default=50,
        help="Number of grid points in the x direction in the lowest resolution (default: 50)",
    )

    parser.add_argument(
        "--ny",
        type=int,
        default=75,
        help="Number of grid points in the y direction in the lowest resolution (default: 75)",
    )

    parser.add_argument(
        "--levels",
        type=int,
        default=2,
        help="Number of grid refinements to perform (default: 2)",
    )

    parser.add_argument(
        "--excitation_duration",
        type=float,
        default=1.0,
        help="Duration of the excitation in seconds (default: 1.0)",
    )

    parser.add_argument(
        "--excitation_amplitude",
        type=float,
        default=0.3,
        help="Amplitude of the excitation (default: 0.3)",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file name (default: benchmark_input_NNN.mat where NNN is the number of modes)",
    )

    args = parser.parse_args()

    # Generate parameters with the provided arguments
    output_file = generate_parameters(
        n_modes=args.n_modes,
        sampling_rate=args.sampling_rate,
        h=args.h,
        nx=args.nx,
        ny=args.ny,
        levels=args.levels,
        excitation_duration=args.excitation_duration,
        excitation_amplitude=args.excitation_amplitude,
        output_file=args.output_file,
    )

    print(f"Parameters successfully generated and saved to {output_file}")


if __name__ == "__main__":
    main()
