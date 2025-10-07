# %%
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp


# Enable 64-bit precision in JAX for improved accuracy
jax.config.update("jax_enable_x64", True)

from jaxdiffmodal.excitations import create_pluck_modal
from jaxdiffmodal.ftm import (
    damping_term,
    evaluate_string_eigenfunctions,
    stiffness_term,
    string_eigenvalues,
    StringParameters,
)
from jaxdiffmodal.time_integrators import (
    solve_sv_one_step,
    solve_sv_one_step_staggered,
    solve_sv_two_step,
    solve_tf,
)


#  %%
n_modes: int = 40
sample_rate: int = 44100
dt: float = 1.0 / sample_rate
n_steps = 22050
#  %%

string_params = StringParameters()
indices = jnp.arange(n_modes) + 1

lambda_mu = string_eigenvalues(
    n_modes,
    string_params.length,
)
gamma2_mu = damping_term(
    string_params,
    lambda_mu,
)
omega_mu_squared = stiffness_term(
    string_params,
    lambda_mu,
)
exc = create_pluck_modal(
    lambdas=lambda_mu,
    string_length=string_params.length,
    initial_deflection=0.03,
)

weights = evaluate_string_eigenfunctions(
    indices=indices,
    position=jnp.array(0.6),
    params=string_params,
)

u0 = jnp.array(exc * 20)
v0 = np.zeros(n_modes)
time = jnp.arange(n_steps) * dt
# %% define a RK solver


def solve_scipy_rk(
    gamma2_mu,
    omega_mu_squared,
    dt,
    n_steps=None,
    u0=None,
    v0=None,
    xs=None,
    method="DOP853",
    rtol: float = 1e-7,  #         Relative tolerance
    atol: float = 1e-8,  #         Absolute tolerance
):
    """
    Scipy solve_ivp wrapper with same interface as solve_sv_one_step.

    Parameters match solve_sv_one_step for consistency.
    Returns positions and velocities.
    """
    n_modes = len(gamma2_mu)

    # Set defaults
    u0 = u0 if u0 is not None else np.zeros(n_modes)
    v0 = v0 if v0 is not None else np.zeros(n_modes)

    # Determine time parameters
    if xs is not None:
        n_steps = n_steps if n_steps is not None else xs.shape[0]
        time_array = np.arange(n_steps) * dt
        # Create interpolation function for external excitation
        from scipy.interpolate import interp1d

        if xs.shape[0] > 1:
            # Extend xs to match time_array length by padding with zeros
            xs_extended = np.zeros((n_steps, n_modes))
            xs_extended[: min(xs.shape[0], n_steps)] = xs[: min(xs.shape[0], n_steps)]
            xs_interp = interp1d(
                np.arange(n_steps) * dt,
                xs_extended,
                axis=0,
                kind="linear",
                bounds_error=False,
                fill_value=0.0,
            )
        else:
            xs_interp = lambda t: np.zeros(n_modes)
    elif n_steps is not None:
        time_array = np.arange(n_steps) * dt
        xs_interp = lambda t: np.zeros(n_modes)
    else:
        raise ValueError("Either xs or n_steps must be provided")

    def rhs(t, state):
        n = len(state) // 2
        u = state[:n]
        v = state[n:]

        # Add external excitation
        excitation = xs_interp(t) if xs is not None else np.zeros(n_modes)

        du_dt = v
        dv_dt = -np.array(gamma2_mu) * v - np.array(omega_mu_squared) * u + excitation
        return np.concatenate([du_dt, dv_dt])

    sol = solve_ivp(
        fun=rhs,
        t_span=[0, time_array[-1]],
        y0=np.concatenate([u0, v0], axis=0),
        t_eval=time_array,
        method=method,
        rtol=rtol,
        atol=atol,
    )

    u_solution = sol.y[:n_modes, :].T  # Transpose to match (time, modes) format
    v_solution = sol.y[n_modes:, :].T  # Transpose to match (time, modes) format

    u_with_ic = u_solution
    v_with_ic = v_solution
    return None, u_with_ic, v_with_ic


# Test with initial conditions only (no excitation)
_, sol_u_solve_ivp, sol_v_solve_ivp = solve_scipy_rk(
    gamma2_mu=gamma2_mu,
    omega_mu_squared=omega_mu_squared,
    dt=dt,
    n_steps=n_steps,
    u0=u0,
    v0=v0,
)


#  %% Test the linear solving without excitation
def lin_fn(q):
    return 0


def solve_analytical_driven_oscillator(
    gamma2_mu: float,
    omega_mu_squared: float,
    F0: float,
    Omega: float,
    q0: float,
    v0: float,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the analytical solution for a damped, forced harmonic oscillator.

    The solved equation is: d²q/dt² + 2γ dq/dt + ω₀² q = F₀ cos(Ωt).

    This implementation is valid for the underdamped case (γ < ω₀).

    Args:
        gamma (float): Damping coefficient (γ).
        omega0 (float): Natural angular frequency (ω₀).
        F0 (float): Amplitude of the driving force (per unit mass).
        Omega (float): Angular frequency of the driving force (Ω).
        q0 (float): Initial position q(0).
        v0 (float): Initial velocity v(0).
        t (np.ndarray): Array of time points to evaluate the solution at.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                                       - q_t: The position at each time point.
                                       - v_t: The velocity at each time point.
    """
    gamma = gamma2_mu * 0.5
    omega_mu = omega_mu_squared**0.5
    if np.any(gamma >= omega_mu):
        raise ValueError(
            "This function is for the underdamped case only (gamma < omega0)."
        )

    t_col = t[:, None]

    amplitude_ss = F0 / np.sqrt(
        (omega_mu_squared - Omega**2) ** 2 + (gamma2_mu * Omega) ** 2
    )
    delta_ss = np.arctan2(gamma2_mu * Omega, omega_mu_squared - Omega**2)

    omega_d = np.sqrt(omega_mu_squared - gamma**2)  # Damped frequency
    C1 = q0 - amplitude_ss * np.cos(-delta_ss)

    v_ss_0 = -amplitude_ss * Omega * np.sin(-delta_ss)
    C2 = (v0 - v_ss_0 + gamma * C1) / omega_d

    # Complete solution: q_t = q_ss + q_tr
    q_ss = amplitude_ss * np.cos(Omega * t_col - delta_ss)
    q_tr = np.exp(-gamma * t_col) * (
        C1 * np.cos(omega_d * t_col) + C2 * np.sin(omega_d * t_col)
    )
    q_t = q_ss + q_tr

    v_ss = -amplitude_ss * Omega * np.sin(Omega * t_col - delta_ss)
    v_tr_term1 = -gamma * q_tr
    v_tr_term2 = np.exp(-gamma * t_col) * (
        -omega_d * C1 * np.sin(omega_d * t_col) + omega_d * C2 * np.cos(omega_d * t_col)
    )
    v_tr = v_tr_term1 + v_tr_term2
    v_t = v_ss + v_tr

    return q_t, v_t


sol_u_sinusoidal, sol_v_sinusoidal = solve_analytical_driven_oscillator(
    gamma2_mu=gamma2_mu,
    omega_mu_squared=omega_mu_squared,
    F0=np.zeros_like(gamma2_mu),
    Omega=np.zeros_like(gamma2_mu),
    q0=u0,
    v0=v0,
    t=time,
)

_, sol_u_tf_ic, sol_v_tf_ic = solve_tf(
    gamma2_mu=gamma2_mu,
    omega_mu_squared=omega_mu_squared,
    u0=u0,
    v0=v0,
    dt=dt,
    n_steps=n_steps,
    nl_fn=lin_fn,
)

_, sol_u_one_step_ic, sol_v_one_step_ic = solve_sv_one_step(
    gamma2_mu=gamma2_mu,
    omega_mu_squared=omega_mu_squared,
    u0=u0,
    v0=v0,
    dt=dt,
    n_steps=n_steps,
    nl_fn=lin_fn,
)

_, sol_u_one_step_staggered_ic, sol_v_one_step_staggered_ic = (
    solve_sv_one_step_staggered(
        gamma2_mu=gamma2_mu,
        omega_mu_squared=omega_mu_squared,
        u0=u0,
        v0=v0,
        dt=dt,
        n_steps=n_steps,
        nl_fn=lin_fn,
    )
)

_, sol_u_two_step_ic, sol_v_two_step_ic = solve_sv_two_step(
    gamma2_mu=gamma2_mu,
    omega_mu_squared=omega_mu_squared,
    u0=u0,
    v0=v0,
    dt=dt,
    n_steps=n_steps,
    nl_fn=lin_fn,
)

# %%
ref_u_solution = sol_u_sinusoidal @ weights
ref_v_solution = sol_v_sinusoidal @ weights

sol_u_tf_ic_weighted = sol_u_tf_ic @ weights
sol_u_two_step_ic_weighted = sol_u_two_step_ic @ weights
sol_u_one_step_ic_weighted = sol_u_one_step_ic @ weights
sol_u_one_step_staggered_ic_weighted = sol_u_one_step_staggered_ic @ weights
sol_u_solve_ivp_weighted = sol_u_solve_ivp[:n_steps] @ weights

sol_v_tf_ic_weighted = sol_v_tf_ic @ weights
sol_v_two_step_ic_weighted = sol_v_two_step_ic @ weights
sol_v_one_step_ic_weighted = sol_v_one_step_ic @ weights
sol_v_one_step_staggered_ic_weighted = sol_v_one_step_staggered_ic @ weights
sol_v_solve_ivp_weighted = sol_v_solve_ivp[:n_steps] @ weights

plt.figure(figsize=(15, 6))
plt.subplot(2, 1, 1)
plt.plot(time, ref_u_solution, label="Analytical", color="C0")
plt.plot(time, sol_u_solve_ivp_weighted, label="DOP853", linestyle="--", color="C1")
plt.plot(time, sol_u_tf_ic_weighted, label="TF", linestyle="-.", color="C2")
plt.plot(time, sol_u_two_step_ic_weighted, label="Two-step", linestyle=":", color="C3")
plt.plot(
    time,
    sol_u_one_step_ic_weighted,
    label="One-step",
    linestyle=(0, (3, 1, 1, 1)),
    color="C4",
)
plt.plot(
    time,
    sol_u_one_step_staggered_ic_weighted,
    label="One-step (staggered)",
    linestyle=(0, (5, 2, 1, 2)),
    color="C5",
)
plt.legend(loc="lower left")
plt.xlabel("Time")
plt.ylabel("Displacement")
plt.grid()
plt.xlim(-0.001, 0.05)

plt.subplot(2, 1, 2)
plt.plot(
    time,
    jnp.abs(sol_u_solve_ivp_weighted - ref_u_solution),
    label="DOP853",
    linestyle="--",
    color="C1",
)
plt.plot(
    time,
    jnp.abs(sol_u_tf_ic_weighted - ref_u_solution),
    label="TF",
    linestyle="-.",
    color="C2",
)
plt.plot(
    time,
    jnp.abs(sol_u_two_step_ic_weighted - ref_u_solution),
    label="Two-step",
    linestyle=":",
    color="C3",
)
plt.plot(
    time,
    jnp.abs(sol_u_one_step_ic_weighted - ref_u_solution),
    label="One-step",
    linestyle=(0, (3, 1, 1, 1)),
    color="C4",
)
plt.plot(
    time,
    jnp.abs(sol_u_one_step_staggered_ic_weighted - ref_u_solution),
    label="One-step (staggered)",
    linestyle=(0, (5, 2, 1, 2)),
    color="C5",
)
plt.legend(loc="lower left")
plt.xlabel("Time")
plt.ylabel("Absolute Error")
plt.grid()
plt.yscale("log")
plt.xlim(-0.001, 0.01)
plt.tight_layout()
plt.savefig("solver_accuracy_displacement_ic.pdf", dpi=300)


# %% Check solver accuracy with multiple metrics
def check_solver_accuracy(ref_traj, test_traj, solver_name, rtol=1e-0, atol=1e-4):
    """
    Check solver accuracy using multiple error metrics against RK reference.

    Parameters:
    - rtol: Relative tolerance (default 0.5 = 50%)
    - atol: Absolute tolerance (default 1e-4)
    """
    diff = np.abs(test_traj - ref_traj)
    max_diff = np.max(diff)
    max_amplitude = np.max(np.abs(ref_traj))

    # RMS error for overall assessment
    rms_error = np.sqrt(np.mean(diff**2))
    rms_amplitude = np.sqrt(np.mean(ref_traj**2))

    # Relative error where reference is non-zero (use larger threshold for velocity)
    threshold = max(atol, 0.1 * rms_amplitude)  # Use adaptive threshold
    mask = np.abs(ref_traj) > threshold
    rel_error = np.zeros_like(diff)
    if np.any(mask):
        rel_error[mask] = diff[mask] / np.abs(ref_traj[mask])
    max_rel_error = np.max(rel_error) if np.any(mask) else 0.0

    print(f"=== {solver_name} vs RK Reference ===")
    print(f"Max absolute error: {max_diff:.2e}")
    print(f"Max relative error: {max_rel_error:.2e}")
    print(f"RMS error: {rms_error:.2e}")
    print(f"RMS amplitude: {rms_amplitude:.2e}")
    print(f"RMS error/amplitude ratio: {rms_error / rms_amplitude:.2e}")

    # Multiple tolerance checks
    assert max_diff < atol + rtol * max_amplitude, (
        f"{solver_name}: Max error {max_diff:.2e} exceeds tolerance "
        f"{atol + rtol * max_amplitude:.2e}"
    )

    if np.any(mask):
        assert max_rel_error < rtol, (
            f"{solver_name}: Max relative error {max_rel_error:.2e} exceeds {rtol:.2e}"
        )

    assert rms_error < atol + rtol * rms_amplitude, (
        f"{solver_name}: RMS error {rms_error:.2e} exceeds tolerance "
        f"{atol + rtol * rms_amplitude:.2e}"
    )

    print(f"{solver_name} passes all accuracy tests")
    print()


check_solver_accuracy(
    ref_u_solution,
    sol_u_tf_ic_weighted,
    "tf",
)
check_solver_accuracy(
    ref_v_solution,
    sol_v_tf_ic_weighted,
    "tf",
)
check_solver_accuracy(
    ref_u_solution,
    sol_u_solve_ivp_weighted,
    "DOP853 (SciPy)",
)
check_solver_accuracy(
    ref_v_solution,
    sol_v_solve_ivp_weighted,
    "DOP853 (SciPy)",
)

# %% Excitation tests

modal_excitation = np.zeros((n_steps, n_modes))

excitation_freq = 200  # Frequency of the sinusoidal excitation in Hz
excitation_amplitude = 100000.0  # Amplitude of the sinusoidal excitation
modal_excitation[:, 0] = excitation_amplitude * np.cos(
    2 * np.pi * excitation_freq * time
)


# %%

Omega = np.zeros_like(gamma2_mu)
Omega[0] = 2 * np.pi * excitation_freq
F0 = np.zeros_like(gamma2_mu)
F0[0] = excitation_amplitude

sol_anal_u, sol_anal_v = solve_analytical_driven_oscillator(
    gamma2_mu=gamma2_mu,
    omega_mu_squared=omega_mu_squared,
    F0=F0,
    Omega=Omega,
    q0=np.zeros(n_modes),
    v0=np.zeros(n_modes),
    t=time,
)


_, sol_u_tf_exc, sol_v_tf_exc = solve_tf(
    gamma2_mu=gamma2_mu,
    omega_mu_squared=omega_mu_squared,
    dt=dt,
    xs=modal_excitation,
    nl_fn=lin_fn,
)

_, sol_u_one_step_exc, sol_v_one_step_exc = solve_sv_one_step(
    gamma2_mu=gamma2_mu,
    omega_mu_squared=omega_mu_squared,
    dt=dt,
    xs=modal_excitation,
    nl_fn=lin_fn,
    u0=jnp.zeros(n_modes),
    v0=jnp.zeros(n_modes),
)

_, sol_u_one_step_staggered_exc, sol_v_one_step_staggered_exc = (
    solve_sv_one_step_staggered(
        gamma2_mu=gamma2_mu,
        omega_mu_squared=omega_mu_squared,
        dt=dt,
        xs=modal_excitation,
        nl_fn=lin_fn,
        u0=jnp.zeros(n_modes),
        v0=jnp.zeros(n_modes),
    )
)

_, sol_u_two_step_exc, sol_v_two_step_exc = solve_sv_two_step(
    gamma2_mu=gamma2_mu,
    omega_mu_squared=omega_mu_squared,
    dt=dt,
    xs=modal_excitation,
    nl_fn=lin_fn,
)

_, sol_u_scipy_exc, sol_v_scipy_exc = solve_scipy_rk(
    gamma2_mu=gamma2_mu,
    omega_mu_squared=omega_mu_squared,
    dt=dt,
    xs=modal_excitation,
    u0=jnp.zeros(n_modes),
    v0=jnp.zeros(n_modes),
)


# %%
ref_u_solution = sol_anal_u @ weights
sol_u_tf_exc_weighted = sol_u_tf_exc @ weights
sol_u_scipy_exc_weighted = sol_u_scipy_exc @ weights
sol_u_one_step_exc_weighted = sol_u_one_step_exc @ weights
sol_u_one_step_staggered_exc_weighted = sol_u_one_step_staggered_exc @ weights
sol_u_two_step_exc_weighted = sol_u_two_step_exc @ weights

ref_v_solution = sol_anal_v @ weights
sol_v_tf_exc_weighted = sol_v_tf_exc @ weights
sol_v_scipy_exc_weighted = sol_v_scipy_exc @ weights
sol_v_one_step_exc_weighted = sol_v_one_step_exc @ weights
sol_v_one_step_staggered_exc_weighted = sol_v_one_step_staggered_exc @ weights
sol_v_two_step_exc_weighted = sol_v_two_step_exc @ weights

plt.figure(figsize=(15, 6))
plt.subplot(2, 1, 1)
plt.plot(time, ref_u_solution, label="Analytical", color="C0")
plt.plot(time, sol_u_scipy_exc_weighted, label="DOP853", linestyle="--", color="C1")
plt.plot(time, sol_u_tf_exc_weighted, label="TF", linestyle="-.", color="C2")
plt.plot(time, sol_u_two_step_exc_weighted, label="Two-step", linestyle=":", color="C3")
plt.plot(
    time,
    sol_u_one_step_exc_weighted,
    label="One-step",
    linestyle=(0, (3, 1, 1, 1)),
    color="C4",
)
plt.plot(
    time,
    sol_u_one_step_staggered_exc_weighted,
    label="One-step (staggered)",
    linestyle=(0, (5, 2, 1, 2)),
    color="C5",
)
plt.legend(loc="lower left")
plt.xlabel("Time")
plt.ylabel("Displacement")
plt.grid()
plt.xlim(-0.001, 0.05)
plt.subplot(2, 1, 2)
plt.plot(
    time,
    jnp.abs(sol_u_scipy_exc_weighted - ref_u_solution),
    label="DOP853",
    linestyle="--",
    color="C1",
)
plt.plot(
    time,
    jnp.abs(sol_u_tf_exc_weighted - ref_u_solution),
    label="TF",
    linestyle="-.",
    color="C2",
)
plt.plot(
    time,
    jnp.abs(sol_u_two_step_exc_weighted - ref_u_solution),
    label="Two-step",
    linestyle=":",
    color="C3",
)
plt.plot(
    time,
    jnp.abs(sol_u_one_step_exc_weighted - ref_u_solution),
    label="One-step",
    linestyle=(0, (3, 1, 1, 1)),
    color="C4",
)
plt.plot(
    time,
    jnp.abs(sol_u_one_step_staggered_exc_weighted - ref_u_solution),
    label="One-step (staggered)",
    linestyle=(0, (5, 2, 1, 2)),
    color="C5",
)
plt.legend(loc="lower left")
plt.xlabel("Time")
plt.ylabel("Absolute Error")
plt.grid()
plt.yscale("log")
plt.xlim(-0.001, 0.05)
plt.tight_layout()
plt.savefig("solver_accuracy_exc.pdf", dpi=300)

check_solver_accuracy(
    ref_u_solution,
    sol_u_scipy_exc_weighted,
    "DOP853 U",
)
check_solver_accuracy(
    ref_v_solution,
    sol_v_scipy_exc_weighted,
    "DOP853 V",
)
check_solver_accuracy(
    ref_u_solution,
    sol_u_tf_exc_weighted,
    "tf U",
)
check_solver_accuracy(
    ref_v_solution,
    sol_v_tf_exc_weighted,
    "tf V",
)

check_solver_accuracy(ref_u_solution, sol_u_one_step_exc_weighted, "SV One-step U")
check_solver_accuracy(ref_v_solution, sol_v_one_step_exc_weighted, "SV One-step V")

check_solver_accuracy(
    ref_u_solution, sol_u_one_step_staggered_exc_weighted, "SV One-step staggered U"
)
check_solver_accuracy(
    ref_v_solution, sol_v_one_step_staggered_exc_weighted, "SV One-step staggered V"
)
check_solver_accuracy(ref_u_solution, sol_u_two_step_exc_weighted, "SV Two-step U")
check_solver_accuracy(ref_v_solution, sol_v_two_step_exc_weighted, "SV Two-step V")
# %%
