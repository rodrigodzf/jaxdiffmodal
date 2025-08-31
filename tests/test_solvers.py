# This is a test for the solvers

# %%
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

from jaxdiffmodal.excitations import create_1d_raised_cosine, create_pluck_modal
from jaxdiffmodal.ftm import (
    damping_term,
    evaluate_string_eigenfunctions,
    stiffness_term,
    string_eigenvalues,
    StringParameters,
)
from jaxdiffmodal.time_integrators import (
    solve_sv_excitation,
    solve_sv_ic,
    solve_sv_leapfrog,
    solve_sv_two_step,
)


#  %%
n_modes: int = 3
sample_rate: int = 16000
dt: float = 1.0 / sample_rate
n_steps = 100
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

u0 = jnp.array(exc)
v0 = 10 * jnp.sin(jnp.pi * jnp.arange(1, n_modes + 1) / n_modes)
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
    rtol: float = 1e-12,  #         Relative tolerance
    atol: float = 1e-14,  #         Absolute tolerance
):
    """
    Scipy solve_ivp wrapper with same interface as solve_sv_leapfrog.

    Parameters match solve_sv_leapfrog for consistency.
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

    # Concatenate initial conditions at the start to match other solvers
    # u_with_ic = np.concatenate([u0[None], u_solution], axis=0)
    v_with_ic = np.concatenate([v0[None], v_solution], axis=0)
    u_with_ic = u_solution  # Initial condition already included in solve_ivp output
    return None, u_with_ic, v_with_ic  # Return format matching other solvers


# Test with initial conditions only (no excitation)
_, sol_u_solve_ivp, sol_v_solve_ivp = solve_scipy_rk(
    gamma2_mu=gamma2_mu,
    omega_mu_squared=omega_mu_squared,
    dt=dt,
    n_steps=n_steps,
    u0=u0,
    v0=v0,
)

# Note: SciPy solver now returns (time, modes) format directly, no transpose needed


#  %% Test the linear solving without excitation
def lin_fn(q):
    return 0


_, traj = solve_sv_ic(
    gamma2_mu=gamma2_mu,
    omega_mu_squared=omega_mu_squared,
    u0=u0,
    v0=v0,
    dt=dt,
    n_steps=n_steps,
    nl_fn=lin_fn,
)

_, sol_u_leapfrog, sol_v_leapfrog = solve_sv_leapfrog(
    gamma2_mu=gamma2_mu,
    omega_mu_squared=omega_mu_squared,
    u0=u0,
    v0=v0,
    dt=dt,
    n_steps=n_steps,
    nl_fn=lin_fn,
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
n_mode = 2

# print(traj.shape, sol.shape)
plt.figure(figsize=(15, 6))
plt.subplot(1, 3, 1)
# plt.plot(time, traj[:, n_mode], label="Implicit")
# plt.plot(time, sol_u_leapfrog[:, n_mode], label="Leapfrog", linestyle="--")
plt.plot(time, sol_u_two_step_ic[:, n_mode], label="Two-step", linestyle="-.")
plt.plot(time, sol_u_solve_ivp[:n_steps, n_mode], label="SciPy", linestyle=":")
plt.legend()
plt.title("Initial Conditions Test - Position")
plt.xlabel("Time")
plt.ylabel("Displacement")

plt.subplot(1, 3, 2)
# plt.plot(time, sol_v_leapfrog[:, n_mode], label="Leapfrog", linestyle="--")
plt.plot(time, sol_v_two_step_ic[:n_steps, n_mode], label="Two-step", linestyle="-.")
plt.plot(time, sol_v_solve_ivp[:n_steps, n_mode], label="SciPy", linestyle=":")
plt.legend()
plt.title("Initial Conditions Test - Velocity")
plt.xlabel("Time")
plt.ylabel("Velocity")


# %% Check differences are within one order of magnitude
def check_solver_accuracy(ref_traj, test_traj, solver_name):
    diff = np.abs(test_traj - ref_traj)
    max_diff = np.max(diff)
    max_amplitude = np.max(np.abs(ref_traj))

    print(f"Max difference ({solver_name} vs Reference): {max_diff:.2e}")
    print(f"Max amplitude: {max_amplitude:.2e}")

    assert max_diff < max_amplitude, (
        f"{solver_name} solver differs by more than one order of magnitude"
    )

    print(f"✓ {solver_name} solver agrees within one order of magnitude")


print("=== Initial Conditions Test ===")
check_solver_accuracy(
    sol_u_solve_ivp[:n_steps, 2],
    traj[:, 2],
    "Implicit",
)
check_solver_accuracy(
    sol_u_solve_ivp[:n_steps, 2],
    sol_u_leapfrog[:, 2],
    "Leapfrog",
)
check_solver_accuracy(
    sol_u_solve_ivp[:n_steps, 2],
    sol_u_two_step_ic[:, 2],
    "Two-step IC",
)

print("\n=== Velocity Comparisons (Initial Conditions) ===")
check_solver_accuracy(
    sol_v_solve_ivp[:n_steps, 2],
    sol_v_leapfrog[:, 2],
    "Leapfrog velocity",
)
check_solver_accuracy(
    sol_v_solve_ivp[:n_steps, 2],
    sol_v_two_step_ic[:, 2],
    "Two-step velocity",
)

# %% test using excitation
#  %%
rc = create_1d_raised_cosine(
    duration=time[-1].item() + dt,
    start_time=time[20].item(),
    end_time=time[30].item(),
    amplitude=1.0,
    sample_rate=sample_rate,
)

modal_excitation = exc * rc[..., None]  # (n_steps, n_modes)
# %%
_, sol_u_exc = solve_sv_excitation(
    gamma2_mu=gamma2_mu,
    omega_mu_squared=omega_mu_squared,
    modal_excitation=modal_excitation,
    dt=dt,
    nl_fn=lin_fn,
    u0=jnp.zeros(n_modes),
    v0=jnp.zeros(n_modes),
)
_, sol_u_leapfrog, sol_v_leapfrog_exc = solve_sv_leapfrog(
    gamma2_mu=gamma2_mu,
    omega_mu_squared=omega_mu_squared,
    dt=dt,
    xs=modal_excitation,
    nl_fn=lin_fn,
    u0=jnp.zeros(n_modes),
    v0=jnp.zeros(n_modes),
)

_, sol_u_two_step, sol_v_two_step = solve_sv_two_step(
    gamma2_mu=gamma2_mu,
    omega_mu_squared=omega_mu_squared,
    dt=dt,
    xs=modal_excitation,
    nl_fn=lin_fn,
    u0=jnp.zeros(n_modes),
    v0=jnp.zeros(n_modes),
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
print("Excitation shape:", sol_u_exc.shape)
print("Leapfrog shape:", sol_u_leapfrog.shape)
print("Two-step shape:", sol_u_two_step.shape)
print("Two-step exc shape:", sol_u_two_step_exc.shape)
print("SciPy exc shape:", sol_u_scipy_exc.shape)
print("SciPy velocities shape (ICs):", sol_v_solve_ivp.shape)
print("SciPy velocities shape (exc):", sol_v_scipy_exc.shape)
print("Leapfrog velocities shape (ICs):", sol_v_leapfrog.shape)
print("Leapfrog velocities shape (exc):", sol_v_leapfrog_exc.shape)
print("Two-step velocities shape (ICs):", sol_v_two_step_ic.shape)
print("Two-step velocities shape (exc):", sol_v_two_step_exc.shape)

plt.subplot(1, 3, 3)
plt.plot(time, sol_u_exc[:n_steps, n_mode], label="Excitation")
plt.plot(time, sol_u_leapfrog[:, n_mode], label="Leapfrog", linestyle="--")
plt.plot(time, sol_u_two_step[:n_steps, n_mode], label="Two-step (ICs)", linestyle="-.")
plt.plot(
    time, sol_u_two_step_exc[:n_steps, n_mode], label="Two-step (exc)", linestyle=":"
)
plt.plot(
    time,
    sol_u_scipy_exc[:n_steps, n_mode],
    label="SciPy (exc)",
    linestyle="-",
    alpha=0.7,
)
plt.legend()
plt.title("Excitation Test - Position")
plt.xlabel("Time")
plt.ylabel("Displacement")

# Add velocity subplot for excitation
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(time, sol_v_leapfrog_exc[:, n_mode], label="Leapfrog", linestyle="--")
plt.plot(
    time, sol_v_two_step_exc[:n_steps, n_mode], label="Two-step (exc)", linestyle=":"
)
plt.plot(
    time,
    sol_v_scipy_exc[:n_steps, n_mode],
    label="SciPy (exc)",
    linestyle="-",
    alpha=0.7,
)
plt.legend()
plt.title("Excitation Test - Velocity")
plt.xlabel("Time")
plt.ylabel("Velocity")

plt.subplot(1, 2, 2)
plt.plot(time, sol_u_exc[:n_steps, n_mode], label="Excitation")
plt.plot(time, sol_u_leapfrog[:, n_mode], label="Leapfrog", linestyle="--")
plt.plot(
    time, sol_u_two_step_exc[:n_steps, n_mode], label="Two-step (exc)", linestyle=":"
)
plt.plot(
    time,
    sol_u_scipy_exc[:n_steps, n_mode],
    label="SciPy (exc)",
    linestyle="-",
    alpha=0.7,
)
plt.legend()
plt.title("Excitation Test - Position")
plt.xlabel("Time")
plt.ylabel("Displacement")

plt.tight_layout()
plt.show()

print("\n=== Excitation Test ===")
# Test two-step solver accuracy - now both should have same shape (101,)
check_solver_accuracy(
    sol_u_exc[:, n_mode], sol_u_two_step_exc[:, n_mode], "Two-step (exc only)"
)
# For the ICs case, we need to slice to match the time array
check_solver_accuracy(
    sol_u_exc[:n_steps, n_mode], sol_u_two_step[:n_steps, n_mode], "Two-step (with ICs)"
)
# Test SciPy excitation solver vs reference
check_solver_accuracy(
    sol_u_exc[:n_steps, n_mode], sol_u_scipy_exc[:n_steps, n_mode], "SciPy (exc)"
)

print("\n=== Velocity Comparisons (Excitation) ===")
# Compare velocities for excitation case
check_solver_accuracy(
    sol_v_leapfrog_exc[:, n_mode],
    sol_v_two_step_exc[:n_steps, n_mode],
    "Two-step vs Leapfrog velocity (exc)",
)
check_solver_accuracy(
    sol_v_leapfrog_exc[:, n_mode],
    sol_v_scipy_exc[:n_steps, n_mode],
    "SciPy vs Leapfrog velocity (exc)",
)
# %%
