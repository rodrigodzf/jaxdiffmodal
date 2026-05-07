from collections.abc import Callable
from functools import partial

import einops
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxdiffmodal.ftm import PlateParameters, StringParameters


def A_inv_vector(
    h: float,
    damping: Float[Array, " N"],
) -> Float[Array, " N"]:
    return 2.0 * h**2 / (2.0 + damping * h)


def B_vector(
    h: float,
    stiffness: Float[Array, " N"],
) -> Float[Array, " N"]:
    return 2.0 / h**2 - stiffness


def C_vector(
    h: float,
    damping: Float[Array, " N"],
) -> Float[Array, " N"]:
    return -1.0 / h**2 + (damping / (2.0 * h))


def make_identity_nl_fn() -> Callable[[Float[Array, " N"]], Float[Array, " N"]]:
    def nl_fn(q: Float[Array, " N"]) -> Float[Array, " N"]:
        return q

    return nl_fn


def make_vk_nl_fn(
    H: Float[Array, "N N N"],
) -> Callable[[Float[Array, " N"]], Float[Array, " N"]]:
    r"""
    Create nonlinear function for von Kármán plate dynamics.

    Constructs a function that computes the nonlinear coupling term
    in the von Kármán plate equations using the H-tensor.

    Parameters
    ----------
    H : jax.numpy.ndarray
        Nonlinear coupling tensor of shape (n_modes, n_modes, n_modes)

    Returns
    -------
    callable
        Function that computes nonlinear term: nl_fn(q) -> jax.numpy.ndarray
        The returned function takes modal amplitudes q and returns the
        nonlinear coupling terms.

    Notes
    -----
    The nonlinear term is computed as:

    $$\text{nl}_s = \sum_{n,p,q,r} H_{npq} H_{nrs} q_p q_q q_r$$

    This represents the cubic nonlinearity in the von Kármán plate equations
    arising from geometric nonlinearities.
    """

    def nl_fn(q: Float[Array, " N"]) -> Float[Array, " N"]:
        # Type ignore for einops.einsum as it has complex overload matching
        return einops.einsum(  # type: ignore
            H,
            H,
            q,
            q,
            q,
            "n p q, n r s, p, q, r -> s",
        )

    return nl_fn


def make_tm_nl_fn(
    lambda_mu: Float[Array, " N"], factors: Float[Array, "N N"]
) -> Callable[[Float[Array, " N"]], Float[Array, " N"]]:
    r"""
    Create nonlinear function for Timoshenko beam dynamics.

    Constructs a function that computes the nonlinear coupling term
    for Timoshenko beam equations with geometric nonlinearities.

    Parameters
    ----------
    lambda_mu : jax.numpy.ndarray
        Modal coupling coefficients for nonlinear terms
    factors : jax.numpy.ndarray
        Coupling matrix between modes

    Returns
    -------
    callable
        Function that computes nonlinear term: nl_fn(q) -> jax.numpy.ndarray
        The returned function takes modal amplitudes q and returns
        nonlinear coupling terms.

    Notes
    -----
    The nonlinear term is computed as:

    $$\text{nl}_\mu = \lambda_\mu q_\mu \sum_\nu F_{\mu\nu} q_\nu^2$$

    where $F_{\mu\nu}$ are the coupling factors between modes.
    """

    def nl_fn(q: Float[Array, " N"]) -> Float[Array, " N"]:
        return lambda_mu * q * (factors @ q**2)

    return nl_fn


def plate_tau_with_density(plate_params: PlateParameters) -> float:
    r"""
    Compute normalized time constant for plate dynamics.

    Calculates the characteristic time scale for plate vibrations
    normalized by the material density.

    Parameters
    ----------
    plate_params : PlateParameters
        Plate parameters containing material properties and dimensions

    Returns
    -------
    float
        Normalized time constant $\tau/\rho$ where $\tau$ is the
        characteristic time scale and $\rho$ is the density

    Notes
    -----
    The time constant is computed as:

    $$\frac{\tau}{\rho} = \frac{Eh}{2\rho l_1 l_2 (1-\nu^2)}$$

    where $E$ is Young's modulus, $h$ is thickness, $l_1, l_2$ are
    plate dimensions, $\nu$ is Poisson's ratio, and $\rho$ is density.
    """
    plate_tau = (plate_params.E * plate_params.h) / (
        2 * plate_params.l1 * plate_params.l2 * (1 - plate_params.nu**2)
    )
    return plate_tau / plate_params.density


def string_tau_with_density(string_params: StringParameters) -> float:
    r"""
    Compute normalized time constant for string dynamics.

    Calculates the characteristic time scale for string vibrations
    normalized by the material density.

    Parameters
    ----------
    string_params : StringParameters
        String parameters containing material properties and dimensions

    Returns
    -------
    float
        Normalized time constant $\tau/\rho$ where $\tau$ is the
        characteristic time scale and $\rho$ is the density

    Notes
    -----
    The time constant is computed as:

    $$\frac{\tau}{\rho} = \frac{EA}{2\rho L}$$

    where $E$ is Young's modulus, $A$ is cross-sectional area,
    $L$ is string length, and $\rho$ is density.
    """
    string_tau = (
        string_params.E * string_params.A / (string_params.length * 2)
    ) / string_params.density
    return string_tau


def rk4_step(
    u0: Float[Array, " N"],
    v0: Float[Array, " N"],
    dt: float,
    gamma2_mu: Float[Array, " N"],
    omega_mu_squared: Float[Array, " N"],
    nl_fn: Callable[[Float[Array, " N"]], Float[Array, " N"]] | None = None,
    excitation: Float[Array, " N"] | None = None,
) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
    """
    One step of RK4 for the second-order damped oscillator.
    Returns: (u1, v1)
    """
    n_modes = u0.shape[0]
    apply_nl = nl_fn if nl_fn is not None else lambda q: 0.0
    exc = excitation if excitation is not None else 0.0

    def f(x):
        u = x[:n_modes]
        v = x[n_modes:]
        return jnp.concatenate(
            [v, -gamma2_mu * v - omega_mu_squared * u - apply_nl(u) + exc]
        )

    x = jnp.concatenate([u0, v0])
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)

    x_next = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    u1 = x_next[:n_modes]
    v1 = x_next[n_modes:]
    return u1, v1


@partial(
    jax.jit,
    static_argnames=(
        "n_steps",
        "nl_fn",
    ),
)
def solve_sv_one_step(
    gamma2_mu: Float[Array, " N"],
    omega_mu_squared: Float[Array, " N"],
    dt: float,
    n_steps: int | None = None,
    nl_fn: Callable[[Float[Array, " N"]], Float[Array, " N"]] | None = None,
    u0: Float[Array, " N"] | None = None,
    v0: Float[Array, " N"] | None = None,
    xs: Float[Array, "T N"] | None = None,
) -> tuple[
    tuple[Float[Array, " N"], Float[Array, " N"]],
    Float[Array, "T N"],
    Float[Array, "T N"],
]:
    r"""
    Solve using one-step "leapfrog" Verlet scheme with initial conditions
    and external forces.

    Implements the one-step Verlet scheme using an integer time grid
    where positions and velocities are at integer steps.
    See:
    - "Geometric numerical integration illustrated by the Stoermer-Verlet method", Hairer et al. 2003
    - "Learning Nonlinear Dynamics in Physical Modelling Synthesis using Neural Ordinary Differential Equations", Zheleznov et al. 2025

    Parameters
    ----------
    gamma2_mu : jax.numpy.ndarray
        Damping coefficients (2*gamma), shape (n_modes,)
    omega_mu_squared : jax.numpy.ndarray
        Squared natural frequencies, shape (n_modes,)
    u0 : jax.numpy.ndarray
        Initial displacement, shape (n_modes,)
    v0 : jax.numpy.ndarray
        Initial velocity, shape (n_modes,)
    xs: jax.numpy.ndarray
        External force, shape (T, n_modes)
    dt : float
        Time step size
    n_steps : int
        Number of time steps
    nl_fn : callable
        Nonlinear function

    Returns
    -------
    tuple
        Final state, time series of positions, and time series of velocities
    """
    n_modes = gamma2_mu.shape[0]

    # Set defaults for optional parameters
    u0 = u0 if u0 is not None else jnp.zeros(n_modes)
    v0 = v0 if v0 is not None else jnp.zeros(n_modes)

    # Determine number of steps
    if xs is not None:
        n_steps = n_steps if n_steps is not None else xs.shape[0]
        # Create scan inputs for the external force
        # we need the (f_n, f_{n+1}) pairs for the scheme
        # Pad xs with final value to ensure we process n_steps
        xs_padded = jnp.concatenate([xs, xs[-1:]], axis=0)
        scan_inputs = (xs_padded[:-1], xs_padded[1:])
    elif n_steps is None:
        raise ValueError("Either xs or n_steps must be provided")

    damping_factor = 1.0 + gamma2_mu * dt / 2.0

    apply_nl = nl_fn if nl_fn is not None else lambda q: 0.0

    def advance_state(
        state: tuple[Float[Array, " N"], Float[Array, " N"]],
        excitation,  # External force (array or None)
    ) -> tuple[
        tuple[Float[Array, " N"], Float[Array, " N"]],
        tuple[Float[Array, " N"], Float[Array, " N"]],
    ]:
        q, v = state
        if excitation is not None:
            f_curr, f_next = excitation
        else:
            f_curr, f_next = 0.0, 0.0

        # kick
        v_half_next = v + 0.5 * dt * (
            -gamma2_mu * v - omega_mu_squared * q - apply_nl(q) + f_curr
        )

        # drift
        q_next = q + dt * v_half_next

        # kick
        a = (-omega_mu_squared * q_next) - apply_nl(q_next) + f_next
        v_next = (v_half_next + 0.5 * dt * a) / damping_factor

        return (q_next, v_next), (q_next, v_next)

    if xs is not None:
        state, outputs = jax.lax.scan(
            advance_state,
            (u0, v0),
            scan_inputs,
            unroll=8,
        )
        final_positions, final_velocities = outputs
    else:
        assert n_steps is not None
        state, outputs = jax.lax.scan(
            advance_state,
            (u0, v0),
            None,
            length=n_steps - 1,
            unroll=8,
        )
        final_positions, final_velocities = outputs

    # Always concatenate initial state and slice to exactly n_steps
    full_positions = jnp.concatenate([u0[None], final_positions], axis=0)
    positions = full_positions[:n_steps]

    # For velocities, we need the initial velocity v0
    full_velocities = jnp.concatenate([v0[None], final_velocities], axis=0)
    velocities = full_velocities[:n_steps]

    return state, positions, velocities


@partial(
    jax.jit,
    static_argnames=(
        "n_steps",
        "nl_fn",
    ),
)
def solve_sv_one_step_staggered(
    gamma2_mu: Float[Array, " N"],
    omega_mu_squared: Float[Array, " N"],
    dt: float,
    n_steps: int | None = None,
    nl_fn: Callable[[Float[Array, " N"]], Float[Array, " N"]] | None = None,
    u0: Float[Array, " N"] | None = None,
    v0: Float[Array, " N"] | None = None,
    xs: Float[Array, "T N"] | None = None,
) -> tuple[
    tuple[Float[Array, " N"], Float[Array, " N"]],
    Float[Array, "T N"],
    Float[Array, "T N"],
]:
    r"""
    Solve using one-step "leapfrog" Verlet scheme with initial conditions
    and external forces.

    Implements the one-step Verlet scheme using staggered time grid
    where positions are at integer steps and velocities at half-steps.
    See "Geometric numerical integration illustrated by the Stoermer-Verlet method",
    Hairer et al. 2003

    Parameters
    ----------
    gamma2_mu : jax.numpy.ndarray
        Damping coefficients (2*gamma), shape (n_modes,)
    omega_mu_squared : jax.numpy.ndarray
        Squared natural frequencies, shape (n_modes,)
    u0 : jax.numpy.ndarray
        Initial displacement, shape (n_modes,)
    v0 : jax.numpy.ndarray
        Initial velocity, shape (n_modes,)
    xs: jax.numpy.ndarray
        External force, shape (T, n_modes)
    dt : float
        Time step size
    n_steps : int
        Number of time steps
    nl_fn : callable
        Nonlinear function

    Returns
    -------
    tuple
        Final state, time series of positions, and time series of velocities
    """
    n_modes = gamma2_mu.shape[0]

    # Set defaults for optional parameters
    u0 = u0 if u0 is not None else jnp.zeros(n_modes)
    v0 = v0 if v0 is not None else jnp.zeros(n_modes)

    # Determine number of steps
    if xs is not None:
        n_steps = n_steps if n_steps is not None else xs.shape[0]
    elif n_steps is None:
        raise ValueError("Either xs or n_steps must be provided")

    apply_nl = nl_fn if nl_fn is not None else lambda q: 0.0

    # Initial conditions for leapfrog scheme
    f0 = (xs[0] if xs is not None else 0.0) - omega_mu_squared * u0 - apply_nl(u0)
    a0 = -gamma2_mu * v0 + f0
    v_half_prev = v0 - 0.5 * dt * a0

    # Leapfrog coefficients
    damping_factor = 1.0 + gamma2_mu * dt / 2.0
    alpha = (1.0 - gamma2_mu * dt / 2.0) / damping_factor
    beta = dt / damping_factor

    def advance_state(
        state: tuple[Float[Array, " N"], Float[Array, " N"]],
        excitation,  # External force (array or None)
    ) -> tuple[
        tuple[Float[Array, " N"], Float[Array, " N"]],
        tuple[Float[Array, " N"], Float[Array, " N"]],
    ]:
        q, v_half = state

        # Force calculation: F = excitation - k*q - nl(q)
        f = (
            (excitation if excitation is not None else 0.0)
            - omega_mu_squared * q
            - apply_nl(q)
        )

        # Leapfrog update
        v_half_next = alpha * v_half + beta * f
        q_next = q + dt * v_half_next

        # Convert half-step velocity to full-step velocity: v_n = (v_half_{n-1/2} + v_half_{n+1/2}) / 2
        v_next = (v_half + v_half_next) / 2.0

        return (q_next, v_half_next), (q_next, v_next)

    if xs is not None:
        state, outputs = jax.lax.scan(
            advance_state,
            (u0, v_half_prev),
            xs,
            unroll=8,
        )
        final_positions, final_velocities = outputs
    else:
        assert n_steps is not None
        state, outputs = jax.lax.scan(
            advance_state,
            (u0, v_half_prev),
            None,
            length=n_steps,
            unroll=8,
        )
        final_positions, final_velocities = outputs

    # Always concatenate initial state and slice to exactly n_steps
    full_positions = jnp.concatenate(
        [u0[None], final_positions[:-1]],
        axis=0,
    )
    positions = full_positions[:n_steps]

    # The initial velocity is calculated in the loop
    velocities = final_velocities[:n_steps]

    return state, positions, velocities


@partial(
    jax.jit,
    static_argnames=(
        "n_steps",
        "nl_fn",
    ),
)
def solve_sv_two_step(
    gamma2_mu: Float[Array, " N"],
    omega_mu_squared: Float[Array, " N"],
    dt: float,
    n_steps: int | None = None,
    nl_fn: Callable[[Float[Array, " N"]], Float[Array, " N"]] | None = None,
    u0: Float[Array, " N"] | None = None,
    v0: Float[Array, " N"] | None = None,
    xs: Float[Array, "T N"] | None = None,
) -> tuple[
    tuple[Float[Array, " N"], Float[Array, " N"], Float[Array, " N"]],
    Float[Array, "T N"],
    Float[Array, "T N"],
]:
    r"""
    Solve using two-step Störmer-Verlet scheme with same interface as solve_sv_leapfrog.

    Implements the two-step Verlet scheme that can handle both initial conditions
    and external excitations. Combines functionality of solve_sv_excitation and solve_sv_ic.

    Parameters
    ----------
    gamma2_mu : jax.numpy.ndarray
        Damping coefficients (2*gamma), shape (n_modes,)
    omega_mu_squared : jax.numpy.ndarray
        Squared natural frequencies, shape (n_modes,)
    dt : float
        Time step size
    n_steps : int | None
        Number of time steps (optional if xs is provided)
    nl_fn : callable | None
        Nonlinear function (optional, defaults to zero)
    u0 : jax.numpy.ndarray | None
        Initial displacement, shape (n_modes,) (optional, defaults to zeros)
    v0 : jax.numpy.ndarray | None
        Initial velocity, shape (n_modes,) (optional, defaults to zeros)
    xs : jax.numpy.ndarray | None
        External excitation, shape (T, n_modes) (optional)

    Returns
    -------
    tuple
        Final state, time series of positions, and time series of velocities
    """
    n_modes = gamma2_mu.shape[0]

    # Set defaults for optional parameters
    u0 = u0 if u0 is not None else jnp.zeros(n_modes)
    v0 = v0 if v0 is not None else jnp.zeros(n_modes)
    apply_nl = nl_fn if nl_fn is not None else lambda q: jnp.zeros_like(q)

    # Determine number of steps
    if xs is not None:
        n_steps = n_steps if n_steps is not None else xs.shape[0]
    elif n_steps is None:
        raise ValueError("Either xs or n_steps must be provided")

    # Discretisation using Stoermer-Verlet method
    gamma_mu = gamma2_mu * 0.5
    b1 = dt**2 / (1.0 + gamma_mu * dt)
    a1 = (2.0 - dt**2 * omega_mu_squared) / (1.0 + gamma_mu * dt)
    a2 = (gamma_mu * dt - 1.0) / (1.0 + gamma_mu * dt)

    # Initialize first two states - note: q0 is at t=0, q1 should include first excitation
    q0 = u0
    first_excitation = xs[0] if xs is not None else None

    # Compute q1 using RK4 with nonlinearity and excitation
    q1, v1 = rk4_step(
        u0,
        v0,
        dt,
        gamma2_mu,
        omega_mu_squared,
        nl_fn=nl_fn,
        excitation=first_excitation,
    )

    def advance_state(
        state: tuple[Float[Array, " N"], Float[Array, " N"], Float[Array, " N"]],
        x: Float[Array, " N"] | None,
    ) -> tuple[
        tuple[Float[Array, " N"], Float[Array, " N"], Float[Array, " N"]],
        tuple[Float[Array, " N"], Float[Array, " N"]],
    ]:
        q_prev, q, v = state
        nl = apply_nl(q)
        excitation = x if x is not None else jnp.zeros(n_modes)
        q_next = a1 * q + a2 * q_prev - b1 * nl + excitation
        # Compute velocity as (q_next - q_prev) / (2 * dt)
        v_next = (q_next - q_prev) / (2.0 * dt)
        return (q, q_next, v_next), (q_next, v_next)

    if xs is not None:
        # With external excitation - process from step 1 onwards (we already handled step 0)
        remaining_xs = xs[1:] * b1
        state, outputs = jax.lax.scan(
            advance_state,
            (q0, q1, v1),
            remaining_xs,
            unroll=8,
        )
        # Extract positions and velocities from outputs
        final_positions, final_velocities = outputs
        # Concatenate initial state + all computed states (including q1 and final)
        full_positions = jnp.concatenate(
            [q0[None], q1[None], final_positions[:-1]],
            axis=0,
        )
        full_velocities = jnp.concatenate([v0[None], final_velocities], axis=0)
    else:
        # No external excitation - use None as input
        state, outputs = jax.lax.scan(
            advance_state,
            (q0, q1, v1),
            None,
            length=n_steps - 1,
            unroll=8,
        )
        # Extract positions and velocities from outputs
        final_positions, final_velocities = outputs
        # Concatenate initial states with computed trajectory
        full_positions = jnp.concatenate(
            [q0[None], q1[None], final_positions[:-1]],
            axis=0,
        )
        full_velocities = jnp.concatenate(
            [v0[None], final_velocities],
            axis=0,
        )

    # Ensure output matches requested n_steps, but don't truncate if we have excitation
    if xs is not None:
        # For excitation case, return full result to match solve_sv_excitation behavior
        final_positions_output = full_positions
        final_velocities_output = full_velocities
    else:
        # For n_steps case, truncate to exactly n_steps
        final_positions_output = full_positions[:n_steps]
        final_velocities_output = full_velocities[:n_steps]

    return state, final_positions_output, final_velocities_output


@partial(
    jax.jit,
    static_argnames=(
        "n_steps",
        "nl_fn",
    ),
)
def solve_tf(
    gamma2_mu: Float[Array, " N"],
    omega_mu_squared: Float[Array, " N"],
    dt: float,
    n_steps: int | None = None,
    nl_fn: Callable[[Float[Array, " N"]], Float[Array, " N"]] | None = None,
    u0: Float[Array, " N"] | None = None,
    v0: Float[Array, " N"] | None = None,
    xs: Float[Array, "T N"] | None = None,
) -> tuple[
    tuple[Float[Array, " N"], Float[Array, " N"], Float[Array, " N"]],
    Float[Array, "T N"],
    Float[Array, "T N"],
]:
    """Solve using transfer-function (TF) based recurrence."""
    n_modes = gamma2_mu.shape[0]

    # Set defaults for optional parameters
    u0 = u0 if u0 is not None else jnp.zeros(n_modes)
    v0 = v0 if v0 is not None else jnp.zeros(n_modes)
    apply_nl = nl_fn if nl_fn is not None else lambda q: jnp.zeros_like(q)

    # Determine number of steps
    if xs is not None:
        n_steps = n_steps if n_steps is not None else xs.shape[0]
    elif n_steps is None:
        raise ValueError("Either xs or n_steps must be provided")

    gamma_mu = gamma2_mu / 2.0
    omega_mu_damped = jnp.sqrt(omega_mu_squared - gamma_mu**2)
    radius = jnp.exp(-gamma_mu * dt)
    imag = radius * jnp.sin(omega_mu_damped * dt)
    real = radius * jnp.cos(omega_mu_damped * dt)

    b1 = dt * imag / omega_mu_damped

    a1 = 2.0 * real
    a2 = -(radius**2)

    # Initialize first two states
    q0 = u0
    first_excitation = xs[0] if xs is not None else None

    q1, v1 = rk4_step(
        u0,
        v0,
        dt,
        gamma2_mu,
        omega_mu_squared,
        nl_fn=nl_fn,
        excitation=first_excitation,
    )

    def advance_state(
        state: tuple[Float[Array, " N"], Float[Array, " N"], Float[Array, " N"]],
        x: Float[Array, " N"] | None,
    ) -> tuple[
        tuple[Float[Array, " N"], Float[Array, " N"], Float[Array, " N"]],
        tuple[Float[Array, " N"], Float[Array, " N"]],
    ]:
        q_prev, q_curr, v = state
        nl = apply_nl(q_curr)
        excitation = x if x is not None else jnp.zeros(n_modes)
        q_next = a1 * q_curr + a2 * q_prev - b1 * nl + excitation
        v_next = (q_next - q_prev) / (2.0 * dt)
        return (q_curr, q_next, v_next), (q_next, v_next)

    if xs is not None:
        remaining_xs = xs[1:] * b1
        state, outputs = jax.lax.scan(
            advance_state,
            (q0, q1, v1),
            remaining_xs,
            unroll=8,
        )
        final_positions, final_velocities = outputs
        full_positions = jnp.concatenate(
            [q0[None], q1[None], final_positions[:-1]],
            axis=0,
        )
        full_velocities = jnp.concatenate([v0[None], final_velocities], axis=0)
    else:
        state, outputs = jax.lax.scan(
            advance_state,
            (q0, q1, v1),
            None,
            length=n_steps - 1,
            unroll=8,
        )
        final_positions, final_velocities = outputs
        full_positions = jnp.concatenate(
            [q0[None], q1[None], final_positions[:-1]],
            axis=0,
        )
        full_velocities = jnp.concatenate(
            [v0[None], final_velocities],
            axis=0,
        )

    if xs is not None:
        final_positions_output = full_positions
        final_velocities_output = full_velocities
    else:
        final_positions_output = full_positions[:n_steps]
        final_velocities_output = full_velocities[:n_steps]

    return state, final_positions_output, final_velocities_output


def solve_sinusoidal(
    gamma2_mu: Float[Array, " N"],
    omega_mu_squared: Float[Array, " N"],
    u0: Float[Array, " N"],
    v0: Float[Array, " N"],
    n_steps: int,
    dt: float,
) -> Float[Array, "T N"]:
    """
    Solve the system of ODEs using complex exponentials

    Parameters
    ----------
    gamma2_mu : jnp.ndarray
        Damping coefficients
    omega_mu_squared : jnp.ndarray
        Squared frequencies
    ic : jnp.ndarray
        Initial conditions
    n_steps : int
        Number of steps
    dt : float
        Time step

    Returns
    -------
    jnp.ndarray
        Modal solution
    """
    gamma_mu = gamma2_mu * 0.5
    omega_damped = jnp.sqrt(omega_mu_squared - gamma_mu**2)
    s_mu = -gamma_mu + 1j * omega_damped
    z_mu = jnp.exp(s_mu * dt)
    C = u0 - ((v0 + gamma_mu * u0) / omega_damped) * 1j

    sol = C * z_mu ** jnp.arange(n_steps)[:, None]
    return jnp.real(sol)
