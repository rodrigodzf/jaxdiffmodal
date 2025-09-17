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
) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
    """
    One step of RK4 for the second-order damped oscillator.
    Returns: (u1, v1)
    """
    n_modes = u0.shape[0]

    def f(x):
        u = x[:n_modes]
        v = x[n_modes:]
        return jnp.concatenate([v, -gamma2_mu * v - omega_mu_squared * u])

    x = jnp.concatenate([u0, v0])
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)

    x_next = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    u1 = x_next[:n_modes]
    v1 = x_next[n_modes:]
    return u1, v1


def second_order_step(
    u0: Float[Array, " N"],
    v0: Float[Array, " N"],
    dt: float,
    gamma2_mu: Float[Array, " N"],
    omega_mu_squared: Float[Array, " N"],
) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
    r"""
    Perform one step of second-order Taylor expansion for damped oscillator.

    Advances the solution of the second-order differential equation
    $\ddot{u} + \gamma \dot{u} + \omega^2 u = 0$ by one time step using
    a second-order Taylor expansion method.

    Parameters
    ----------
    u0 : jax.numpy.ndarray
        Initial displacement, shape (n_modes,)
    v0 : jax.numpy.ndarray
        Initial velocity, shape (n_modes,)
    dt : float
        Time step size
    gamma2_mu : jax.numpy.ndarray
        Damping coefficients, shape (n_modes,)
    omega_mu_squared : jax.numpy.ndarray
        Squared natural frequencies, shape (n_modes,)

    Returns
    -------
    tuple[jax.numpy.ndarray, jax.numpy.ndarray]
        Updated displacement and velocity (u1, v1)

    Notes
    -----
    Uses second-order Taylor expansion:

    $$u_1 = u_0 + v_0 \Delta t + \frac{1}{2} a_0 (\Delta t)^2$$
    $$v_1 = v_0 + a_0 \Delta t$$

    where $a_0 = -\gamma v_0 - \omega^2 u_0$ is the initial acceleration.
    """
    a = -gamma2_mu * v0 - omega_mu_squared * u0  # ddot(q)
    u1 = u0 + v0 * dt + 0.5 * a * dt**2
    v1 = v0 + a * dt
    return u1, v1


@partial(jax.jit, static_argnames=("nl_fn",))
def solve_sv_vk_jax_scan(
    A_inv: Float[Array, " N"],
    B: Float[Array, " N"],
    C: Float[Array, " N"],
    modal_excitation: Float[Array, "T N"],
    nl_fn: Callable[[Float[Array, " N"]], Float[Array, " N"]],
) -> tuple[tuple[Float[Array, " N"], Float[Array, " N"]], Float[Array, "T N"]]:
    n_modes = A_inv.shape[0]
    q = jnp.zeros((n_modes,))  # Modal displacement vector at n (n_modes, 1)
    q_prev = jnp.zeros((n_modes,))  # Modal displacement vector at n-1

    def advance_state(
        state: tuple[Float[Array, " N"], Float[Array, " N"]],
        x: Float[Array, " N"],
    ) -> tuple[tuple[Float[Array, " N"], Float[Array, " N"]], Float[Array, " N"]]:
        # unpack state
        q_prev, q = state

        nl = nl_fn(q)

        # compute the next state
        q_next = B * q + C * q_prev - A_inv * nl + x

        # return the next state and the output
        return (q, q_next), q_next

    state, final = jax.lax.scan(
        advance_state,
        (q_prev, q),
        modal_excitation,  # (T, n_modes)
        unroll=8,
    )
    return state, final


@partial(jax.jit, static_argnames=("nl_fn",))
def solve_sv_excitation(
    gamma2_mu: Float[Array, " N"],
    omega_mu_squared: Float[Array, " N"],
    modal_excitation: Float[Array, "T N"],
    dt: float,
    nl_fn: Callable,
    u0: Float[Array, " N"] | None = None,
    v0: Float[Array, " N"] | None = None,
):
    # Discretisation using Stoermer-Verlet method
    A_inv = A_inv_vector(dt, gamma2_mu)
    B = B_vector(dt, omega_mu_squared) * A_inv
    C = C_vector(dt, gamma2_mu) * A_inv

    n_modes = A_inv.shape[0]

    # Set initial conditions
    u0 = u0 if u0 is not None else jnp.zeros(n_modes)
    v0 = v0 if v0 is not None else jnp.zeros(n_modes)

    # Initialize for two-step Verlet scheme
    # q_prev = u0 - dt * v0 + 0.5 * dt^2 * a0 where a0 = -gamma*v0 - omega^2*u0
    a0 = -gamma2_mu * v0 - omega_mu_squared * u0
    q_prev = u0 - dt * v0 + 0.5 * dt**2 * a0
    q = u0

    def advance_state(
        state: tuple[Float[Array, " N"], Float[Array, " N"]],  # initial state
        x: Float[Array, " N"],  # input
    ) -> tuple[tuple[Float[Array, " N"], Float[Array, " N"]], Float[Array, " N"]]:
        q_prev, q = state
        nl = nl_fn(q)
        q_next = B * q + C * q_prev - A_inv * nl + A_inv * x
        return (q, q_next), q_next

    state, final = jax.lax.scan(
        advance_state,
        (q_prev, q),
        modal_excitation,
        unroll=8,
    )

    # Include initial condition at the beginning like solve_sv_leapfrog
    full_result = jnp.concatenate([q[None], final], axis=0)
    n_steps = modal_excitation.shape[0] + 1
    final = full_result[:n_steps]

    return state, final


@partial(
    jax.jit,
    static_argnames=(
        "n_steps",
        "nl_fn",
    ),
)
def solve_sv_leapfrog(
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
        scan_inputs = (xs[:-1], xs[1:])
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
def solve_sv_leapfrog_2(
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

    # Initial conditions for leapfrog scheme
    a0 = -gamma2_mu * v0 - omega_mu_squared * u0  # Initial acceleration
    v_half_prev = v0 - 0.5 * dt * a0  # Half-step velocity at t_{-1/2}

    # Leapfrog coefficients
    damping_factor = 1.0 + gamma2_mu * dt / 2.0
    alpha = (1.0 - gamma2_mu * dt / 2.0) / damping_factor
    beta = dt / damping_factor

    apply_nl = nl_fn if nl_fn is not None else lambda q: 0.0

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
def solve_sv_ic(
    gamma2_mu: Float[Array, " N"],
    omega_mu_squared: Float[Array, " N"],
    u0: Float[Array, " N"],
    v0: Float[Array, " N"],
    dt: float,
    n_steps: int,
    nl_fn: Callable[[Float[Array, " N"]], Float[Array, " N"]],
) -> tuple[tuple[Float[Array, " N"], Float[Array, " N"]], Float[Array, "T N"]]:
    r"""
    Solve using two-step Störmer-Verlet scheme with initial conditions.

    Implements the two-step Verlet scheme.

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
    A_inv = A_inv_vector(dt, gamma2_mu)
    B = B_vector(dt, omega_mu_squared) * A_inv
    C = C_vector(dt, gamma2_mu) * A_inv

    q0 = u0
    q1, _ = rk4_step(u0, v0, dt, gamma2_mu, omega_mu_squared)

    def advance_state(
        state: tuple[Float[Array, " N"], Float[Array, " N"]],
        _: None,
    ) -> tuple[tuple[Float[Array, " N"], Float[Array, " N"]], Float[Array, " N"]]:
        q_prev, q = state
        nl = nl_fn(q)
        q_next = B * q + C * q_prev - A_inv * nl
        return (q, q_next), q_next

    state, final = jax.lax.scan(
        advance_state,
        (q0, q1),
        length=n_steps - 2,
        unroll=8,
    )
    final = jnp.concatenate([q0[None], q1[None], final], axis=0)

    return state, final


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
    A_inv = A_inv_vector(dt, gamma2_mu)
    B = B_vector(dt, omega_mu_squared) * A_inv
    C = C_vector(dt, gamma2_mu) * A_inv

    # Initialize first two states - note: q0 is at t=0, q1 should include first excitation
    q0 = u0
    if xs is not None:
        # For excitation case, compute q1 including the first excitation step
        first_excitation = xs[0] if xs.shape[0] > 0 else jnp.zeros(n_modes)
    else:
        first_excitation = jnp.zeros(n_modes)

    # Compute q1 using RK4 but include first excitation
    q1_no_exc, v1 = rk4_step(u0, v0, dt, gamma2_mu, omega_mu_squared)
    # Add first excitation contribution
    q1 = q1_no_exc + A_inv_vector(dt, gamma2_mu) * first_excitation

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
        q_next = B * q + C * q_prev - A_inv * nl + A_inv * excitation
        # Compute velocity as (q_next - q_prev) / (2 * dt)
        v_next = (q_next - q_prev) / (2.0 * dt)
        return (q, q_next, v_next), (q_next, v_next)

    if xs is not None:
        # With external excitation - process from step 1 onwards (we already handled step 0)
        remaining_xs = xs[1:] if xs.shape[0] > 1 else jnp.zeros((0, n_modes))
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
            [q0[None], q1[None], final_positions],
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


@partial(jax.jit, static_argnames=("nl_fn",))
def solve_tf_excitation(
    gamma2_mu: Float[Array, " N"],
    omega_mu_squared: Float[Array, " N"],
    modal_excitation: Float[Array, "T N"],
    dt: float,
    nl_fn: Callable[[Float[Array, " N"]], Float[Array, " N"]],
) -> tuple[tuple[Float[Array, " N"], Float[Array, " N"]], Float[Array, "T N"]]:
    """Solve using transfer-function (TF) based recurrence."""
    gamma_mu = gamma2_mu / 2.0
    omega_mu_damped = jnp.sqrt(omega_mu_squared - gamma_mu**2)
    radius = jnp.exp(-gamma_mu * dt)
    imag = radius * jnp.sin(omega_mu_damped * dt)
    real = radius * jnp.cos(omega_mu_damped * dt)

    b1_exc = dt * imag / omega_mu_damped

    a1 = 2.0 * real
    a2 = -(radius**2)

    n_modes = modal_excitation.shape[-1]
    q = jnp.zeros((n_modes,))  # Modal displacement vector at n (n_modes, 1)
    q_prev = jnp.zeros((n_modes,))  # Modal displacement vector at n-1

    def advance_state(
        state: tuple[Float[Array, " N"], Float[Array, " N"]],
        x: Float[Array, " N"],
    ) -> tuple[
        tuple[
            Float[Array, " N"],
            Float[Array, " N"],
        ],
        Float[Array, " N"],
    ]:
        q_prev, q_curr = state
        nl = nl_fn(q_curr)
        q_next = a1 * q_curr + a2 * q_prev - b1_exc * nl + b1_exc * x
        return (q_curr, q_next), q_next

    state, final = jax.lax.scan(
        advance_state,
        (q_prev, q),
        modal_excitation,
        unroll=8,
    )
    return state, final


@partial(
    jax.jit,
    static_argnames=(
        "n_steps",
        "nl_fn",
    ),
)
def solve_tf_ic(
    gamma2_mu: Float[Array, " N"],
    omega_mu_squared: Float[Array, " N"],
    u0: Float[Array, " N"],
    v0: Float[Array, " N"],
    dt: float,
    n_steps: int,
    nl_fn: Callable[[Float[Array, " N"]], Float[Array, " N"]],
) -> tuple[tuple[Float[Array, " N"], Float[Array, " N"]], Float[Array, "T N"]]:
    """Solve using transfer-function (TF) based recurrence."""
    gamma_mu = gamma2_mu / 2.0
    omega_mu_damped = jnp.sqrt(omega_mu_squared - gamma_mu**2)
    radius = jnp.exp(-gamma_mu * dt)
    imag = radius * jnp.sin(omega_mu_damped * dt)
    real = radius * jnp.cos(omega_mu_damped * dt)

    b1_exc = dt * imag / omega_mu_damped

    a1 = 2.0 * real
    a2 = -(radius**2)

    q0 = u0
    q1, _ = rk4_step(u0, v0, dt, gamma2_mu, omega_mu_squared)

    def advance_state(
        state: tuple[Float[Array, " N"], Float[Array, " N"]],
        _: None,
    ) -> tuple[tuple[Float[Array, " N"], Float[Array, " N"]], Float[Array, " N"]]:
        q_prev, q_curr = state
        nl = nl_fn(q_curr)
        q_next = a1 * q_curr + a2 * q_prev - b1_exc * nl
        return (q_curr, q_next), q_next

    state, final = jax.lax.scan(
        advance_state,
        (q0, q1),
        length=n_steps - 2,
        unroll=8,
    )
    final = jnp.concatenate([q0[None], q1[None], final], axis=0)
    return state, final


def solve_sinusoidal(
    gamma2_mu: Float[Array, " N"],
    omega_mu_squared: Float[Array, " N"],
    ic: Float[Array, " N"],
    n_steps: int,
    dt: float,
) -> Float[Array, "N T"]:
    """
    Solve the system of ODEs using complex exponentials
    NB: this assumes the ic is only for positions and that the initial velocities are 0

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
    gamma_mu = gamma2_mu / 2.0
    omega_mu = jnp.sqrt(omega_mu_squared - gamma_mu**2)
    s_mu = -gamma_mu + 1j * omega_mu
    z_mu = jnp.exp(s_mu * dt)

    z_mu_sequence = jnp.repeat(z_mu[None, :], n_steps - 1, axis=0)
    # include the initial condition
    z_mu_sequence = jnp.concatenate(
        [
            jnp.ones(
                shape=(1, gamma2_mu.shape[0]),
                dtype=gamma2_mu.dtype,
            ),
            z_mu_sequence,
        ],
        axis=0,
    )
    z_mu_sequence = jax.lax.associative_scan(
        jnp.multiply,
        z_mu_sequence,
        axis=0,
    )
    modal_sol = z_mu_sequence.real.T * ic[:, None]
    return modal_sol


def solve_sinusoidal_excitation(
    gamma2_mu: Float[Array, " N"],
    omega_mu_squared: Float[Array, " N"],
    modal_excitation: Float[Array, "T N"],
    dt: float,
) -> Float[Array, "T N"]:
    """
    Solve the modal system with sinusoidal response for external excitation using parallel scan.

    Parameters
    ----------
    gamma2_mu : jnp.ndarray
        Damping coefficients (n_modes,)
    omega_mu_squared : jnp.ndarray
        Squared frequencies (n_modes,)
    modal_excitation : jnp.ndarray
        Modal excitation (T, n_modes)
    dt : float
        Time step

    Returns
    -------
    jnp.ndarray
        Modal solution (T, n_modes)
    """

    gamma_mu = gamma2_mu / 2.0
    omega_mu = jnp.sqrt(omega_mu_squared - gamma_mu**2)
    s_mu = -gamma_mu + 1j * omega_mu
    z_mu = jnp.exp(s_mu * dt)

    # Number of time steps
    n_steps = modal_excitation.shape[0]

    z_mu_sequence = jnp.repeat(z_mu[None, :], n_steps, axis=0)

    # Define binary operator for parallel scan
    def binary_op(elem_i, elem_j):
        a_i, b_i = elem_i
        a_j, b_j = elem_j
        return a_i * a_j, b_i * a_j + b_j

    _, output = jax.lax.associative_scan(
        binary_op,
        (z_mu_sequence, modal_excitation.astype(z_mu_sequence.dtype)),
        axis=0,
    )

    modal_solution = output.imag

    return modal_solution
