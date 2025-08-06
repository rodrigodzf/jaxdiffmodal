from collections.abc import Callable
from functools import partial

import einops
import jax
import jax.numpy as jnp


def A_inv_vector(
    h: float,  # temporal grid spacing
    damping: jnp.ndarray,  # damping term vector
) -> jnp.ndarray:
    return 2.0 * h**2 / (2.0 + damping * h)


def B_vector(
    h,  # temporal grid spacing (scalar)
    stiffness,  # stiffness term (vector)
):
    return 2.0 / h**2 - stiffness


def C_vector(
    h,  # temporal grid spacing (scalar)
    damping,  # damping term (vector)
):
    return -1.0 / h**2 + (damping / (2.0 * h))


def make_identity_nl_fn():
    def nl_fn(q):
        return q

    return nl_fn


def make_vk_nl_fn(H: jnp.ndarray) -> Callable[[jnp.ndarray], jnp.ndarray]:
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

    def nl_fn(q: jnp.ndarray) -> jnp.ndarray:
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
    lambda_mu: jnp.ndarray, factors: jnp.ndarray
) -> Callable[[jnp.ndarray], jnp.ndarray]:
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

    def nl_fn(q):
        return lambda_mu * q * (factors @ q**2)

    return nl_fn


def plate_tau_with_density(plate_params) -> float:
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


def string_tau_with_density(string_params) -> float:
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
    u0: jnp.ndarray,  # initial conditions (n_modes,)
    v0: jnp.ndarray,  # initial conditions (n_modes,)
    dt: float,  # time step
    gamma2_mu: jnp.ndarray,  # damping (n_modes,)
    omega_mu_squared: jnp.ndarray,  # frequency (n_modes,)
):
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
    u0: jnp.ndarray,
    v0: jnp.ndarray,
    dt: float,
    gamma2_mu: jnp.ndarray,
    omega_mu_squared: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
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
    A_inv: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    modal_excitation: jnp.ndarray,  # (T, n_modes)
    nl_fn: Callable,
):
    n_modes = A_inv.shape[0]
    q = jnp.zeros((n_modes,))  # Modal displacement vector at n (n_modes, 1)
    q_prev = jnp.zeros((n_modes,))  # Modal displacement vector at n-1

    def advance_state(
        state: tuple[jnp.ndarray, jnp.ndarray],  # initial state
        x: jnp.ndarray,  # input
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:  # carry, output
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
    gamma2_mu,  # (n_modes,)
    omega_mu_squared,  # (n_modes,)
    modal_excitation: jnp.ndarray,  # (T, n_modes)
    dt: float,
    nl_fn: Callable,
):
    A_inv = A_inv_vector(dt, gamma2_mu)
    B = B_vector(dt, omega_mu_squared) * A_inv
    C = C_vector(dt, gamma2_mu) * A_inv

    n_modes = A_inv.shape[0]
    q = jnp.zeros((n_modes,))  # Modal displacement vector at n (n_modes, 1)
    q_prev = jnp.zeros((n_modes,))  # Modal displacement vector at n-1

    def advance_state(
        state: tuple[jnp.ndarray, jnp.ndarray],  # initial state
        x: jnp.ndarray,  # input
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:  # carry, output
        q_prev, q = state
        nl = nl_fn(q)
        q_next = B * q + C * q_prev - A_inv * nl + A_inv * x
        return (q, q_next), q_next

    state, final = jax.lax.scan(
        advance_state,
        (q_prev, q),
        modal_excitation,  # (T, n_modes)
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
def solve_sv_initial_conditions(
    gamma2_mu,  # (n_modes,)
    omega_mu_squared,  # (n_modes,)
    u0: jnp.ndarray,  # initial conditions (n_modes,)
    v0: jnp.ndarray,  # initial conditions (n_modes,)
    dt: float,
    n_steps: int,
    nl_fn: Callable,
):
    A_inv = A_inv_vector(dt, gamma2_mu)
    B = B_vector(dt, omega_mu_squared) * A_inv
    C = C_vector(dt, gamma2_mu) * A_inv

    q0 = u0
    q1, _ = rk4_step(u0, v0, dt, gamma2_mu, omega_mu_squared)

    def advance_state(
        state: tuple[jnp.ndarray, jnp.ndarray],  # initial state
        _: None,  # input
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:  # carry, output
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


@partial(jax.jit, static_argnames=("nl_fn",))
def solve_tf_excitation(
    gamma2_mu,
    omega_mu_squared,
    modal_excitation: jnp.ndarray,  # (T, n_modes)
    dt: float,
    nl_fn: Callable,
):
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
        state: tuple[jnp.ndarray, jnp.ndarray],  # initial carry
        x: jnp.ndarray,  # input
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:  # carry, output
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
def solve_tf_initial_conditions(
    gamma2_mu,
    omega_mu_squared,
    u0: jnp.ndarray,  # initial conditions (n_modes,)
    v0: jnp.ndarray,  # initial conditions (n_modes,)
    dt: float,
    n_steps: int,
    nl_fn: Callable,
):
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
        state: tuple[jnp.ndarray, jnp.ndarray],  # initial carry
        _: None,  # input (unused)
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:  # carry, output
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
    gamma2_mu,
    omega_mu_squared,
    ic,
    n_steps,
    dt,
):
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
    gamma2_mu,
    omega_mu_squared,
    modal_excitation: jnp.ndarray,  # (T, n_modes)
    dt: float,
):
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

    # Number of time steps and modes
    n_steps = modal_excitation.shape[0]
    n_modes = modal_excitation.shape[1]

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


def solve_tf_ic(gamma2_mu, omega_mu_squared, ic, n_steps, dt, nl_fn):
    gamma_mu = gamma2_mu / 2.0
    omega_mu_damped = jnp.sqrt(omega_mu_squared - gamma_mu**2)

    radius = jnp.exp(-gamma_mu * dt)
    imag = radius * jnp.sin(omega_mu_damped * dt)
    real = radius * jnp.cos(omega_mu_damped * dt)

    b1_ic = imag / omega_mu_damped * gamma_mu - real

    b1_exc = dt * imag / omega_mu_damped

    b1_ic = -b1_ic * ic

    a1 = 2.0 * real
    a2 = -(radius**2)

    # initial condition: q0 = ic
    q0 = ic

    # compute q1 using Taylor expansion
    v0 = jnp.zeros_like(ic)
    ddq0 = -gamma2_mu * v0 - omega_mu_squared * ic
    q1 = ic + dt * v0 + 0.5 * dt**2 * ddq0

    # recurrence loop
    def step_fn(q_past, _):
        q_prev, q_curr = q_past

        nl = nl_fn(q_curr)

        q_next = a1 * q_curr + a2 * q_prev - b1_exc * nl
        return (q_curr, q_next), q_next

    (_, _), q_rest = jax.lax.scan(
        step_fn,
        (q0, q1),
        xs=None,
        length=n_steps - 2,
    )
    return jnp.concatenate([q0[:, None], q1[:, None], q_rest.T], axis=1)
