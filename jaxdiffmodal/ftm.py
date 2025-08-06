from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Int
from opt_einsum import contract as einsum
from scipy.integrate import simpson, trapezoid
from scipy.signal import cont2discrete
from scipy.special import jn_zeros, jv
from tabulate import tabulate


@dataclass
class PhysicalParameters(ABC):
    def asdict(self):
        return asdict(self)

    @property
    @abstractmethod
    def d1(self) -> float:
        """Frequency independent damping coefficient"""
        pass

    @property
    @abstractmethod
    def d3(self) -> float:
        """Frequency dependent damping coefficient"""
        pass

    @property
    @abstractmethod
    def Ts0(self) -> float:
        """Initial tension"""
        pass

    @property
    @abstractmethod
    def density(self) -> float:
        """Surface or area density"""
        pass

    @property
    @abstractmethod
    def bending_stiffness(self) -> float:
        """Bending stiffness"""
        pass

    def tabulate(self) -> str:
        param_symbols: dict[str, str] = {
            "A": "$A$",
            "I": "$I$",
            "rho": "$\\rho$",
            "E": "$E$",
            "d1": "$d_1$",
            "d3": "$d_3$",
            "Ts0": "$T_{s0}$",
            "length": "$\\ell$",
            "h": "$h$",
            "l1": "$l_1$",
            "l2": "$l_2$",
            "nu": "$\\nu$",
            "d0": "$d_0$",
            "d2": "$d_2$",
            "Tm": "$T_m$",
            "f_1": "$f_1$",
            "r0": "$r_0$",
            "f0": "$f_0$",
        }

        param_units: dict[str, str] = {
            "A": "$m^2$",
            "I": "$m^4$",
            "rho": "$kg/m^3$",
            "E": "$Pa$",
            "d1": "$kg/(ms)$",
            "d3": "$kg\\cdot m/s$",
            "Ts0": "$N$",
            "length": "$m$",
            "h": "$m$",
            "l1": "$m$",
            "l2": "$m$",
            "nu": "dimensionless",
            "d0": "$kg/(m^2 s)$",
            "d2": "$kg/s$",
            "Tm": "$N/m$",
            "f_1": "$Hz$",
            "r0": "$m$",
            "f0": "$Hz$",
        }

        table_data = []

        for field_name, field_value in self.asdict().items():
            if field_name in param_symbols:
                symbol = param_symbols[field_name]
                unit = param_units.get(field_name, "")
                table_data.append([symbol, field_value, unit])

        return tabulate(
            table_data,
            headers=["Parameter", "Value", "Units"],
            tablefmt="github",
        )


@dataclass
class StringParameters(PhysicalParameters):
    """
    Dataclass to store the parameters of the string.
    """

    # fmt: off
    A: float = 0.5188e-6        # m**2    Cross section area
    I: float = 0.141e-12        # m**4    Moment of inertia  # noqa: E741
    rho: float = 1140           # kg/m**3 Density
    E: float = 5.4e9            # Pa      Young's modulus
    d1: float= 8e-5             # kg/(ms) Frequency independent loss
    d3: float = 1.4e-5          # kg m/s  Frequency dependent loss
    Ts0: float = 60.97          # N       Tension
    length: float = 0.65        # m       Length of the string
    # fmt: on

    @staticmethod
    def piano_string():
        """
        From Table 5.1 of Digital Sound Synthesis using the FTM
        """
        return StringParameters(
            A=1.54e-6,
            I=4.12e-12,
            rho=57.0e3,
            E=19.5e9,
            d1=3e-3,
            d3=2e-5,
            Ts0=2104,
            length=1.08,
        )

    @staticmethod
    def bass_string():
        """
        From Table 5.1 of Digital Sound Synthesis using the FTM
        """
        return StringParameters(
            A=2.4e-6,
            I=0.916e-12,
            rho=6300,
            E=5e9,
            d1=6e-3,
            d3=1e-3,
            Ts0=114,
            length=1.05,
        )

    @staticmethod
    def guitar_string_D():
        """
        From Table 8.2 of Simulation of Distributed Parameter Systems by
        Transfer Function Models
        """
        return StringParameters(
            A=7.96e-7,
            I=0.171e-12,
            rho=1140,
            E=5.4e9,
            d1=8e-5,
            d3=1.4e-5,
            Ts0=13.35,
            length=0.65,
        )

    @staticmethod
    def guitar_string_B_schafer():
        """
        From A String In a Room: Mixed-Dimensional Transfer Function Models
        for Sound Synthesis
        """
        return StringParameters(
            A=0.5e-6,
            I=0.17e-12,
            rho=1140,
            E=5.4e9,
            d1=8e-5,
            d3=1.4e-5,
            Ts0=60.97,
            length=0.65,
        )

    @property
    def density(self):
        """
        Area density of the string
        """
        return self.rho * self.A

    @property
    def bending_stiffness(self):
        return self.E * self.I


@dataclass
class PlateParameters(PhysicalParameters):
    """
    Physical parameters for a rectangular plate simulation.

    Properties
    ----------
    density : float
        Mass per unit area (kg/m²)
    D : float
        Flexural rigidity (N·m)
    """

    # fmt: off
    h: float = 5e-4             # m           Thickness
    l1: float = 0.2             # m           Width
    l2: float = 0.3             # m           Height
    rho: float = 7.8e3          # (kg/m³)     Density
    E: float = 2e12             # Pa          Young's modulus
    nu: float = 0.3             #             Poisson's ratio
    d1: float= 4.2e-2           #             Frequency independent loss
    d3: float = 2.3e-3          #             Frequency dependent loss
    Ts0: float = 100            # N/m         Surface Tension
    # fmt: on

    @property
    def density(self):
        return self.rho * self.h

    @property
    def bending_stiffness(self):
        return self.E * self.h**3 / (12 * (1 - self.nu**2))


@dataclass
class CircularDrumHeadParameters(PhysicalParameters):
    """
    Kettle drum head, from Digital sound synthesis of string instruments with
    the functional transformation method Table 5.2.
    """

    # fmt: off
    h: float = 1.9e-4           # m           Thickness
    r0: float = 0.328           # m           Radius
    I: float = 0.57e-12         # m**4        Moment of intertia # noqa: E741
    rho: float = 1.38e3         # kg/m**3     Density
    E: float = 3.5e9            # Pa          Young's modulus
    nu: float = 0.35            #             Poisson's ratio
    d1: float= 0.14             # kg/(m**2 s) Frequency independent loss
    d3: float = 0.32            # kg/s        Frequency dependent loss
    Ts0: float = 3990           # N/m         Surface Tension
    f0: float = 143.95          # Hz          Fundamental frequency
    # fmt: on

    @property
    def density(self):
        return self.rho * self.h

    @property
    def bending_stiffness(self):
        return self.E * self.h**3 / (12 * (1 - self.nu**2))

    @staticmethod
    def avanzini():
        """
        From Section VI of "A Modular Physically Based Approach to
        the Sound Synthesis of Membrane Percussion Instruments"
        """
        return CircularDrumHeadParameters(
            h=2e-4,
            r0=0.20,
            rho=1350.0,
            E=3.5e9,
            nu=0.2,
            d1=1.25,
            d3=5e-4,
            Ts0=1500,
        )


def string_eigenvalues(n_modes: int, length: float):
    r"""
    Compute the eigenvalues of a string with fixed ends.

    The eigenvalues are given by:

    $$\lambda_\mu = \left(\frac{\mu \pi}{L}\right)^2$$

    where $\mu$ is the mode number and $L$ is the length of the string.

    Parameters
    ----------
    n_modes : int
        Number of modes to compute
    length : float
        Length of the string

    Returns
    -------
    jnp.ndarray
        Array of eigenvalues with shape (n_modes,)
    """
    mu = jnp.arange(1, n_modes + 1)
    return (mu * jnp.pi / length) ** 2


def string_eigenfunctions(
    wavenumbers: Float[Array, " modes"],
    grid: Float[Array, " grid"],
) -> Float[Array, "modes grid"]:
    r"""
    Compute the modes of the string.
    The modes of the string are given by:

    $$
    K = \sin(\pi x k)
    $$

    where $k$ is the wavenumber and $x$ is the grid positions.

    Parameters
    ----------
    wavenumbers : np.ndarray
        The wavenumbers of the string.
    grid : np.ndarray
        The grid positions of the string where to compute the modes.

    Returns
    -------
    np.ndarray
        The modes of the string at the given grid positions.
    """
    return jnp.sin(jnp.outer(wavenumbers, grid))


def plate_wavenumbers(
    n_max_modes_x: int,
    n_max_modes_y: int,
    l1: float,
    l2: float,
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    r"""
    Compute the wavenumbers of a rectangular plate with clamped edges.

    The wavenumbers are given by:

    $$k_x = \frac{\mu \pi}{L_1}, \quad k_y = \frac{\nu \pi}{L_2}$$

    where $\mu$ and $\nu$ are the mode numbers, and $L_1$ and $L_2$ are the
    plate dimensions.

    Parameters
    ----------
    n_max_modes_x : int
        Number of modes in x direction
    n_max_modes_y : int
        Number of modes in y direction
    l1 : float
        Width of the plate
    l2 : float
        Height of the plate

    Returns
    -------
    tuple[Array, Array]
        Wavenumbers in x and y directions with shapes
        (n_max_modes_x,) and (n_max_modes_y,). Returns JAX arrays that are
        compatible with NumPy operations.
    """

    mu_x = jnp.arange(1, n_max_modes_x + 1)
    mu_y = jnp.arange(1, n_max_modes_y + 1)
    wavenumbers_x = mu_x * jnp.pi / l1
    wavenumbers_y = mu_y * jnp.pi / l2
    return wavenumbers_x, wavenumbers_y


def plate_eigenvalues(
    wavenumbers_x: Float[Array, " mx"],
    wavenumbers_y: Float[Array, " my"],
) -> Float[Array, "mx my"]:
    r"""

    Compute the eigenvalues of the plate.
    The eigenvalues of the plate are given by:

    \begin{equation}
    \lambda_{\mu, \nu}
    = \left(\frac{\mu \pi}{L_1}\right)^2 + \left(\frac{\nu \pi}{L_2}\right)^2
    \end{equation}

    where $\mu$ and $\nu$ are the mode numbers
    and $L_1$ and $L_2$ are the width and height of the plate.

    Parameters
    ----------
    wavenumbers_x: np.ndarray
        The wavenumbers in the x direction.
    wavenumbers_y: np.ndarray
        The wavenumbers in the y direction

    Returns
    -------
    np.ndarray
        The eigenvalues
    """
    wn_x, wn_y = jnp.meshgrid(wavenumbers_x, wavenumbers_y)

    return wn_x**2 + wn_y**2


def plate_eigenfunctions(
    wavenumbers_x: Float[Array, " modes_x"],
    wavenumbers_y: Float[Array, " modes_y"],
    x: Float[Array, " grid_x"],
    y: Float[Array, " grid_y"],
) -> Float[Array, "modes_x modes_y grid_x grid_y"]:
    r"""
    Compute the modes of the plate.
    The modes of the plate are given by:

    $$
    K = \sin(\pi x k) \sin(\pi y k)
    $$

    where $k$ is the wavenumber and $x$ and $y$ are the grid positions.
    """

    # Compute the sine values using broadcasting
    sin_wx_grid_x = jnp.sin(wavenumbers_x[:, None] * x[None, :])
    sin_wy_grid_y = jnp.sin(wavenumbers_y[:, None] * y[None, :])

    # Use einsum to compute the outer product and obtain the modes
    return einsum("m x, n y -> m n x y", sin_wx_grid_x, sin_wy_grid_y)


def dblintegral(
    integrand: Float[Array, "x y"],
    x: Float[Array, " x"],
    y: Float[Array, " y"],
    method: str = "simpson",
) -> float | Float[Array, "..."]:
    """
    Compute the double integral of a function K over the domain x and y.
    """

    if method == "simpson":
        integral_y = simpson(integrand, x=y, axis=1)
        return jnp.array(simpson(integral_y, x=x))
    elif method == "trapezoid":
        integral_y = trapezoid(integrand, x=y, axis=1)
        return trapezoid(integral_y, x=x)
    else:
        raise ValueError("Method not supported")


def forward_STL(
    K: Float[Array, "modes grid"],  # (n_modes, n_gridpoints)
    u: Float[Array, "grid samples"],  # (n_gridpoints, n_samples) or (n_gridpoints,)
    dx: float,  # grid spacing
) -> Float[Array, "modes samples"]:
    """
    Compute the forward STL transform. The integration is done using the
    trapezoidal rule.

    Parameters
    ----------
    K: Array
        The sampled eigenfunctions of the string. Shape (n_modes, n_gridpoints)
    u: Array
        The input signal. Shape (n_gridpoints, n_samples) or (n_gridpoints,)
    dx: float
        The grid spacing of the sampled string.

    Returns
    -------
    Array
        The transformed signal. Shape (n_modes, n_samples) or (n_modes,)
    """
    if u.ndim == 1:
        u = u[:, None]
    transformed_signal = dx * einsum("m n, n s -> m s", K, u)
    return transformed_signal if u.shape[1] > 1 else transformed_signal[:, 0]


def inverse_STL(
    K: Float[Array, "modes grid"],  # (n_modes, n_gridpoints)
    u_bar: Float[Array, "modes samples"],  # (n_modes, n_samples) or (n_modes,)
    length: float,  # length of the string
) -> Float[Array, "grid samples"]:
    """
    Compute the inverse STL transform using the formula of Rabenstein et al. (2000).

    Parameters
    -----------

    K: np.ndarray
        The sampled eigenfunctions of the string. Shape (n_modes, n_gridpoints)
    u_bar: np.ndarray
        The transformed signal. Shape (n_modes, n_samples) or (n_modes,)
    length: float
        The length of the string.

    Returns
    -------
    np.ndarray
        The reconstructed signal. Shape (n_gridpoints, n_samples) or (n_gridpoints,)
    """
    if u_bar.ndim == 1:
        u_bar = u_bar[:, None]  # Convert to (n_modes, 1) if input is (n_modes,)

    N = length / 2.0
    reconstructed_signal = einsum("m n, m s -> n s", K, u_bar) / N
    return reconstructed_signal if u_bar.shape[1] > 1 else reconstructed_signal[:, 0]


def forward_STL_2d(
    K: Float[Array, "modes_x modes_y grid_x grid_y"],  # 4D tensor
    u: Float[
        Array, "grid_x grid_y samples"
    ],  # (n_gridpoints_x, n_gridpoints_y, n_samples) or
    # (n_gridpoints_x, n_gridpoints_y)
    x: Float[Array, " grid_x"],  # grid points in x direction
    y: Float[Array, " grid_y"],  # grid points in y direction
    use_simpson: bool = False,
) -> Float[Array, "modes_x modes_y samples"]:
    """
    Compute the forward STL transform. The integration is done using the
    trapezoidal rule.

    Parameters
    ----------
    K : Array
        The sampled eigenfunctions of the string. Shape (n_modes, n_gridpoints)
    u : Array
        The input signal. Shape (n_gridpoints, n_samples) or (n_gridpoints,)
    x : Array
        The grid points in x direction.
    y : Array
        The grid points in y direction.

    Returns
    -------
    Array
        The transformed signal. Shape (n_modes, n_samples) or (n_modes,)
    """
    if u.ndim == 2:
        u = u[..., None]

    if use_simpson:
        n_modes_x, n_modes_y, _, _ = K.shape
        _, _, n_samples = u.shape

        transformed_signal = jnp.zeros((n_modes_x, n_modes_y, n_samples))

        for mode_x in range(n_modes_x):
            for mode_y in range(n_modes_y):
                for sample in range(n_samples):
                    # Perform 2D Simpson's integration
                    uu = K[mode_x, mode_y] * u[:, :, sample]

                    # integral_x = simpson([simpson(uu_y, dx=dy) for uu_y in uu], dx=dx)
                    transformed_signal[mode_x, mode_y, sample] = dblintegral(
                        uu,
                        x,
                        y,
                        method="simpson",
                    )
    # else use trapezoidal rule
    else:
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        transformed_signal = dx * dy * einsum("m n x y, x y s -> m n s", K, u)

    return (
        transformed_signal.squeeze()
        if transformed_signal.shape[-1] == 1
        else transformed_signal
    )


def inverse_STL_2d(
    K: Float[Array, "modes_x modes_y grid_x grid_y"],
    u_bar: Float[Array, "modes_x modes_y samples"],
    l1: float,
    l2: float,
) -> Float[Array, "grid_x grid_y samples"]:
    """
    Compute the inverse STL transform.

    Parameters
    ----------
    K : np.ndarray
        The sampled eigenfunctions of the string. Shape (n_modes, n_gridpoints)
    u_bar : np.ndarray
        The transformed signal. Shape (n_modes, n_samples) or (n_modes,)
    l1 : float
        The length in x direction.
    l2 : float
        The length in y direction.

    Returns
    -------
    np.ndarray
        The reconstructed signal. Shape (n_gridpoints, n_samples) or (n_gridpoints,)
    """
    if u_bar.ndim == 2:
        u_bar = u_bar[
            ..., None
        ]  # Convert to (n_modes_x, n_modes_y, 1) if input is (n_modes_x, n_modes_y)

    N = 0.25 * (l1 * l2)
    reconstructed_signal = einsum("m n x y, m n s -> x y s", K, u_bar) / N
    return (
        reconstructed_signal.squeeze() if u_bar.shape[-1] == 1 else reconstructed_signal
    )


def evaluate_string_eigenfunctions(
    indices: Int[Array, " modes"],
    position: Float[Array, " position"],
    params: StringParameters,
) -> Float[Array, " modes"]:
    r"""
    Evaluate string eigenfunctions at a given position.

    Computes the eigenfunction values for selected string modes at
    a specific spatial position along the string.

    Parameters
    ----------
    indices : numpy.ndarray
        Mode indices to evaluate, shape (n_modes,)
    position : numpy.ndarray
        Position along string to evaluate at, shape (1,)
    params : StringParameters
        String physical parameters including length

    Returns
    -------
    numpy.ndarray
        Eigenfunction values at the position, shape (n_modes,)

    Notes
    -----
    The string eigenfunctions are:

    $$\phi_\mu(x) = \sin\left(\frac{\mu \pi x}{L}\right)$$

    where $\mu$ is the mode index and $L$ is the string length.
    """
    return jnp.sin(indices * jnp.pi * position / params.length)


def evaluate_rectangular_eigenfunctions(
    mn_indices: Int[Array, "modes 2"],
    position: Float[Array, " 2"],
    params: PlateParameters,
) -> Float[Array, " modes"]:
    r"""
    Evaluate rectangular plate eigenfunctions at a given position.

    Computes the eigenfunction values for selected rectangular plate modes
    at a specific spatial position on the plate surface.

    Parameters
    ----------
    mn_indices : numpy.ndarray
        Mode indices (m,n) to evaluate, shape (n_modes, 2)
    position : numpy.ndarray
        Position (x,y) on plate to evaluate at, shape (2,)
    params : PlateParameters
        Plate physical parameters including dimensions l1, l2

    Returns
    -------
    numpy.ndarray
        Eigenfunction values at the position, shape (n_modes,)

    Notes
    -----
    The rectangular plate eigenfunctions are:

    $$\phi_{mn}(x,y) = \sin\left(\frac{m \pi x}{l_1}\right)
    \sin\left(\frac{n \pi y}{l_2}\right)$$

    where $(m,n)$ are the mode indices and $(l_1, l_2)$ are the plate dimensions.
    """
    return jnp.sin(mn_indices[:, 0] * jnp.pi * position[0] / params.l1) * jnp.sin(
        mn_indices[:, 1] * jnp.pi * position[1] / params.l2
    )


def forward_STL_drumhead(
    K: Float[Array, "n_modes m_modes r_grid theta_grid"],
    u: Float[Array, "r_grid theta_grid samples"],
    r: Float[Array, " r_grid"],
    theta: Float[Array, " theta_grid"],
    use_simpson: bool = False,
) -> Float[ArrayLike, "n_modes m_modes samples"]:
    """
    Compute the forward STL transform. The integration is done using the
    trapezoidal rule or Simpson's rule.

    Parameters
    -----------

    K : np.ndarray
        The sampled eigenfunctions of the string. Shape
        (n_modes_r, n_modes_theta, n_gridpoints_r, n_gridpoints_theta)
    u : np.ndarray
        The input signal. Shape (n_gridpoints_r, n_gridpoints_theta,
        n_samples) or (n_gridpoints_r, n_gridpoints_theta)
    r : np.ndarray
        The radial grid.
    theta : np.ndarray
        The angular grid.
    use_simpson : bool, optional
        Whether to use Simpson's rule for integration (default is False).

    Returns
    -------
    np.ndarray
        The transformed signal. Shape (n_modes_r, n_modes_theta, n_samples)
        or (n_modes_r, n_modes_theta)
    """
    if u.ndim == 2:
        u = u[..., None]

    r_grid, _ = np.meshgrid(r, theta, indexing="ij")

    if use_simpson:
        max_n, max_m, _, _ = K.shape
        _, _, n_samples = u.shape

        transformed_signal = np.zeros((max_n, max_m, n_samples))

        for n in range(max_n):
            for m in range(max_m):
                for sample in range(n_samples):
                    integrand = (
                        u[..., sample] * K[n, m] * r_grid
                    )  # notice the r_grid (Jacobian determinant)
                    transformed_signal[n, m] = dblintegral(
                        integrand, x=r, y=theta, method="trapezoid"
                    )

    else:
        # integrand has shape
        # (n_modes_r, n_modes_theta, n_gridpoints_r, n_gridpoints_theta)
        integrand = u[..., None].transpose(2, 3, 0, 1) * K * r_grid

        integral_y = jnp.trapezoid(integrand, x=theta, axis=-1)
        transformed_signal = jnp.trapezoid(integral_y, x=r, axis=-1)

    return (
        transformed_signal.squeeze()
        if transformed_signal.shape[-1] == 1
        else transformed_signal
    )


def inverse_STL_drumhead(
    K_inv: Float[
        Array, "n_modes m_modes r_grid theta_grid"
    ],  # (n_modes_x, n_modes_y, n_gridpoints_x, n_gridpoints_y)
    u_bar: Float[
        Array, "n_modes m_modes samples"
    ],  # (n_modes_x, n_modes_y, n_samples) or (n_modes_x, n_modes_y)
) -> Float[Array, "r_grid theta_grid samples"]:
    """
    Compute the inverse STL transform using the formula of Rabenstein et al. (2000).

    Parameters
    ----------
    K_inv : np.ndarray
        The inverse eigenfunctions. Shape (n_modes_x, n_modes_y,
        n_gridpoints_x, n_gridpoints_y)
    u_bar : np.ndarray
        The transformed signal. Shape (n_modes_x, n_modes_y, n_samples) or
        (n_modes_x, n_modes_y)

    Returns
    -------
    np.ndarray
        The reconstructed signal. Shape (n_gridpoints, n_samples) or (n_gridpoints,)
    """
    if u_bar.ndim == 2:
        u_bar = u_bar[
            ..., None
        ]  # Convert to (n_modes_x, n_modes_y, 1) if input is (n_modes_x, n_modes_y)

    reconstructed_signal = einsum("n m x y, n m s -> x y s", K_inv, u_bar)
    return (
        reconstructed_signal.squeeze() if u_bar.shape[-1] == 1 else reconstructed_signal
    )


def damping_term(
    params: PhysicalParameters,
    lambda_mu: jnp.ndarray,
):
    return (params.d1 + params.d3 * lambda_mu) / params.density


def damping_term_simple(
    lambda_mu: jnp.ndarray,
    factor: float = 1e-3,
):
    return factor * lambda_mu


def stiffness_term(
    params: PhysicalParameters,
    lambda_mu: jnp.ndarray,
):
    omega_mu = params.bending_stiffness * lambda_mu**2 + params.Ts0 * lambda_mu
    omega_mu = omega_mu / params.density
    return omega_mu


def eigenvalues_from_pde(
    pars: PhysicalParameters,
    lambda_mu: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the positive imaginary side of the eigenvalues of the
    continuous-time system from the PDE parameters.

    Parameters
    ----------
    pars : PhysicalParameters
        The physical parameters of the system.
    lambda_mu : jnp.ndarray
        The eigenvalues of the decompostion of the Laplacian operator.

    Returns
    -------
    np.ndarray
        The eigenvalues of the continuous-time system.
    """

    gamma_mu = damping_term(pars, lambda_mu) / 2
    omega_mu_squared = stiffness_term(pars, lambda_mu)
    omega_mu_damped = np.sqrt(omega_mu_squared - gamma_mu**2)

    return -gamma_mu + 1j * omega_mu_damped


def sample_parallel_tf(
    num: Float[Array, " modes"],
    den: Float[Array, " modes"],
    dt: float,
    method: str = "impulse",
) -> tuple[Float[Array, "modes coeff"], Float[Array, "modes coeff"]]:
    """
    Sample multiple transfer functions.

    Parameters
    ----------
    num : np.ndarray
        The numerator of the transfer function.
    den : np.ndarray
        The denominator of the transfer function.

    Returns
    -------
    num_d : np.ndarray
        The numerator of the discrete-time transfer function.
    den_d : np.ndarray
        The denominator of the discrete-time transfer function.
    """

    def sample(n, d):
        b, a, _ = cont2discrete((n, d), dt, method=method)  # type: ignore
        return b.flatten(), a.flatten()

    b, a = zip(*[sample(n, d) for n, d in zip(num, den)])

    return jnp.array(b), jnp.array(a)


def tf_initial_conditions_continuous_2(
    D: float,
    density: float,
    d1: float,
    d3: float,
    Ts0: float,
    lambda_mu: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Conpute the continuous-time initial condition transfer function.
    This is an alternative to the function `tf_initial_conditions_continuous`
    that eigenvalues of the PDE as input.

    Parameters
    ----------

    D : float
        The bending stiffness of the string or plate.
    density : float
        The area or surface density of the string or plate.
    d1 : float
        The linear damping coefficient, or frequency-independent damping.
    d3 : float
        The cubic damping coefficient, or frequency-dependent damping.
    Ts0 : float
        The initial tension of the string or plate.
    lambda_mu : jnp.array
        The eigenvalues from the decomposition of the Laplacian operator.

    Returns
    -------
    tuple[jnp.array, jnp.array]
        The numerator and denominator of the transfer function.
    """

    sigma_mu = (d1 + d3 * lambda_mu) / density

    omega_mu_squared = D * lambda_mu**2 + Ts0 * lambda_mu
    omega_mu_squared = omega_mu_squared / density

    a1 = sigma_mu
    a2 = omega_mu_squared
    ones = jnp.ones(lambda_mu.shape[0])

    # assemble the numerator and denominator for the transfer function
    # starting with highest order term
    b = jnp.stack([ones, a1], axis=-1)
    a = jnp.stack([ones, a1, a2], axis=-1)

    return (b, a)


def tf_excitation_continuous(
    eigenvalues: Float[Array, " modes"],
    density: float,  # surface or area density
) -> tuple[Float[Array, " modes"], Float[Array, "modes coeff"]]:
    """
    Compute the continuous excitation transfer function.

    Parameters
    ----------
    eigenvalues : np.ndarray
        The eigenvalues of the system.
    density : float
        The surface density of a membrane (rho * h) or area density of a
        string (rho * A)

    Returns
    -------
    np.ndarray
        The numerator of the discrete-time transfer function.
    np.ndarray
        The denominator of the discrete-time transfer function
    """
    sigma_mu = -1.0 * eigenvalues.real
    omega_mu = eigenvalues.imag

    a1 = sigma_mu * 2
    a2 = sigma_mu**2 + omega_mu**2

    ones = jnp.ones_like(sigma_mu)
    b = ones / density
    a = jnp.stack([ones, a1, a2], axis=-1)
    return b, a


def tf_excitation_discrete(
    eigenvalues: Float[Array, " modes"],
    density: float,  # surface or area density
    dt: float,  # time step
) -> tuple[Float[Array, "modes coeff"], Float[Array, "modes coeff"]]:
    """
    Compute the discrete-time excitation transfer function of a system.

    Parameters
    ----------
    eigenvalues : np.ndarray
        The eigenvalues of the system.
    density: float
        The mass density of the system.
    dt : float
        The time step size.

    Returns
    -------
    np.ndarray
        The numerator of the discrete-time transfer function.
    np.ndarray
        The denominator of the discrete-time transfer function
    """
    b, a = tf_excitation_continuous(eigenvalues, density)

    # Discretize the system
    tf_d = sample_parallel_tf(b, a, dt)

    return tf_d


def tf_initial_conditions_continuous(
    eigenvalues: Float[Array, " modes"],
) -> tuple[Float[Array, "modes coeff"], Float[Array, "modes coeff"]]:
    """
    Compute the continuos "initial-conditions" transfer function from the
    eigenvalues of the system.

    Parameters
    ----------
    eigenvalues : Array
        The eigenvalues of the system.

    Returns
    -------
    np.ndarray
        The numerator of the discrete-time transfer function.
    np.ndarray
        The denominator of the discrete-time transfer function
    """

    sigma_mu = -1.0 * eigenvalues.real
    omega_mu = eigenvalues.imag

    a1 = sigma_mu * 2
    a2 = sigma_mu**2 + omega_mu**2

    ones = jnp.ones_like(sigma_mu)
    b1 = a1

    b = jnp.stack([ones, b1], axis=-1)
    a = jnp.stack([ones, a1, a2], axis=-1)

    return (b, a)


def tf_initial_conditions_discrete(
    eigenvalues: Float[Array, " modes"],
    dt: float,  # time step
) -> tuple[Float[Array, "modes coeff"], Float[Array, "modes coeff"]]:
    """
    Compute the discrete-time initial conditions transfer function of a system.

    Parameters
    ----------
    eigenvalues : np.ndarray
        The eigenvalues of the system.
    dt : float
        The time step size.

    Returns
    -------
    np.ndarray
        The numerator of the discrete-time transfer function.
    np.ndarray
        The denominator of the discrete-time transfer function
    """
    b, a = tf_initial_conditions_continuous(eigenvalues)

    # Discretize the system
    tf_d = sample_parallel_tf(b, a, dt)

    return tf_d


def select_modes_from_eigenvalues(
    lambda_mu_2d: Float[Array, "modes_x modes_y"],
    n_modes: int,
) -> tuple[
    Int[ArrayLike, " modes"],
    Int[ArrayLike, " modes"],
    Float[ArrayLike, " modes"],
    Int[ArrayLike, " modes"],
]:
    """
    Select the lowest n_modes from a 2D eigenvalue array and return sorted
    indices and eigenvalues.

    Parameters
    ----------
    lambda_mu_2d : np.ndarray
        2D array of eigenvalues with shape (n_modes_x, n_modes_y)
    n_modes : int
        Number of modes to select (lowest eigenvalues)

    Returns
    -------
    tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]
        - kx_indices : ArrayLike, shape (n_modes,)
            Selected mode indices in x-direction (1-based)
        - ky_indices : ArrayLike, shape (n_modes,)
            Selected mode indices in y-direction (1-based)
        - lambda_mu : ArrayLike, shape (n_modes,)
            Sorted eigenvalues corresponding to selected modes
        - sorted_indices : ArrayLike, shape (n_modes,)
            Indices of the selected modes in the flattened array

    Examples
    --------
    >>> wnx, wny = plate_wavenumbers(30, 30, 0.2, 0.3)
    >>> lambda_mu_2d = plate_eigenvalues(wnx, wny)
    >>> kx_indices, ky_indices, lambda_mu = select_modes_from_eigenvalues(
    ...     lambda_mu_2d, 64)
    >>> selected_indices = np.stack([kx_indices, ky_indices], axis=-1)
    """
    # Flatten eigenvalues and get sorted indices of n_modes lowest values
    lambda_flat = lambda_mu_2d.ravel()
    sorted_indices = np.argsort(lambda_flat)[:n_modes]

    # Convert flat indices to 2D coordinates and convert to 1-based indexing
    ky_indices, kx_indices = np.unravel_index(sorted_indices, lambda_mu_2d.shape)

    return (
        kx_indices + 1,
        ky_indices + 1,
        lambda_flat[sorted_indices],
        sorted_indices,
    )


def circ_laplacian_wavenumbers(
    n_max_modes: int,
    m_max_modes: int,
    radius: float = 1.0,
) -> Float[Array, "n_modes m_modes"]:
    r"""
    Compute wavenumbers for circular Laplacian eigenvalue problem.

    Calculates wavenumbers for the 2D Laplacian eigenvalue problem
    ∇²φ + k²φ = 0 on a circular domain with Dirichlet boundary conditions.
    This corresponds to the membrane/drumhead approximation for circular plates.

    **Physical Context**: Valid when bending effects are negligible:
    - **Low bending stiffness** (thin plates, low Young's modulus)
    - **High tension** (pre-stressed membrane, large in-plane forces)
    - Wavelength >> plate thickness

    For full plate theory, use `circ_plate_transverse_eigenvalues()`.

    Parameters
    ----------
    n_max_modes : int
        Number of angular modes (azimuthal order n = 0, 1, 2, ...)
    m_max_modes : int
        Number of radial modes for each angular mode
    radius : float, optional
        Radius of the circular domain, default: 1.0

    Returns
    -------
    numpy.ndarray
        Wavenumbers array of shape (n_max_modes, m_max_modes)
        where wavenumbers[n, m] = j_{nm}/R

    Notes
    -----
    Eigenvalue problem: J_n(kR) = 0, giving k_{nm} = j_{nm}/R
    where j_{nm} is the m-th positive zero of J_n(x).
    """

    wavenumbers = np.zeros((n_max_modes, m_max_modes))
    for n in range(n_max_modes):
        bessel_roots = jn_zeros(n, m_max_modes)
        wavenumbers[n, :] = bessel_roots / radius
    return jnp.array(wavenumbers)


def circ_laplacian_eigenvalues(
    wavenumbers: Float[Array, "n_modes m_modes"],
) -> Float[Array, "n_modes m_modes"]:
    r"""
    Compute eigenvalues for circular Laplacian (membrane approximation).

    Converts wavenumbers to eigenvalues λ = k² for the 2D Laplacian eigenvalue
    problem. This represents the membrane/drumhead approximation.

    **Physical Validity**: Most accurate for tensioned membranes:
    - **Low bending stiffness** (thin plates, low Young's modulus)
    - **High tension** (pre-stressed membrane, large in-plane forces)
    - Wavelength >> plate thickness

    For full plate dynamics, use `circ_plate_transverse_eigenvalues()`.

    Parameters
    ----------
    wavenumbers : numpy.ndarray
        Wavenumbers from `circ_laplacian_wavenumbers()`

    Returns
    -------
    numpy.ndarray
        Eigenvalues λ = k²

    Notes
    -----
    Solves ∇²φ + k²φ = 0 with φ = 0 at r = R.
    Neglects bending stiffness and Poisson coupling.
    """
    return wavenumbers**2


def circ_laplacian_eigenfunctions(
    wavenumbers: Float[Array, "n_modes m_modes"],
    r: Float[Array, " r_grid"],
    theta: Float[Array, " theta_grid"],
) -> Float[Array, "n_modes m_modes 2 r_grid theta_grid"]:
    r"""
    Compute eigenfunctions for circular Laplacian (membrane modes).

    Evaluates φ_{nm}(r,θ) for the 2D Laplacian eigenvalue problem.
    These are the mode shapes of a tensioned circular membrane.

    Parameters
    ----------
    wavenumbers : numpy.ndarray
        Wavenumbers from `circ_laplacian_wavenumbers()`
    r : numpy.ndarray
        Radial grid points (0 ≤ r ≤ R)
    theta : numpy.ndarray
        Angular grid points (0 ≤ θ ≤ 2π)

    Returns
    -------
    numpy.ndarray
        Eigenfunction values φ_{nm}(r,θ) = J_n(k_{nm}r) × [cos(nθ) or sin(nθ)]

    Notes
    -----
    Eigenfunctions:
    - n = 0: φ₀ₘ(r,θ) = J₀(k₀ₘr) (axisymmetric)
    - n > 0: φₙₘ^c(r,θ) = Jₙ(kₙₘr)cos(nθ), φₙₘ^s(r,θ) = Jₙ(kₙₘr)sin(nθ)
    """

    n_max_modes, m_max_modes = wavenumbers.shape
    modes = jnp.zeros((n_max_modes, m_max_modes, 2, len(r), len(theta)))

    R, THETA = jnp.meshgrid(r, theta, indexing="ij")

    for n in range(n_max_modes):
        for m in range(m_max_modes):
            k_nm = wavenumbers[n, m]
            if k_nm == 0:
                continue

            radial = jv(n, k_nm * R)

            if n == 0:
                modes[n, m, 0] = radial
                modes[n, m, 1] = 0
            else:
                modes[n, m, 0] = radial * np.cos(n * THETA)
                modes[n, m, 1] = radial * np.sin(n * THETA)

    return modes
