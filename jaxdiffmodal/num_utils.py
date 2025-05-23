# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/api/num_utils.ipynb.

# %% auto 0
__all__ = [
    "pad_upper",
    "pad_lower",
    "second_derivative",
    "second_derivative_mixed",
    "vkoperator",
    "double_trapezoid",
    "double_trapezoid_flat",
    "compute_coupling_matrix_numerical",
    "polarisation",
    "eigenMAC",
    "biharmonic_eigendecomposition",
    "multiresolution_eigendecomposition",
]

# %% ../nbs/api/num_utils.ipynb 3
import numpy as np
import scipy.sparse as sp
from magpie import bhmat
from scipy.integrate import trapezoid
from scipy.interpolate import RectBivariateSpline
from scipy.sparse.linalg import eigs

from jaxdiffmodal.ftm import PlateParameters


# %% ../nbs/api/num_utils.ipynb 4
def pad_upper(vec, pad):
    return np.pad(vec[::-1], (0, pad))


def pad_lower(vec, pad):
    return np.pad(vec, (pad, 0))


# %% ../nbs/api/num_utils.ipynb 5
def second_derivative(Nx, Ny, h, direction="x"):
    """

    Construct a higher-order second derivative operator matching the MATLAB
    implementation of vkplate.

    Parameters
    ----------
        Nx : int
            Number of intervals in the x-direction.
        Ny : int
            Number of intervals in the y-direction.
        h : float
            Grid spacing.
        direction : str
            Direction of the second derivative operator. Can be 'x' or 'y'.

    Returns
    -------

        scipy.sparse.spmatrix
            The $(Nx+1)(Ny+1) \\times (Nx+1)(Ny+1)$ second derivative operator.
    """
    # Total number of grid points
    N = (Nx + 1) * (Ny + 1)

    # Main diagonal d0: initially -2 everywhere, then modify boundaries.
    d0 = -2 * np.ones(N)
    if direction == "x":
        d0[: Ny + 1] = 2  # First Ny+1 entries (top boundary in MATLAB indexing)
        d0[-(Ny + 1) :] = 2  # Last Ny+1 entries (bottom boundary)
    elif direction == "y":
        d0[0 :: (Ny + 1)] = 2
        d0[Ny :: (Ny + 1)] = 2
    elif direction == "xy":
        d0 = np.zeros(N)
        d0[0] = 2.25
        d0[-1] = 2.25
        d0[Ny] = -2.25
        d0[N - Ny - 1] = -2.25

    # First off-diagonal
    if direction == "x":
        d1 = np.ones(Nx * (Ny + 1))
        d1[: Ny + 1] = -5  # Set the first Ny+1 entries to -5
    elif direction == "y":
        d1 = np.ones(Ny + 1)
        d1[0] = -5
        d1[-1] = 0
        d1 = np.tile(d1, Nx + 1)  # length becomes (Nx+1)*(Ny+1)
        d1 = d1[:-1]  # final length: N - 1
    elif direction == "xy":
        d1 = np.zeros(N - 1)
        d1[1:Ny] = -0.75
        d1[N - Ny :] = 0.75
        d1[0] = -3.0
        d1[N - Ny - 1] = 3.0

    # Second off-diagonal
    if direction == "x":
        d2 = np.zeros((Nx - 1) * (Ny + 1))
        d2[: Ny + 1] = 4  # First Ny+1 entries set to 4
    elif direction == "y":
        # d2: vector of zeros of length N - 2; set every (Ny+1)th entry to 4.
        d2 = np.zeros(N - 2)
        d2[0 :: (Ny + 1)] = 4
    elif direction == "xy":
        d2 = np.zeros(N - 2)
        d2[0] = 0.75  # d2(1)
        d2[N - Ny - 1] = -0.75  # d2(end-Ny+2)

    # Third off-diagonal
    if direction == "x":
        d3 = np.zeros((Nx - 2) * (Ny + 1))
        d3[: Ny + 1] = -1  # First Ny+1 entries set to -1
    elif direction == "y":
        d3 = np.zeros(N - 3)
        d3[0 :: (Ny + 1)] = -1

    if direction == "x":
        col1 = pad_upper(d3, 3 * (Ny + 1))
        col2 = pad_upper(d2, 2 * (Ny + 1))
        col3 = pad_upper(d1, Ny + 1)
        col4 = d0  # Already full length (N,)
        col5 = pad_lower(d1, Ny + 1)
        col6 = pad_lower(d2, 2 * (Ny + 1))
        col7 = pad_lower(d3, 3 * (Ny + 1))
    elif direction == "y":
        col1 = pad_upper(d3, 3)
        col2 = pad_upper(d2, 2)
        col3 = pad_upper(d1, 1)
        col4 = d0
        col5 = pad_lower(d1, 1)
        col6 = pad_lower(d2, 2)
        col7 = pad_lower(d3, 3)

    d = np.vstack([col1, col2, col3, col4, col5, col6, col7])

    if direction == "x":
        dN = (Ny + 1) * np.arange(-3, 4)
    elif direction == "y":
        dN = np.arange(-3, 4)

    Dxx = (1 / h**2) * sp.spdiags(d, dN, N, N)
    return Dxx


def second_derivative_mixed(Nx, Ny, h):
    N = (Nx + 1) * (Ny + 1)

    # d0: central diagonal (length N)
    d0 = np.zeros(N)
    d0[0], d0[-1] = 2.25, 2.25
    d0[Ny] = -2.25
    d0[N - Ny - 1] = -2.25

    # d1: length N-1
    d1 = np.zeros(N - 1)
    d1[0] = -3.0
    d1[N - Ny] = 3.0
    d1[1:Ny] = -0.75
    d1[-(Ny - 1) :] = 0.75

    # d2: length N-2
    d2 = np.zeros(N - 2)
    d2[0] = 0.75
    d2[N - Ny - 1] = -0.75

    # dNym1: length N-(Ny-1)
    dNym1 = np.zeros(N - (Ny - 1))
    dNym1[Ny] = 1.0
    dNym1[2 * Ny + 1 :: (Ny + 1)] = 0.25

    # dNy: length N-Ny
    mot = np.concatenate(([0.0], -0.25 * np.ones(Ny - 1), [-1.0]))
    dNy = np.concatenate((np.tile(mot, Nx), [0.0]))
    dNy[1:Ny] = -1.0
    dNy[Ny] = -4.0

    # dNyp1: length N-(Ny+1)
    dNyp1 = np.zeros(N - (Ny + 1))
    dNyp1[0] = -3.0
    dNyp1[Ny] = 3.0
    dNyp1[2 * Ny + 1 :: (Ny + 1)] = 0.75
    dNyp1[Ny + 1 :: (Ny + 1)] = -0.75

    # dNyp2: length N-(Ny+2)
    mot = np.concatenate(([1.0], 0.25 * np.ones(Ny - 1), [0.0]))
    dNyp2 = np.tile(mot, Nx)[:-1]
    dNyp2[1:Ny] = 1.0
    dNyp2[0] = 4.0

    # dNyp3: length N-(Ny+3)
    dNyp3 = np.zeros(N - (Ny + 3))
    dNyp3[0] = -1.0
    dNyp3[Ny + 1 :: (Ny + 1)] = -0.25

    # d2Ny: length N-2*Ny
    d2Ny = np.zeros(N - 2 * Ny)
    d2Ny[Ny] = -0.25

    # d2Nyp1: length N-(2*Ny+1)
    d2Nyp1 = np.zeros(N - (2 * Ny + 1))
    d2Nyp1[1:Ny] = 0.25
    d2Nyp1[Ny] = 1.0

    # d2Nyp2: length N-(2*Ny+2)
    d2Nyp2 = np.zeros(N - (2 * Ny + 2))
    d2Nyp2[0] = 0.75
    d2Nyp2[Ny] = -0.75

    # d2Nyp3: length N-(2*Ny+3)
    d2Nyp3 = np.zeros(N - (2 * Ny + 3))
    d2Nyp3[0] = -1.0
    d2Nyp3[1:Ny] = -0.25

    # d2Nyp4: length N-(2*Ny+4)
    d2Nyp4 = np.zeros(N - (2 * Ny + 4))
    d2Nyp4[0] = 0.25

    # Assemble the 25 diagonals with appropriate padding.
    diags = np.vstack(
        [
            pad_upper(d2Nyp4, 2 * Ny + 4),
            pad_upper(d2Nyp3, 2 * Ny + 3),
            pad_upper(d2Nyp2, 2 * Ny + 2),
            pad_upper(d2Nyp1, 2 * Ny + 1),
            pad_upper(d2Ny, 2 * Ny),
            pad_upper(dNyp3, Ny + 3),
            pad_upper(dNyp2, Ny + 2),
            pad_upper(dNyp1, Ny + 1),
            pad_upper(dNy, Ny),
            pad_upper(dNym1, (Ny - 1)),
            pad_upper(d2, 2),
            pad_upper(d1, 1),
            d0,
            pad_lower(d1, 1),
            pad_lower(d2, 2),
            pad_lower(dNym1, Ny - 1),
            pad_lower(dNy, Ny),
            pad_lower(dNyp1, Ny + 1),
            pad_lower(dNyp2, Ny + 2),
            pad_lower(dNyp3, Ny + 3),
            pad_lower(d2Ny, 2 * Ny),
            pad_lower(d2Nyp1, 2 * Ny + 1),
            pad_lower(d2Nyp2, 2 * Ny + 2),
            pad_lower(d2Nyp3, 2 * Ny + 3),
            pad_lower(d2Nyp4, 2 * Ny + 4),
        ]
    )

    dN = np.array(
        [
            -(2 * Ny + 4),
            -(2 * Ny + 3),
            -(2 * Ny + 2),
            -(2 * Ny + 1),
            -(2 * Ny),
            -(Ny + 3),
            -(Ny + 2),
            -(Ny + 1),
            -Ny,
            -(Ny - 1),
            -2,
            -1,
            0,
            1,
            2,
            (Ny - 1),
            Ny,
            (Ny + 1),
            (Ny + 2),
            (Ny + 3),
            2 * Ny,
            2 * Ny + 1,
            2 * Ny + 2,
            2 * Ny + 3,
            2 * Ny + 4,
        ]
    )

    Dxy = sp.spdiags(diags, dN, N, N)
    return (1 / h**2) * Dxy


# %% ../nbs/api/num_utils.ipynb 7
def vkoperator(
    phi1: np.ndarray,
    phi2: np.ndarray,
    Dxx: sp.spmatrix,
    Dyy: sp.spmatrix,
    Dxy: sp.spmatrix,
) -> np.ndarray:
    r"""
    Compute the numerical approximation of the von Kármán operator $[\phi_1,\phi_2]$.

    The operator is defined as:

    $$
    [\phi_1,\phi_2] = \frac{\partial^2\phi_1}{\partial x^2}\frac{\partial^2\phi_2}{\partial y^2} +
    \frac{\partial^2\phi_1}{\partial y^2}\frac{\partial^2\phi_2}{\partial x^2} -
    2\frac{\partial^2\phi_1}{\partial x\partial y}\frac{\partial^2\phi_2}{\partial x\partial y}
    $$

    Parameters
    ----------
    phi1 : numpy.ndarray
        First function discretized on grid
    phi2 : numpy.ndarray
        Second function discretized on grid
    Dxx : scipy.sparse.spmatrix
        Second derivative operator in x direction
    Dyy : scipy.sparse.spmatrix
        Second derivative operator in y direction
    Dxy : scipy.sparse.spmatrix
        Mixed derivative operator

    Returns
    -------
    numpy.ndarray
        Discretized von Kármán operator evaluated at grid points
    """
    phi1x = Dxx @ phi1
    phi1y = Dyy @ phi1

    phi2x = Dxx @ phi2
    phi2y = Dyy @ phi2

    phi1xy = Dxy @ phi1
    phi2xy = Dxy @ phi2

    return phi1x * phi2y + phi1y * phi2x - 2 * phi1xy * phi2xy


# %% ../nbs/api/num_utils.ipynb 8
def double_trapezoid(f, dx, dy=None):
    return trapezoid(trapezoid(f, dx=dx if dy is None else dy), dx=dx)


def double_trapezoid_flat(
    f: np.ndarray, dx: float, dy: float, Ny: int, Nx: int
) -> float:
    """
    Compute double trapezoid integration on flattened array.

    Parameters
    ----------
    f : np.ndarray
        Flattened array to integrate
    dx : float
        Grid spacing in x direction
    dy : float
        Grid spacing in y direction
    Ny : int
        Number of points in y direction
    Nx : int
        Number of points in x direction

    Returns
    -------
    float
        Result of double integration
    """
    F = f.reshape((Ny, Nx), order="F")
    return double_trapezoid(F, dx, dy)


# %% ../nbs/api/num_utils.ipynb 9
def compute_coupling_matrix_numerical(
    psi: np.ndarray,
    phi: np.ndarray,
    h: float,
    nx: int,
    ny: int,
):
    r"""

    Compute the coupling matrix for the given in-plane and out-of-plane modes.

    The modal coupling matrix is computed as

    $$
    H_{p, q}^k =
    \frac{\int_S \Psi_k L\left(\Phi_p, \Phi_q\right) \mathrm{d} S}{\left\|\Psi_k\right\|\left\|\Phi_p\right\|\left\|\Phi_q\right\|}
    $$

    Here however we compute

    $$
    H_{p, q}^k= \int_S \Psi_k L\left(\Phi_p, \Phi_q\right) \mathrm{d} S
    $$

    since the $\Psi$ and $\Phi$ functions are normalised elsewhere.

    Parameters
    ----------
    psi : np.ndarray
        The **normalised** in-plane modes with shape (ny+1 * nx+1, n_modes.
        These are stored in a flattened array column-wise.
    phi : np.ndarray
        The **normalised** out-of-plane modes with shape (ny+1 * nx+1, n_modes).
        These are stored in a flattened array column-wise.
    h : float
        The grid spacing.
    nx : int
        The number of intervals in the x-direction.
    ny : int
        The number of intervals in the y-direction.

    Returns
    -------
    np.ndarray
        The coupling matrix with shape (n_modes, n_modes, n_modes).

    """

    Dxx = second_derivative(
        nx,
        ny,
        h,
        direction="x",
    )
    Dyy = second_derivative(
        nx,
        ny,
        h,
        direction="y",
    )
    Dxy = second_derivative_mixed(
        nx,
        ny,
        h,
    )

    n_modes = psi.shape[1]

    # Compute the norms of the in-plane modes
    # psi_norms = np.array(
    #     [
    #         double_trapezoid_flat(
    #             psi[:, k] * psi[:, k],
    #             h,
    #             h,
    #             Ny=ny + 1,
    #             Nx=nx + 1,
    #         )
    #         for k in range(n_modes)
    #     ]
    # )

    # Compute the coupling matrix
    H = np.zeros((n_modes, n_modes, n_modes))
    for k in range(n_modes):
        psik = psi[:, k]
        # norm_k = psi_norms[k]
        for p in range(n_modes):
            phip = phi[:, p]
            # phi_norm_p = phi_norms[p]
            for q in range(p, n_modes):
                phiq = phi[:, q]
                # phi_norm_q = phi_norms[q]
                vkop = vkoperator(phip, phiq, Dxx, Dyy, Dxy)
                coupling = double_trapezoid_flat(
                    psik * vkop,
                    h,
                    h,
                    Ny=ny + 1,
                    Nx=nx + 1,
                )  # / (np.sqrt(norm_k) * np.sqrt(phi_norm_p) * np.sqrt(phi_norm_q))

                # ensure symmetry in the last two indices
                H[k, q, p] = coupling
                H[k, p, q] = coupling

    # Zero small values
    threshold = 1e-8
    H[np.abs(H / np.max(np.abs(H))) < threshold] = 0
    return H


# %% ../nbs/api/num_utils.ipynb 11
def polarisation(
    interpolated_eigenvectors,
    eigenvectors,
    h,
):
    negative = interpolated_eigenvectors - eigenvectors
    positive = interpolated_eigenvectors + eigenvectors

    sgn = np.sign(
        -np.abs(double_trapezoid(negative * negative, dx=h))
        + np.abs(double_trapezoid(positive * positive, dx=h))
    )
    return sgn * eigenvectors


# %% ../nbs/api/num_utils.ipynb 12
def eigenMAC(
    ref_eigenvectors,
    ref_nx,
    ref_ny,
    eigenvectors,
    eigenvalues,
    nx,
    ny,
    n_modes,
    Lx,
    Ly,
    h,
):
    r"""

    Computes the Modal Assurance Criterion (MAC) between reference
    eigenvectors and given eigenvectors.

    The Modal Assurance Criterion (MAC) between two eigenvectors
    (mode shapes) $\Phi_i$ and $\Phi_j$ is:

    $$
    \mathrm{MAC}(\Phi_i,\Phi_j) =
    \frac{|\Phi_i^{T}\,\Phi_j|^{2}}
        {\left(\Phi_i^{T}\,\Phi_i\right)\,\left(\Phi_j^{T}\,\Phi_j\right)}.
    $$

    MAC measures the degree of similarity (or consistency) between the two mode shapes.
    A value of 1 indicates identical shapes (up to a scalar),
    while a value near 0 indicates they are nearly orthogonal.


    Parameters
    ----------
    ref_eigenvectors : ndarray
        Reference eigenvectors (reshaped for interpolation).
    ref_nx : int
        Number of reference grid points along the x-axis.
    ref_ny : int
        Number of reference grid points along the y-axis.
    eigenvectors : ndarray
        Eigenvectors to compare against the reference.
    eigenvalues : ndarray
        Corresponding eigenvalues of the eigenvectors.
    nx : int
        Number of grid points along the x-axis for interpolation.
    ny : int
        Number of grid points along the y-axis for interpolation.
    n_modes : int
        Number of modes to compare.
    Lx : float
        Length of the domain along the x-axis.
    Ly : float
        Length of the domain along the y-axis.

    Returns
    -------
    eigenvectors_swapped : ndarray
        Reordered eigenvectors after MAC computation.
    eigenvalues_swapped : ndarray
        Reordered eigenvalues after MAC computation.
    """
    # Define reference and target grids
    xref = np.linspace(0, Lx, ref_nx + 1)
    yref = np.linspace(0, Ly, ref_ny + 1)

    x = np.linspace(0, Lx, nx + 1)
    y = np.linspace(0, Ly, ny + 1)

    # Interpolate eigenvectors
    interpolated_eigenvectors = np.zeros(((nx + 1) * (ny + 1), n_modes))
    for mode in range(n_modes):
        Z = ref_eigenvectors[:, mode].reshape(ref_nx + 1, ref_ny + 1)

        interpolator = RectBivariateSpline(
            xref,
            yref,
            Z,
            kx=1,
            ky=1,
        )

        interpolated_eigenvectors[:, mode] = interpolator(x, y).ravel()

    # Compute MAC matrix
    norm_eigenvectors = np.sum(eigenvectors**2, axis=0, keepdims=True)
    norm_interpolated = np.sum(interpolated_eigenvectors**2, axis=0, keepdims=True)

    num = np.abs(interpolated_eigenvectors.T @ eigenvectors) ** 2
    den = norm_interpolated.T @ norm_eigenvectors  # Shape (n_modes, n_modes)

    MAC_matrix = num / den
    MAC_matrix[MAC_matrix < 0.1] = 0
    np.fill_diagonal(MAC_matrix, 0)

    # Find matching indices
    rows, cols = np.where(MAC_matrix > 0)
    lmc = len(cols)

    if lmc > 0:
        swap_indices = np.arange(n_modes)
        check = rows[0]
        for i in range(lmc - 1):
            if check != cols[i]:
                swap_indices[cols[i]], swap_indices[rows[i]] = (
                    swap_indices[rows[i]],
                    swap_indices[cols[i]],
                )
                check = rows[i]

        eigenvectors = polarisation(interpolated_eigenvectors, eigenvectors, h)

        # Reorder eigenvectors and eigenvalues
        eigenvectors_swapped = eigenvectors[:, swap_indices]
        eigenvalues_swapped = eigenvalues[swap_indices]

    else:
        eigenvectors_swapped = eigenvectors
        eigenvalues_swapped = eigenvalues
    return eigenvectors_swapped, eigenvalues_swapped


# %% ../nbs/api/num_utils.ipynb 15
def biharmonic_eigendecomposition(
    params: PlateParameters,
    n_modes: int,
    bcs: np.ndarray,
    nx: int,
    ny: int,
    h: float,
    normalise_eigenvectors=True,
):
    """
    Computes the eigenvalue decomposition of the biharmonic operator for a
    plate with the given parameters and boundary conditions.

    Additionally it sorts the eigenvalues and eigenvectors in ascending order,
    and normalises the eigenvectors if requested.

    Parameters
    ----------
    params : PlateParameters
        The parameters of the plate.
    n_modes : int
        The number of modes to compute.
    bcs : np.ndarray
        The boundary conditions of the plate.
    nx : int
        The number of points in the x direction.
    ny : int
        The number of points in the y direction.
    h : float
        The spacing between points.
    normalise_eigenvectors : bool
        Whether to normalise the eigenvectors.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        The eigenvalues, eigenvectors and norms of the eigenvectors.
    """
    biharm = bhmat(bcs, [nx + 1, ny + 1], h, params.h, params.E, params.nu)

    [eigenvalues, eigenvectors] = eigs(biharm, k=n_modes, sigma=0, which="LR")

    indSort = np.argsort(eigenvalues)

    eigenvalues = eigenvalues[indSort]
    eigenvectors = eigenvectors[:, indSort]

    # sometimes these might be complex due to numerical errors in bhmat?
    # TODO: check if the biharmonic matrix is symmetric always
    # if so we take the real part
    eigenvectors = np.real(eigenvectors)
    eigenvalues = np.real(eigenvalues)

    norms = np.zeros(n_modes)
    for i in range(n_modes):
        norm = double_trapezoid_flat(
            eigenvectors[:, i] ** 2,
            h,
            h,
            ny + 1,
            nx + 1,
        )
        norms[i] = norm
        if normalise_eigenvectors:
            eigenvectors[:, i] /= np.sqrt(norm)
    return eigenvalues, eigenvectors, norms


# %% ../nbs/api/num_utils.ipynb 16
def multiresolution_eigendecomposition(
    params: PlateParameters,
    n_modes: int,
    bcs: np.ndarray,
    h: float,
    nx: int,
    ny: int,
    levels: int = 2,
):
    """
    Runs the biharmonic eigendecomposition and eigenvector alignment on multiple grid resolutions.

    Parameters
    ----------
      params: PlateParameters
        Parameters object containing domain lengths (e.g. params.lx and params.ly).
      n_modes : int
        Number of eigenmodes to compute.
      bcs : np.ndarray
        Boundary conditions.
      h : float
        Initial grid spacing.
      nx : int
        Number of grid points in the x-direction.
      ny : int
        Number of grid points in the y-direction.
      levels : int
        Total number of resolutions to run (default 2).
        The first level is the coarse grid, and each subsequent level
        uses h/2 and double the grid points to cover the same domain.

    Returns
    -------
      swapped_eigenvectors, swapped_eigenvalues from the last refinement.
    """
    # Store the coarse grid values to use as a reference for eigenMAC.
    ref_h, ref_nx, ref_ny = h, nx, ny

    # Run the coarse-grid eigen-decomposition.
    _, ref_eigenvectors, norms = biharmonic_eigendecomposition(
        params,
        n_modes,
        bcs,
        nx,
        ny,
        h,
    )

    # For each subsequent refinement level, halve h and double the grid points.
    for _ in range(1, levels):
        print(f"Refining grid to h = {h / 2}, nx = {nx * 2}, ny = {ny * 2}")
        h = h / 2
        nx = int(nx * 2)
        ny = int(ny * 2)

        omega_mu, eigenvectors, norms = biharmonic_eigendecomposition(
            params,
            n_modes,
            bcs,
            nx,
            ny,
            h,
        )

        swapped_eigenvectors, swapped_eigenvalues = eigenMAC(
            ref_eigenvectors=ref_eigenvectors,
            ref_nx=ref_nx,
            ref_ny=ref_ny,
            eigenvectors=eigenvectors,
            eigenvalues=omega_mu,
            nx=nx,
            ny=ny,
            n_modes=n_modes,
            Lx=params.l1,
            Ly=params.l2,
            h=h,
        )

        # Optionally, update the reference eigenvectors for the next level.
        ref_eigenvectors = swapped_eigenvectors

    return swapped_eigenvectors, swapped_eigenvalues, nx, ny, h, norms
