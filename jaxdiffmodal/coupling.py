import math

import numpy as np
from scipy import sparse
from scipy.linalg import eig  # or eigh if K,M guaranteed symmetric
from scipy.special import iv, jv  # Bessel functions for circular functions


def int4(
    m: int,
    p: int,
    L: float,
) -> float:
    r"""
    Compute the integral of Xd(m,x)*Xd(p,x) from 0 to L, where X is the clamped-plate
    function and d denotes derivative in x.

    The integral is computed for different cases based on the mode numbers m and p.

    Parameters
    ----------
    m : int
        First mode number
    p : int
        Second mode number
    L : float
        Length of the domain

    Returns
    -------
    float
        Value of the integral
    """
    if m == 0 and p == 0:
        y = 120.0 / (7.0 * L)
    elif (m == p) and (m != 0):
        # (768*pi^2*m^2 - 47040*(-1)^m + 35*pi^4*m^4 + 432*(-1)^m*pi^2*m^2 - 53760) / (70*L*pi^2*m^2)
        y = (
            768.0 * math.pi**2 * m * m
            - 47040.0 * ((-1) ** m)
            + 35.0 * math.pi**4 * m**4
            + 432.0 * ((-1) ** m) * math.pi**2 * m * m
            - 53760.0
        ) / (70.0 * L * math.pi**2 * m * m)
    elif m == 0:
        # (60*((-1)^p + 1)*(pi^2*p^2 - 42)) / (7*L*pi^2*p^2)
        num = 60.0 * (((-1) ** p + 1.0) * (math.pi**2 * p**2 - 42.0))
        den = 7.0 * L * math.pi**2 * p**2
        y = num / den
    elif p == 0:
        # (60*((-1)^m + 1)*(pi^2*m^2 - 42)) / (7*L*pi^2*m^2)
        num = 60.0 * (((-1) ** m + 1.0) * (math.pi**2 * m**2 - 42.0))
        den = 7.0 * L * math.pi**2 * m**2
        y = num / den
    else:
        # 192/35/L*(1 + (-1)^m*(-1)^p)
        # - 192/(m^2*p^2*L*pi^2)*((p^2+m^2)*(1 + (-1)^m*(-1)^p))
        # - 168/(m^2*p^2*L*pi^2)*((p^2+m^2)*((-1)^m + (-1)^p))
        # + 108/35/L*((-1)^m + (-1)^p)
        term1 = 192.0 / 35.0 / L * (1.0 + (-1) ** m * (-1) ** p)
        term2 = (
            -192.0
            / (m * m * p * p * L * math.pi**2)
            * ((p * p + m * m) * (1.0 + (-1) ** m * (-1) ** p))
        )
        term3 = (
            -168.0
            / (m * m * p * p * L * math.pi**2)
            * ((p * p + m * m) * (((-1) ** m) + ((-1) ** p)))
        )
        term4 = 108.0 / 35.0 / L * (((-1) ** m) + ((-1) ** p))
        y = term1 + term2 + term3 + term4
    return y


def int1(m, p, L):
    r"""

    Parameters
    ----------
    m : int
        First mode number
    p : int
        Second mode number
    L : float
        Length of the domain

    Returns
    -------
    float
        Value of the integral
    """
    if m == 0 and p == 0:
        # y = 720 / L^3
        y = 720.0 / (L**3)
    elif m == p:
        # (pi^4*m^4 - 672*(-1)^m - 768)/(2*L^3)
        y = ((math.pi**4) * (m**4) - 672.0 * ((-1) ** m) - 768.0) / (2.0 * (L**3))
    elif m == 0 or p == 0:
        # y=0
        y = 0.0
    else:
        # -(24*(7*(-1)^m + 7*(-1)^p + 8*(-1)^m*(-1)^p + 8))/L^3
        val = (
            7.0 * ((-1) ** m)
            + 7.0 * ((-1) ** p)
            + 8.0 * ((-1) ** m) * ((-1) ** p)
            + 8.0
        )
        y = -24.0 * val / (L**3)
    return y


def int2(m, p, L):
    """
    Function int2(m,p,L). Piecewise definition from the Matlab code.
    """
    if m == 0 and p == 0:
        y = (10.0 * L) / 7.0
    elif m == p:
        # (67*L)/70 - ((-1)^m*L)/35 - (768*L)/(pi^4*m^4) - (672*(-1)^m*L)/(pi^4*m^4)
        y = (
            (67.0 * L) / 70.0
            - ((-1) ** m * L) / 35.0
            - (768.0 * L) / (math.pi**4 * m**4)
            - (672.0 * ((-1) ** m) * L) / (math.pi**4 * m**4)
        )
    elif m == 0:
        # (3*L*((-1)^p + 1)*(pi^4*p^4 - 1680)) / (14*pi^4*p^4)
        num = 3.0 * L * (((-1) ** p + 1.0) * ((math.pi**4) * (p**4) - 1680.0))
        den = 14.0 * math.pi**4 * (p**4)
        y = num / den
    elif p == 0:
        # (3*L*((-1)^m + 1)*(pi^4*m^4 - 1680)) / (14*pi^4*m^4)
        num = 3.0 * L * (((-1) ** m + 1.0) * ((math.pi**4) * (m**4) - 1680.0))
        den = 14.0 * math.pi**4 * (m**4)
        y = num / den
    else:
        # Big piecewise "else" expression from the Matlab code
        # -(L*(11760*(-1)^m + 11760*(-1)^p - 16*pi^4*m^4 + ...)) / (70*pi^4*m^4)
        #  - (L*(13440*m^4 + 11760*(-1)^m*m^4 + 11760*(-1)^p*m^4 + ... )) / (70*pi^4*m^4*p^4)
        part1 = (
            11760.0 * ((-1) ** m)
            + 11760.0 * ((-1) ** p)
            - 16.0 * (math.pi**4) * m**4
            + 13440.0 * ((-1) ** m) * ((-1) ** p)
            + ((-1) ** m) * (math.pi**4) * (m**4)
            + ((-1) ** p) * (math.pi**4) * (m**4)
            - 16.0 * ((-1) ** m) * ((-1) ** p) * (math.pi**4) * (m**4)
            + 13440.0
        )
        part2 = (
            13440.0 * m**4
            + 11760.0 * ((-1) ** m) * m**4
            + 11760.0 * ((-1) ** p) * m**4
            + 13440.0 * ((-1) ** m) * ((-1) ** p) * m**4
        )
        y = -(L * part1) / (70.0 * (math.pi**4) * (m**4)) - (L * part2) / (
            70.0 * (math.pi**4) * (m**4) * (p**4)
        )
    return y


def int2_mat(N, L):
    """
    Builds the N x N matrix whose (m,p) entry is int2(m,p,L).
    Mirrors the logic of the original Matlab int2_mat function exactly,
    but we can do it more simply by calling int2 in a loop.
    """
    y = np.zeros((N, N), dtype=float)
    for m in range(N):
        for p in range(N):
            y[m, p] = int2(m, p, L)
    return y


def build_I1(N, L):
    """Returns the N x N matrix whose (m,p) entry = int1(m,p,L)."""
    I = np.zeros((N, N), dtype=float)
    for m in range(N):
        for p in range(N):
            I[m, p] = int1(m, p, L)
    return I


def build_I2(N, L):
    """Returns the N x N matrix whose (m,p) entry = int2(m,p,L)."""
    I = np.zeros((N, N), dtype=float)
    for m in range(N):
        for p in range(N):
            I[m, p] = int2(m, p, L)
    return I


def build_I4(N, L):
    """Returns the N x N matrix whose (m,p) entry = int4(m,p,L)."""
    I = np.zeros((N, N), dtype=float)
    for m in range(N):
        for p in range(N):
            I[m, p] = int4(m, p, L)
    return I


def assemble_K_and_M(Npsi: int, Lx: float, Ly: float) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Assemble stiffness ($K$) and mass ($M$) matrices for plate dynamics.

    Constructs the global stiffness and mass matrices for rectangular plate
    vibration analysis using Kronecker products of 1D integral matrices.
    The matrices are built from precomputed integrals in x and y directions.

    Parameters
    ----------
    Npsi : int
        Number of modes in each spatial direction for the in-plane problem
    Lx : float
        Length of the plate in x-direction
    Ly : float
        Length of the plate in y-direction

    Returns
    -------
    K : numpy.ndarray
        Stiffness matrix of shape (Npsi², Npsi²)
    M : numpy.ndarray
        Mass matrix of shape (Npsi², Npsi²)

    Notes
    -----
    The matrices are assembled using:

    $$K = I_1^x \otimes I_2^y + I_2^x \otimes I_1^y + 2 I_4^x \otimes I_4^y$$

    $$M = I_2^x \otimes I_2^y$$

    where $\otimes$ denotes the Kronecker product and $I_j^{x,y}$ are
    1D integral matrices computed using functions int1, int2, and int4.
    """
    # 1) Precompute integrals in the x-direction:
    I1x = build_I1(Npsi, Lx)
    I2x = build_I2(Npsi, Lx)
    I4x = build_I4(Npsi, Lx)

    # 2) Precompute integrals in the y-direction:
    I1y = build_I1(Npsi, Ly)
    I2y = build_I2(Npsi, Ly)
    I4y = build_I4(Npsi, Ly)

    # 3) Build the K, M matrices via Kronecker products:
    K = np.kron(I1x, I2y) + np.kron(I2x, I1y) + 2.0 * np.kron(I4x, I4y)
    M = np.kron(I2x, I2y)

    return K, M


def airy_stress_coefficients(
    n_psi: int,
    vals: np.ndarray,
    vecs: np.ndarray,
    Lx: float,
    Ly: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Compute Airy stress function coefficients from eigendecomposition.

    Processes eigenvalues and eigenvectors from the generalized eigenvalue problem
    $K\phi = \lambda M\phi$ to compute normalized coefficients for the Airy stress
    function representation. Filters out negative and complex eigenvalues.

    Parameters
    ----------
    n_psi : int
        Number of modes in each spatial direction
    vals : numpy.ndarray
        Eigenvalues from the generalized eigenvalue problem, shape (n_psi²,)
    vecs : numpy.ndarray
        Eigenvectors from the generalized eigenvalue problem, shape (n_psi², n_psi²)
    Lx : float
        Length of the plate in x-direction
    Ly : float
        Length of the plate in y-direction

    Returns
    -------
    coeff0 : numpy.ndarray
        Base coefficients normalized by eigenvector norm, shape (S/2, S/2)
    coeff1 : numpy.ndarray
        Coefficients normalized by eigenvector norm and sqrt(eigenvalue), shape (S/2, S/2)
    coeff2 : numpy.ndarray
        Coefficients normalized by eigenvector norm and eigenvalue, shape (S/2, S/2)

    Notes
    -----
    The coefficients are computed using three different normalizations:

    - $\text{coeff0}_{ij} = \frac{\phi_i}{\|\phi_i\|}$
    - $\text{coeff1}_{ij} = \frac{\phi_i}{\|\phi_i\| \sqrt{\lambda_i}}$
    - $\text{coeff2}_{ij} = \frac{\phi_i}{\|\phi_i\| \lambda_i}$

    where $\phi_i$ are the eigenvectors and $\lambda_i$ are the eigenvalues.
    Only the first $S/2$ modes are retained where $S$ is the number of
    positive real eigenvalues.
    """

    # Remove negative or imaginary eigenvalues:
    #    (1) negative real part
    #    (2) any non-zero imaginary part
    real_vals = vals.real
    imag_vals = vals.imag

    # Indices of negative or significantly imaginary eigenvalues
    eps_imag = 1e-12  # small threshold
    bad_neg = np.where(real_vals < 0.0)[0]
    bad_imag = np.where(abs(imag_vals) > eps_imag)[0]
    bad_indices = np.unique(np.concatenate((bad_neg, bad_imag)))

    # Filter out the unwanted eigenvalues and eigenvectors
    good_mask = np.ones(vals.shape, dtype=bool)
    good_mask[bad_indices] = False
    good_vals = vals[good_mask]
    good_vecs = vecs[:, good_mask]

    # Sort them
    idx_sort = np.argsort(good_vals.real)  # sort by real part
    auto = good_vals.real[idx_sort]
    coeff = good_vecs.real[:, idx_sort]

    # Build the factor arrays (coeff0, coeff1, coeff2)
    dim = n_psi**2
    coeff0 = np.zeros((dim, coeff.shape[0]), dtype=float)
    coeff1 = np.zeros((dim, coeff.shape[0]), dtype=float)
    coeff2 = np.zeros((dim, coeff.shape[0]), dtype=float)

    NN = int2_mat(n_psi, Lx)
    MM = int2_mat(n_psi, Ly)

    NN = np.tile(NN.reshape(-1, 1), dim)
    MM = np.tile(MM.reshape(1, -1), (dim, 1))

    nmatr = NN * MM
    nmatr = np.reshape(nmatr, (n_psi, n_psi, n_psi, n_psi))

    nmatr = np.transpose(nmatr, (3, 0, 2, 1))
    nmatr = np.reshape(nmatr, (1, n_psi**4))

    nmatr = sparse.csr_matrix(nmatr)

    S = auto.shape[0]

    for d in range(S):
        temp = coeff[:, d]  # shape (DIM,)

        temp = temp.reshape(n_psi**2, 1)
        temp_big = np.tile(temp, (1, n_psi**2))
        temp2 = temp_big.T
        temp3 = temp_big * temp2
        temp3 = temp3.reshape(n_psi**4, 1)
        norms = nmatr.dot(temp3)

        # Then do the same normalizations:
        coeff0[:, d] = coeff[:, d] / np.sqrt(norms)
        coeff1[:, d] = coeff[:, d] / np.sqrt(norms) / np.sqrt(auto[d])
        coeff2[:, d] = coeff[:, d] / np.sqrt(norms) / auto[d]

    # "S = floor(S/2);"
    S2 = S // 2

    coeff0 = coeff0[:S2, :S2]
    coeff1 = coeff1[:S2, :S2]
    coeff2 = coeff2[:S2, :S2]

    return coeff0, coeff1, coeff2


def _basis(m, x, Lx):
    """
    Evaluate the 1D basis function in the x or y direction
    for integer index m at location x, with length Lx.
    """
    # Python integer exponent (-1)**m is fine for integer m
    return (
        np.cos(m * np.pi * x / Lx)
        + (15 * (1 + (-1) ** m) / Lx**4) * x**4
        - (4 * (8 + 7 * (-1) ** m) / Lx**3) * x**3
        + (6 * (3 + 2 * (-1) ** m) / Lx**2) * x**2
        - 1.0
    )


def basis(
    m: int, n: int, x: float | np.ndarray, y: float | np.ndarray, Lx: float, Ly: float
) -> float | np.ndarray:
    r"""
    Evaluate the full 2D basis function for indices (m, n) at point (x, y).

    The basis function is defined as:

    .. math::
        \phi_{mn}(x,y) = X_m(x)Y_n(y)

    where X_m and Y_n are the 1D basis functions in x and y directions.

    Parameters
    ----------
    m : int
        Mode number in x direction
    n : int
        Mode number in y direction
    x : float or array_like
        x coordinate(s) where to evaluate
    y : float or array_like
        y coordinate(s) where to evaluate
    Lx : float
        Length in x direction
    Ly : float
        Length in y direction

    Returns
    -------
    float or array_like
        Value of the basis function at (x,y)
    """
    return _basis(m, x, Lx) * _basis(n, y, Ly)


def i1_mat(Npsi, Nphi, L):
    # Initialize a 3D array of zeros.
    s = np.zeros((Npsi, Nphi, Nphi))

    # Loop over indices.
    # In MATLAB: m = 1:Npsi, here m = 0,...,Npsi-1, and m1 = m (since MATLAB m1 = m-1)
    for m in range(1, Npsi + 1):
        m1 = m - 1
        for n in range(1, Nphi + 1):
            for p in range(1, Nphi + 1):
                if m1 == 0 and n == p:
                    s[m - 1, n - 1, p - 1] = L / 2.0
                elif m1 == (p - n) or m1 == (n - p):
                    s[m - 1, n - 1, p - 1] = L / 4.0
                elif m1 == (-n - p) or m1 == (n + p):
                    s[m - 1, n - 1, p - 1] = -L / 4.0
    return s


def i2_mat(Npsi, Nphi, L):
    """ """
    s = np.zeros((Npsi, Nphi, Nphi))

    # Loop over m, n, and p (using 1-indexing, then subtract 1 for Python indexing)
    for m in range(1, Npsi + 1):
        m1 = m - 1
        for n in range(1, Nphi + 1):
            for p in range(1, Nphi + 1):
                # The condition "if n==0 || p==0" is omitted since n,p >= 1.
                if n == p:
                    # s(m,n,p) = (15/L^4*((-1)^(m1) + 1))*(L^5*(4*pi^5*p^5 - 20*pi^3*p^3 + 30*pi*p))/(40*pi^5*p^5);
                    numerator = L**5 * (
                        4 * np.pi**5 * p**5 - 20 * np.pi**3 * p**3 + 30 * np.pi * p
                    )
                    s[m - 1, n - 1, p - 1] = (15 / L**4 * (((-1) ** m1) + 1)) * (
                        numerator / (40 * np.pi**5 * p**5)
                    )
                else:
                    # Else branch: a long expression.
                    # First compute the common terms:
                    np_sum = n + p
                    np_diff = (
                        n - p
                    )  # note: may be negative, but the formula handles it.

                    term1 = (
                        np.sin(np.pi * np_sum)
                        * (
                            (1713638851887625 * L**4 * np_sum**4) / 17592186044416
                            - (8334140006820045 * L**4 * np_sum**2) / 70368744177664
                            + 24 * L**4
                        )
                    ) + 4 * np.pi * L**2 * np.cos(np.pi * np_sum) * np_sum * (
                        ((2778046668940015 * L**2 * np_sum**2) / 281474976710656)
                        - 6 * L**2
                    )
                    term1 /= np_sum**5

                    term2 = (
                        np.sin(np.pi * np_diff)
                        * (
                            (1713638851887625 * L**4 * np_diff**4) / 17592186044416
                            - (8334140006820045 * L**4 * np_diff**2) / 70368744177664
                            + 24 * L**4
                        )
                    ) + 4 * np.pi * L**2 * np.cos(np.pi * np_diff) * np_diff * (
                        ((2778046668940015 * L**2 * np_diff**2) / 281474976710656)
                        - 6 * L**2
                    )
                    term2 /= np_diff**5

                    big_term = 8796093022208 * L * (term1 - term2)
                    s[m - 1, n - 1, p - 1] = (
                        -(15 / L**4 * (((-1) ** m1) + 1)) * big_term / 5383555227996211
                    )
    return s


def i3_mat(Npsi, Nphi, L):
    """
    Auxiliary integral for the computation of the coupling coefficient H.
    """
    s = np.zeros((Npsi, Nphi, Nphi))

    for m in range(1, Npsi + 1):
        m1 = m - 1
        for n in range(1, Nphi + 1):
            for p in range(1, Nphi + 1):
                # The first condition "if n==0 && p==0" never occurs because n, p start at 1.
                if n == p:
                    # Compute:
                    # s(m,n,p) = -(-4/L^3*(7*(-1)^(m1)+8))*(L^4*(6*pi^2*p^2 - 2*pi^4*p^4))/(16*pi^4*p^4)
                    # The double negative simplifies to a positive.
                    s[m - 1, n - 1, p - 1] = (
                        (4 / L**3 * (7 * (-1) ** m1 + 8))
                        * (L**4 * (6 * np.pi**2 * p**2 - 2 * np.pi**4 * p**4))
                        / (16 * np.pi**4 * p**4)
                    )
                else:
                    # Else branch: compute two terms and add them.
                    term1 = (
                        (-4 / L**3 * (7 * (-1) ** m1 + 8))
                        * (L * ((6 * L**3) / (n - p) ** 4 - (6 * L**3) / (n + p) ** 4))
                        / (2 * np.pi**4)
                    )

                    term2 = (
                        (-4 / L**3 * (7 * (-1) ** m1 + 8))
                        * (
                            L
                            * (
                                (
                                    3
                                    * L
                                    * np.cos(np.pi * (n + p))
                                    * (2 * L**2 - L**2 * np.pi**2 * (n + p) ** 2)
                                )
                                / (n + p) ** 4
                                - (
                                    3
                                    * L
                                    * np.cos(np.pi * (n - p))
                                    * (2 * L**2 - L**2 * np.pi**2 * (n - p) ** 2)
                                )
                                / (n - p) ** 4
                            )
                        )
                        / (2 * np.pi**4)
                    )

                    s[m - 1, n - 1, p - 1] = term1 + term2
    return s


def i4_mat(Npsi, Nphi, L):
    """
    Auxiliary integral for the computation of the coupling coefficient H.
    """
    s = np.zeros((Npsi, Nphi, Nphi))
    for m in range(1, Npsi + 1):
        m1 = m - 1
        for n in range(1, Nphi + 1):
            for p in range(1, Nphi + 1):
                if n == p:
                    s[m - 1, n - 1, p - 1] = (
                        -(6 / L**2 * (2 * (-1) ** m1 + 3))
                        * (L**3 * (6 * np.pi * p - 4 * np.pi**3 * p**3))
                        / (24 * np.pi**3 * p**3)
                    )
                else:
                    term1 = (
                        (6 / L**2 * (2 * (-1) ** m1 + 3))
                        * (L**3 * np.cos(np.pi * (n - p)))
                        / (np.pi**2 * (n - p) ** 2)
                    )
                    term2 = (
                        (6 / L**2 * (2 * (-1) ** m1 + 3))
                        * (L**3 * np.cos(np.pi * (n + p)))
                        / (np.pi**2 * (n + p) ** 2)
                    )
                    s[m - 1, n - 1, p - 1] = term1 - term2
    return s


def i5_mat(Npsi, Nphi, L):
    s = np.zeros((Npsi, Nphi, Nphi))
    for m in range(1, Npsi + 1):
        for n in range(1, Nphi + 1):
            for p in range(1, Nphi + 1):
                if n == p:
                    s[m - 1, n - 1, p - 1] = L / 2.0
    return -s


def i9_mat(Npsi, Nphi, L):
    # Initialize a 3D array of zeros.
    s = np.zeros((Npsi, Nphi, Nphi))

    for m in range(1, Npsi + 1):
        m1 = m - 1
        for n in range(1, Nphi + 1):
            for p in range(1, Nphi + 1):
                if m1 == 0 and n == p:
                    s[m - 1, n - 1, p - 1] = L / 2.0
                elif m1 == (p - n) or m1 == (n - p):
                    s[m - 1, n - 1, p - 1] = L / 4.0
                elif m1 == (-n - p) or m1 == (n + p):
                    s[m - 1, n - 1, p - 1] = L / 4.0
    return s


def i10_mat(Npsi, Nphi, L):
    s = np.zeros((Npsi, Nphi, Nphi))
    for m in range(1, Npsi + 1):
        m1 = m - 1
        for n in range(1, Nphi + 1):
            for p in range(1, Nphi + 1):
                if n == p and p != 0:
                    s[m - 1, n - 1, p - 1] = (
                        (15 / L**4 * (((-1) ** m1) + 1))
                        * (
                            L**5
                            * (
                                4 * np.pi**5 * n**5
                                + 20 * np.pi**3 * n**3
                                - 30 * np.pi * n
                            )
                        )
                        / (40 * np.pi**5 * n**5)
                    )
                else:
                    term1 = (
                        4
                        * np.pi
                        * L**2
                        * np.cos(np.pi * (n + p))
                        * (6 * L**2 - L**2 * np.pi**2 * (n + p) ** 2)
                    ) / (n + p) ** 4
                    term2 = (
                        4
                        * np.pi
                        * L**2
                        * np.cos(np.pi * (n - p))
                        * (6 * L**2 - L**2 * np.pi**2 * (n - p) ** 2)
                    ) / (n - p) ** 4
                    s[m - 1, n - 1, p - 1] = (
                        -(15 / L**4 * (((-1) ** m1) + 1))
                        * (L * (term1 + term2))
                        / (2 * np.pi**5)
                    )
    return s


def i11_mat(Npsi, Nphi, L):
    s = np.zeros((Npsi, Nphi, Nphi))
    for m in range(1, Npsi + 1):
        m1 = m - 1
        for n in range(1, Nphi + 1):
            for p in range(1, Nphi + 1):
                if n == p and p != 0:
                    s[m - 1, n - 1, p - 1] = (
                        (-4 / L**3 * (7 * ((-1) ** m1) + 8)) * L**4
                    ) / 8 + ((-4 / L**3 * (7 * ((-1) ** m1) + 8)) * (3 * L**4)) / (
                        8 * np.pi**2 * p**2
                    )
                else:
                    term1 = (
                        (-4 / L**3 * (7 * ((-1) ** m1) + 8))
                        * (L * ((6 * L**3) / (n - p) ** 4 + (6 * L**3) / (n + p) ** 4))
                        / (2 * np.pi**4)
                    )
                    term2 = (
                        (-4 / L**3 * (7 * ((-1) ** m1) + 8))
                        * (
                            L
                            * (
                                (
                                    3
                                    * L
                                    * np.cos(np.pi * (n + p))
                                    * (2 * L**2 - L**2 * np.pi**2 * (n + p) ** 2)
                                )
                                / (n + p) ** 4
                                + (
                                    3
                                    * L
                                    * np.cos(np.pi * (n - p))
                                    * (2 * L**2 - L**2 * np.pi**2 * (n - p) ** 2)
                                )
                                / (n - p) ** 4
                            )
                        )
                        / (2 * np.pi**4)
                    )
                    s[m - 1, n - 1, p - 1] = term1 - term2
    return s


def i12_mat(Npsi, Nphi, L):
    s = np.zeros((Npsi, Nphi, Nphi))
    for m in range(1, Npsi + 1):
        m1 = m - 1
        for n in range(1, Nphi + 1):
            for p in range(1, Nphi + 1):
                if n == p and p != 0:
                    s[m - 1, n - 1, p - 1] = (
                        6 / L**2 * (2 * ((-1) ** m1) + 3)
                    ) * L**3 / 6 + (6 / L**2 * (2 * ((-1) ** m1) + 3)) * L**3 / (
                        4 * np.pi**2 * p**2
                    )
                else:
                    s[m - 1, n - 1, p - 1] = (
                        6 / L**2 * (2 * ((-1) ** m1) + 3)
                    ) * L**3 * np.cos(np.pi * (n - p)) / (
                        np.pi**2 * (n - p) ** 2
                    ) + (
                        6 / L**2 * (2 * ((-1) ** m1) + 3)
                    ) * L**3 * np.cos(
                        np.pi * (n + p)
                    ) / (
                        np.pi**2 * (n + p) ** 2
                    )
    return s


def i13_mat(Npsi, Nphi, L):
    s = np.zeros((Npsi, Nphi, Nphi))
    for m in range(1, Npsi + 1):
        for n in range(1, Nphi + 1):
            for p in range(1, Nphi + 1):
                if n == p:
                    s[m - 1, n - 1, p - 1] = L / 2.0
    return -s


def compute_partial_integrals(
    Npsi,
    Nphi,
    Lx,
    Ly,
):
    """
    Precompute and store all partial-integral matrices needed.
    Returns them in a dictionary or a custom object.
    """
    partials = {}
    partials["i1_Lx"] = i1_mat(Npsi, Nphi, Lx)
    partials["i2_Lx"] = i2_mat(Npsi, Nphi, Lx)
    partials["i3_Lx"] = i3_mat(Npsi, Nphi, Lx)
    partials["i4_Lx"] = i4_mat(Npsi, Nphi, Lx)
    partials["i5_Lx"] = i5_mat(Npsi, Nphi, Lx)

    partials["i9_Lx"] = i9_mat(Npsi, Nphi, Lx)
    partials["i10_Lx"] = i10_mat(Npsi, Nphi, Lx)
    partials["i11_Lx"] = i11_mat(Npsi, Nphi, Lx)
    partials["i12_Lx"] = i12_mat(Npsi, Nphi, Lx)
    partials["i13_Lx"] = i13_mat(Npsi, Nphi, Lx)

    # For Ly-based:
    partials["i1_Ly"] = i1_mat(Npsi, Nphi, Ly)
    partials["i2_Ly"] = i2_mat(Npsi, Nphi, Ly)
    partials["i3_Ly"] = i3_mat(Npsi, Nphi, Ly)
    partials["i4_Ly"] = i4_mat(Npsi, Nphi, Ly)
    partials["i5_Ly"] = i5_mat(Npsi, Nphi, Ly)

    partials["i9_Ly"] = i9_mat(Npsi, Nphi, Ly)
    partials["i10_Ly"] = i10_mat(Npsi, Nphi, Ly)
    partials["i11_Ly"] = i11_mat(Npsi, Nphi, Ly)
    partials["i12_Ly"] = i12_mat(Npsi, Nphi, Ly)
    partials["i13_Ly"] = i13_mat(Npsi, Nphi, Ly)

    return partials


def build_s_matrix(Npsi, Nphi, partials, idx_array, factor_mode):
    """
    Summation of partial integrals in a 3D array with a factor.

    Returns
    -------
    s : np.ndarray of shape (Npsi, Nphi, Nphi)
        The 3D tensor after summation and factor application.
    """
    s = np.zeros((Npsi, Nphi, Nphi), dtype=float)

    for m in range(Npsi):
        for n in range(Nphi):
            n_idx = idx_array[n]
            for p in range(Nphi):
                p_idx = idx_array[p]

                val = 0.0
                for mat in partials:
                    val += mat[m, n_idx - 1, p_idx - 1]

                # Apply factor
                if factor_mode == "n":
                    val *= n_idx**2
                elif factor_mode == "p":
                    val *= p_idx**2
                elif factor_mode == "np":
                    val *= n_idx * p_idx

                s[m, n, p] = val

    return s


def g1(Npsi, Nphi, S, kx, cache):
    partials = [
        cache["i1_Lx"],
        cache["i2_Lx"],
        cache["i3_Lx"],
        cache["i4_Lx"],
        cache["i5_Lx"],
    ]
    s = build_s_matrix(Npsi, Nphi, partials, idx_array=kx, factor_mode="n")
    s_reshaped = s.reshape((Npsi, Nphi**2), order="F")
    m_mat = np.repeat(s_reshaped, Npsi, axis=0)
    return m_mat[:S, :]


def g2(Npsi, Nphi, S, kx, cache):
    partials = [
        cache["i1_Lx"],
        cache["i2_Lx"],
        cache["i3_Lx"],
        cache["i4_Lx"],
        cache["i5_Lx"],
    ]
    s = build_s_matrix(Npsi, Nphi, partials, idx_array=kx, factor_mode="p")
    s_reshaped = s.reshape((Npsi, Nphi**2), order="F")
    m_mat = np.repeat(s_reshaped, Npsi, axis=0)
    return m_mat[:S, :]


def g3(Npsi, Nphi, S, ky, cache):
    partials = [
        cache["i1_Ly"],
        cache["i2_Ly"],
        cache["i3_Ly"],
        cache["i4_Ly"],
        cache["i5_Ly"],
    ]
    s = build_s_matrix(Npsi, Nphi, partials, idx_array=ky, factor_mode="n")
    s_reshaped = s.reshape((Npsi, Nphi**2), order="F")
    m_mat = np.tile(s_reshaped, (Npsi, 1))
    return m_mat[:S, :]


def g4(Npsi, Nphi, S, ky, cache):
    partials = [
        cache["i1_Ly"],
        cache["i2_Ly"],
        cache["i3_Ly"],
        cache["i4_Ly"],
        cache["i5_Ly"],
    ]
    s = build_s_matrix(Npsi, Nphi, partials, idx_array=ky, factor_mode="p")
    s_reshaped = s.reshape((Npsi, Nphi**2), order="F")
    m_mat = np.tile(s_reshaped, (Npsi, 1))
    return m_mat[:S, :]


def g5(Npsi, Nphi, S, kx, cache):
    partials = [
        cache["i9_Lx"],
        cache["i10_Lx"],
        cache["i11_Lx"],
        cache["i12_Lx"],
        cache["i13_Lx"],
    ]
    s = build_s_matrix(Npsi, Nphi, partials, idx_array=kx, factor_mode="np")
    s_reshaped = s.reshape((Npsi, Nphi**2), order="F")
    m_mat = np.repeat(s_reshaped, Npsi, axis=0)
    return m_mat[:S, :]


def g6(Npsi, Nphi, S, ky, cache):
    partials = [
        cache["i9_Ly"],
        cache["i10_Ly"],
        cache["i11_Ly"],
        cache["i12_Ly"],
        cache["i13_Ly"],
    ]
    s = build_s_matrix(Npsi, Nphi, partials, idx_array=ky, factor_mode="np")
    s_reshaped = s.reshape((Npsi, Nphi**2), order="F")
    m_mat = np.tile(s_reshaped, (Npsi, 1))
    return m_mat[:S, :]


def H_tensor_rectangular(
    coeff0: np.ndarray,
    coeff1: np.ndarray,
    coeff2: np.ndarray,
    Nphi: int,
    Npsi: int,
    Lx: float,
    Ly: float,
    kx: np.ndarray,
    ky: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Compute nonlinear coupling H-tensor for rectangular plates.

    Calculates the three-dimensional tensor representing nonlinear coupling
    between transverse and in-plane modes in rectangular plate dynamics.
    The H-tensor quantifies the quadratic nonlinearity in the von Kármán equations.

    Parameters
    ----------
    coeff0 : numpy.ndarray
        Base Airy stress coefficients, shape (S, S)
    coeff1 : numpy.ndarray
        First normalized Airy stress coefficients, shape (S, S)
    coeff2 : numpy.ndarray
        Second normalized Airy stress coefficients, shape (S, S)
    Nphi : int
        Number of transverse modes
    Npsi : int
        Number of in-plane modes in each direction
    Lx : float
        Length of the plate in x-direction
    Ly : float
        Length of the plate in y-direction
    kx : numpy.ndarray
        Mode indices in x-direction for transverse modes, shape (Nphi,)
    ky : numpy.ndarray
        Mode indices in y-direction for transverse modes, shape (Nphi,)

    Returns
    -------
    H0 : numpy.ndarray
        H-tensor computed with coeff0, shape (S, Nphi²)
    H1 : numpy.ndarray
        H-tensor computed with coeff1, shape (S, Nphi²)
    H2 : numpy.ndarray
        H-tensor computed with coeff2, shape (S, Nphi²)

    Notes
    -----
    The H-tensor represents the coupling between transverse and in-plane vibrations
    in the von Kármán plate equations. Each tensor component is computed as:

    $$H_{ijk} = \frac{4\pi^4}{L_x^3 L_y^3} \sum_{\alpha} \text{coeff}_{\alpha i}
    \left(m_1 m_4 + m_2 m_3 - 2 m_5 m_6\right)_{jk}$$

    where $m_1, m_2, ..., m_6$ are integral matrices computed from the g-functions
    and the coupling between mode indices $(j,k)$ and in-plane mode $i$.
    """
    S = coeff1.shape[1]  # Number of modes

    # Initialize H tensors
    H0 = np.zeros((S, Nphi * Nphi))
    H1 = np.zeros((S, Nphi * Nphi))
    H2 = np.zeros((S, Nphi * Nphi))

    partials = compute_partial_integrals(Npsi, Nphi, Lx, Ly)

    # Compute integral components
    m1 = g1(Npsi, Nphi, S, kx, partials)
    m2 = g2(Npsi, Nphi, S, kx, partials)

    m3 = g3(Npsi, Nphi, S, ky, partials)
    m4 = g4(Npsi, Nphi, S, ky, partials)

    m5 = g5(Npsi, Nphi, S, kx, partials)
    m6 = g6(Npsi, Nphi, S, ky, partials)

    # Compute H tensor
    for n in range(S):
        f0 = coeff0[:, n].T @ (m1 * m4 + m2 * m3 - 2 * m5 * m6)
        f1 = coeff1[:, n].T @ (m1 * m4 + m2 * m3 - 2 * m5 * m6)
        f2 = coeff2[:, n].T @ (m1 * m4 + m2 * m3 - 2 * m5 * m6)

        H0[n, :] = f0
        H1[n, :] = f1
        H2[n, :] = f2

    # Normalize with constants
    const_factor = 4 * np.pi**4 / (Lx**3 * Ly**3)
    H0 *= const_factor
    H1 *= const_factor
    H2 *= const_factor

    # Zero small values
    threshold = 1e-8
    H0[np.abs(H0 / np.max(np.abs(H0))) < threshold] = 0
    H1[np.abs(H1 / np.max(np.abs(H1))) < threshold] = 0
    H2[np.abs(H2 / np.max(np.abs(H2))) < threshold] = 0
    return H0, H1, H2


def compute_coupling_matrix(
    n_psi,
    n_phi,
    lx,
    ly,
    kx_indices,
    ky_indices,
):
    K, M = assemble_K_and_M(n_psi, lx, ly)

    # this will give different results than in MATLAB
    vals, vecs = eig(K, M)

    coeff0, coeff1, coeff2 = airy_stress_coefficients(n_psi, vals, vecs, lx, ly)
    H0, H1, H2 = H_tensor_rectangular(
        coeff0=coeff0,
        coeff1=coeff1,
        coeff2=coeff2,
        Npsi=n_psi,
        Nphi=n_phi,
        Lx=lx,
        Ly=ly,
        kx=kx_indices,
        ky=ky_indices,
    )

    H0 = H0[:n_psi, : n_phi * n_phi]
    H1 = H1[:n_psi, : n_phi * n_phi]
    H2 = H2[:n_psi, : n_phi * n_phi]

    H0 = H0.reshape((n_psi, n_phi, n_phi))
    H1 = H1.reshape((n_psi, n_phi, n_phi))
    H2 = H2.reshape((n_psi, n_phi, n_phi))
    return H0, H1, H2


# =============================================================================
# Circular Plate Functions
# =============================================================================


def cos_cos_cos_integration(
    k: int | np.ndarray,
    l: int | np.ndarray,
    m: int | np.ndarray,
) -> float | np.ndarray:
    r"""
    Compute the analytical integral ∫₀²π cos(kθ)cos(lθ)cos(mθ) dθ.

    Evaluates the analytical integral of the product of three cosine functions
    over the interval [0, 2π]. This function is essential for computing angular
    coupling coefficients in circular plate dynamics.

    Parameters
    ----------
    k : int or numpy.ndarray
        First mode number (can be scalar or array)
    l : int or numpy.ndarray
        Second mode number (can be scalar or array)
    m : int or numpy.ndarray
        Third mode number (can be scalar or array)

    Returns
    -------
    float or numpy.ndarray
        Value of the integral. Returns:
        - 2π if k = l = m = 0
        - π if one parameter is 0 and the other two are equal (and non-zero)
        - π/2 if m = |l±k| (and not covered by above cases)
        - 0 otherwise

    Notes
    -----
    This integral arises from the orthogonality relationships of trigonometric
    functions and is computed using analytical formulas rather than numerical
    integration for exact results.

    The analytical result follows from:
    cos(a)cos(b)cos(c) = ¼[cos(a+b+c) + cos(a+b-c) + cos(a-b+c) + cos(-a+b+c)]

    Examples
    --------
    >>> cos_cos_cos_integration(0, 0, 0)
    6.283185307179586  # 2π
    >>> cos_cos_cos_integration(1, 1, 0)
    3.141592653589793  # π
    >>> cos_cos_cos_integration(1, 2, 3)
    1.5707963267948966  # π/2
    """
    # Ensure inputs are numpy arrays for vectorized operations
    k = np.asarray(k)
    l = np.asarray(l)
    m = np.asarray(m)

    # Initialize result with zeros
    S = np.zeros_like(k, dtype=float)

    # Case 1: m = |l±k| (fundamental coupling condition)
    condition1 = (m == l + k) | (m == np.abs(l - k))
    S = np.where(condition1, np.pi / 2, S)

    # Case 2: One parameter is 0 and the other two are equal (and non-zero)
    condition2 = ((k == 0) & (l == m)) | ((l == 0) & (k == m)) | ((m == 0) & (k == l))
    S = np.where(condition2, np.pi, S)

    # Case 3: All parameters are zero
    condition3 = (k == 0) & (l == 0) & (m == 0)
    S = np.where(condition3, 2 * np.pi, S)

    # Return scalar if input was scalar
    if S.shape == ():
        return float(S)
    return S


def cos_sin_sin_integration(
    k: int | np.ndarray, l: int | np.ndarray, m: int | np.ndarray
) -> float | np.ndarray:
    r"""
    Compute the analytical integral ∫₀²π cos(kθ)sin(lθ)sin(mθ) dθ.

    Evaluates the analytical integral of the product of one cosine and two sine
    functions over the interval [0, 2π]. This function is essential for computing
    angular coupling coefficients in circular plate dynamics.

    Parameters
    ----------
    k : int or numpy.ndarray
        Mode number for cosine term (can be scalar or array)
    l : int or numpy.ndarray
        Mode number for first sine term (can be scalar or array)
    m : int or numpy.ndarray
        Mode number for second sine term (can be scalar or array)

    Returns
    -------
    float or numpy.ndarray
        Value of the integral. Returns:
        - π if k = 0 and l = m ≠ 0
        - π/2 if k = |l-m| and l ≠ m ≠ 0
        - -π/2 if k = l+m and l ≠ 0, m ≠ 0
        - 0 otherwise

    Notes
    -----
    This integral arises from the orthogonality relationships of trigonometric
    functions and is computed using analytical formulas. The result follows from:
    cos(a)sin(b)sin(c) = ¼[sin(b+c-a) - sin(b-c-a) + sin(c-b-a) - sin(-b-c-a)]

    The sign differences compared to cos_cos_cos_integration reflect the different
    parity properties of sine and cosine functions.

    Examples
    --------
    >>> cos_sin_sin_integration(0, 1, 1)
    3.141592653589793  # π
    >>> cos_sin_sin_integration(1, 2, 1)
    1.5707963267948966  # π/2
    >>> cos_sin_sin_integration(3, 1, 2)
    -1.5707963267948966  # -π/2
    """
    # Ensure inputs are numpy arrays for vectorized operations
    k = np.asarray(k)
    l = np.asarray(l)
    m = np.asarray(m)

    # Initialize result with zeros
    S = np.zeros_like(k, dtype=float)

    # Case 1: k = |l-m| and l ≠ m ≠ 0
    condition1 = (k == np.abs(l - m)) & (l != m) & (l != 0) & (m != 0)
    S = np.where(condition1, np.pi / 2, S)

    # Case 2: k = l+m and l ≠ 0, m ≠ 0
    condition2 = (k == l + m) & (l != 0) & (m != 0)
    S = np.where(condition2, -np.pi / 2, S)

    # Case 3: k = 0 and l = m ≠ 0
    condition3 = (k == 0) & (l == m) & (l != 0)
    S = np.where(condition3, np.pi, S)

    # Return scalar if input was scalar
    if S.shape == ():
        return float(S)
    return S


def hcoefficient_circular(
    kp: int,
    kq: int,
    cp: int,
    cq: int,
    xip: float,
    xiq: float,
    ki: int,
    ci: int,
    zeta: float,
    nu: float,
    KR: float,
    dr_H: float,
) -> float:
    r"""
    Compute H^i_pq tensor coefficient for circular plates.

    Calculates the coupling coefficient H^i_pq = ∫_S Ψ_i L(Φ_p,Φ_q) dS
    for circular plate dynamics, where Ψ_i are in-plane modes, Φ_p,Φ_q are
    transverse modes, and L is the nonlinear operator from von Kármán theory.

    Parameters
    ----------
    kp, kq : int
        Number of nodal diameters for transverse modes p and q
    cp, cq : int
        Configuration of transverse modes (1=cos, 2=sin)
    xip, xiq : float
        Square root of angular eigenfrequencies for transverse modes p and q
    ki : int
        Number of nodal diameters for in-plane mode i
    ci : int
        Configuration of in-plane mode (1=cos, 2=sin)
    zeta : float
        Square root of angular eigenfrequency for in-plane mode i
    nu : float
        Poisson ratio
    KR : float
        Normalized rotational stiffness (KR = Kr/D)
    dr_H : float
        Integration step size for radial integration

    Returns
    -------
    float
        Value of the H^i_pq coupling coefficient

    Notes
    -----
    The coefficient is computed through radial integration involving:
    - Bessel function solutions for plate bending (J_k, I_k)
    - Boundary condition enforcement via KR parameter
    - Mode shape normalization
    - Angular coupling through trigonometric integrals

    The function returns 0 if the angular coupling conditions are not satisfied:
    - ki must equal kp±kq or |kp-kq|
    - Specific combinations of cos/sin configurations lead to zero coupling
    """

    # Check angular coupling conditions - MATLAB compatibility
    # ki must equal kp+kq or |kp-kq|
    valid_ki_values = [kp + kq, abs(kp - kq)]
    if ki not in valid_ki_values:
        return 0.0

    # Specific combinations of cos/sin configurations lead to zero coupling:
    # - If cp == cq (both transverse modes same type) AND ci == 2 (in-plane is sin)
    # - If cp != cq (transverse modes different types) AND ci == 1 (in-plane is cos)
    if ((cp == cq) and (ci == 2)) or ((cp != cq) and (ci == 1)):
        return 0.0

    # Set up radial grid
    rr = np.arange(0, 1 + dr_H, dr_H)
    rr[0] = 1e-8  # Avoid singularity at r=0

    # Initialize arrays for transverse mode calculations
    R = np.zeros((2, len(rr)))
    dR = np.zeros((2, len(rr)))
    ddR = np.zeros((2, len(rr)))

    # Calculate R, dR, ddR for both transverse modes (p and q)
    kk = [kp, kq]
    xi = [xip, xiq]

    for ii in range(2):
        xkn = xi[ii]
        k = kk[ii]

        # Bessel functions and derivatives
        JJ0 = jv(k, xkn * rr)
        JJ1 = jv(k + 1, xkn * rr)
        dJJ0 = -JJ1 * xkn + k / rr * JJ0
        ddJJ0 = -(xkn**2 + k / rr**2 - k**2 / rr**2) * JJ0 + xkn / rr * JJ1

        II0 = iv(k, xkn * rr)
        II1 = iv(k + 1, xkn * rr)
        dII0 = II1 * xkn + k / rr * II0
        ddII0 = (xkn**2 - k / rr**2 + k**2 / rr**2) * II0 - xkn / rr * II1

        # Apply boundary conditions
        if np.isinf(KR):  # Clamped boundary conditions
            Jkn = jv(k, xkn)
            Ikn = iv(k, xkn)

            R[ii, :] = Ikn * JJ0 - Jkn * II0
            dR[ii, :] = Ikn * dJJ0 - Jkn * dII0
            ddR[ii, :] = Ikn * ddJJ0 - Jkn * ddII0

        else:  # Free/elastic boundary conditions
            # Compute boundary condition coefficients
            # Note: MATLAB computes negative order Bessel functions directly
            J2 = jv(k - 2, xkn)
            J1 = jv(k - 1, xkn)
            J0 = jv(k, xkn)
            I2 = iv(k - 2, xkn)
            I1 = iv(k - 1, xkn)
            I0 = iv(k, xkn)

            Jtild = (
                xkn**2 * J2
                + ((nu - 2 * k + 1) * xkn + KR) * J1
                + (k * (k + 1) * (1 - nu) - KR * k) * J0
            )
            Itild = (
                xkn**2 * I2
                + ((nu - 2 * k + 1) * xkn + KR) * I1
                + (k * (k + 1) * (1 - nu) - KR * k) * I0
            )

            R[ii, :] = Itild * JJ0 - Jtild * II0
            dR[ii, :] = Itild * dJJ0 - Jtild * dII0
            ddR[ii, :] = Itild * ddJJ0 - Jtild * ddII0

        # Normalize R (∫ phi^2 dS = 1)
        rR2 = rr * R[ii, :] ** 2
        integral = np.trapezoid(rR2, x=rr)  # MATLAB: trapz(rr, rR2)

        if integral <= 0 or not np.isfinite(integral):
            # Skip normalization for problematic cases - return zero coupling
            return 0.0

        Kkn = np.sqrt(1.0 / integral)

        if k == 0:
            Kkn = Kkn / np.sqrt(2 * np.pi)
        else:
            Kkn = Kkn / np.sqrt(np.pi)

        R[ii, :] *= Kkn
        dR[ii, :] *= Kkn
        ddR[ii, :] *= Kkn

    # Calculate S, S', S'' for in-plane mode
    if np.isinf(KR):  # Clamped boundary conditions
        # Note: MATLAB computes negative order Bessel functions directly
        J2 = jv(ki - 2, zeta)
        J1 = jv(ki - 1, zeta)
        J0 = jv(ki, zeta)

        I2 = iv(ki - 2, zeta)
        I1 = iv(ki - 1, zeta)
        I0 = iv(ki, zeta)

        Jtild = (
            zeta**2 * J2
            + (-nu - 2 * ki + 1) * zeta * J1
            + (ki * (ki + 1) + nu * ki * (1 - ki)) * J0
        )
        Itild = (
            zeta**2 * I2
            + (-nu - 2 * ki + 1) * zeta * I1
            + (ki * (ki + 1) + nu * ki * (1 - ki)) * I0
        )

        J = jv(ki, zeta * rr)
        I = iv(ki, zeta * rr)

        S = Itild * J - Jtild * I

    else:  # Free/elastic boundary conditions
        J = jv(ki, zeta)
        I = iv(ki, zeta)

        JJ0 = jv(ki, zeta * rr)
        II0 = iv(ki, zeta * rr)

        S = I * JJ0 - J * II0

    # Normalize S (∫ psi^2 dS = 1)
    rS2 = rr * S**2
    integral_S = np.trapezoid(rS2, x=rr)

    if integral_S <= 0 or not np.isfinite(integral_S):
        # Skip normalization for problematic cases - return zero coupling
        return 0.0

    Lkn = np.sqrt(1.0 / integral_S)

    if ki == 0:
        Lkn = Lkn / np.sqrt(2 * np.pi)
    else:
        Lkn = Lkn / np.sqrt(np.pi)

    S = Lkn * S

    # Compute the coefficient components
    fctH1 = S * (
        ddR[0, :] * (dR[1, :] - kq**2 * R[1, :] / rr)
        + ddR[1, :] * (dR[0, :] - kp**2 * R[0, :] / rr)
    )
    fctH2 = S * (dR[0, :] - R[0, :] / rr) * (dR[1, :] - R[1, :] / rr) / rr

    # Radial integration
    H1 = np.trapezoid(fctH1, x=rr)
    H2 = np.trapezoid(fctH2, x=rr)

    # Angular integration (theta-dependent term)
    if cp == 1:  # p is cos mode
        if cq == 1:  # q is cos mode
            beta1 = cos_cos_cos_integration(kp, kq, ki)
            beta2 = cos_sin_sin_integration(ki, kp, kq)
        else:  # q is sin mode
            beta1 = cos_sin_sin_integration(kp, kq, ki)
            beta2 = -cos_sin_sin_integration(kq, kp, ki)
    else:  # p is sin mode
        if cq == 1:  # q is cos mode
            beta1 = cos_sin_sin_integration(kq, kp, ki)
            beta2 = -cos_sin_sin_integration(kp, kq, ki)
        else:  # q is sin mode
            beta1 = cos_sin_sin_integration(ki, kp, kq)
            beta2 = cos_cos_cos_integration(kq, kp, ki)

    # Final result
    H = H1 * beta1 - 2 * kp * kq * H2 * beta2

    return H


def H_tensor_circular(
    mode_t: np.ndarray,
    mode_l: np.ndarray,
    Nphi: int,
    Npsi: int,
    nu: float,
    KR: float,
    dr_H: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Compute H-tensor matrices for circular plates.

    Calculates the three H-tensor matrices (H0, H1, H2) that represent the
    nonlinear coupling between transverse and in-plane modes in circular plate
    dynamics according to von Kármán theory.

    Parameters
    ----------
    mode_t : numpy.ndarray
        Transverse mode data with shape (Nphi, 6) containing:
        [Index, xi, k, n, c, xi^2] where:
        - Index: mode number (1-based)
        - xi: square root of eigenfrequency
        - k: number of nodal diameters
        - n: number of nodal circles
        - c: configuration (1=cos, 2=sin)
        - xi^2: eigenfrequency
    mode_l : numpy.ndarray
        In-plane mode data with shape (Npsi, 6) containing:
        [Index, zeta, k, n, c, zeta^2] with similar structure
    Nphi : int
        Number of transverse modes to compute
    Npsi : int
        Number of in-plane modes to compute
    nu : float
        Poisson ratio
    KR : float
        Normalized rotational stiffness (KR = Kr/D)
        Use KR = 0 for free boundaries, KR = inf for clamped boundaries
    dr_H : float
        Integration step size for radial integration

    Returns
    -------
    H0 : numpy.ndarray
        Base H-tensor of shape (Npsi, Nphi, Nphi)
    H1 : numpy.ndarray
        H-tensor normalized by in-plane eigenfrequency: H1 = H0/zeta_i^2
    H2 : numpy.ndarray
        H-tensor normalized by in-plane eigenfrequency squared: H2 = H0/zeta_i^4

    Notes
    -----
    The H-tensor quantifies the quadratic nonlinearity in circular plate dynamics:
    H^i_pq represents the coupling strength between in-plane mode i and the
    quadratic interaction of transverse modes p and q.

    The computation involves:
    - Radial integration using Bessel function mode shapes
    - Angular integration using trigonometric orthogonality
    - Proper handling of boundary conditions through KR parameter
    - Mode shape normalization

    For computational efficiency, only the upper triangular part is computed
    and symmetry H^i_pq = H^i_qp is used to fill the full tensor.

    Examples
    --------
    >>> # For clamped circular plate
    >>> H0, H1, H2 = H_tensor_circular(mode_t, mode_l, 10, 5, 0.3, np.inf, 0.01)
    >>> print(f"H0 shape: {H0.shape}")  # (5, 10, 10)
    """
    # Extract mode parameters
    # mode_t columns: [Index, xi, k, n, c, xi^2]
    k_t = mode_t[:Nphi, 2].astype(int)  # Number of nodal diameters (0-based indexing)
    c_t = mode_t[:Nphi, 4].astype(int)  # cos(1)/sin(2) mode
    xi_t = mode_t[:Nphi, 1]  # Square root of eigenfrequency

    # mode_l columns: [Index, zeta, k, n, c, zeta^2]
    k_l = mode_l[:Npsi, 2].astype(int)  # Number of nodal diameters
    c_l = mode_l[:Npsi, 4].astype(int)  # cos(1)/sin(2) mode
    zeta_l = mode_l[:Npsi, 1]  # Square root of eigenfrequency

    # Initialize H tensors
    H0 = np.zeros((Npsi, Nphi, Nphi))
    H1 = np.zeros((Npsi, Nphi, Nphi))
    H2 = np.zeros((Npsi, Nphi, Nphi))

    # Compute H tensor elements
    # Only compute upper triangular part and use symmetry
    for p in range(Nphi):
        for q in range(p, Nphi):  # Start from p to utilize symmetry
            for i in range(Npsi):
                # Compute H^i_pq coefficient
                H0_val = hcoefficient_circular(
                    kp=k_t[p],
                    kq=k_t[q],
                    cp=c_t[p],
                    cq=c_t[q],
                    xip=xi_t[p],
                    xiq=xi_t[q],
                    ki=k_l[i],
                    ci=c_l[i],
                    zeta=zeta_l[i],
                    nu=nu,
                    KR=KR,
                    dr_H=dr_H,
                )

                # Fill tensor using correct indexing: H0[i, p, q] and H0[i, q, p]
                # Shape is (Npsi, Nphi, Nphi) so first index is in-plane mode
                H0[i, p, q] = H0_val
                if p != q:  # Only fill symmetric element if not on diagonal
                    H0[i, q, p] = H0_val

                # Compute normalized versions
                if zeta_l[i] != 0:  # Avoid division by zero
                    H1[i, p, q] = H0_val / (zeta_l[i] ** 2)
                    if p != q:
                        H1[i, q, p] = H0_val / (zeta_l[i] ** 2)

                    H2[i, p, q] = H0_val / (zeta_l[i] ** 4)
                    if p != q:
                        H2[i, q, p] = H0_val / (zeta_l[i] ** 4)

    return H0, H1, H2


def find_zeros(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Find zeros of a function using linear interpolation between sign changes.

    Locates the roots of function f(x) by detecting sign changes and using
    linear interpolation to estimate the zero crossings. This is used for
    finding eigenvalue zeros in characteristic equations.

    Parameters
    ----------
    x : numpy.ndarray
        Vector of x-coordinates (abscissa points)
    f : numpy.ndarray
        Function values evaluated at x points

    Returns
    -------
    numpy.ndarray
        Array of x-values where f(x) ≈ 0

    Notes
    -----
    The algorithm:
    1. Detects sign changes in consecutive function values
    2. Uses linear interpolation between sign-change points to estimate zeros
    3. Formula: x_zero = (x1*f2 - x2*f1)/(f2 - f1) where f1, f2 have opposite signs

    This method assumes the function is reasonably smooth and changes sign
    at each root (no multiple roots with even multiplicity).
    """
    if len(x) != len(f) or len(x) < 2:
        return np.array([])

    # Shift function values to detect sign changes
    f_shifted = np.zeros_like(f)
    f_shifted[:-1] = f[1:]  # f_shifted[i] = f[i+1]
    f_shifted[-1] = 0  # Pad with zero

    # Find points where function changes sign (and is not zero)
    sign_changes = ((np.sign(f) + np.sign(f_shifted)) == 0) & (np.sign(f) != 0)
    indices = np.where(sign_changes)[0]

    if len(indices) == 0:
        return np.array([])

    # Get corresponding indices for the next points
    indices_next = indices + 1

    # Linear interpolation to find zero crossings
    # x_zero = (x1*f2 - x2*f1)/(f2 - f1)
    x1 = x[indices]
    x2 = x[indices_next]
    f1 = f[indices]
    f2 = f[indices_next]

    # Avoid division by zero
    valid = np.abs(f2 - f1) > 1e-12
    if not np.any(valid):
        return np.array([])

    zeros = (x1[valid] * f2[valid] - x2[valid] * f1[valid]) / (f2[valid] - f1[valid])

    return zeros


def sort_zeros(zeros_matrix: np.ndarray) -> np.ndarray:
    """
    Sort eigenvalue zeros and organize them into mode table format.

    Takes a matrix of zeros found for different mode numbers k and organizes
    them into a sorted mode table with appropriate mode classifications.

    Parameters
    ----------
    zeros_matrix : numpy.ndarray
        Matrix where zeros_matrix[k, n] contains the nth zero for mode number k.
        Zero entries indicate no zero found. Shape: (num_k_values, max_zeros_per_k)

    Returns
    -------
    numpy.ndarray
        Sorted mode table with columns:
        [Index, xi, k, n, c, xi^2] where:
        - Index: Sequential mode number (1-based)
        - xi: Square root of eigenfrequency (the zero value)
        - k: Number of nodal diameters (0-based)
        - n: Number of nodal circles (1-based)
        - c: Configuration (1=cos, 2=sin)
        - xi^2: Eigenfrequency (square of xi)

    Notes
    -----
    The sorting algorithm:
    1. Replaces zeros with infinity for proper sorting
    2. Finds minimum eigenvalue iteratively
    3. For k > 0, creates both cos (c=1) and sin (c=2) modes
    4. For k = 0, creates only cos modes (axisymmetric)

    The resulting table is sorted by increasing eigenvalue magnitude.
    """
    if zeros_matrix.size == 0:
        return np.array([]).reshape(0, 6)

    # Work with a copy to avoid modifying the input
    zeros_work = zeros_matrix.copy()

    # Replace zeros with infinity for proper sorting
    zeros_work[zeros_work == 0] = np.inf

    mode_table = []
    mode_index = 1

    # Process zeros in ascending order
    min_zero = np.min(zeros_work)

    while np.isfinite(min_zero):
        # Find the position of the minimum zero
        k_idx, n_idx = np.unravel_index(np.argmin(zeros_work), zeros_work.shape)

        # k is 0-based (k_idx), n is 1-based (n_idx + 1)
        k = k_idx
        n = n_idx + 1
        xi = min_zero

        # Add cos mode (c=1)
        mode_table.append([mode_index, xi, k, n, 1, xi**2])
        mode_index += 1

        # For k > 0, also add sin mode (c=2)
        if k > 0:
            mode_table.append([mode_index, xi, k, n, 2, xi**2])
            mode_index += 1

        # Mark this zero as used
        zeros_work[k_idx, n_idx] = np.inf

        # Find next minimum
        min_zero = np.min(zeros_work)

    if not mode_table:
        return np.array([]).reshape(0, 6)

    return np.array(mode_table)


def circ_plate_transverse_eigenvalues(
    dx: float,
    xmax: float,
    BC: str,
    nu: float,
    KR: float = 0.0,
    KT: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Compute transverse eigenvalues for circular plates using full plate theory.

    Solves the characteristic equations for transverse vibrations of circular plates
    with various boundary conditions. Uses Bessel function solutions and finds zeros
    of the characteristic equations to determine eigenfrequencies.

    Parameters
    ----------
    dx : float
        Precision step for numerical root finding
    xmax : float
        Upper limit for eigenvalue search (maximum √ω)
    BC : str
        Boundary conditions: "free", "clamped", or "elastic"
    nu : float
        Poisson ratio
    KR : float, optional
        Normalized rotational stiffness KR = Kr/D (for elastic BC)
        Default: 0.0
    KT : float, optional
        Normalized translational stiffness KT = Kt/D (for elastic BC)
        Default: 0.0

    Returns
    -------
    mode_t : numpy.ndarray
        Transverse mode table with columns:
        [Index, xi, k, n, c, xi^2] where:
        - Index: Sequential mode number (1-based)
        - xi: √ω (square root of angular eigenfrequency)
        - k: Number of nodal diameters (0-based)
        - n: Number of nodal circles (1-based)
        - c: Configuration (1=cos, 2=sin)
        - xi^2: ω (angular eigenfrequency)
    zeros_matrix : numpy.ndarray
        Matrix of found zeros organized by k and n indices

    Notes
    -----
    The characteristic equations solved are:

    **Free/Elastic boundaries**:
    Jtild·Itild2 - Itild·Jtild2 = 0

    where:
    - Jtild = x²J₂ + ((ν-2k+1)x + KR)J₁ + (k(k+1)(1-ν) - KRk)J₀
    - Itild = x²I₂ + ((ν-2k+1)x + KR)I₁ + (k(k+1)(1-ν) - KRk)I₀
    - Jtild2, Itild2: Similar expressions with KT terms

    **Clamped boundaries**:
    J₁I₀ - J₀I₁ = 0

    The solutions represent eigenvalues ξ² = ω of the plate equation:
    ∇⁴w - ξ²∇²w = 0 in polar coordinates with appropriate boundary conditions.

    Examples
    --------
    >>> # Clamped circular plate
    >>> mode_t, zeros = circ_plate_transverse_eigenvalues(0.01, 20.0, "clamped", 0.3)
    >>> print(f"Found {len(mode_t)} modes")

    >>> # Free circular plate
    >>> mode_t, zeros = circ_plate_transverse_eigenvalues(0.01, 20.0, "free", 0.3)
    """
    # Set boundary condition parameters
    if BC.lower() == "free":
        KR = 0.0
        KT = 0.0
    elif BC.lower() == "clamped":
        KR = np.inf
        KT = np.inf
    elif BC.lower() != "elastic":
        raise ValueError(
            f"Unknown boundary condition: {BC}. Use 'free', 'clamped', or 'elastic'"
        )

    # Create x grid for root finding
    x = np.arange(0, xmax + dx, dx)
    x = x[1:]  # Remove x=0 to avoid singularities

    zeros_list = []
    k = 0

    # Find zeros for increasing values of k until no more zeros are found
    while True:
        if BC.lower() in ["free", "elastic"]:
            # Free/elastic boundary conditions
            # Compute Bessel functions (scipy handles negative orders correctly)
            J3 = jv(k - 3, x)
            J2 = jv(k - 2, x)
            J1 = jv(k - 1, x)
            J0 = jv(k, x)

            I3 = iv(k - 3, x)
            I2 = iv(k - 2, x)
            I1 = iv(k - 1, x)
            I0 = iv(k, x)

            # Characteristic equation terms
            Jtild = (
                x**2 * J2
                + ((nu - 2 * k + 1) + KR) * x * J1
                + (k * (k + 1) * (1 - nu) - KR * k) * J0
            )
            Itild = (
                x**2 * I2
                + ((nu - 2 * k + 1) + KR) * x * I1
                + (k * (k + 1) * (1 - nu) - KR * k) * I0
            )

            Jtild2 = (
                x**3 * J3
                + (4 - 3 * k) * x**2 * J2
                + k * (k * (1 + nu) - 2) * x * J1
                + ((1 - nu) * k**2 * (1 + k) - KT) * J0
            )
            Itild2 = (
                x**3 * I3
                + (4 - 3 * k) * x**2 * I2
                + k * (k * (1 + nu) - 2) * x * I1
                + ((1 - nu) * k**2 * (1 + k) - KT) * I0
            )

            # Characteristic equation: Jtild*Itild2 - Itild*Jtild2 = 0
            f = Jtild * Itild2 - Itild * Jtild2

        elif BC.lower() == "clamped":
            # Clamped boundary conditions
            J0 = jv(k - 1, x)
            J1 = jv(k, x)

            I0 = iv(k - 1, x)
            I1 = iv(k, x)

            # Characteristic equation: J1*I0 - J0*I1 = 0
            f = J1 * I0 - J0 * I1

        # Find zeros of the characteristic equation
        zeros_k = find_zeros(x, f)

        if len(zeros_k) == 0:
            break

        zeros_list.append(zeros_k)
        k += 1

    # Organize zeros into matrix format
    if not zeros_list:
        return np.array([]).reshape(0, 6), np.array([]).reshape(0, 0)

    max_zeros = max(len(z) for z in zeros_list)
    zeros_matrix = np.zeros((len(zeros_list), max_zeros))

    for i, zeros_k in enumerate(zeros_list):
        zeros_matrix[i, : len(zeros_k)] = zeros_k

    # Sort zeros into mode table
    mode_t = sort_zeros(zeros_matrix)

    # Adjust mode numbering for free boundary conditions
    if BC.lower() == "free":
        # For free boundary, modes with k≠1 have n reduced by 1
        mask = mode_t[:, 2] != 1  # k != 1
        mode_t[mask, 3] -= 1  # n -= 1

    return mode_t, zeros_matrix


def circ_plate_inplane_eigenvalues(
    dx: float,
    xmax: float,
    BC: str,
    nu: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Compute in-plane eigenvalues for circular plates using full plate theory.

    Solves the characteristic equations for in-plane vibrations of circular plates
    with various boundary conditions. Uses Bessel function solutions and finds zeros
    of the characteristic equations to determine eigenfrequencies.

    Parameters
    ----------
    dx : float
        Precision step for numerical root finding
    xmax : float
        Upper limit for eigenvalue search (maximum √ω)
    BC : str
        Boundary conditions: "free", "clamped", or "elastic"
    nu : float
        Poisson ratio

    Returns
    -------
    mode_l : numpy.ndarray
        In-plane mode table with columns:
        [Index, zeta, k, n, c, zeta^2] where:
        - Index: Sequential mode number (1-based)
        - zeta: √ω (square root of angular eigenfrequency)
        - k: Number of nodal diameters (0-based)
        - n: Number of nodal circles (1-based)
        - c: Configuration (1=cos, 2=sin)
        - zeta^2: ω (angular eigenfrequency)
    zeros_matrix : numpy.ndarray
        Matrix of found zeros organized by k and n indices

    Notes
    -----
    The characteristic equations solved are:

    **Free/Elastic boundaries**:
    J₁I₀ - J₀I₁ = 0

    **Clamped boundaries**:
    Itild·(x³J₃ + (4-3k)x²J₂ + k(k(1-ν)-2)xJ₁ + (1+ν)k²(1+k)J₀) -
    Jtild·(x³I₃ + (4-3k)x²I₂ + k(k(1-ν)-2)xI₁ + (1+ν)k²(1+k)I₀) = 0

    where:
    - Jtild = x²J₂ + (-ν-2k+1)xJ₁ + (k(k+1) + νk(1-k))J₀
    - Itild = x²I₂ + (-ν-2k+1)xI₁ + (k(k+1) + νk(1-k))I₀

    Note: The boundary conditions are **opposite** to transverse modes:
    - Simple equation for free boundaries (in-plane) vs clamped (transverse)
    - Complex equation for clamped boundaries (in-plane) vs free (transverse)

    Examples
    --------
    >>> # Free circular plate (in-plane)
    >>> mode_l, zeros = circ_plate_inplane_eigenvalues(0.01, 20.0, "free", 0.3)
    >>> print(f"Found {len(mode_l)} in-plane modes")

    >>> # Clamped circular plate (in-plane)
    >>> mode_l, zeros = circ_plate_inplane_eigenvalues(0.01, 20.0, "clamped", 0.3)
    """
    # Set boundary condition parameters (note: no KR, KT for in-plane modes)
    if BC.lower() not in ["free", "clamped", "elastic"]:
        raise ValueError(
            f"Unknown boundary condition: {BC}. Use 'free', 'clamped', or 'elastic'"
        )

    # Create x grid for root finding
    x = np.arange(0, xmax + dx, dx)
    x = x[1:]  # Remove x=0 to avoid singularities

    zeros_list = []
    k = 0

    # Find zeros for increasing values of k until no more zeros are found
    while True:
        if BC.lower() in ["free", "elastic"]:
            # Free/elastic boundary conditions (simple equation)
            J0 = jv(k - 1, x)
            J1 = jv(k, x)

            I0 = iv(k - 1, x)
            I1 = iv(k, x)

            # Characteristic equation: J1*I0 - J0*I1 = 0
            f = J1 * I0 - J0 * I1

        elif BC.lower() == "clamped":
            # Clamped boundary conditions (complex equation)
            J3 = jv(k - 3, x)
            J2 = jv(k - 2, x)
            J1 = jv(k - 1, x)
            J0 = jv(k, x)

            I3 = iv(k - 3, x)
            I2 = iv(k - 2, x)
            I1 = iv(k - 1, x)
            I0 = iv(k, x)

            # Characteristic equation terms
            Jtild = (
                x**2 * J2
                + (-nu - 2 * k + 1) * x * J1
                + (k * (k + 1) + nu * k * (1 - k)) * J0
            )
            Itild = (
                x**2 * I2
                + (-nu - 2 * k + 1) * x * I1
                + (k * (k + 1) + nu * k * (1 - k)) * I0
            )

            # Additional terms for clamped boundary
            Jterm2 = (
                x**3 * J3
                + (4 - 3 * k) * x**2 * J2
                + k * (k * (1 - nu) - 2) * x * J1
                + (1 + nu) * k**2 * (1 + k) * J0
            )
            Iterm2 = (
                x**3 * I3
                + (4 - 3 * k) * x**2 * I2
                + k * (k * (1 - nu) - 2) * x * I1
                + (1 + nu) * k**2 * (1 + k) * I0
            )

            # Characteristic equation: Itild*Jterm2 - Jtild*Iterm2 = 0
            f = Itild * Jterm2 - Jtild * Iterm2

        # Find zeros of the characteristic equation
        zeros_k = find_zeros(x, f)

        if len(zeros_k) == 0:
            break

        zeros_list.append(zeros_k)
        k += 1

    # Organize zeros into matrix format
    if not zeros_list:
        return np.array([]).reshape(0, 6), np.array([]).reshape(0, 0)

    max_zeros = max(len(z) for z in zeros_list)
    zeros_matrix = np.zeros((len(zeros_list), max_zeros))

    for i, zeros_k in enumerate(zeros_list):
        zeros_matrix[i, : len(zeros_k)] = zeros_k

    # Sort zeros into mode table
    mode_l = sort_zeros(zeros_matrix)

    # Adjust mode numbering for free boundary conditions
    if BC.lower() in ["free", "elastic"]:
        # For free boundary in-plane modes, modes with k≠1 have n reduced by 1
        mask = mode_l[:, 2] != 1  # k != 1
        mode_l[mask, 3] -= 1  # n -= 1

    return mode_l, zeros_matrix


def evaluate_circular_modes(
    mode_t: np.ndarray,
    nu: float,
    KR: float,
    op: np.ndarray,
    BC: str,
    R: float,
    dr: float,
) -> np.ndarray:
    """Evaluate circular plate eigenfunctions at given point.

    This function computes the modal weights (eigenfunction values) for all
    circular plate modes at a specified radial and angular coordinate.

    Parameters
    ----------
    mode_t : np.ndarray
        Mode table with columns [Index, xkn, k, n, c, xkn^2] where:
        - Index: mode number (1-based)
        - xkn: square root of eigenfrequency
        - k: number of nodal diameters
        - n: number of nodal circles
        - c: configuration (1=cos, 2=sin)
        - xkn^2: eigenfrequency
    nu : float
        Poisson ratio
    KR : float
        Normalized rotational stiffness (KR = Kr/D)
    op : np.ndarray
        Output point [theta, r] where:
        - theta: angular coordinate (radians)
        - r: radial coordinate (normalized, 0 <= r <= 1)
    BC : str
        Boundary condition: 'free', 'elastic', or 'clamped'
    R : float
        Dimensional plate radius (will be normalized internally)
    dr : float
        Dimensional integration step (will be normalized internally)

    Returns
    -------
    np.ndarray
        Normalized modal weights at the specified point (Nphi x 1 vector)

    Examples
    --------
    >>> # Evaluate all modes at a single point
    >>> op = np.array([np.pi/4, 0.7])  # theta = pi/4, r = 0.7
    >>> weights = evaluate_circular_modes(mode_t, 0.3, np.inf, op, 'clamped', 0.2, dr)
    """
    # Input validation
    if len(op) != 2:
        raise ValueError("op must be a vector with 2 elements [theta, r]")

    # Extract coordinates
    theta = op[0]
    r = op[1]

    # Normalize coordinates and parameters (following MATLAB plate_def_circ.m)
    R_norm = 1.0  # R = Rd/Rd = 1
    dr_norm = dr / R  # dr = dr/Rd

    # Extract mode parameters from mode_t
    Nphi = mode_t.shape[0]
    c_t = mode_t[:, 4]  # Configuration (cos/sin) - 0-indexed in Python
    k_t = mode_t[:, 2]  # Number of nodal diameters
    xkn = mode_t[:, 1]  # Square root of eigenfrequency

    # Initialize output
    weights = np.zeros(Nphi)

    # Create radial grid for normalization (shared across all modes)
    rr = np.arange(0, R_norm + dr_norm, dr_norm)
    rr[0] = 1e-10  # Avoid singularity at r=0

    # Compute mode shape over full domain for normalization
    JJ0_norm = np.zeros((Nphi, len(rr)))
    II0_norm = np.zeros((Nphi, len(rr)))
    for i in range(Nphi):
        k = k_t[i]
        xkn_i = xkn[i]

        JJ0_norm[i] = jv(k, xkn_i * rr)
        II0_norm[i] = iv(k, xkn_i * rr)

    # Loop over all modes (following plate_def_circ.m pattern)
    if BC.lower() == "clamped":
        Jkn = jv(k_t, xkn)  # Bessel functions for all modes
        Ikn = iv(k_t, xkn)  # Bessel functions for all modes

        # Compute mode shape at the specific point
        J = jv(k_t, xkn * r)
        I = iv(k_t, xkn * r)

        # Modal deformation at output point
        rp_raw = (Ikn * J - Jkn * I) * np.cos(k_t * theta - (c_t - 1) / 2 * np.pi)

        # Compute normalization factor
        Rkn_norm = JJ0_norm * Ikn[...,None] - Jkn[...,None] * II0_norm


    elif BC.lower() in ["free", "elastic"]:
        J0 = jv(k_t, xkn)  # Bessel function J0 for all modes
        J1 = jv(k_t - 1, xkn)  # Bessel function J1 for all modes
        J2 = jv(k_t - 2, xkn)  # Bessel function J2 for all modes

        I0 = iv(k_t, xkn)  # Bessel function I0 for all modes
        I1 = iv(k_t - 1, xkn)  # Bessel function I1 for all modes
        I2 = iv(k_t - 2, xkn)  # Bessel function I2 for all modes

        Jtild = (
            xkn**2 * J2
            + ((nu - 2 * k_t + 1) * xkn + KR) * J1
            + (k_t * (k_t + 1) * (1 - nu) - KR * k_t) * J0
        )
        Itild = (
            xkn**2 * I2
            + ((nu - 2 * k_t + 1) * xkn + KR) * I1
            + (k_t * (k_t + 1) * (1 - nu) - KR * k_t) * I0
        )

        # Compute mode shape at the specific point
        JJ0 = jv(k_t, xkn * r)
        II0 = iv(k_t, xkn * r)

        # Modal deformation at output point
        rp_raw = (JJ0 - (Jtild * II0 / Itild)) * np.cos(
            k_t * theta - (c_t - 1) / 2 * np.pi
        )

        # Compute normalization factor
        Rkn_norm = JJ0_norm - (Jtild[...,None] * II0_norm / Itild[...,None])

    else:
        raise ValueError(f"Unknown boundary condition: {BC}")

    # Compute normalization integral
    rR2 = rr[None] * Rkn_norm**2
    Kkn = np.sqrt(1.0 / np.trapezoid(rR2, x=rr[None], axis=1))

    # Apply angular normalization
    # Alternative: Kkn = np.where(k_t == 0, Kkn / np.sqrt(2 * np.pi), Kkn / np.sqrt(np.pi))
    zero_indices = k_t == 0
    Kkn[zero_indices] = Kkn[zero_indices] / np.sqrt(2 * np.pi)
    Kkn[~zero_indices] = Kkn[~zero_indices] / np.sqrt(np.pi)

    # Apply normalization to the result
    weights = rp_raw * Kkn

    return weights
