# Utils


::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’ 6=‘i’}

``` python
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import RectBivariateSpline
```

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

``` python
def to_db(x):
    return 20 * jnp.log10(x)
```

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

``` python
def impulse_response(
    b: jnp.ndarray,  # numerators of the discrete transfer function
    a: jnp.ndarray,  # denominators of the discrete transfer function
    n=4410,
):
    """
    Compute the impulse response of a discrete time system

    Parameters
    ----------
    b : jnp.ndarray
        The numerator of the discrete transfer function, with shape (n_modes, n)
    a : jnp.ndarray
        The denominator of the discrete transfer function, with shape (n_modes, n)

    Returns
    -------
    jnp.ndarray
        The impulse response of the system, with shape (n_modes, n)

    """
    # sample the discrete time systems
    ret = jnp.fft.rfft(b, n=n) / jnp.fft.rfft(a, n=n)
    return jnp.fft.irfft(ret)
```

:::

::: {.cell 0=‘e’ 1=‘x’ 2=‘p’ 3=‘o’ 4=‘r’ 5=‘t’}

``` python
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
):
    """
    Computes the Modal Assurance Criterion (MAC) between reference eigenvectors and given eigenvectors.

    Parameters:
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

    Returns:
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

        # Reorder eigenvectors and eigenvalues
        eigenvectors_swapped = eigenvectors[:, swap_indices]
        eigenvalues_swapped = eigenvalues[swap_indices]

    return eigenvectors_swapped, eigenvalues_swapped
```

:::
