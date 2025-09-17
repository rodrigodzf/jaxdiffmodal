import numpy as np


def create_1d_raised_cosine(
    duration: float,
    start_time: float,
    end_time: float,
    amplitude: float,
    sample_rate: float,
):
    """
    Create a 1D raised cosine excitation with time parameters in seconds.

    Parameters
    ----------
    duration : float
        Total duration of the excitation (in seconds).
    start_time : float
        Start time of the excitation (in seconds).
    end_time : float
        End time of the excitation (in seconds).
    amplitude : float
        Amplitude of the excitation.
    sample_rate : float
        Sample rate (samples per second).

    Returns
    -------
    excitation : ndarray
        The excitation signal.
    """
    n_samples = int(duration * sample_rate)
    fex = np.zeros(n_samples)

    width_samples = round((end_time - start_time) * sample_rate)
    start_sample = round(start_time * sample_rate)

    n_indices = np.arange(2 * width_samples + 1)
    cosine_values = (
        amplitude
        / 2
        * (1 + np.cos(np.pi * (n_indices - width_samples) / width_samples))
    )

    placement_indices = n_indices + start_sample
    valid_mask = (placement_indices >= 0) & (placement_indices < n_samples)

    if np.any(valid_mask):
        fex[placement_indices[valid_mask]] = cosine_values[valid_mask]

    return fex


def create_raised_cosine(Nx, Ny, h, ctr, epsilon, wid):
    """
    Create a raised cosine function on a 2D grid.

    Parameters
        Nx (int): Number of grid points in the x-direction.
        Ny (int): Number of grid points in the y-direction.
        h (float): Grid spacing.
        ctr (tuple): Center of the raised cosine (x, y).
        epsilon (float): Scaling parameter.
        wid (float): Width of the cosine.

    Returns:
        np.ndarray: Flattened raised cosine array.
    """
    # Create the grid
    X, Y = np.meshgrid(np.arange(0, Nx + 1) * h, np.arange(0, Ny + 1) * h)

    # Compute the distance
    dist_x = (X - ctr[0]) ** 2
    dist_y = (Y - ctr[1]) ** 2
    dist = np.sqrt(dist_x + dist_y)

    # Compute the indicator function
    ind = np.sign(np.maximum(-dist + wid / 2, 0))

    # Compute the raised cosine
    rc = 0.5 * ind.T * (1 + np.cos(2 * np.pi * dist.T / wid))

    # Flatten the array
    # rc = rc.ravel()
    return rc, X, Y, dist, dist_x, dist_y


def create_pluck_modal(
    lambdas: np.ndarray,
    pluck_position: float = 0.28,
    initial_deflection: float = 0.03,
    string_length: float = 1.0,
) -> np.ndarray:
    """
    Create a pluck excitation for a string with a given length and pluck position.
    The pluck is modeled in the modal domain.

    Parameters
    ----------
    lambdas : np.ndarray
        The eigenvalues.
    pluck_position : float
        The position of the pluck in meters.
    initial_deflection : float
        The initial deflection of the string in meters.
    string_length : float
        The length of the string in meters.

    Returns
    -------
    np.ndarray
        The pluck excitation in the modal domain.
    """

    lambdas_sqrt = np.sqrt(lambdas)

    # Scaling factor for the initial deflection
    deflection_scaling = initial_deflection * (
        string_length / (string_length - pluck_position)
    )

    # Compute the coefficients
    coefficients = (
        deflection_scaling
        * np.sin(lambdas_sqrt * pluck_position)
        / (lambdas_sqrt * pluck_position)
    )
    coefficients /= lambdas_sqrt

    return coefficients
