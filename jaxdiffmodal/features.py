import warnings
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from scipy.signal import get_window


def nextpow2(A: float) -> int:
    r"""
    Compute the exponent of the next power of 2 greater than or equal to A.

    Parameters
    ----------
    A : float
        Input value

    Returns
    -------
    int
        Exponent $n$ such that $2^n \geq A$

    Notes
    -----
    This function computes $\lceil \log_2(A) \rceil$.
    """
    return int(np.ceil(np.log2(A)))


def broadcast_dim(x: jnp.ndarray) -> jnp.ndarray:
    """
    Broadcast input array to 3D format for CQT processing.

    Converts 1D or 2D input arrays to 3D format (batch, channels, length)
    required for CQT transform operations.

    Parameters
    ----------
    x : jax.numpy.ndarray
        Input array with shape (len,), (batch, len), or (batch, channels, len)

    Returns
    -------
    jax.numpy.ndarray
        Broadcasted array with shape (batch, channels, len)

    Raises
    ------
    ValueError
        If input array has unsupported number of dimensions
    """
    if x.ndim == 2:
        x = x[:, None, :]
    elif x.ndim == 1:
        x = x[None, None, :]
    elif x.ndim == 3:
        pass
    else:
        raise ValueError(
            "Only support input with shape = (batch, len) or shape = (len)"
        )
    return x


def get_window_dispatch(window: str | tuple | float, N: int, fftbins: bool = True) -> np.ndarray:
    """
    Get window function with enhanced Gaussian parameter handling.

    Wrapper around scipy's get_window with special handling for Gaussian
    windows to compute optimal sigma parameter from dB specification.

    Parameters
    ----------
    window : str, tuple, or float
        Window specification. Can be window name, (name, param) tuple,
        or Kaiser beta parameter
    N : int
        Window length
    fftbins : bool, default=True
        If True, create a periodic window for FFT use

    Returns
    -------
    numpy.ndarray
        Window coefficients

    Notes
    -----
    For Gaussian windows specified as ("gaussian", dB), the sigma parameter
    is automatically computed to achieve the specified dB attenuation.
    """
    if isinstance(window, str):
        return get_window(window, N, fftbins=fftbins)
    elif isinstance(window, tuple):
        if window[0] == "gaussian":
            assert window[1] >= 0
            sigma = np.floor(-N / 2 / np.sqrt(-2 * np.log(10 ** (-window[1] / 20))))
            return get_window(("gaussian", sigma), N, fftbins=fftbins)
        else:
            warnings.warn(
                "Tuple windows may have undesired behaviour regarding Q factor"
            )
            return get_window(window, N, fftbins=fftbins)
    elif isinstance(window, float):
        warnings.warn(
            "You are using Kaiser window with beta factor "
            + str(window)
            + ". Correct behaviour not checked."
        )
        return get_window(window, N, fftbins=fftbins)
    else:
        raise Exception(
            "The function get_window from scipy only supports strings, tuples and floats."
        )


def create_cqt_kernels(
    Q: float,
    fs: float,
    fmin: float,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    norm: int = 1,
    window: str = "hann",
    fmax: float | None = None,
    topbin_check: bool = True,
    gamma: float = 0,
) -> tuple[np.ndarray, int, jnp.ndarray, np.ndarray]:
    r"""
    Create Constant-Q Transform kernels in time domain.

    Generates complex-valued temporal kernels for computing the CQT.
    Each kernel corresponds to a different frequency bin with constant
    Q factor (quality factor = frequency/bandwidth).

    Parameters
    ----------
    Q : float
        Quality factor (frequency/bandwidth ratio)
    fs : float
        Sampling frequency in Hz
    fmin : float
        Minimum frequency in Hz
    n_bins : int, default=84
        Number of frequency bins (7 octaves × 12 bins/octave)
    bins_per_octave : int, default=12
        Number of frequency bins per octave
    norm : int, default=1
        Normalization mode (1 for L2 normalization)
    window : str, default="hann"
        Window function name
    fmax : float, optional
        Maximum frequency in Hz (overrides n_bins if provided)
    topbin_check : bool, default=True
        Check if highest frequency exceeds Nyquist limit
    gamma : float, default=0
        Bandwidth scaling factor

    Returns
    -------
    tuple
        (kernels, fftLen, lengths, freqs) where:
        - kernels: Complex CQT kernels, shape (n_bins, fftLen)
        - fftLen: FFT length used
        - lengths: Kernel lengths for each bin
        - freqs: Center frequencies for each bin

    Notes
    -----
    The CQT provides logarithmic frequency resolution with constant Q:
    
    $$Q = \frac{f_k}{\Delta f_k} = \text{constant}$$
    
    where $f_k$ is the center frequency and $\Delta f_k$ is the bandwidth.
    """

    if (fmax is not None) and (n_bins is None):
        n_bins = np.ceil(bins_per_octave * np.log2(fmax / fmin))
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.double(bins_per_octave))
    elif (fmax is None) and (n_bins is not None):
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.double(bins_per_octave))
    else:
        warnings.warn("If fmax is given, n_bins will be ignored", SyntaxWarning)
        n_bins = np.ceil(bins_per_octave * np.log2(fmax / fmin))
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.double(bins_per_octave))

    if np.max(freqs) > fs / 2 and topbin_check:
        raise ValueError(
            f"The top bin {np.max(freqs)}Hz has exceeded the Nyquist frequency, "
            "please reduce the n_bins"
        )

    alpha = 2.0 ** (1.0 / bins_per_octave) - 1.0
    lengths = np.ceil(Q * fs / (freqs + gamma / alpha))

    # get max window length depending on gamma value
    max_len = int(max(lengths))
    fftLen = int(2 ** (np.ceil(np.log2(max_len))))

    tempKernel = np.zeros((int(n_bins), int(fftLen)), dtype=np.complex64)

    for k in range(0, int(n_bins)):
        freq = freqs[k]
        l = int(lengths[k])

        # Centering the kernels
        if l % 2 == 1:  # pad more zeros on RHS
            start = int(np.ceil(fftLen / 2.0 - l / 2.0)) - 1
        else:
            start = int(np.ceil(fftLen / 2.0 - l / 2.0))

        window_func = get_window_dispatch(window, l, fftbins=True)
        sig = (
            window_func
            * np.exp(np.r_[-l // 2 : l // 2] * 1j * 2 * np.pi * freq / fs)
            / l
        )

        if norm:  # Normalizing the filter
            tempKernel[k, start : start + l] = sig / np.linalg.norm(sig, norm)
        else:
            tempKernel[k, start : start + l] = sig

    return tempKernel, fftLen, jnp.array(lengths, dtype=jnp.float32), freqs


# More efficient implementation using JAX's lax.conv_general_dilated
def conv1d_efficient(x, kernel, stride=1):
    # Get dimensions
    batch_size, channels_in, width = x.shape
    channels_out, kernel_channels, kernel_width = kernel.shape

    # Ensure kernel channels match input channels
    assert channels_in == kernel_channels, (
        f"Input channels ({channels_in}) must match kernel channels ({kernel_channels})"
    )

    # Reshape for JAX's conv_general_dilated
    # Move channel dimension for proper convolution
    x = x.transpose(0, 2, 1)  # [batch, width, channels_in]

    # Reshape kernel: [out_channels, in_channels, kernel_width] -> [kernel_width, in_channels, out_channels]
    kernel = kernel.transpose(2, 1, 0)  # [kernel_width, in_channels, out_channels]

    # Define dimension numbers for 1D convolution
    dimension_numbers = lax.ConvDimensionNumbers(
        lhs_spec=(0, 2, 1),  # batch, features, spatial dims
        rhs_spec=(2, 1, 0),  # output features, input features, spatial dims
        out_spec=(0, 2, 1),  # batch, features, spatial dims
    )

    # Perform convolution
    output = lax.conv_general_dilated(
        x,  # input
        kernel,  # kernel
        (stride,),  # stride
        "VALID",  # padding
        dimension_numbers=dimension_numbers,
    )

    # Transpose back to match expected output format [batch, channels, width]
    output = output.transpose(0, 2, 1)

    return output


class CQT1992v2:
    """JAX implementation of CQT1992v2 from nnAudio."""

    def __init__(
        self,
        sr=22050,
        hop_length=512,
        fmin=32.70,
        fmax=None,
        n_bins=84,
        bins_per_octave=12,
        filter_scale=1,
        norm=1,
        window="hann",
        center=True,
        pad_mode="reflect",
        trainable=False,
        output_format="Magnitude",
    ):
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.output_format = output_format
        self.trainable = trainable

        # Creating kernels for CQT
        Q = float(filter_scale) / (2 ** (1 / bins_per_octave) - 1)

        print("Creating CQT kernels...")
        cqt_kernels, self.kernel_width, self.lengths, self.frequencies = (
            create_cqt_kernels(Q, sr, fmin, n_bins, bins_per_octave, norm, window, fmax)
        )

        # Convert to JAX arrays
        self.cqt_kernels_real = jnp.array(cqt_kernels.real)[:, None, :]
        self.cqt_kernels_imag = jnp.array(cqt_kernels.imag)[:, None, :]
        print("CQT kernels created")

    def __call__(self, x, output_format=None, normalization_type="librosa"):
        """Forward pass of the CQT transform."""
        return self.forward(x, output_format, normalization_type)

    def forward(self, x, output_format=None, normalization_type="librosa"):
        """
        Convert a batch of waveforms to CQT spectrograms.

        Parameters
        ----------
        x : jax.numpy.ndarray
            Input signal should be in either of the following shapes:
            1. (len_audio)
            2. (num_audio, len_audio)
            3. (num_audio, 1, len_audio)

        normalization_type : str
            Type of the normalisation. Options:
            'librosa' : output fits the librosa implementation
            'convolutional' : output conserves the convolutional inequalities
            'wrap' : wraps positive and negative frequencies into positive frequencies
        """
        output_format = output_format or self.output_format

        # Convert numpy arrays to JAX arrays if needed
        if not isinstance(x, jnp.ndarray):
            x = jnp.array(x)

        # Broadcast dimensions
        x = broadcast_dim(x)

        # Apply padding if center is True
        if self.center:
            pad_width = self.kernel_width // 2
            if self.pad_mode == "constant":
                x = jnp.pad(
                    x, ((0, 0), (0, 0), (pad_width, pad_width)), mode="constant"
                )
            elif self.pad_mode == "reflect":
                x = jnp.pad(x, ((0, 0), (0, 0), (pad_width, pad_width)), mode="reflect")

        # CQT computation
        # Use the efficient convolution implementation
        CQT_real = conv1d_efficient(
            x,
            self.cqt_kernels_real,
            stride=self.hop_length,
        )
        CQT_imag = -conv1d_efficient(
            x,
            self.cqt_kernels_imag,
            stride=self.hop_length,
        )

        # Apply normalization
        if normalization_type == "librosa":
            CQT_real = CQT_real * jnp.sqrt(self.lengths.reshape(-1, 1))
            CQT_imag = CQT_imag * jnp.sqrt(self.lengths.reshape(-1, 1))
        elif normalization_type == "convolutional":
            pass  # No normalization
        elif normalization_type == "wrap":
            CQT_real = CQT_real * 2
            CQT_imag = CQT_imag * 2
        else:
            raise ValueError(
                f"The normalization_type {normalization_type} is not part of our current options."
            )

        # Return the appropriate output format
        if output_format == "Magnitude":
            if not self.trainable:
                # Getting CQT Amplitude
                CQT = jnp.sqrt(CQT_real**2 + CQT_imag**2)
            else:
                CQT = jnp.sqrt(CQT_real**2 + CQT_imag**2 + 1e-8)
            return CQT

        elif output_format == "Complex":
            return jnp.stack((CQT_real, CQT_imag), axis=-1)

        elif output_format == "Phase":
            phase_real = jnp.cos(jnp.arctan2(CQT_imag, CQT_real))
            phase_imag = jnp.sin(jnp.arctan2(CQT_imag, CQT_real))
            return jnp.stack((phase_real, phase_imag), axis=-1)


# JIT-compiled version for faster execution
@partial(jax.jit, static_argnums=(1, 2))
def cqt_transform(x, cqt_instance, output_format=None):
    """JIT-compiled CQT transform function."""
    return cqt_instance.forward(x, output_format)
