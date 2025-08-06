import marimo

__generated_with = "0.14.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Fitting a real plucked guitar string""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""This notebook shows how to fit the physical parameters of a guitar string to a real recording.""")
    return


@app.cell(hide_code=True)
def _():
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import soundfile as sf
    import soxr
    from matplotlib import pyplot as plt
    from scipy.signal import freqz
    from tqdm import tqdm

    from jaxdiffmodal.ftm import (
        string_eigenvalues,
    )
    from jaxdiffmodal.losses import (
        spectral_convergence_loss,
        spectral_wasserstein,
    )
    from jaxdiffmodal.lpc import lpc_cpu_solve
    from jaxdiffmodal.utils import (
        bark2hz,
        display_audio_with_title,
        hz2bark,
        iir_filter_parallel,
        safe_log,
        tf_freqz,
        to_db,
    )
    return (
        bark2hz,
        display_audio_with_title,
        freqz,
        hz2bark,
        iir_filter_parallel,
        jax,
        jnp,
        lpc_cpu_solve,
        np,
        optax,
        plt,
        safe_log,
        sf,
        soxr,
        spectral_convergence_loss,
        spectral_wasserstein,
        string_eigenvalues,
        tf_freqz,
        to_db,
        tqdm,
    )


@app.cell(hide_code=True)
def _():
    sample_rate_opt = 44100
    dt_opt = 1 / sample_rate_opt
    return dt_opt, sample_rate_opt


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Load a recording of a plucked guitar string.""")
    return


@app.cell
def _(np, sample_rate_opt, sf, soxr):
    stiff_string_real, file_sr = sf.read("docs/examples/audio/G53-50205-1111-00019.wav")

    if file_sr != sample_rate_opt:
        print(f"Resampling from {file_sr} to {sample_rate_opt}")
        stiff_string_real = soxr.resample(
            stiff_string_real,
            in_rate=file_sr,
            out_rate=sample_rate_opt,
        )

    print("The sample rate is", sample_rate_opt)

    duration_opt = 1.0
    offset_opt = int(0.00 * file_sr)
    stop_opt = int(1 * file_sr)
    stiff_string_real = stiff_string_real[offset_opt : offset_opt + stop_opt]
    u_stiff_string_rfft = np.fft.rfft(stiff_string_real)
    return stiff_string_real, u_stiff_string_rfft


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Fit the real data using LPC to get an spectral envelope.""")
    return


@app.cell
def _(
    display_audio_with_title,
    freqz,
    lpc_cpu_solve,
    np,
    plt,
    sample_rate_opt,
    stiff_string_real,
    to_db,
    u_stiff_string_rfft,
):
    a_lpc_cpu_solve_autocorr, g_lpc_solve_autocorr = lpc_cpu_solve(
        stiff_string_real,
        512,
        method="autocorrelation",
        biased=False,
    )

    w_lpc, h_lpc = freqz(
        b=g_lpc_solve_autocorr,
        a=np.concatenate([[1], a_lpc_cpu_solve_autocorr]),
        worN=u_stiff_string_rfft.shape[0],
        fs=sample_rate_opt,
    )

    # impulse response
    H_lpc = g_lpc_solve_autocorr / np.fft.rfft(
        np.concatenate([[1], a_lpc_cpu_solve_autocorr]),
        n=sample_rate_opt,
    )
    y_lpc = np.fft.irfft(H_lpc, n=sample_rate_opt)
    y_rfft_lpc = np.abs(np.fft.rfft(y_lpc))

    fig_lpc, ax_lpc = plt.subplots(1, 1, figsize=(10, 5))
    ax_lpc.semilogx(w_lpc, to_db(np.abs(u_stiff_string_rfft)), label="RFFT")
    ax_lpc.semilogx(w_lpc, to_db(np.abs(h_lpc)), label="LPC spectral envelope", ls="--")
    ax_lpc.grid("both")
    ax_lpc.legend()
    ax_lpc

    display_audio_with_title(stiff_string_real, sample_rate_opt, "Original")
    display_audio_with_title(y_lpc, sample_rate_opt, "LPC fit")
    return a_lpc_cpu_solve_autocorr, g_lpc_solve_autocorr, y_lpc


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Sample the envelope using the bark scale.""")
    return


@app.cell
def _(
    a_lpc_cpu_solve_autocorr,
    bark2hz,
    freqz,
    g_lpc_solve_autocorr,
    hz2bark,
    jnp,
    np,
    sample_rate_opt,
):
    hz_range_opt = np.array([50, 15000])
    melrange_opt = hz2bark(hz_range_opt)
    worN_opt = bark2hz(np.linspace(melrange_opt[0], melrange_opt[1], 20_000))

    w_bark, h_bark = freqz(
        g_lpc_solve_autocorr,
        a=np.concatenate([[1], a_lpc_cpu_solve_autocorr]),
        worN=worN_opt,
        fs=sample_rate_opt,
    )

    h_normalized = jnp.abs(h_bark) / jnp.max(jnp.abs(h_bark))
    return h_normalized, worN_opt


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Define the initial parameters and constraints.""")
    return


@app.cell
def _(jax, jnp, np):
    n_modes_opt = 64
    rng_opt = np.random.default_rng(654)
    pars_initial = {
        "bending_stiffness": rng_opt.normal(scale=1e-3),
        "gamma_mu": rng_opt.uniform(size=(n_modes_opt)),
        "zero_radii": rng_opt.normal(size=(n_modes_opt)).astype(np.float32),
        "zero_angles": rng_opt.normal(size=(n_modes_opt)).astype(np.float32),
        "Ts0": rng_opt.normal(scale=1e-3),
        "length": 0.65,
        "z0": rng_opt.normal(size=(n_modes_opt, 1)).astype(np.float32),
        "gain": rng_opt.normal(scale=1e-4),
    }

    def get_z0(params):
        return jax.nn.sigmoid(params["z0"])

    def get_gamma_mu(params):
        return -jax.nn.relu(params["gamma_mu"])

    def get_radii(params):
        return jax.nn.sigmoid(params["radii"])

    def get_Ts0(params):
        return jax.nn.sigmoid(params["Ts0"]) * 50_000

    def get_gain(params):
        return jax.nn.sigmoid(params["gain"]) * 0.001

    def get_length(params):
        return jax.nn.sigmoid(params["length"])

    def get_bending_stiffness(params):
        return jax.nn.sigmoid(params["bending_stiffness"]) * 10

    def get_zero_radii(params):
        return jax.nn.sigmoid(params["zero_radii"])

    def get_zero_angles(params):
        return jax.nn.sigmoid(params["zero_angles"])

    def get_zeros(pars):
        return jax.nn.sigmoid(pars["zero_radii"]) * jnp.exp(
            2j * np.pi * jax.nn.sigmoid(pars["zero_angles"])
        )

    return (
        get_Ts0,
        get_bending_stiffness,
        get_gain,
        get_gamma_mu,
        get_length,
        get_z0,
        get_zeros,
        n_modes_opt,
        pars_initial,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Simulate the string using the initial parameters.""")
    return


@app.cell
def _(
    dt_opt,
    get_Ts0,
    get_bending_stiffness,
    get_gain,
    get_gamma_mu,
    get_length,
    get_z0,
    get_zeros,
    jnp,
    n_modes_opt,
    sample_rate_opt,
    string_eigenvalues,
    tf_freqz,
    worN_opt,
):
    def tf_modified(
        pars,
        lambda_mu,
        dt,
    ):
        omega_mu_squared = (
            get_bending_stiffness(pars) * lambda_mu**2 + get_Ts0(pars) * lambda_mu
        )
        gamma_mu = get_gamma_mu(pars)
        omega_mu = jnp.sqrt(omega_mu_squared - gamma_mu**2)

        # discretise
        radius = jnp.exp(gamma_mu * dt)
        real = radius * jnp.cos(omega_mu * dt)

        zeros = get_zeros(pars)
        b1 = -2.0 * zeros.real
        b2 = zeros.real**2 + zeros.imag**2

        a1 = -2.0 * real
        a2 = radius**2

        ones = jnp.ones_like(lambda_mu)

        b = jnp.stack([ones, b1, b2], axis=-1)
        a = jnp.stack([ones, a1, a2], axis=-1)
        return b, a

    def simulate_string(pars):
        lambdas = string_eigenvalues(n_modes_opt, length=get_length(pars))
        b, a = tf_modified(pars, lambdas, dt_opt)
        b = b * get_z0(pars) * get_gain(pars)
        h = tf_freqz(b, a, worN_opt, sample_rate_opt)
        pred_freq_response = jnp.mean(jnp.abs(h), axis=0)
        return pred_freq_response, b, a

    return (simulate_string,)


@app.cell
def _(
    display_audio_with_title,
    dt_opt,
    h_normalized,
    iir_filter_parallel,
    jnp,
    np,
    pars_initial,
    plt,
    sample_rate_opt,
    simulate_string,
    stiff_string_real,
    to_db,
    worN_opt,
    y_lpc,
):
    initial_freq_response, b_initial, a_initial = simulate_string(pars_initial)
    u_stiff_string_rfft_opt = np.fft.rfft(stiff_string_real)
    fft_freqs_opt = np.fft.rfftfreq(len(stiff_string_real), dt_opt)

    target_freq_resp = h_normalized

    fig_init, ax_init = plt.subplots(1, 1, figsize=(10, 5))
    ax_init.semilogx(
        worN_opt,
        to_db(target_freq_resp),
        label="Target",
    )
    ax_init.semilogx(
        worN_opt,
        to_db(initial_freq_response),
        label="Initial",
        ls="--",
    )
    ax_init.grid(which="both")
    ax_init.legend()
    ax_init

    x_impulse = jnp.zeros(shape=(sample_rate_opt), dtype=jnp.float32)
    x_impulse = x_impulse.at[0].set(1.0)
    mean_sol_pred_initial = iir_filter_parallel(b_initial, a_initial, x_impulse).mean(axis=1)

    display_audio_with_title(y_lpc, sample_rate_opt, "Target")
    display_audio_with_title(mean_sol_pred_initial, sample_rate_opt, "Initial")
    return initial_freq_response, target_freq_resp, x_impulse


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Optimise the parameters using gradient descent.""")
    return


@app.cell
def _(
    get_Ts0,
    get_bending_stiffness,
    get_length,
    jax,
    jnp,
    optax,
    pars_initial,
    safe_log,
    simulate_string,
    spectral_convergence_loss,
    spectral_wasserstein,
    target_freq_resp,
    tqdm,
):
    iterations_opt = 20_000
    learning_rate_opt = 3e-2
    scheduler_opt = optax.cosine_onecycle_schedule(
        transition_steps=iterations_opt,
        peak_value=learning_rate_opt,
    )
    optimiser_opt = optax.chain(
        optax.clip_by_global_norm(2.0),
        optax.adam(learning_rate=scheduler_opt),
    )
    state_opt = optimiser_opt.init(pars_initial)

    def loss_fn(pars):
        pred_freq_resp, b, a = simulate_string(pars)

        log_diff = safe_log(pred_freq_resp) - safe_log(target_freq_resp)
        log_l1_loss = jnp.mean(
            jnp.abs(
                log_diff,
            ),
        )
        sc_loss = spectral_convergence_loss(
            pred_freq_resp,
            target_freq_resp,
        )
        ot_loss = jnp.mean(
            spectral_wasserstein(
                pred_freq_resp,
                target_freq_resp,
                is_mag=True,
            )
        )

        return log_l1_loss * 0.1 + sc_loss + ot_loss

    @jax.jit
    def train_step(pars, state):
        loss_val, grads = jax.value_and_grad(loss_fn)(pars)
        updates, state = optimiser_opt.update(grads, state, pars)
        pars = optax.apply_updates(pars, updates)
        return pars, state, loss_val

    # Run optimization
    pars_optimized = pars_initial.copy()
    state_current = state_opt

    bar = tqdm(range(iterations_opt))
    for i in bar:
        pars_optimized, state_current, loss_val = train_step(pars_optimized, state_current)
        bar.set_description(
            f"Loss: {loss_val:.3f}, length: {get_length(pars_optimized):.3f}, Ts0: {get_Ts0(pars_optimized):.3f}, bending stiffness: {get_bending_stiffness(pars_optimized):.3f}"
        )

    return (pars_optimized,)


@app.cell
def _(
    display_audio_with_title,
    get_Ts0,
    get_bending_stiffness,
    get_length,
    iir_filter_parallel,
    initial_freq_response,
    np,
    pars_optimized,
    plt,
    sample_rate_opt,
    simulate_string,
    stiff_string_real,
    target_freq_resp,
    to_db,
    worN_opt,
    x_impulse,
    y_lpc,
):
    pred_freq_response_final, b_final, a_final = simulate_string(pars_optimized)
    u_stiff_string_rfft_final = np.fft.rfft(stiff_string_real)
    fft_freqs_final = np.fft.rfftfreq(len(stiff_string_real), 1 / sample_rate_opt)

    fig_final, ax_final = plt.subplots(1, 1, figsize=(16, 8))
    ax_final.semilogx(
        worN_opt,
        to_db(target_freq_resp),
        label="Target",
    )
    ax_final.semilogx(
        worN_opt,
        to_db(initial_freq_response),
        label="Initial",
        ls="--",
        alpha=0.5,
    )
    ax_final.semilogx(
        worN_opt,
        to_db(pred_freq_response_final),
        label="Pred",
        ls="--",
    )
    ax_final.set_xlabel("Frequency [Hz]")
    ax_final.set_ylabel("Magnitude [dB]")
    ax_final.grid(which="both")
    ax_final.legend(loc="upper right")
    ax_final

    mean_sol_pred_final = iir_filter_parallel(b_final, a_final, x_impulse).mean(axis=1)

    display_audio_with_title(y_lpc, sample_rate_opt, "Target")
    display_audio_with_title(mean_sol_pred_final, sample_rate_opt, "Optimised")

    print(f"Final parameters:")
    print(f"Length: {get_length(pars_optimized):.3f}")
    print(f"Ts0: {get_Ts0(pars_optimized):.3f}")
    print(f"Bending stiffness: {get_bending_stiffness(pars_optimized):.3f}")
    return


if __name__ == "__main__":
    app.run()
