import marimo

__generated_with = "0.14.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Fitting to a real (thick) plate""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""This notebook shows how to fit the physical parameters of a plate to a real recording.""")
    return


@app.cell(hide_code=True)
def _():
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import soundfile as sf
    import soxr
    from IPython.display import Audio
    from matplotlib import pyplot as plt
    from scipy.signal import butter, freqz, lfilter
    from tqdm import tqdm

    from jaxdiffmodal.ftm import (
        PlateParameters,
        plate_eigenvalues,
        plate_wavenumbers,
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
        Audio,
        PlateParameters,
        bark2hz,
        butter,
        display_audio_with_title,
        freqz,
        hz2bark,
        iir_filter_parallel,
        jax,
        jnp,
        lfilter,
        lpc_cpu_solve,
        np,
        optax,
        plate_eigenvalues,
        plate_wavenumbers,
        plt,
        safe_log,
        sf,
        soxr,
        spectral_convergence_loss,
        spectral_wasserstein,
        tf_freqz,
        to_db,
        tqdm,
    )


@app.cell(hide_code=True)
def _(PlateParameters):
    n_max_modes_x_plate = 10
    n_max_modes_y_plate = 10
    n_modes_plate_opt = 100
    sample_rate_plate_opt = 44100
    dt_plate_opt = 1 / sample_rate_plate_opt

    params_plate = PlateParameters(
        E=2e12,
        nu=0.3,
        rho=7850,
        h=5e-4,
        l1=0.2,
        l2=0.3,
        Ts0=0,
        d1=4e-2,
    )
    return (
        dt_plate_opt,
        n_max_modes_x_plate,
        n_max_modes_y_plate,
        n_modes_plate_opt,
        params_plate,
        sample_rate_plate_opt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Load a recording of a struck plate and preprocess it""")
    return


@app.cell
def _(
    butter,
    display_audio_with_title,
    freqz,
    lfilter,
    lpc_cpu_solve,
    np,
    plt,
    sample_rate_plate_opt,
    sf,
    soxr,
    to_db,
):
    stiff_string_real_plate, file_sr_plate = sf.read("audio/single.wav")
    if file_sr_plate != sample_rate_plate_opt:
        print(f"Resampling from {file_sr_plate} to {sample_rate_plate_opt}")
        stiff_string_real_plate = soxr.resample(
            stiff_string_real_plate,
            in_rate=file_sr_plate,
            out_rate=sample_rate_plate_opt,
        )

    print("The sample rate is", sample_rate_plate_opt)

    scale_plate = 1
    duration_plate = 2.0
    offset_plate = int(0.00 * sample_rate_plate_opt)
    stop_plate = int(1 * sample_rate_plate_opt)
    # ensure the audio has exactly the same length
    stiff_string_real_plate = stiff_string_real_plate[offset_plate : offset_plate + stop_plate]

    # high pass filter the audio
    b_filter, a_filter = butter(N=4, Wn=800, btype="high", fs=sample_rate_plate_opt)
    stiff_string_real_plate = lfilter(b_filter, a_filter, stiff_string_real_plate)

    # get the rfft of the real audio
    u_stiff_string_rfft_plate = np.fft.rfft(stiff_string_real_plate)

    # get the spectral envelope
    a_lpc_cpu_solve_autocorr_plate, g_lpc_solve_autocorr_plate = lpc_cpu_solve(
        stiff_string_real_plate,
        128,
        method="autocorrelation",
        biased=False,
    )

    w_plate, h_plate = freqz(
        b=g_lpc_solve_autocorr_plate,
        a=np.concatenate([[1], a_lpc_cpu_solve_autocorr_plate]),
        worN=u_stiff_string_rfft_plate.shape[0],
        fs=sample_rate_plate_opt,
    )

    # impulse response
    H_plate = g_lpc_solve_autocorr_plate / np.fft.rfft(
        np.concatenate([[1], a_lpc_cpu_solve_autocorr_plate]),
        n=sample_rate_plate_opt,
    )
    y_plate = np.fft.irfft(H_plate, n=sample_rate_plate_opt)
    y_rfft_plate = np.abs(np.fft.rfft(y_plate))

    t_plate = np.linspace(0, duration_plate, len(stiff_string_real_plate))
    fig_plate_load, ax_plate_load = plt.subplots(1, 1, figsize=(10, 5))
    ax_plate_load.set_title("RFFT and spectral envelope")
    ax_plate_load.semilogx(to_db(np.abs(u_stiff_string_rfft_plate)), label="RFFT")
    ax_plate_load.semilogx(w_plate, to_db(np.abs(h_plate)), label="LPC spectral envelope", ls="--")
    ax_plate_load.grid("both")
    ax_plate_load.legend()
    fig_plate_load.tight_layout()
    ax_plate_load

    display_audio_with_title(stiff_string_real_plate, sample_rate_plate_opt, "Original")
    display_audio_with_title(y_plate, sample_rate_plate_opt, "LPC fit and filtered")
    return (
        a_lpc_cpu_solve_autocorr_plate,
        g_lpc_solve_autocorr_plate,
        h_plate,
        stiff_string_real_plate,
        y_plate,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Sample the envelope using the bark scale.""")
    return


@app.cell
def _(
    a_lpc_cpu_solve_autocorr_plate,
    bark2hz,
    freqz,
    g_lpc_solve_autocorr_plate,
    hz2bark,
    jnp,
    np,
    sample_rate_plate_opt,
):
    hz_range_plate = np.array([200, 20000])
    barkrange_plate = hz2bark(hz_range_plate)
    worN_plate = bark2hz(np.linspace(barkrange_plate[0], barkrange_plate[1], 20000))

    w_bark_plate, h_bark_plate = freqz(
        g_lpc_solve_autocorr_plate,
        a=np.concatenate([[1], a_lpc_cpu_solve_autocorr_plate]),
        worN=worN_plate,
        fs=sample_rate_plate_opt,
    )
    target_freq_resp_plate = jnp.abs(h_bark_plate) / jnp.max(jnp.abs(h_bark_plate))
    return target_freq_resp_plate, worN_plate


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Define the initial parameters and constraints.""")
    return


@app.cell
def _(jax, jnp, n_modes_plate_opt, np):
    RANGE_BENDING_STIFFNESS_PLATE = 15000

    rng_plate = np.random.default_rng(654)
    pars_plate = {
        "bending_stiffness": rng_plate.normal(),
        "gamma_mu": jnp.linspace(5, 15, n_modes_plate_opt),
        "d1": rng_plate.normal(),
        "d3": rng_plate.normal(),
        "Ts0": rng_plate.normal(),
        "l1": rng_plate.normal(),
        "l2": rng_plate.normal(),
        "z0": rng_plate.normal(size=(n_modes_plate_opt, 1)).astype(np.float32),
        "gain": rng_plate.normal(scale=1e-3),
        "zero_radii": rng_plate.normal(size=(n_modes_plate_opt)).astype(np.float32),
        "zero_angles": rng_plate.normal(size=(n_modes_plate_opt)).astype(np.float32),
    }

    def get_bending_stiffness_plate(params):
        return jax.nn.sigmoid(params["bending_stiffness"]) * RANGE_BENDING_STIFFNESS_PLATE

    def get_l1_plate(params):
        return jax.nn.sigmoid(params["l1"])

    def get_l2_plate(params):
        return jax.nn.sigmoid(params["l2"])

    def get_z0_plate(params):
        return params["z0"]

    def get_gamma_mu_plate(params):
        return -jax.nn.relu(params["gamma_mu"])

    def get_Ts0_plate(params):
        return 0.0

    def get_gain_plate(params):
        return params["gain"]

    def get_zeros_plate(pars):
        return jax.nn.sigmoid(pars["zero_radii"]) * jnp.exp(
            2j * np.pi * jax.nn.sigmoid(pars["zero_angles"])
        )

    return (
        RANGE_BENDING_STIFFNESS_PLATE,
        get_Ts0_plate,
        get_bending_stiffness_plate,
        get_gain_plate,
        get_gamma_mu_plate,
        get_l1_plate,
        get_l2_plate,
        get_z0_plate,
        get_zeros_plate,
        pars_plate,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Simulate the plate using the initial parameters.""")
    return


@app.cell
def _(
    dt_plate_opt,
    get_Ts0_plate,
    get_bending_stiffness_plate,
    get_gamma_mu_plate,
    get_l1_plate,
    get_l2_plate,
    get_z0_plate,
    get_gain_plate,
    get_zeros_plate,
    jnp,
    n_max_modes_x_plate,
    n_max_modes_y_plate,
    n_modes_plate_opt,
    plate_eigenvalues,
    plate_wavenumbers,
    sample_rate_plate_opt,
    tf_freqz,
):
    def tf_modified_plate(
        pars,
        lambda_mu,
        dt,
    ):
        omega_mu_squared = (
            get_bending_stiffness_plate(pars) * lambda_mu**2 + get_Ts0_plate(pars) * lambda_mu
        )
        gamma_mu = get_gamma_mu_plate(pars)
        omega_mu = jnp.sqrt(omega_mu_squared - gamma_mu**2)

        # discretise
        radius = jnp.exp(gamma_mu * dt)
        real = radius * jnp.cos(omega_mu * dt)

        zeros = get_zeros_plate(pars)
        b1 = -2.0 * zeros.real
        b2 = zeros.real**2 + zeros.imag**2

        a1 = -2.0 * real
        a2 = radius**2

        ones = jnp.ones_like(lambda_mu)

        b = jnp.stack([ones, b1, b2], axis=-1)
        a = jnp.stack([ones, a1, a2], axis=-1)
        return b, a

    def simulate_membrane(pars):
        wnx, wny = plate_wavenumbers(
            n_max_modes_x_plate,
            n_max_modes_y_plate,
            get_l1_plate(pars),
            get_l2_plate(pars),
        )
        lambda_mu = plate_eigenvalues(wnx, wny).reshape(-1)
        lambda_mu = lambda_mu.reshape(-1).sort()[:n_modes_plate_opt]

        b, a = tf_modified_plate(pars, lambda_mu, dt_plate_opt)
        b = b * get_z0_plate(pars) * get_gain_plate(pars)
        h = tf_freqz(b, a, worN_plate, sample_rate_plate_opt)
        pred_freq_resp = jnp.mean(jnp.abs(h), axis=0)
        return pred_freq_resp, b, a

    return simulate_membrane, tf_modified_plate


@app.cell
def _(
    display_audio_with_title,
    iir_filter_parallel,
    jnp,
    pars_plate,
    plt,
    sample_rate_plate_opt,
    simulate_membrane,
    target_freq_resp_plate,
    to_db,
    worN_plate,
    y_plate,
):
    initial_freq_resp_plate, b_plate_init, a_plate_init = simulate_membrane(pars_plate)

    fig_plate_init, ax_plate_init = plt.subplots(1, 1, figsize=(10, 5))
    ax_plate_init.semilogx(
        worN_plate,
        to_db(target_freq_resp_plate),
        label="Target",
    )
    ax_plate_init.semilogx(
        worN_plate,
        to_db(initial_freq_resp_plate),
        label="Initial",
        ls="--",
    )
    ax_plate_init.grid(which="both")
    ax_plate_init.legend()
    ax_plate_init

    x_plate = jnp.zeros(shape=(sample_rate_plate_opt), dtype=jnp.float32)
    x_plate = x_plate.at[0].set(1.0)
    pred_imp_resp_plate_init = iir_filter_parallel(b_plate_init, a_plate_init, x_plate).mean(axis=1)

    display_audio_with_title(y_plate, sample_rate_plate_opt, "Target")
    display_audio_with_title(pred_imp_resp_plate_init, sample_rate_plate_opt, "Initial")
    return initial_freq_resp_plate, pred_imp_resp_plate_init, x_plate


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Optimise the parameters using gradient descent.""")
    return


@app.cell
def _(
    get_bending_stiffness_plate,
    get_l1_plate,
    get_l2_plate,
    jax,
    jnp,
    optax,
    pars_plate,
    safe_log,
    simulate_membrane,
    spectral_convergence_loss,
    spectral_wasserstein,
    target_freq_resp_plate,
    tqdm,
):
    iterations_plate = 30_000
    learning_rate_plate = 1e-2
    scheduler_plate = optax.cosine_onecycle_schedule(
        transition_steps=iterations_plate,
        peak_value=learning_rate_plate,
    )
    optimiser_plate = optax.chain(
        optax.clip_by_global_norm(2.0),
        optax.adam(learning_rate=scheduler_plate),
    )
    state_plate = optimiser_plate.init(pars_plate)

    @jax.jit
    def train_step_plate(pars, state):
        def loss_fn_plate(pars):
            pred_freq_resp, b, a = simulate_membrane(pars)

            log_pred_freq_resp = safe_log(pred_freq_resp)
            log_target_freq_resp = safe_log(target_freq_resp_plate)

            lin_diff = pred_freq_resp - target_freq_resp_plate
            log_diff = log_pred_freq_resp - log_target_freq_resp

            lin_l2_loss = jnp.mean(
                jnp.square(
                    lin_diff,
                ),
            )
            log_l1_loss = jnp.mean(
                jnp.abs(
                    log_diff,
                ),
            )
            sc_loss = spectral_convergence_loss(
                log_pred_freq_resp,
                log_target_freq_resp,
            )
            ot_loss = jnp.mean(
                spectral_wasserstein(
                    pred_freq_resp,
                    target_freq_resp_plate,
                    squared=True,
                    is_mag=True,
                )
            )

            return lin_l2_loss + log_l1_loss * 0.1 + sc_loss + ot_loss * 0.001

        loss_val, grads = jax.value_and_grad(loss_fn_plate)(pars)

        updates, state = optimiser_plate.update(grads, state, pars)
        pars = optax.apply_updates(pars, updates)
        return pars, state, loss_val

    # Run optimization
    pars_plate_opt = pars_plate.copy()
    state_plate_current = state_plate
    
    bar_plate = tqdm(range(iterations_plate))
    for i in bar_plate:
        pars_plate_opt, state_plate_current, loss_val = train_step_plate(pars_plate_opt, state_plate_current)
        bar_plate.set_description(
            f"Loss: {loss_val:.3f}, bending_stiffness: {get_bending_stiffness_plate(pars_plate_opt):.3f}, l1: {get_l1_plate(pars_plate_opt):.3f}, l2: {get_l2_plate(pars_plate_opt):.3f}"
        )
        
    return pars_plate_opt, train_step_plate


@app.cell
def _(
    display_audio_with_title,
    get_bending_stiffness_plate,
    get_l1_plate,
    get_l2_plate,
    iir_filter_parallel,
    initial_freq_resp_plate,
    jnp,
    pars_plate_opt,
    plt,
    sample_rate_plate_opt,
    simulate_membrane,
    target_freq_resp_plate,
    to_db,
    worN_plate,
    x_plate,
    y_plate,
):
    pred_freq_resp_plate_final, b_plate_final, a_plate_final = simulate_membrane(pars_plate_opt)

    fig_plate_final, ax_plate_final = plt.subplots(1, 1, figsize=(16, 8))
    ax_plate_final.semilogx(
        worN_plate,
        to_db(target_freq_resp_plate),
        label="Target",
    )
    ax_plate_final.semilogx(
        worN_plate,
        to_db(initial_freq_resp_plate),
        label="Initial",
        ls="--",
        alpha=0.5,
    )
    ax_plate_final.semilogx(
        worN_plate,
        to_db(pred_freq_resp_plate_final),
        label="Pred",
        ls="--",
    )
    ax_plate_final.set_xlabel("Frequency [Hz]")
    ax_plate_final.set_ylabel("Magnitude [dB]")
    ax_plate_final.grid(which="both")
    ax_plate_final.legend(loc="upper right")
    ax_plate_final

    pred_imp_resp_plate_final = iir_filter_parallel(b_plate_final, a_plate_final, x_plate).mean(axis=1)

    display_audio_with_title(y_plate, sample_rate_plate_opt, "Target")
    display_audio_with_title(pred_imp_resp_plate_final, sample_rate_plate_opt, "Optimised")

    print(f"Final parameters:")
    print(f"Bending stiffness: {get_bending_stiffness_plate(pars_plate_opt):.3f}")
    print(f"L1: {get_l1_plate(pars_plate_opt):.3f}")
    print(f"L2: {get_l2_plate(pars_plate_opt):.3f}")
    return pred_freq_resp_plate_final, pred_imp_resp_plate_final


if __name__ == "__main__":
    app.run()