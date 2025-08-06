import marimo

__generated_with = "0.14.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Optimising the coupling coefficients of the Von Karman plate model""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""We can also optimise the non-linearity part of the plate model, that is the coupling coefficients.""")
    return


@app.cell(hide_code=True)
def _():
    import jax
    import jax.numpy as jnp
    import jax.scipy.signal as jsig
    import numpy as np
    import optax
    from IPython.display import Audio
    from matplotlib import pyplot as plt
    from tqdm import tqdm

    from jaxdiffmodal.excitations import create_1d_raised_cosine
    from jaxdiffmodal.ftm import (
        PlateParameters,
        damping_term_simple,
        stiffness_term,
    )
    from jaxdiffmodal.losses import (
        spectral_convergence_loss,
        spectral_wasserstein,
    )
    from jaxdiffmodal.num_utils import (
        compute_coupling_matrix_numerical,
        multiresolution_eigendecomposition,
    )
    from jaxdiffmodal.sv import (
        make_vk_nl_fn,
        solve_tf_excitation,
    )
    from jaxdiffmodal.utils import compute_spectrogram, display_audio_with_title, safe_log
    return (
        Audio,
        PlateParameters,
        compute_coupling_matrix_numerical,
        compute_spectrogram,
        create_1d_raised_cosine,
        damping_term_simple,
        display_audio_with_title,
        jax,
        jnp,
        jsig,
        make_vk_nl_fn,
        multiresolution_eigendecomposition,
        np,
        optax,
        plt,
        safe_log,
        solve_tf_excitation,
        spectral_convergence_loss,
        spectral_wasserstein,
        stiffness_term,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Again, we generate a target simulation of the plate. We define the parameters of the plate and the excitation.""")
    return


@app.cell
def _(PlateParameters):
    n_modes_h = 20
    sampling_rate_h = 44100
    sampling_period_h = 1 / sampling_rate_h
    h_grid_h = 0.004  # grid spacing in the lowest resolution
    nx_h = 50  # number of grid points in the x direction in the lowest resolution
    ny_h = 75  # number of grid points in the y direction in the lowest resolution
    levels_h = 2  # number of grid refinements to perform
    amplitude_h = 0.5
    params_h = PlateParameters(
        E=2e12,
        nu=0.3,
        rho=7850,
        h=5e-4,
        l1=0.2,
        l2=0.3,
        Ts0=100,
    )
    force_position_h = (0.05, 0.05)
    readout_position_h = (0.1, 0.1)
    return (
        amplitude_h,
        force_position_h,
        h_grid_h,
        levels_h,
        n_modes_h,
        nx_h,
        ny_h,
        params_h,
        readout_position_h,
        sampling_period_h,
        sampling_rate_h,
    )


@app.cell
def _(
    compute_coupling_matrix_numerical,
    h_grid_h,
    jnp,
    levels_h,
    multiresolution_eigendecomposition,
    n_modes_h,
    np,
    nx_h,
    ny_h,
    params_h,
):
    # boundary conditions for the transverse modes
    bcs_phi_h = np.array(
        [
            [1e15, 0],
            [1e15, 0],
            [1e15, 0],
            [1e15, 0],
        ]
    )
    # boundary conditions for the in-plane modes
    bcs_psi_h = np.array(
        [
            [1e15, 1e15],
            [1e15, 1e15],
            [1e15, 1e15],
            [1e15, 1e15],
        ]
    )

    psi_h, zeta_mu_squared_h, nx_final_h, ny_final_h, h_final_h, psi_norms_h = (
        multiresolution_eigendecomposition(
            params_h,
            n_modes_h,
            bcs_psi_h,
            h_grid_h,
            nx_h,
            ny_h,
            levels=2,
        )
    )

    phi_h, lambda_mu_squared_h, nx_final_h, ny_final_h, h_final_h, phi_norms_h = (
        multiresolution_eigendecomposition(
            params_h,
            n_modes_h,
            bcs_phi_h,
            h_grid_h,
            nx_h,
            ny_h,
            levels=2,
        )
    )

    H_coupling = compute_coupling_matrix_numerical(
        psi_h,
        phi_h,
        h_final_h,
        nx_final_h,
        ny_final_h,
    )
    e_h = params_h.E / (2 * params_h.rho)
    H_coupling = H_coupling * np.sqrt(e_h)
    lambda_mu_h = jnp.sqrt(lambda_mu_squared_h)
    return H_coupling, lambda_mu_h, nx_final_h, ny_final_h, phi_h


@app.cell
def _(
    amplitude_h,
    create_1d_raised_cosine,
    force_position_h,
    n_modes_h,
    np,
    nx_final_h,
    ny_final_h,
    params_h,
    phi_h,
    readout_position_h,
):
    # generate a 1d raised cosine excitation
    rc_h = create_1d_raised_cosine(
        duration=1.0,
        start_time=0.001,
        end_time=0.003,
        amplitude=amplitude_h,
        sample_rate=44100,
    )

    phi_reshaped_h = np.reshape(
        phi_h,
        shape=(ny_final_h + 1, nx_final_h + 1, n_modes_h),
        order="F",
    )

    mode_gains_at_pos_h = phi_reshaped_h[
        int(force_position_h[1] * ny_final_h),
        int(force_position_h[0] * nx_final_h),
        :,
    ]

    mode_gains_at_readout_h = phi_reshaped_h[
        int(readout_position_h[1] * ny_final_h),
        int(readout_position_h[0] * nx_final_h),
        :,
    ]
    # the modal excitation needs to be scaled by A_inv and divided by the density
    mode_gains_at_pos_normalised_h = mode_gains_at_pos_h / params_h.density
    modal_excitation_normalised_short_h = rc_h[: 4410 * 3, None] * mode_gains_at_pos_normalised_h
    modal_excitation_normalised_long_h = rc_h[:44100, None] * mode_gains_at_pos_normalised_h
    return (
        modal_excitation_normalised_long_h,
        modal_excitation_normalised_short_h,
        mode_gains_at_readout_h,
    )


@app.cell(hide_code=True)
def _(
    H_coupling,
    damping_term_simple,
    jnp,
    jsig,
    lambda_mu_h,
    make_vk_nl_fn,
    mode_gains_at_readout_h,
    params_h,
    sampling_period_h,
    solve_tf_excitation,
    stiffness_term,
):
    mask_h = H_coupling == 0

    def get_H_opt(pars):
        H_opt = pars["H_opt"].at[mask_h].set(0.0)
        return H_opt * 1e9

    omega_mu_squared_h = stiffness_term(params_h, lambda_mu_h)
    gamma2_mu_h = damping_term_simple(jnp.sqrt(omega_mu_squared_h))

    def simulate_vkplate_h(pars, modal_excitation_normalised):
        _, modal_sol = solve_tf_excitation(
            gamma2_mu_h,
            omega_mu_squared_h,
            modal_excitation_normalised,
            sampling_period_h,
            nl_fn=make_vk_nl_fn(get_H_opt(pars)),
        )

        out_pos = modal_sol @ mode_gains_at_readout_h
        return out_pos

    def stft_h(x):
        _, _, zxx = jsig.stft(
            x,
            nperseg=1024,
            noverlap=512,
            padded=False,
            window="hann",
        )
        return zxx.T

    return get_H_opt, simulate_vkplate_h, stft_h


@app.cell
def _(
    H_coupling,
    jnp,
    modal_excitation_normalised_long_h,
    modal_excitation_normalised_short_h,
    simulate_vkplate_h,
    stft_h,
):
    gt_pars_h = {"H_opt": jnp.array(H_coupling) / 1e9}
    out_pos_gt_h = simulate_vkplate_h(gt_pars_h, modal_excitation_normalised_short_h)
    out_pos_gt_scale_h = jnp.max(jnp.abs(out_pos_gt_h))

    out_pos_gt_fft_h = stft_h(out_pos_gt_h)
    out_pos_gt_fft_mag_h = jnp.abs(out_pos_gt_fft_h)
    out_pos_gt_fft_mag_scale_h = 1.0 / jnp.max(out_pos_gt_fft_mag_h)
    out_pos_gt_fft_mag_h = out_pos_gt_fft_mag_h * out_pos_gt_fft_mag_scale_h
    return gt_pars_h, out_pos_gt_fft_mag_h, out_pos_gt_h


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Plot the intial simulation""")
    return


@app.cell
def _(
    H_coupling,
    compute_spectrogram,
    display_audio_with_title,
    gt_pars_h,
    jax,
    modal_excitation_normalised_long_h,
    np,
    plt,
    sampling_rate_h,
    simulate_vkplate_h,
):
    pars_h_init = {
        "H_opt": jax.random.uniform(jax.random.PRNGKey(0), H_coupling.shape),
    }
    out_pos_pred_initial_long_h = simulate_vkplate_h(pars_h_init, modal_excitation_normalised_long_h)
    out_pos_gt_long_h = simulate_vkplate_h(gt_pars_h, modal_excitation_normalised_long_h)

    n_fft_h = 4096
    hop_length_h = 128
    max_freq_h = 16000  # Maximum frequency for spectrogram
    scale_h = 1e5

    S_db_x_h = compute_spectrogram(
        np.array(out_pos_gt_long_h) * scale_h,
        sampling_rate_h,
        n_fft_h,
        hop_length_h,
        max_freq_h,
    )

    S_db_x_hat_h = compute_spectrogram(
        np.array(out_pos_pred_initial_long_h) * scale_h,
        sampling_rate_h,
        n_fft_h,
        hop_length_h,
        max_freq_h,
    )
    # Find global min and max for color bar scaling
    vmin_h = min(S_db_x_h.min(), S_db_x_hat_h.min())
    vmax_h = max(S_db_x_h.max(), S_db_x_hat_h.max())

    # Plot side-by-side spectrograms
    fig_h_init, ax_h_init = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    def plot_spectrogram_h(ax, S_db, title):
        import librosa

        freq_bins = librosa.fft_frequencies(sr=sampling_rate_h, n_fft=n_fft_h)
        max_freq_idx = np.sum(freq_bins <= 5000)

        # Calculate time values for x-axis
        times = np.arange(S_db.shape[1]) * hop_length_h / sampling_rate_h

        img = ax.pcolormesh(
            times,  # Use time values for x-axis
            freq_bins[:max_freq_idx],  # Use frequency bins for y-axis
            S_db[:max_freq_idx],
            cmap="viridis",
            shading="auto",
            rasterized=True,
            vmin=vmin_h,
            vmax=vmax_h,
        )

        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel("Time (s)")

        return img

    # Plot spectrograms
    img1_h = plot_spectrogram_h(ax_h_init[0], S_db_x_h, "Target")
    img2_h = plot_spectrogram_h(ax_h_init[1], S_db_x_hat_h, "Initial")
    ax_h_init[0].set_ylabel("Frequency (Hz)")
    fig_h_init.tight_layout()
    ax_h_init[0]

    display_audio_with_title(out_pos_gt_long_h, sampling_rate_h, "Target")
    display_audio_with_title(out_pos_pred_initial_long_h, sampling_rate_h, "Initial")
    return out_pos_gt_long_h, out_pos_pred_initial_long_h, pars_h_init, plot_spectrogram_h


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Optimisation loop""")
    return


@app.cell
def _(
    jax,
    jnp,
    modal_excitation_normalised_short_h,
    optax,
    out_pos_gt_fft_mag_h,
    out_pos_gt_fft_mag_scale_h,
    pars_h_init,
    simulate_vkplate_h,
    spectral_convergence_loss,
    spectral_wasserstein,
    stft_h,
    tqdm,
):
    learning_rate_h = 5e-1
    iterations_h = 2000
    scheduler_h = optax.cosine_onecycle_schedule(
        transition_steps=iterations_h,
        peak_value=learning_rate_h,
    )
    optimiser_h = optax.adam(learning_rate=scheduler_h)
    state_h = optimiser_h.init(pars_h_init)

    def loss_fn_h(pars):
        out_pos = simulate_vkplate_h(pars, modal_excitation_normalised_short_h)
        out_pos_fft_mag = jnp.abs(stft_h(out_pos)) * out_pos_gt_fft_mag_scale_h

        ot_loss = jnp.mean(
            jax.vmap(spectral_wasserstein, in_axes=(0, 0, None, None))(
                out_pos_fft_mag,
                out_pos_gt_fft_mag_h,
                True,
                True,
            )
        )
        sc_loss = spectral_convergence_loss(
            out_pos_fft_mag,
            out_pos_gt_fft_mag_h,
        )

        return sc_loss + ot_loss

    @jax.jit
    def train_step_h(pars, state):
        loss_val, grads = jax.value_and_grad(loss_fn_h)(pars)
        updates, state = optimiser_h.update(grads, state, pars)
        pars = optax.apply_updates(pars, updates)
        return pars, state, loss_val

    # Run optimization
    pars_h_opt = pars_h_init.copy()
    state_h_current = state_h
    
    bar_h = tqdm(range(iterations_h))
    for i in bar_h:
        pars_h_opt, state_h_current, loss_val = train_step_h(pars_h_opt, state_h_current)
        bar_h.set_description(f"Loss: {loss_val:.3f}")
        
    return loss_fn_h, pars_h_opt, train_step_h


@app.cell
def _(
    compute_spectrogram,
    display_audio_with_title,
    modal_excitation_normalised_long_h,
    np,
    out_pos_gt_long_h,
    out_pos_pred_initial_long_h,
    pars_h_opt,
    plot_spectrogram_h,
    plt,
    sampling_rate_h,
    simulate_vkplate_h,
):
    n_fft_final_h = 4096
    hop_length_final_h = 128
    max_freq_final_h = 16000
    scale_final_h = 1e8

    out_pos_pred_long_h = simulate_vkplate_h(pars_h_opt, modal_excitation_normalised_long_h)

    S_db_x_final_h = compute_spectrogram(
        np.array(out_pos_gt_long_h) * scale_final_h,
        sampling_rate_h,
        n_fft_final_h,
        hop_length_final_h,
        max_freq_final_h,
    )
    S_db_x_initial_final_h = compute_spectrogram(
        np.array(out_pos_pred_initial_long_h) * scale_final_h,
        sampling_rate_h,
        n_fft_final_h,
        hop_length_final_h,
        max_freq_final_h,
    )
    S_db_x_hat_final_h = compute_spectrogram(
        np.array(out_pos_pred_long_h) * scale_final_h,
        sampling_rate_h,
        n_fft_final_h,
        hop_length_final_h,
        max_freq_final_h,
    )
    # Find global min and max for color bar scaling
    vmin_final_h = min(S_db_x_final_h.min(), S_db_x_hat_final_h.min())
    vmax_final_h = max(S_db_x_final_h.max(), S_db_x_hat_final_h.max())

    # Plot side-by-side spectrograms
    fig_final_h, ax_final_h = plt.subplots(1, 3, figsize=(6.9 * 3, 2.5 * 3), sharey=True)

    def plot_spectrogram_final_h(ax, S_db, title):
        import librosa

        freq_bins = librosa.fft_frequencies(sr=sampling_rate_h, n_fft=n_fft_final_h)
        max_freq_idx = np.sum(freq_bins <= 5000)

        # Calculate time values for x-axis
        times = np.arange(S_db.shape[1]) * hop_length_final_h / sampling_rate_h

        img = ax.pcolormesh(
            times,  # Use time values for x-axis
            freq_bins[:max_freq_idx],  # Use frequency bins for y-axis
            S_db[:max_freq_idx],
            cmap="viridis",
            shading="auto",
            rasterized=True,
            vmin=vmin_final_h,
            vmax=vmax_final_h,
        )

        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel("Time (s)")

        return img

    # Plot spectrograms
    img1_final_h = plot_spectrogram_final_h(ax_final_h[0], S_db_x_final_h, "Target")
    img2_final_h = plot_spectrogram_final_h(ax_final_h[1], S_db_x_initial_final_h, "Initial")
    img3_final_h = plot_spectrogram_final_h(ax_final_h[2], S_db_x_hat_final_h, "Optimised")

    # Only set y-label on the first plot
    ax_final_h[0].set_ylabel("Frequency (Hz)")

    fig_final_h.tight_layout()
    ax_final_h[0]

    display_audio_with_title(out_pos_gt_long_h, sampling_rate_h, "Target")
    display_audio_with_title(out_pos_pred_long_h, sampling_rate_h, "Optimised")
    return out_pos_pred_long_h


if __name__ == "__main__":
    app.run()