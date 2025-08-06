import marimo

__generated_with = "0.14.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Optimising the parameters of the Von Karman plate""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""In this notebook we optimise the bending stiffness of a Von Karman plate using backpropagation through time.""")
    return


@app.cell(hide_code=True)
def _():
    from functools import partial

    import jax
    import jax.numpy as jnp
    import jax.scipy.signal as jsig
    import numpy as np
    import optax
    from matplotlib import pyplot as plt
    from tqdm import tqdm

    from jaxdiffmodal.excitations import create_1d_raised_cosine
    from jaxdiffmodal.ftm import (
        PlateParameters,
        damping_term_simple,
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
        partial,
        plt,
        safe_log,
        solve_tf_excitation,
        spectral_convergence_loss,
        spectral_wasserstein,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""First, we generate a target simulation of the plate. We define the parameters of the plate and the excitation.""")
    return


@app.cell
def _(PlateParameters):
    n_modes_vk = 20
    sampling_rate_vk = 44100
    sampling_period_vk = 1 / sampling_rate_vk
    h_vk = 0.004  # grid spacing in the lowest resolution
    nx_vk = 50  # number of grid points in the x direction in the lowest resolution
    ny_vk = 75  # number of grid points in the y direction in the lowest resolution
    levels_vk = 2  # number of grid refinements to perform
    amplitude_vk = 0.5
    params_vk = PlateParameters(
        E=2e12,
        nu=0.3,
        rho=7850,
        h=5e-4,
        l1=0.2,
        l2=0.3,
        Ts0=100,
    )
    force_position_vk = (0.05, 0.05)
    readout_position_vk = (0.1, 0.1)
    return (
        amplitude_vk,
        force_position_vk,
        h_vk,
        levels_vk,
        n_modes_vk,
        nx_vk,
        ny_vk,
        params_vk,
        readout_position_vk,
        sampling_period_vk,
        sampling_rate_vk,
    )


@app.cell
def _(
    compute_coupling_matrix_numerical,
    h_vk,
    jnp,
    levels_vk,
    multiresolution_eigendecomposition,
    n_modes_vk,
    np,
    nx_vk,
    ny_vk,
    params_vk,
):
    # boundary conditions for the transverse modes
    bcs_phi_vk = np.array(
        [
            [1e15, 0],
            [1e15, 0],
            [1e15, 0],
            [1e15, 0],
        ]
    )
    # boundary conditions for the in-plane modes
    bcs_psi_vk = np.array(
        [
            [1e15, 1e15],
            [1e15, 1e15],
            [1e15, 1e15],
            [1e15, 1e15],
        ]
    )

    psi_vk, zeta_mu_squared_vk, nx_final_vk, ny_final_vk, h_final_vk, psi_norms_vk = (
        multiresolution_eigendecomposition(
            params_vk,
            n_modes_vk,
            bcs_psi_vk,
            h_vk,
            nx_vk,
            ny_vk,
            levels=2,
        )
    )

    phi_vk, lambda_mu_squared_vk, nx_final_vk, ny_final_vk, h_final_vk, phi_norms_vk = (
        multiresolution_eigendecomposition(
            params_vk,
            n_modes_vk,
            bcs_phi_vk,
            h_vk,
            nx_vk,
            ny_vk,
            levels=2,
        )
    )

    H_vk = compute_coupling_matrix_numerical(
        psi_vk,
        phi_vk,
        h_final_vk,
        nx_final_vk,
        ny_final_vk,
    )
    e_vk = params_vk.E / (2 * params_vk.rho)
    H_vk = H_vk * np.sqrt(e_vk)
    lambda_mu_vk = jnp.sqrt(lambda_mu_squared_vk)
    return H_vk, lambda_mu_vk, nx_final_vk, ny_final_vk, phi_vk


@app.cell
def _(
    amplitude_vk,
    create_1d_raised_cosine,
    force_position_vk,
    n_modes_vk,
    np,
    nx_final_vk,
    ny_final_vk,
    params_vk,
    phi_vk,
    readout_position_vk,
):
    # generate a 1d raised cosine excitation
    rc_vk = create_1d_raised_cosine(
        duration=1.0,
        start_time=0.001,
        end_time=0.003,
        amplitude=amplitude_vk,
        sample_rate=44100,
    )

    phi_reshaped_vk = np.reshape(
        phi_vk,
        shape=(ny_final_vk + 1, nx_final_vk + 1, n_modes_vk),
        order="F",
    )

    mode_gains_at_pos_vk = phi_reshaped_vk[
        int(force_position_vk[1] * ny_final_vk),
        int(force_position_vk[0] * nx_final_vk),
        :,
    ]

    mode_gains_at_readout_vk = phi_reshaped_vk[
        int(readout_position_vk[1] * ny_final_vk),
        int(readout_position_vk[0] * nx_final_vk),
        :,
    ]
    # the modal excitation needs to be scaled by A_inv and divided by the density
    mode_gains_at_pos_normalised_vk = mode_gains_at_pos_vk / params_vk.density
    modal_excitation_normalised_short_vk = rc_vk[: 4410 * 3, None] * mode_gains_at_pos_normalised_vk
    modal_excitation_normalised_long_vk = rc_vk[:44100, None] * mode_gains_at_pos_normalised_vk
    return (
        modal_excitation_normalised_long_vk,
        modal_excitation_normalised_short_vk,
        mode_gains_at_readout_vk,
    )


@app.cell(hide_code=True)
def _(
    H_vk,
    damping_term_simple,
    jax,
    jnp,
    jsig,
    lambda_mu_vk,
    make_vk_nl_fn,
    mode_gains_at_readout_vk,
    params_vk,
    sampling_period_vk,
    solve_tf_excitation,
):
    nl_fn_vk = jax.jit(make_vk_nl_fn(jnp.array(H_vk)))

    def get_bending_stiffness_vk(pars):
        return pars["bending_stiffness"]

    def get_Ts0_vk(pars):
        return pars["Ts0"]

    def simulate_vkplate(pars, modal_excitation_normalised):
        omega_mu_squared = (
            get_bending_stiffness_vk(pars) * lambda_mu_vk**2 + get_Ts0_vk(pars) * lambda_mu_vk
        )
        gamma2_mu = damping_term_simple(jnp.sqrt(omega_mu_squared))

        _, modal_sol = solve_tf_excitation(
            gamma2_mu,
            omega_mu_squared,
            modal_excitation_normalised,
            sampling_period_vk,
            nl_fn=nl_fn_vk,
        )

        out_pos = modal_sol @ mode_gains_at_readout_vk
        return out_pos

    def stft_vk(x):
        _, _, zxx = jsig.stft(
            x,
            nperseg=1024,
            noverlap=512,
            padded=False,
            window="hann",
        )
        return zxx.T

    return get_Ts0_vk, get_bending_stiffness_vk, simulate_vkplate, stft_vk


@app.cell
def _(
    get_Ts0_vk,
    get_bending_stiffness_vk,
    jnp,
    modal_excitation_normalised_long_vk,
    modal_excitation_normalised_short_vk,
    params_vk,
    simulate_vkplate,
    stft_vk,
):
    gt_pars_vk = {
        "bending_stiffness": params_vk.bending_stiffness / params_vk.density,
        "Ts0": params_vk.Ts0 / params_vk.density,
    }

    out_pos_gt_vk = simulate_vkplate(gt_pars_vk, modal_excitation_normalised_short_vk)
    out_pos_gt_long_vk = simulate_vkplate(gt_pars_vk, modal_excitation_normalised_long_vk)
    out_pos_gt_fft_vk = stft_vk(out_pos_gt_vk)
    out_pos_gt_fft_mag_vk = jnp.abs(out_pos_gt_fft_vk)
    out_pos_gt_fft_mag_scale_vk = 1.0 / jnp.max(out_pos_gt_fft_mag_vk)
    out_pos_gt_fft_mag_vk = out_pos_gt_fft_mag_vk * out_pos_gt_fft_mag_scale_vk
    return gt_pars_vk, out_pos_gt_fft_mag_vk, out_pos_gt_long_vk, out_pos_gt_vk


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Loss landscape\nLet's take a small detour to explore how the loss function varies with respect to a single parameter. First define the loss function.""")
    return


@app.cell
def _(
    jax,
    jnp,
    modal_excitation_normalised_short_vk,
    out_pos_gt_fft_mag_scale_vk,
    out_pos_gt_fft_mag_vk,
    out_pos_gt_vk,
    safe_log,
    simulate_vkplate,
    spectral_convergence_loss,
    spectral_wasserstein,
    stft_vk,
):
    def combined_loss_fn_vk(
        pars,
        lm_loss_weight=1.0,
        ot_loss_weight=1.0,
        sc_loss_weight=1.0,
        time_loss_weight=1.0,
    ):
        out_pos = simulate_vkplate(pars, modal_excitation_normalised_short_vk)

        out_pos_fft_mag = jnp.abs(stft_vk(out_pos)) * out_pos_gt_fft_mag_scale_vk

        log_diff = safe_log(out_pos_gt_fft_mag_vk + 1e-10) - safe_log(out_pos_fft_mag + 1e-10)
        lm_loss = jnp.mean(jnp.abs(log_diff))

        ot_loss = jnp.mean(
            jax.vmap(spectral_wasserstein, in_axes=(0, 0, None, None))(
                out_pos_fft_mag,
                out_pos_gt_fft_mag_vk,
                True,
                True,
            )
        )
        time_loss = jnp.mean(jnp.square(out_pos - out_pos_gt_vk))
        sc_loss = spectral_convergence_loss(
            out_pos_fft_mag,
            out_pos_gt_fft_mag_vk,
        )

        combined_loss = (
            lm_loss * lm_loss_weight
            + ot_loss * ot_loss_weight
            + sc_loss * sc_loss_weight
            + time_loss * time_loss_weight
        )
        return combined_loss, (lm_loss, ot_loss, sc_loss, time_loss)

    return (combined_loss_fn_vk,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Plot the loss landscape for the bending stiffness""")
    return


@app.cell
def _(
    combined_loss_fn_vk,
    gt_pars_vk,
    jax,
    jnp,
    params_vk,
    partial,
):
    def compute_losses_for_stiffness_vk(
        bending_stiffness,
        loss_fn,
    ):
        pars = {
            "bending_stiffness": bending_stiffness,
            "Ts0": params_vk.Ts0 / params_vk.density,
        }
        return loss_fn(pars)

    bending_stiffness_normalised_range_vk = jnp.linspace(
        gt_pars_vk["bending_stiffness"] - 4.5,
        gt_pars_vk["bending_stiffness"] + 4.5,
        100,
    )

    losses_combined_vk, (losses_lm_vk, losses_ot_vk, losses_sc_vk, losses_time_vk) = jax.vmap(
        partial(
            compute_losses_for_stiffness_vk,
            loss_fn=combined_loss_fn_vk,
        )
    )(bending_stiffness_normalised_range_vk)
    return (
        bending_stiffness_normalised_range_vk,
        losses_lm_vk,
        losses_ot_vk,
        losses_sc_vk,
        losses_time_vk,
    )


@app.cell
def _(
    bending_stiffness_normalised_range_vk,
    gt_pars_vk,
    jnp,
    losses_lm_vk,
    losses_ot_vk,
    losses_sc_vk,
    losses_time_vk,
    plt,
):
    plt.rcParams["text.usetex"] = False
    plt.rcParams["font.size"] = 15
    fig_loss, ax_loss = plt.subplots(1, 1, figsize=(10, 5))
    ax_loss.plot(
        bending_stiffness_normalised_range_vk,
        losses_ot_vk / jnp.max(losses_ot_vk),
        label="$\mathcal{L}_{\\text{sot}}$",
    )
    ax_loss.plot(
        bending_stiffness_normalised_range_vk,
        losses_sc_vk / jnp.max(losses_sc_vk),
        linestyle="--",
        label="$\mathcal{L}_{\\text{sc}}$",
    )
    ax_loss.plot(
        bending_stiffness_normalised_range_vk,
        losses_lm_vk / jnp.max(losses_lm_vk),
        linestyle="-.",
        label="$\mathcal{L}_{\\text{log}}$",
        alpha=0.8,
    )
    ax_loss.plot(
        bending_stiffness_normalised_range_vk,
        losses_time_vk / jnp.max(losses_time_vk),
        linestyle=":",
        label="$\mathcal{L}_{\\text{time}}$",
    )

    ax_loss.set_xlabel("Bending stiffness (normalised)")
    ax_loss.set_yticklabels([])
    ax_loss.set_yticks([])
    ax_loss.axvline(gt_pars_vk["bending_stiffness"], color="k", linestyle="--")
    ax_loss.legend(loc="upper right")
    ax_loss.grid()
    fig_loss.tight_layout()
    ax_loss
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Optimise the bending stiffness\n\nNow we optimise the bending stiffness. Starting from an initial value of 10.0""")
    return


@app.cell
def _(
    compute_spectrogram,
    display_audio_with_title,
    modal_excitation_normalised_long_vk,
    np,
    out_pos_gt_long_vk,
    params_vk,
    plt,
    sampling_rate_vk,
    simulate_vkplate,
):
    pars_vk_init = {
        "bending_stiffness": 10.0,
        "Ts0": params_vk.Ts0 / params_vk.density,
    }
    out_pos_pred_initial_long_vk = simulate_vkplate(pars_vk_init, modal_excitation_normalised_long_vk)

    n_fft_vk = 4096
    hop_length_vk = 128
    max_freq_vk = 16000  # Maximum frequency for spectrogram

    scale_vk = 1e5
    S_db_x_vk = compute_spectrogram(
        np.array(out_pos_gt_long_vk) * scale_vk,
        sampling_rate_vk,
        n_fft_vk,
        hop_length_vk,
        max_freq_vk,
    )
    S_db_x_initial_vk = compute_spectrogram(
        np.array(out_pos_pred_initial_long_vk) * scale_vk,
        sampling_rate_vk,
        n_fft_vk,
        hop_length_vk,
        max_freq_vk,
    )

    # Plot side-by-side spectrograms
    fig_init_vk, ax_init_vk = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    # Find global min and max for color bar scaling
    vmin_vk = S_db_x_vk.min()
    vmax_vk = S_db_x_vk.max()

    def plot_spectrogram_vk(ax, S_db, title):
        import librosa

        freq_bins = librosa.fft_frequencies(sr=sampling_rate_vk, n_fft=n_fft_vk)
        max_freq_idx = np.sum(freq_bins <= 5000)

        # Calculate time values for x-axis
        times = np.arange(S_db.shape[1]) * hop_length_vk / sampling_rate_vk

        img = ax.pcolormesh(
            times,  # Use time values for x-axis
            freq_bins[:max_freq_idx],  # Use frequency bins for y-axis
            S_db[:max_freq_idx],
            cmap="viridis",
            shading="auto",
            rasterized=True,
            vmin=vmin_vk,
            vmax=vmax_vk,
        )

        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel("Time (s)")

        return img

    img1_vk = plot_spectrogram_vk(ax_init_vk[0], S_db_x_vk, "Target")
    img2_vk = plot_spectrogram_vk(ax_init_vk[1], S_db_x_initial_vk, "Initial")
    ax_init_vk[0].set_ylabel("Frequency (Hz)")
    fig_init_vk.tight_layout()
    ax_init_vk[0]

    display_audio_with_title(out_pos_gt_long_vk, sampling_rate_vk, "Target")
    display_audio_with_title(out_pos_pred_initial_long_vk, sampling_rate_vk, "Initial")
    return out_pos_pred_initial_long_vk, pars_vk_init, plot_spectrogram_vk


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Optimisation loop. This might take a while to run, depending on how long is the sequence we want to optimise over and the number of iterations. Here we optimise over a 0.3 second sequence (13230 samples), which is on the longer side for this sort of optimisation.""")
    return


@app.cell
def _(
    combined_loss_fn_vk,
    gt_pars_vk,
    jax,
    optax,
    pars_vk_init,
    partial,
    tqdm,
):
    learning_rate_vk = 2e-1
    iterations_vk = 1000
    scheduler_vk = optax.cosine_onecycle_schedule(
        transition_steps=iterations_vk,
        peak_value=learning_rate_vk,
    )
    optimiser_vk = optax.adam(learning_rate=scheduler_vk)

    state_vk = optimiser_vk.init(pars_vk_init)
    value_and_grad_vk = jax.value_and_grad(
        partial(
            combined_loss_fn_vk,
            lm_loss_weight=0.0,
            ot_loss_weight=1.0,
            sc_loss_weight=0.001,
            time_loss_weight=0.0,
        ),
        has_aux=True,
    )

    @jax.jit
    def train_step_vk(pars, state):
        (loss_val, _), grads = value_and_grad_vk(pars)
        updates, state = optimiser_vk.update(grads, state, pars)
        pars = optax.apply_updates(pars, updates)
        return pars, state, loss_val

    # Run optimization
    pars_vk_opt = pars_vk_init.copy()
    state_vk_current = state_vk
    
    bar_vk = tqdm(range(iterations_vk))
    for i in bar_vk:
        pars_vk_opt, state_vk_current, loss_val = train_step_vk(pars_vk_opt, state_vk_current)
        bar_vk.set_description(
            f"Loss: {loss_val:.3f}, bending stiffness: {pars_vk_opt['bending_stiffness']:.4f}, ground truth: {gt_pars_vk['bending_stiffness']:.4f}",
        )
        
    return pars_vk_opt, train_step_vk


@app.cell
def _(
    compute_spectrogram,
    display_audio_with_title,
    modal_excitation_normalised_long_vk,
    np,
    out_pos_gt_long_vk,
    out_pos_pred_initial_long_vk,
    pars_vk_opt,
    plot_spectrogram_vk,
    plt,
    sampling_rate_vk,
    scale_vk,
    simulate_vkplate,
):
    out_pos_pred_long_vk = simulate_vkplate(pars_vk_opt, modal_excitation_normalised_long_vk)

    scale_final_vk = 1e8
    S_db_x_final_vk = compute_spectrogram(
        np.array(out_pos_gt_long_vk) * scale_final_vk,
        sampling_rate_vk,
        n_fft_vk,
        hop_length_vk,
        max_freq_vk,
    )
    S_db_x_initial_final_vk = compute_spectrogram(
        np.array(out_pos_pred_initial_long_vk) * scale_final_vk,
        sampling_rate_vk,
        n_fft_vk,
        hop_length_vk,
        max_freq_vk,
    )
    S_db_x_hat_vk = compute_spectrogram(
        np.array(out_pos_pred_long_vk) * scale_final_vk,
        sampling_rate_vk,
        n_fft_vk,
        hop_length_vk,
        max_freq_vk,
    )
    # Find global min and max for color bar scaling
    vmin_final_vk = min(S_db_x_final_vk.min(), S_db_x_hat_vk.min())
    vmax_final_vk = max(S_db_x_final_vk.max(), S_db_x_hat_vk.max())

    # Plot side-by-side spectrograms
    fig_final_vk, ax_final_vk = plt.subplots(1, 3, figsize=(6.9 * 3, 2.5 * 3), sharey=True)
    
    def plot_spectrogram_final_vk(ax, S_db, title):
        import librosa

        freq_bins = librosa.fft_frequencies(sr=sampling_rate_vk, n_fft=n_fft_vk)
        max_freq_idx = np.sum(freq_bins <= 5000)

        # Calculate time values for x-axis
        times = np.arange(S_db.shape[1]) * hop_length_vk / sampling_rate_vk

        img = ax.pcolormesh(
            times,  # Use time values for x-axis
            freq_bins[:max_freq_idx],  # Use frequency bins for y-axis
            S_db[:max_freq_idx],
            cmap="viridis",
            shading="auto",
            rasterized=True,
            vmin=vmin_final_vk,
            vmax=vmax_final_vk,
        )

        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel("Time (s)")

        return img

    img1_final_vk = plot_spectrogram_final_vk(ax_final_vk[0], S_db_x_final_vk, "Ground Truth")
    img2_final_vk = plot_spectrogram_final_vk(ax_final_vk[1], S_db_x_initial_final_vk, "Initial")
    img3_final_vk = plot_spectrogram_final_vk(ax_final_vk[2], S_db_x_hat_vk, "Optimised")
    ax_final_vk[0].set_ylabel("Frequency (Hz)")
    fig_final_vk.tight_layout()
    ax_final_vk[0]

    display_audio_with_title(out_pos_gt_long_vk, sampling_rate_vk, "Target")
    display_audio_with_title(out_pos_pred_long_vk, sampling_rate_vk, "Optimised")

    print(f"Optimized bending stiffness: {pars_vk_opt['bending_stiffness']:.4f}")
    print(f"Ground truth: {gt_pars_vk['bending_stiffness']:.4f}")
    return out_pos_pred_long_vk


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Multistart parallel optimisation""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""We can also optimise over a larger range of initial bending stiffness values, using parallel multiple starts in parallel. Here we use 50 random initial values between 2 and 50. NB: with longer sequences, this will take longer to run also because the XLA compilation will take longer to find an appropriate implementation because of the STFT.""")
    return


@app.cell
def _(
    combined_loss_fn_vk,
    gt_pars_vk,
    jax,
    jnp,
    optax,
    params_vk,
    partial,
    tqdm,
):
    value_and_grad_multi_vk = jax.value_and_grad(
        partial(
            combined_loss_fn_vk,
            lm_loss_weight=0.0,
            ot_loss_weight=1.0,
            sc_loss_weight=0.001,
            time_loss_weight=0.0,
        ),
        has_aux=True,
    )

    def losses_and_grads_for_single_stiffness_vk(bending_stiffness):
        pars = {
            "bending_stiffness": bending_stiffness,
            "Ts0": params_vk.Ts0 / params_vk.density,
        }

        (loss_val, _), grads = value_and_grad_multi_vk(pars)

        # return loss and gradient for bending stiffness only
        return loss_val, grads["bending_stiffness"]

    compute_vec_loss_grad_vk = jax.vmap(losses_and_grads_for_single_stiffness_vk)

    # Generate starting points
    num_starts_vk = 50
    start_points_vk = jnp.linspace(2.0, 50.0, num_starts_vk)

    learning_rate_multi_vk = 1e-2
    iterations_multi_vk = 100
    scheduler_multi_vk = optax.cosine_onecycle_schedule(
        transition_steps=iterations_multi_vk,
        peak_value=learning_rate_multi_vk,
    )
    optimiser_multi_vk = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=scheduler_multi_vk),
    )
    opt_state_multi_vk = optimiser_multi_vk.init(start_points_vk)

    @jax.jit
    def train_step_multi_vk(pars, state):
        losses, grads = compute_vec_loss_grad_vk(pars)
        updates, state = optimiser_multi_vk.update(grads, state, pars)
        pars = optax.apply_updates(pars, updates)
        return pars, state, losses

    # Run multistart optimization
    start_points_current_vk = start_points_vk
    opt_state_current_vk = opt_state_multi_vk
    
    bar_multi_vk = tqdm(range(iterations_multi_vk))
    for i in bar_multi_vk:
        start_points_current_vk, opt_state_current_vk, losses = train_step_multi_vk(start_points_current_vk, opt_state_current_vk)

        best_idx = jnp.argmin(losses)
        best_loss = losses[best_idx]
        best_param = start_points_current_vk[best_idx]
        bar_multi_vk.set_description(f"Best loss: {best_loss:.6f}, Best param: {best_param:.6f}")

    # Get final results
    final_losses_vk, _ = compute_vec_loss_grad_vk(start_points_current_vk)
    best_idx_final = jnp.argmin(final_losses_vk)
    best_param_final = start_points_current_vk[best_idx_final]
    best_loss_final = final_losses_vk[best_idx_final]

    print(f"Best bending stiffness found: {best_param_final:.6f}")
    print(f"Ground truth: {gt_pars_vk['bending_stiffness']:.6f}")
    print(f"Best loss: {best_loss_final:.6f}")
    return best_loss_final, best_param_final


if __name__ == "__main__":
    app.run()