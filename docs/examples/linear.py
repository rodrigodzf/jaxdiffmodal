import marimo

__generated_with = "0.14.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Linear models""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    For the linear models we want to solve the following equation:

    \begin{equation}
    \rho \ddot{w} + \left(d_1 + d_3 \Delta\right)\dot{w} + (D \Delta \Delta - T_0 \Delta) w = f_{\text{ext}},
    \end{equation}

    which in modal coordinates $q_{\mu}$ results in uncoupled damped harmonic oscillators:

    \begin{equation}
    \ddot{q}_{\mu} + 2\gamma_{\mu}\dot{q}_{\mu} + \omega_{\mu}^2 q_{\mu} = \bar{f}_{\text{ext},\mu},
    \end{equation}

    where the coefficients are given by:

    \begin{align}
    \omega_{\mu}^2 &= \frac{D\lambda_{\mu}^2 + T_0\lambda_{\mu}}{\rho},\\
    \gamma_{\mu} &= \frac{d_1 + d_3\lambda_{\mu}}{2\rho}.
    \end{align}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### String""")
    return


@app.cell(hide_code=True)
def _():
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    from IPython.display import HTML, Audio
    from matplotlib import animation

    from jaxdiffmodal.excitations import create_1d_raised_cosine, create_pluck_modal
    from jaxdiffmodal.ftm import (
        PlateParameters,
        StringParameters,
        damping_term,
        evaluate_rectangular_eigenfunctions,
        evaluate_string_eigenfunctions,
        inverse_STL,
        inverse_STL_2d,
        plate_eigenfunctions,
        plate_eigenvalues,
        plate_wavenumbers,
        stiffness_term,
        string_eigenfunctions,
        string_eigenvalues,
    )
    from jaxdiffmodal.time_integrators import (
        solve_sinusoidal,
        solve_sv_excitation,
        solve_tf_excitation,
    )
    return (
        Audio,
        PlateParameters,
        StringParameters,
        animation,
        create_1d_raised_cosine,
        create_pluck_modal,
        damping_term,
        evaluate_rectangular_eigenfunctions,
        evaluate_string_eigenfunctions,
        inverse_STL,
        inverse_STL_2d,
        jnp,
        np,
        plate_eigenfunctions,
        plate_eigenvalues,
        plate_wavenumbers,
        plt,
        solve_sinusoidal,
        solve_tf_excitation,
        stiffness_term,
        string_eigenfunctions,
        string_eigenvalues,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Some parameters for the string""")
    return


@app.cell
def _(StringParameters):
    n_modes = 50
    n_steps = 44100
    sample_rate = 44100
    dt = 1.0 / sample_rate
    excitation_position = 0.2
    readout_position = 0.5
    initial_deflection = 0.03
    n_gridpoints = 101  # number of gridpoints for evaluating the eigenfunctions
    string_params = StringParameters()
    return (
        dt,
        excitation_position,
        initial_deflection,
        n_gridpoints,
        n_modes,
        n_steps,
        readout_position,
        sample_rate,
        string_params,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Getting the eigenpairs""")
    return


@app.cell
def _(
    n_gridpoints,
    n_modes,
    np,
    string_eigenfunctions,
    string_eigenvalues,
    string_params,
):
    lambda_mu = string_eigenvalues(n_modes, string_params.length)
    wn = np.sqrt(lambda_mu)
    grid = np.linspace(0, string_params.length, n_gridpoints)
    K = string_eigenfunctions(wn, grid)
    return K, grid, lambda_mu


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Get the initial conditions or the excitation""")
    return


@app.cell(hide_code=True)
def _(
    K,
    create_pluck_modal,
    excitation_position,
    grid,
    initial_deflection,
    inverse_STL,
    lambda_mu,
    plt,
    string_params,
):
    u0_modal = create_pluck_modal(
        lambda_mu,
        pluck_position=excitation_position,
        initial_deflection=initial_deflection,
        string_length=string_params.length,
    )
    u0 = inverse_STL(K, u0_modal, string_params.length)
    fig_, ax_ = plt.subplots(1, 1, figsize=(6, 2))
    ax_.plot(grid, u0)
    ax_.set_xlabel("Position (m)")
    ax_.set_ylabel("Deflection (m)")
    ax_.set_title("Initial deflection of the string")
    ax_.grid(True)
    ax_
    return (u0_modal,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Get $\\gamma_{\\mu}$ and $\\omega_{\\mu}$ and integrate in time from the initial conditions defined above. This should be very fast, even for a large number of modes.""")
    return


@app.cell
def _(
    damping_term,
    dt,
    lambda_mu,
    n_steps,
    solve_sinusoidal,
    stiffness_term,
    string_params,
    u0_modal,
):
    gamma2_mu = damping_term(
        string_params,
        lambda_mu,
    )
    omega_mu_squared = stiffness_term(
        string_params,
        lambda_mu,
    )

    modal_sol_string = solve_sinusoidal(
        gamma2_mu,
        omega_mu_squared,
        u0_modal,
        n_steps,
        dt,
    )
    return (modal_sol_string,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Transform the modal solution back to the physical space using the precomputed eigenfunctions, or evaluate at a single point.""")
    return


@app.cell
def _(
    K,
    evaluate_string_eigenfunctions,
    inverse_STL,
    modal_sol_string,
    n_modes,
    np,
    readout_position,
    string_params,
):
    mu = np.arange(1, n_modes + 1)  # mode indices

    readout_weights = evaluate_string_eigenfunctions(
        mu,
        readout_position,
        string_params,
    )

    # at a single point
    u_readout = readout_weights @ modal_sol_string

    # at all points
    sol_string = inverse_STL(K, modal_sol_string, string_params.length)
    return sol_string, u_readout


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Single point:""")
    return


@app.cell
def _(Audio, plt, sample_rate, u_readout):
    audio_output = Audio(u_readout, rate=sample_rate)
    audio_output
    def plot_single(u_readout):
        fig_single, ax_single = plt.subplots(1, 1, figsize=(6, 2))
        ax_single.plot(u_readout)
        ax_single.set_xlabel("Sample")
        ax_single.set_ylabel("Deflection (m)")
        ax_single.set_title("Deflection of the string at a single point")
        ax_single.set_xlim(-2, sample_rate // 10)
        ax_single.grid(True)
        return ax_single

    plot_single(u_readout)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""All points:""")
    return


@app.cell(hide_code=True)
def _(animation, mo, np, plt, sol_string, string_params):
    n_frames = 2000
    frame_stride = 15
    positions = np.linspace(0, string_params.length, sol_string.shape[0])
    time_samples = sol_string[:, :n_frames]

    fig_anim, ax_anim = plt.subplots(figsize=(6, 3))
    (line,) = ax_anim.plot(positions, time_samples[:, 0])
    ax_anim.set_ylim(np.min(time_samples) * 1.1, np.max(time_samples) * 1.1)
    ax_anim.set_xlabel("Position (m)")
    ax_anim.set_ylabel("Deflection (m)")
    ax_anim.set_title("String Vibration")
    ax_anim.grid(True)
    fig_anim.tight_layout()


    def update(i):
        frame = i * frame_stride
        line.set_ydata(time_samples[:, frame])
        return (line,)


    ani_string = animation.FuncAnimation(
        fig_anim,
        update,
        frames=n_frames // frame_stride,
        interval=25,
        blit=True,
    )

    mo.Html(ani_string.to_html5_video())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ### Plate

    We do a very similar thing for plates. First define the parameters
    """
    )
    return


@app.cell
def _(PlateParameters):
    n_modes_x = 8
    n_modes_y = 8
    n_modes_plate = n_modes_x * n_modes_y
    n_steps_plate = 44100
    sample_rate_plate = 44100
    dt_plate = 1.0 / sample_rate_plate
    excitation_duration = 1.0  # seconds
    excitation_amplitude = 1.0
    force_position = (0.05, 0.05)
    readout_position_plate = (0.1, 0.1)
    plate_params = PlateParameters(
        Ts0=0,
        d1=4e-3,
        d3=3e-2,
    )
    return (
        dt_plate,
        excitation_amplitude,
        excitation_duration,
        force_position,
        n_modes_plate,
        n_modes_x,
        n_modes_y,
        plate_params,
        readout_position_plate,
        sample_rate_plate,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Get the eigenpairs""")
    return


@app.cell
def _(
    n_modes_plate,
    n_modes_x,
    n_modes_y,
    np,
    plate_eigenfunctions,
    plate_eigenvalues,
    plate_params,
    plate_wavenumbers,
):
    wnx, wny = plate_wavenumbers(
        n_modes_x,
        n_modes_y,
        plate_params.l1,
        plate_params.l2,
    )
    lambda_mu_2d = plate_eigenvalues(wnx, wny)
    n_gridpoints_x = 101
    n_gridpoints_y = 151
    x = np.linspace(0, plate_params.l1, n_gridpoints_x)
    y = np.linspace(0, plate_params.l2, n_gridpoints_y)
    K_plate = plate_eigenfunctions(wnx, wny, x, y)

    # Use the new function to select modes and get eigenvalues
    from jaxdiffmodal.ftm import select_modes_from_eigenvalues

    kx_indices, ky_indices, lambda_mu_plate, sorted_indices = select_modes_from_eigenvalues(
        lambda_mu_2d, n_modes_plate
    )
    selected_indices = np.stack([kx_indices, ky_indices], axis=-1)
    return K_plate, lambda_mu_plate, selected_indices, sorted_indices


@app.cell(hide_code=True)
def _(mo):
    mo.md("""This time we will use an excitation force instead on initial conditions. We define the force as a 1D raised cosine at a single point on the plate.""")
    return


@app.cell
def _(
    create_1d_raised_cosine,
    evaluate_rectangular_eigenfunctions,
    excitation_amplitude,
    excitation_duration,
    force_position,
    np,
    plate_params,
    readout_position_plate,
    sample_rate_plate,
    selected_indices,
):
    rc = create_1d_raised_cosine(
        duration=excitation_duration,
        start_time=0.010,
        end_time=0.012,
        amplitude=excitation_amplitude,
        sample_rate=sample_rate_plate,
    )

    weights_at_ex = (
        evaluate_rectangular_eigenfunctions(
            selected_indices,
            force_position,
            params=plate_params,
        )
        / plate_params.density
    )

    weights_at_readout_plate = evaluate_rectangular_eigenfunctions(
        selected_indices,
        readout_position_plate,
        params=plate_params,
    )

    modal_excitation = np.outer(rc, weights_at_ex)
    return modal_excitation, weights_at_readout_plate


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Get $\\gamma_{\\mu}$ and $\\omega_{\\mu}$ and integrate in time using the excitation force.""")
    return


@app.cell
def _(
    damping_term,
    dt_plate,
    lambda_mu_plate,
    modal_excitation,
    plate_params,
    solve_tf_excitation,
    stiffness_term,
):
    gamma2_mu_plate = damping_term(
        plate_params,
        lambda_mu_plate,
    )
    omega_mu_squared_plate = stiffness_term(
        plate_params,
        lambda_mu_plate,
    )


    def nl_fn(modal_sol):
        return 0.0


    _, modal_sol_plate = solve_tf_excitation(
        gamma2_mu_plate,
        omega_mu_squared_plate,
        modal_excitation,
        dt_plate,
        nl_fn=nl_fn,
    )
    return (modal_sol_plate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Get the solution in physical space""")
    return


@app.cell(hide_code=True)
def _(
    Audio,
    jnp,
    modal_sol_plate,
    plt,
    sample_rate_plate,
    weights_at_readout_plate,
):
    sol_plate = modal_sol_plate @ weights_at_readout_plate

    fig_plate, ax_plate = plt.subplots(1, 1, figsize=(6, 2))
    ax_plate.plot(sol_plate)
    ax_plate.grid(True)
    ax_plate.set_xlim(200, 4410 * 2)
    sol_vel = jnp.diff(sol_plate)
    audio_output_plate = Audio(sol_vel, rate=sample_rate_plate)
    audio_output_plate
    return


@app.cell(hide_code=True)
def _(
    K_plate,
    inverse_STL_2d,
    modal_sol_plate,
    n_modes_x,
    n_modes_y,
    np,
    plate_params,
    sorted_indices,
):
    modal_sol_unsorted = np.zeros_like(modal_sol_plate)
    modal_sol_unsorted[:, sorted_indices] = modal_sol_plate
    modal_sol_unsorted = modal_sol_unsorted.reshape(-1, n_modes_x, n_modes_y)
    modal_sol_unsorted = modal_sol_unsorted.transpose(1, 2, 0)
    sol_2d = inverse_STL_2d(K_plate, modal_sol_unsorted, plate_params.l1, plate_params.l2)
    return (sol_2d,)


@app.cell(hide_code=True)
def _(animation, mo, plate_params, plt, sol_2d):
    # Create a figure and axis for the animation

    # Get dimensions
    n_samples = 2000
    sol_reduced = sol_2d[:, :, 400:n_samples]
    nx, ny, nt = sol_reduced.shape

    # Downsample in time to make animation faster
    # Using a larger step to skip more frames since this is audio data (44.1kHz)
    step = 15  # Show every 1000th frame (about 44 frames per second)
    frames = nt // step

    fig_plate_anim, ax_plate_anim = plt.subplots(figsize=(6.4, 4.8))
    im = ax_plate_anim.imshow(
        sol_reduced[:, :, 0],
        cmap="viridis",
        origin="lower",
        extent=[0, plate_params.l1, 0, plate_params.l2],
        vmin=sol_2d.min(),
        vmax=sol_2d.max(),
    )

    # Add a colorbar
    cbar = fig_plate_anim.colorbar(im, ax=ax_plate_anim)
    cbar.set_label("Displacement")

    # Add labels and title
    ax_plate_anim.set_xlabel("x (m)")
    ax_plate_anim.set_ylabel("y (m)")
    ax_plate_anim.set_title("Plate Vibration")


    # Animation function
    def update_plate(frame):
        # Update the image data
        im.set_array(sol_reduced[:, :, frame * step])
        return [im]


    # Create the animation
    ani_plate = animation.FuncAnimation(
        fig_plate_anim,
        update_plate,
        frames=frames,
        interval=25,
        blit=True,
    )

    mo.Html(ani_plate.to_html5_video())
    return


if __name__ == "__main__":
    app.run()
