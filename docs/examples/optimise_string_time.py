
# %% [markdown]
# # Optimizing String Dynamics with a Differentiable ODE solver
#
# This example shows how to use the differentiable ODEs to learn nonlinear string dynamics, we can optionally replace the nonlinear function in the Störmer-Verlet integration scheme with a learnable neural network, therefore becoming a Neural ODE.

# %% Imports
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import scienceplots  # noqa: F401
from IPython.display import Audio, display
from jaxtyping import Array, ArrayLike, Float
from tqdm import tqdm

from jaxdiffmodal.excitations import create_pluck_modal
from jaxdiffmodal.ftm import (
    damping_term,
    evaluate_string_eigenfunctions,
    stiffness_term,
    string_eigenvalues,
    StringParameters,
)
from jaxdiffmodal.time_integrators import (
    solve_sv_ic,
    string_tau_with_density,
)


plt.style.use(["ieee", "no-latex"])
plt.rcParams["legend.framealpha"] = 1.0
plt.rcParams["legend.fancybox"] = True
plt.ioff()

# %% [markdown]
# ## Step 1: Generate Synthetic String Data
#
# First, we'll create synthetic string data using jaxdiffmodal's physical model.
# We'll generate both linear and nonlinear dynamics to have target data for training.

n_modes: int = 10
sample_rate: int = 16000
dt: float = 1.0 / sample_rate
n_steps_train: int = 1000
n_steps_test: int = 2000
n_steps_vis = 1000


# %% Generate the data
string_params = StringParameters()
indices = jnp.arange(n_modes) + 1

lambda_mu = string_eigenvalues(
    n_modes,
    string_params.length,
)
gamma2_mu = damping_term(
    string_params,
    lambda_mu,
)
omega_mu_squared = stiffness_term(
    string_params,
    lambda_mu,
)
exc = create_pluck_modal(
    lambdas=lambda_mu,
    string_length=string_params.length,
    initial_deflection=0.03,
)

weights = evaluate_string_eigenfunctions(
    indices=indices,
    position=jnp.array(0.6),
    params=string_params,
)
u0 = jnp.array(exc)
v0 = jnp.zeros_like(u0)
time = jnp.arange(n_steps_train) * dt

# %%

# %%
def create_static_filter(
    model,
    static_params_lambda,
):
    # Create a filter where everything is False (trainable) by default
    is_static_filter = jax.tree_util.tree_map(lambda _: False, model)

    # Get the parameters selected by the lambda to determine how many True values we need
    selected_params = static_params_lambda(model)

    # Create tuple of True values matching the number of selected parameters
    if isinstance(selected_params, tuple):
        true_values = tuple(True for _ in selected_params)
    else:
        # Single parameter case
        true_values = True

    is_static_filter = eqx.tree_at(
        static_params_lambda,
        is_static_filter,
        true_values,
    )
    return is_static_filter


class StringModel(eqx.Module):
    length: ArrayLike
    d3_with_density: ArrayLike
    log_Ts0_with_density: ArrayLike
    bending_stiffness_with_density: ArrayLike
    tau_with_density: ArrayLike
    v0: Array
    u0: Array
    weights: Array  # Modal weights for single position output
    mlp: eqx.Module | None = None

    def __call__(
        self,
        n_steps: int,
        dt: float,
        n_modes: int = 10,
        return_modal: bool = False,
    ) -> Float[Array, " n_steps"] | Float[Array, "n_steps n_modes"]:
        # Unpack parameters
        length: ArrayLike = self.length
        d3_with_density: ArrayLike = self.d3_with_density
        # Convert from log-space
        Ts0_with_density: ArrayLike = jnp.exp(self.log_Ts0_with_density)
        bending_stiffness_with_density: ArrayLike = self.bending_stiffness_with_density
        tau_with_density: ArrayLike = self.tau_with_density
        u0: Array = self.u0
        v0: Array = self.v0

        # get the analytical eigenvalues
        lambda_mu: Array = string_eigenvalues(
            n_modes,
            length,
        )

        # get the damping and stiffness terms
        omega_mu_squared: Array = (
            bending_stiffness_with_density * lambda_mu**2 + Ts0_with_density * lambda_mu
        )
        gamma2_mu: Array = d3_with_density * lambda_mu

        # calculate the factor for the nonlinear term
        string_norm: float = string_params.length / 2
        string_tau: Array = tau_with_density * lambda_mu / string_norm

        def nl_fn(q: ArrayLike) -> Array:
            return lambda_mu * q * (string_tau @ q**2)

        def nl_fn_nn(q: ArrayLike) -> Array:
            return lambda_mu * self.mlp(q)

        _, traj = solve_sv_ic(
            gamma2_mu=gamma2_mu,
            omega_mu_squared=omega_mu_squared,
            u0=u0,
            v0=v0,
            dt=dt,
            n_steps=n_steps,
            nl_fn=nl_fn_nn if self.mlp is not None else nl_fn,
        )

        if return_modal:
            return traj
        else:
            # Apply weights to get single position output
            # traj has shape (n_steps, n_modes)
            # weights has shape (n_modes,)
            # result should have shape (n_steps,)
            return traj @ self.weights

    def vector_field(
        self, q: ArrayLike, v: ArrayLike, use_lin: bool = False
    ) -> tuple[Array, Array]:
        """Compute the vector field for the string dynamics.

        Args:
            q: Position coordinates (modal displacements)
            v: Velocity coordinates (modal velocities)

        Returns:
            Tuple of (dot_q, dot_v) representing the time derivatives
        """
        length: float = self.length
        d3_with_density: float = self.d3_with_density * 200
        # Convert from log-space
        Ts0_with_density: ArrayLike = jnp.exp(self.log_Ts0_with_density)
        bending_stiffness_with_density: float = self.bending_stiffness_with_density
        tau_with_density: float = self.tau_with_density

        # get the analytical eigenvalues
        lambda_mu: Array = string_eigenvalues(
            n_modes,
            length,
        )

        # get the damping and stiffness terms
        omega_mu_squared: Array = (
            bending_stiffness_with_density * lambda_mu**2 + Ts0_with_density * lambda_mu
        )
        gamma2_mu: Array = d3_with_density * lambda_mu

        # calculate the factor for the nonlinear term
        string_norm: float = self.length / 2
        string_tau: Array = tau_with_density * lambda_mu / string_norm

        if not use_lin:

            def nl_fn(q: ArrayLike) -> Array:
                return lambda_mu * q * (string_tau @ q**2)
        else:

            def nl_fn(q: ArrayLike):
                return 0.0

        dot_q = v
        dot_v = -gamma2_mu * v - omega_mu_squared * q - nl_fn(q)
        return dot_q, dot_v


# %%


string_tau: float = string_tau_with_density(string_params)

# Instantiate the ground truth model
gt_model = StringModel(
    length=string_params.length,
    log_Ts0_with_density=jnp.log(string_params.Ts0 / string_params.density),
    d3_with_density=(string_params.d3 / string_params.density),
    bending_stiffness_with_density=(
        string_params.bending_stiffness / string_params.density
    ),
    tau_with_density=string_tau_with_density(string_params),
    u0=u0,
    v0=v0,
    weights=weights,  # Add weights from evaluate_eigenfunctions
    mlp=None,  # No MLP initially
)

# Get modal trajectories for visualization
targ_test_traj_modal: Array = gt_model(
    n_steps=n_steps_test,
    dt=dt,
    n_modes=n_modes,
    return_modal=True,  # Get modal trajectories
)

# Get weighted trajectory at single position for training target
targ_test_traj_position: Array = gt_model(
    n_steps=n_steps_test,
    dt=dt,
    n_modes=n_modes,
    return_modal=False,  # Get weighted single position
)

# slice a section for training
targ_train_traj_modal: Array = targ_test_traj_modal[:n_steps_train]
targ_train_traj_position: Array = targ_test_traj_position[:n_steps_train]

# %% Visualize the vector field in phase space for the first mode
use_lin = False
# Create a grid of points in phase space for the first mode
q_range = jnp.linspace(-0.04, 0.04, 20)
v_range = jnp.linspace(-60.0, 60.0, 20)
Q, V = jnp.meshgrid(q_range, v_range)

# Initialize arrays for the vector field
dQ = jnp.zeros_like(Q)
dV = jnp.zeros_like(V)
q_state = jnp.zeros(n_modes)
v_state = jnp.zeros(n_modes)

# Compute vector field at each grid point
for i in range(Q.shape[0]):
    for j in range(Q.shape[1]):
        # Create state vector with only first mode active
        q_state = q_state.at[0].set(Q[i, j])  # First mode position
        v_state = v_state.at[0].set(V[i, j])  # First mode velocity

        # Compute vector field
        dot_q, dot_v = gt_model.vector_field(q_state, v_state, use_lin=use_lin)

        # Store only the first mode components
        dQ = dQ.at[i, j].set(dot_q[0])
        dV = dV.at[i, j].set(dot_v[0])

# Create the phase space plot
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the vector field as equidistant arrows (convert JAX arrays to numpy)
speed = np.array(jnp.sqrt(dQ**2 + dV**2))

# Scale arrows to reasonable size while preserving relative magnitudes
max_speed = np.max(speed)
scale_factor = 0.0001  # Adjust this to control overall arrow size
dQ_scaled = np.array(dQ) * scale_factor
dV_scaled = np.array(dV) * scale_factor

quiver = ax.quiver(
    np.array(Q),
    np.array(V),
    dQ_scaled,
    dV_scaled,
    speed,
    cmap="viridis",
    scale=1,
    scale_units="xy",
    angles="xy",
    alpha=0.8,
    width=0.002,
    headwidth=3,
    headlength=4,
)

# Add a colorbar
# cbar = fig.colorbar(quiver, ax=ax, label="Speed")

# Plot some example trajectories
n_trajectories = 1
colors = plt.get_cmap("viridis")(jnp.linspace(0.0, 0.8, n_trajectories))

for idx, color in enumerate(colors):
    # Random initial condition for first mode
    q0_traj = jnp.zeros(n_modes)
    v0_traj = jnp.zeros(n_modes)
    q0_traj = q0_traj.at[0].set(0.035)
    v0_traj = v0_traj.at[0].set(0.0)  # Initial velocity

    # Simulate trajectory using solve_sv_ic directly to get both position and velocity
    length = gt_model.length
    d3_with_density = gt_model.d3_with_density * 200
    # Convert from log-space
    Ts0_with_density = jnp.exp(gt_model.log_Ts0_with_density)
    bending_stiffness_with_density = gt_model.bending_stiffness_with_density
    tau_with_density = gt_model.tau_with_density

    lambda_mu = string_eigenvalues(n_modes, length)
    omega_mu_squared = (
        bending_stiffness_with_density * lambda_mu**2 + Ts0_with_density * lambda_mu
    )
    gamma2_mu = d3_with_density * lambda_mu
    string_norm = length / 2
    string_tau = tau_with_density * lambda_mu / string_norm

    def nl_fn(q):
        return lambda_mu * q * (string_tau @ q**2)

    def nl_lin(q):
        return 0.0

    # Get both final state and trajectory
    (final_q, final_v), q_trajectory = solve_sv_ic(
        gamma2_mu=gamma2_mu,
        omega_mu_squared=omega_mu_squared,
        u0=q0_traj,
        v0=v0_traj,
        dt=dt,
        n_steps=1000,
        nl_fn=nl_fn if not use_lin else nl_lin,
    )

    # Extract first mode trajectory (position only from trajectory)
    q_traj = q_trajectory[:, 0]  # First mode position over time

    # For velocity, we need to compute it from the vector field at each time step
    # This is computationally expensive, so let's use a simpler approach:
    # Approximate velocity using finite differences
    v_traj = jnp.concatenate(
        [
            jnp.array([v0_traj[0]]),  # Initial velocity
            jnp.diff(q_traj) / dt,  # Approximate velocity from position differences
        ]
    )

    # Plot trajectory
    ax.plot(
        q_traj,
        v_traj,
        color="red",
        linewidth=1,
        alpha=0.7,
        linestyle="--",
        label=f"Traj. {idx + 1}",
    )
    ax.plot(q_traj[0], v_traj[0], "o", color=color, markersize=8)  # Start point
    ax.plot(q_traj[-1], v_traj[-1], "s", color=color, markersize=8)  # End point

# # Labels and formatting
ax.set_xlabel("Modal Displacement $q_1$", fontsize=18)
ax.set_ylabel("Modal Velocity $v_1$", fontsize=18)
# ax.set_title(
# "Phase Space Portrait - First String Mode\n(Vector Field + Sample Trajectories)",
# fontsize=14,
# )
ax.grid(True, alpha=0.3)
# ax.legend(loc="upper right", fontsize=24)
ax.set_xlim(float(q_range[0]), float(q_range[-1]))
ax.set_ylim(float(v_range[0]), float(v_range[-1]))

plt.tight_layout()
plt.show()
fig.savefig(f"phase_space_portrait_{'lin' if use_lin else 'nonlin'}.png")

# %%

# Initialize model with random weights for optimization
key = jax.random.PRNGKey(12345)
key_weights, key_len, key_Ts0, key_d3, key_u0 = jax.random.split(key, 5)

model = StringModel(
    length=jax.random.uniform(
        shape=(1,),
        minval=0.6,
        maxval=0.8,
        key=key_len,
    ),
    log_Ts0_with_density=jax.random.uniform(
        shape=(1,),
        minval=jnp.log(50_000),  # Initialize in log-space
        maxval=jnp.log(150_000),  # Range around 100_000
        key=key_Ts0,
    ),
    d3_with_density=jax.random.uniform(
        shape=(1,),
        minval=5.0,
        maxval=7.0,
        key=key_d3,
    ),
    bending_stiffness_with_density=(
        string_params.bending_stiffness / string_params.density
    ),
    tau_with_density=string_tau_with_density(string_params),
    u0=u0,  # * jax.random.normal(key_u0, shape=(n_modes,)) * 1.0,  # 1% perturbation
    v0=v0,  # Keep v0 as ground truth (static)
    weights=weights,  # jax.random.normal(key_weights, shape=(n_modes,)) * 0.2,
    mlp=None,
)


# Create the static filter using the wrapper function
# Only v0 is static; u0 and weights should be optimizable
is_static_filter = create_static_filter(
    model=model,
    static_params_lambda=lambda m: (
        m.v0,
        m.u0,
        m.weights,
        m.bending_stiffness_with_density,
    ),  # Only v0 and readout_position are static
)

# Now, partition the model using our custom filter
static_model, diff_model = eqx.partition(
    model,
    is_static_filter,
)

pred_init_traj_modal = model(
    n_steps=n_steps_train,
    dt=dt,
    n_modes=n_modes,
    return_modal=True,  # Get modal trajectories for visualization
)

plt.figure(figsize=(10, 4))
plt.plot(
    time[:n_steps_vis],
    targ_train_traj_modal[:n_steps_vis, :1],
    label="GT",
)

plt.plot(
    time[:n_steps_vis],
    pred_init_traj_modal[:n_steps_vis, :1],
    label="Neural ODE Initial Trajectory",
)

plt.title("Neural ODE Initial Trajectory")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()

targ_test_traj_phys: Array = targ_test_traj_modal @ weights
display(Audio(targ_test_traj_phys, rate=sample_rate))

# %% [markdown]
# Define the training loop and loss function.


# %% Visualization functions
def visualize_results(
    model,
    time: Array,
    losses: Array | None = None,
    weights: Array | None = None,
):
    """Visualize training results and model predictions."""
    print("Generating visualizations...")

    time_test = jnp.arange(n_steps_test) * dt

    # Generate predictions in modal space and single position
    pred_test_traj_modal = model(
        n_steps=n_steps_test,
        dt=dt,
        n_modes=n_modes,
        return_modal=True,  # Get modal trajectories for visualization
    )

    # Get single position predictions (weighted)
    pred_test_traj_phys: Array = model(
        n_steps=n_steps_test,
        dt=dt,
        n_modes=n_modes,
        return_modal=False,  # Get weighted position
    )

    # Target physical position
    targ_test_traj_phys: Array = targ_test_traj_position

    # Check if losses exist and training is complete
    plot_losses = losses is not None and len(losses) > 0

    # Create plots - adjust subplot layout based on whether we're plotting losses
    if plot_losses:
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(15, 10),
        )
        # Hide the bottom-left subplot that we won't use
        axes[1, 0].set_visible(False)
        # Plot loss in bottom right when training is complete
        loss_ax = axes[1, 1]
        loss_ax.semilogy(losses)
        loss_ax.set_title("Training Loss")
        loss_ax.set_xlabel("Epoch")
        loss_ax.set_ylabel("MSE Loss")
        loss_ax.grid(True)
    else:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(15, 5),
        )

    # Plot 1: Physical space comparison
    physical_ax = axes[0, 0] if plot_losses else axes[0]
    physical_ax.plot(
        time_test[: n_steps_vis * 2],
        targ_test_traj_phys[: n_steps_vis * 2],
        "b-",
        label="Target",
    )
    physical_ax.plot(
        time_test[: n_steps_vis * 2],
        pred_test_traj_phys[: n_steps_vis * 2],
        "r--",
        label="Neural ODE",
    )
    physical_ax.set_title("Physical Space Displacement")
    physical_ax.set_xlabel("Time (s)")
    physical_ax.set_ylabel("Displacement (m)")
    physical_ax.set_ylim(-0.0025, 0.0025)
    physical_ax.legend()
    physical_ax.grid(True)
    physical_ax.axvline(
        x=n_steps_train * dt,
        color="k",
        alpha=1.0,
        label="Train/Test Split",
    )

    # Plot 2: Modal amplitudes comparison
    modal_ax = axes[0, 1] if plot_losses else axes[1]
    for mode_idx in range(min(3, n_modes)):
        modal_ax.plot(
            time_test,
            targ_test_traj_modal[:, mode_idx],
            label=f"Target Mode {mode_idx + 1}",
            alpha=0.8,
        )
        modal_ax.plot(
            time_test,
            pred_test_traj_modal[:, mode_idx],
            "--",
            label=f"Pred Mode {mode_idx + 1}",
            alpha=0.8,
        )
    modal_ax.set_title("Multi-Mode Comparison")
    modal_ax.set_xlabel("Time (s)")
    modal_ax.set_ylabel("Amplitude")
    modal_ax.set_ylim(-0.0083, 0.0083)
    modal_ax.legend()
    modal_ax.grid(True)
    modal_ax.axvline(
        x=n_steps_train * dt,
        color="k",
        alpha=1.0,
        label="Train/Test Split",
    )

    plt.tight_layout()
    plt.show()


def save_animation_frame(
    model,
    time: Array,
    weights: Array,
    frame_idx: int,
    gt_model,
    output_dir: str = "tmp_node",
):
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    time_test = jnp.arange(n_steps_test) * dt

    # Generate predictions in modal space and single position
    pred_test_traj_modal = model(
        n_steps=n_steps_test,
        dt=dt,
        n_modes=n_modes,
        return_modal=True,  # Get modal trajectories for visualization
    )

    # Get single position predictions (weighted)
    pred_test_traj_phys: Array = model(
        n_steps=n_steps_test,
        dt=dt,
        n_modes=n_modes,
        return_modal=False,  # Get weighted position
    )

    # Target physical position
    targ_test_traj_phys: Array = targ_test_traj_position

    # Create figure with centered plot and table underneath
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.4)

    # Plot 1: Physical space comparison (centered)
    physical_ax = fig.add_subplot(gs[0, 0])
    physical_ax.plot(
        time_test[: n_steps_vis * 2],
        targ_test_traj_phys[: n_steps_vis * 2],
        "b-",
        label="Target",
    )
    physical_ax.plot(
        time_test[: n_steps_vis * 2],
        pred_test_traj_phys[: n_steps_vis * 2],
        "r--",
        label="Optim",
    )
    physical_ax.set_title("Physical Space Displacement")
    physical_ax.set_xlabel("Time (s)")
    physical_ax.set_ylabel("Displacement (m)")
    physical_ax.set_ylim(-0.0025, 0.0025)
    physical_ax.legend()
    physical_ax.grid(True)
    physical_ax.axvline(
        x=n_steps_train * dt,
        color="k",
        linestyle=":",
        alpha=0.7,
        label="Train/Test Split",
    )

    # Skip modal comparison plot for cleaner animation frames

    # Add parameter table
    table_ax = fig.add_subplot(gs[1, 0])
    table_ax.axis("off")

    # Create table data - handle both JAX arrays and floats
    def format_param(param):
        return param.item() if hasattr(param, "item") else param

    table_data = [
        ["Parameter", "Current", "Ground Truth"],
        [
            "Length",
            f"{format_param(model.length):.4f}",
            f"{format_param(gt_model.length):.4f}",
        ],
        [
            r"$\hat{d}_3$",
            f"{format_param(model.d3_with_density):.6f}",
            f"{format_param(gt_model.d3_with_density):.6f}",
        ],
        [
            r"$\hat{T}_0$",
            # Show actual value, not log
            f"{format_param(jnp.exp(model.log_Ts0_with_density)):.1f}",
            f"{format_param(jnp.exp(gt_model.log_Ts0_with_density)):.1f}",
        ],
    ]

    table = table_ax.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.25, 0.25, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)

    # Style the header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/frame_{frame_idx:05d}.png", dpi=150, bbox_inches="tight")
    plt.close()


# %% Training functions
def train_neural_ode(model, save_frames=False, frame_interval=50):
    print("Training...")

    # normalise target trajectory for better training stability
    # Now using single position target instead of modal trajectories
    scale: float = jnp.max(jnp.abs(targ_train_traj_position)).item()
    targ_train_traj_position_scaled = targ_train_traj_position / scale

    @eqx.filter_jit
    def training_step(
        model,
        optimizer,
        opt_state,
        targ_train_traj_position_scaled,
    ):
        @eqx.filter_value_and_grad
        def loss_fn(
            diff_model,
            static_model,
            targ_train_traj_position_scaled,
        ):
            model: eqx.Module = eqx.combine(diff_model, static_model)

            # Get single position output (weighted modal trajectory)
            pred_train_traj_position: Array = model(
                n_steps=n_steps_train,
                dt=dt,
                n_modes=n_modes,
                return_modal=False,  # Get weighted position
            )

            pred_train_traj_position: Array = (
                pred_train_traj_position / scale
            )  # normalise predictions

            # MSE loss
            mse_loss = jnp.mean(
                (pred_train_traj_position - targ_train_traj_position_scaled) ** 2
            )

            # Soft constraint: penalty increases quadratically outside bounds
            u0_penalty = 1000 * jnp.mean(jnp.square(model.u0))

            # Light regularization on readout position (keep it reasonable within string bounds)
            # position_reg = 0.01 * (model.readout_position - 0.6) ** 2

            total_loss = mse_loss  # + u0_penalty #+ position_reg

            # Check for NaN and return large finite loss if detected
            # total_loss = jnp.where(
            #     jnp.isnan(total_loss),
            #     1e10,  # Large but finite penalty for NaN
            #     total_loss,
            # )

            return total_loss

        static_model, diff_model = eqx.partition(
            model,
            is_static_filter,
        )
        loss_value, grads = loss_fn(
            diff_model,
            static_model,
            targ_train_traj_position_scaled,
        )

        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    # Training setup
    epochs = 10000
    learning_rate = 1e-4  # Very small learning rate for stability
    schedule = optax.cosine_onecycle_schedule(
        transition_steps=epochs,
        peak_value=learning_rate,
    )
    # Add gradient clipping for stability
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adabelief(schedule),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    losses = []

    # Training loop - simplified with fixed regularization
    bar = tqdm(range(epochs))
    for epoch in bar:
        model, opt_state, loss_value = training_step(
            model,
            optimizer,
            opt_state,
            targ_train_traj_position_scaled,
        )
        losses.append(loss_value)

        # Early stopping if NaN detected or loss explodes
        if jnp.isnan(loss_value) or loss_value > 1e8:
            print(
                f"\nWarning: Training stopped early at epoch {epoch + 1} due to instability"
            )
            print(f"Loss value: {loss_value}")
            break

        bar.set_description(f"Epoch {epoch + 1}/{epochs} | Loss: {loss_value:.6f}")

        # Save animation frame periodically
        if save_frames and epoch % frame_interval == 0:
            save_animation_frame(
                model=model,
                time=time,
                weights=weights,
                frame_idx=epoch // frame_interval,
                gt_model=gt_model,
            )

    return model, losses


# %% visualise the initial model predictions
# First visualisation of the initial model
visualize_results(
    model=model,
    time=time,
    weights=weights,
)


# %%
# %% Train the model
print("Starting training...")
trained_model, training_losses = train_neural_ode(
    model,
    save_frames=True,
    frame_interval=100,
)
print(f"Training completed! Final loss: {training_losses[-1]:.6f}")

# %% Generate final results
visualize_results(
    model=trained_model,
    time=time,
    weights=weights,
    losses=training_losses,
)
# %% [markdown]
# %%
# long_traj
print(model)
# %%

plt.plot(weights, label="Initial u0")
plt.plot(model.weights, label="Trained u0")
plt.show()

# %%
render_animation_movie(
    input_dir="tmp_node",
    output_file="node_training.webm",
    framerate=10,
)

# %%
