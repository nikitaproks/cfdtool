import jax
import jax.numpy as jnp
from tqdm import tqdm
from utils.schemas import Lattice, NodeVelocity, SimulationOutput


def euclidean_norm(to_be_normed: jnp.array, norm_axis=-1) -> jnp.array:
    return jnp.linalg.norm(to_be_normed, axis=norm_axis, ord=2)


def get_density(discrete_velocities: jnp.array) -> jnp.array:
    density = jnp.sum(discrete_velocities, axis=-1)
    return jnp.array(density)


def get_macroscopic_velocities(
    discrete_velocities: jnp.array, density: jnp.array
) -> jnp.array:
    macroscopic_velocities = (
        jnp.einsum(
            "NMQ,dQ->NMd",
            discrete_velocities,
            Lattice.VELOCITIES,
        )
        / density[..., jnp.newaxis]
    )
    return macroscopic_velocities


def get_equilibrium_discrete_velocities(
    macroscopic_velocites: jnp.array, density: jnp.array
) -> jnp.array:
    projected_discrete_velocites = jnp.einsum(
        "dQ,NMd -> NMQ", Lattice.VELOCITIES, macroscopic_velocites
    )
    macroscopic_velocites_magnitude = euclidean_norm(
        projected_discrete_velocites
    )
    equilibrium_discrete_velocites = (
        density[..., jnp.newaxis]
        * Lattice.WEIGHTS[jnp.newaxis, jnp.newaxis, :]
        * (
            1
            + 3 * projected_discrete_velocites
            + 9 / 2 * projected_discrete_velocites**2
            - 3 / 2 * macroscopic_velocites_magnitude[..., jnp.newaxis] ** 2
        )
    )
    return equilibrium_discrete_velocites


def calculate_kinematic_viscocity(
    velocity: float, length_scale: float, reynolds_number: float
) -> float:
    """
    Implicit value defined by the Reynolds Number.
    """
    return (velocity * length_scale) / reynolds_number


def relaxation_omega(kinematic_viscocity: float) -> float:
    return 1.0 / (3.0 * kinematic_viscocity * 0.5)


# TODO: unused, delete
def generate_mesh(x_points: int, y_points: int) -> tuple[jnp.array, jnp.array]:
    x = jnp.arrange(x_points)
    y = jnp.arrange(y_points)
    X, Y = jnp.meshgrid(x, y, "ij")
    return (X, Y)


def initialize_velocity(
    inlet_velocity: float, x_points: int, y_points: int
) -> jnp.array:
    velocity_profile = jnp.zeros((x_points, y_points, 2))
    velocity_profile = velocity_profile.at[:, :, 0].set(inlet_velocity)
    return velocity_profile


@jax.jit
def lbm_update(
    discrete_velocity_prev: jnp.array,
    velocity_profile: jnp.array,
    kinematic_viscocity: float,
    obstacle_mask,
) -> jnp.array:
    """
    The Algorithm at every step.
    """
    # 1. Perscribe outflow BC at right boundary
    discrete_velocity_prev = discrete_velocity_prev.at[
        -1,
        :,
        NodeVelocity.LEFT,
    ].set(discrete_velocity_prev[-2, :, NodeVelocity.LEFT])

    # 2. Compute macroscopic velocities
    density_prev = get_density(discrete_velocity_prev)
    macroscopic_velocities_prev = get_macroscopic_velocities(
        discrete_velocity_prev, density_prev
    )

    # 3. Inflow Dirichlet with Zou/He scheme
    macroscopic_velocities_prev = macroscopic_velocities_prev.at[
        0, 1:-1, :
    ].set(velocity_profile[0, 1:-1, :])

    density_prev = density_prev.at[0, :].set(
        (
            get_density(
                discrete_velocity_prev[0, :, NodeVelocity.PURE_VERTICAL].T
            )
            + 2
            * get_density(discrete_velocity_prev[0, :, NodeVelocity.LEFT].T)
        )
        / (1 - macroscopic_velocities_prev[0, :, 0])
    )

    # 4. Compute discrete Equilibria velocites
    equilibrium_discrete_velocites = get_equilibrium_discrete_velocities(
        macroscopic_velocities_prev, density_prev
    )
    discrete_velocity_prev = discrete_velocity_prev.at[
        0, :, NodeVelocity.RIGHT
    ].set(equilibrium_discrete_velocites[0, :, NodeVelocity.RIGHT])

    # 5. Collision Step

    discrete_velocity_post_collision = (
        discrete_velocity_prev
        - relaxation_omega(kinematic_viscocity)
        * (discrete_velocity_prev - equilibrium_discrete_velocites)
    )
    # Obstacle mask BC
    for i in range(9):
        discrete_velocity_post_collision = discrete_velocity_post_collision.at[
            obstacle_mask, Lattice.INDICES[i]
        ].set(
            discrete_velocity_prev[obstacle_mask, Lattice.OPOSITE_INDECES[i]]
        )

    # (7) Stream alongside lattice velocities
    discrete_velocities_streamed = discrete_velocity_post_collision
    for i in range(9):
        discrete_velocities_streamed = discrete_velocities_streamed.at[
            :, :, i
        ].set(
            jnp.roll(
                jnp.roll(
                    discrete_velocity_post_collision[:, :, i],
                    Lattice.VELOCITIES[0, i],
                    axis=0,
                ),
                Lattice.VELOCITIES[1, i],
                axis=1,
            )
        )

    return discrete_velocities_streamed


def run_simulation(
    iterations: int,
    velocity_profile,
    kinematic_viscocity,
    obstacle_mask,
    n_x_points,
    n_y_ponts,
    plot_output_steps=100,
    visualize=True,
    skip_first_index=0,
) -> list[SimulationOutput]:
    """
    The Simulation.
    """
    simulation_output = []
    # inlet vel profile needs to be set. maybe a function that gives curves
    discrete_velocities_prev = get_equilibrium_discrete_velocities(
        velocity_profile,
        jnp.ones((n_x_points, n_y_ponts)),
    )
    for iteration_index in tqdm(range(iterations)):
        discrete_velocities_next = lbm_update(
            discrete_velocities_prev,
            velocity_profile,
            kinematic_viscocity,
            obstacle_mask,
        )
        discrete_velocities_prev = discrete_velocities_next
        if (
            iteration_index % plot_output_steps == 0
            and visualize
            and iteration_index > skip_first_index
        ):
            density = get_density(discrete_velocities_next)
            macroscopic_velocities = get_macroscopic_velocities(
                discrete_velocities_next,
                density,
            )
            velocity_mag = euclidean_norm(macroscopic_velocities)

            # calc_curl - du_dx and dv_dy ignored
            _, d_u__d_y = jnp.gradient(macroscopic_velocities[..., 0])
            d_v__d_x, _ = jnp.gradient(macroscopic_velocities[..., 1])
            curl = d_u__d_y - d_v__d_x

            simulation_output.append(
                SimulationOutput(
                    iteration_index,
                    macroscopic_velocities[..., 0],
                    macroscopic_velocities[..., 1],
                    velocity_mag,
                    curl,
                )
            )
    return simulation_output
