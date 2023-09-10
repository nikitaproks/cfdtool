import time
import logging


import jax
import jax.numpy as jnp
from tqdm import tqdm

# from utils.schemas import SimulationOutput
from utils.constants import Lattice, NodeVelocity

from utils.classes import Store

# logger = logging.getLogger(__name__)
jax_logger = logging.getLogger("jax")
jax_logger.setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_density(discrete_velocities: jnp.array) -> jnp.array:
    density = jnp.sum(discrete_velocities, axis=-1)

    return density


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
    macroscopic_velocities: jnp.array, density: jnp.array
) -> jnp.array:
    projected_discrete_velocities = jnp.einsum(
        "dQ,NMd->NMQ",
        Lattice.VELOCITIES,
        macroscopic_velocities,
    )
    macroscopic_velocity_magnitude = jnp.linalg.norm(
        macroscopic_velocities,
        axis=-1,
        ord=2,
    )
    equilibrium_discrete_velocities = (
        density[..., jnp.newaxis]
        * Lattice.WEIGHTS[jnp.newaxis, jnp.newaxis, :]
        * (
            1
            + 3 * projected_discrete_velocities
            + 9 / 2 * projected_discrete_velocities**2
            - 3 / 2 * macroscopic_velocity_magnitude[..., jnp.newaxis] ** 2
        )
    )
    return equilibrium_discrete_velocities


def simulation_management(store: Store) -> bool:
    if store.is_simulation_paused():
        logger.info("Simulation paused")
        while store.is_simulation_paused():
            if store.is_simulation_reset():
                logger.info("Simulation reset. Exiting simulation loop")
                return False
            time.sleep(0.5)
    # logger.info("Simulation continued running")
    return True


def run_simulation(
    store: Store,
    mesh: jnp.array,
    domain_height: int,
    domain_width: int,
    length_scale: float,
    inlet_velocity: float,
    reynolds_number: int,
    iterations: int,
    visualize: bool,
    output_interval: int,
    delay=4000,
) -> None:
    """
    The Simulation.
    """
    jax.config.update("jax_enable_x64", True)

    kinematic_viscosity = (inlet_velocity * length_scale) / (reynolds_number)
    relaxation_omega = (1.0) / (3.0 * kinematic_viscosity + 0.5)

    velocity_profile = jnp.zeros((domain_width, domain_height, 2))
    velocity_profile = velocity_profile.at[:, :, 0].set(inlet_velocity)

    @jax.jit
    def update(discrete_velocities_prev):
        # (1) Prescribe the outflow BC on the right boundary
        discrete_velocities_prev = discrete_velocities_prev.at[
            -1, :, NodeVelocity.LEFT
        ].set(discrete_velocities_prev[-2, :, NodeVelocity.LEFT])

        # (2) Macroscopic Velocities
        density_prev = get_density(discrete_velocities_prev)
        macroscopic_velocities_prev = get_macroscopic_velocities(
            discrete_velocities_prev,
            density_prev,
        )

        # (3) Prescribe Inflow Dirichlet BC using Zou/He scheme
        macroscopic_velocities_prev = macroscopic_velocities_prev.at[
            0, 1:-1, :
        ].set(velocity_profile[0, 1:-1, :])
        density_prev = density_prev.at[0, :].set(
            (
                get_density(
                    discrete_velocities_prev[
                        0, :, NodeVelocity.PURE_VERTICAL
                    ].T
                )
                + 2
                * get_density(
                    discrete_velocities_prev[0, :, NodeVelocity.LEFT].T
                )
            )
            / (1 - macroscopic_velocities_prev[0, :, 0])
        )

        # (4) Compute discrete Equilibria velocities
        equilibrium_discrete_velocities = get_equilibrium_discrete_velocities(
            macroscopic_velocities_prev,
            density_prev,
        )

        # (3) Belongs to the Zou/He scheme
        discrete_velocities_prev = discrete_velocities_prev.at[
            0, :, NodeVelocity.RIGHT
        ].set(equilibrium_discrete_velocities[0, :, NodeVelocity.RIGHT])

        # (5) Collide according to BGK
        discrete_velocities_post_collision = (
            discrete_velocities_prev
            - relaxation_omega
            * (discrete_velocities_prev - equilibrium_discrete_velocities)
        )

        # (6) Bounce-Back Boundary Conditions to enfore the no-slip
        for i in range(9):
            discrete_velocities_post_collision = (
                discrete_velocities_post_collision.at[
                    mesh, Lattice.INDICES[i]
                ].set(
                    discrete_velocities_prev[mesh, Lattice.OPOSITE_INDECES[i]]
                )
            )

        # (7) Stream alongside lattice velocities
        discrete_velocities_streamed = discrete_velocities_post_collision
        for i in range(9):
            discrete_velocities_streamed = discrete_velocities_streamed.at[
                :, :, i
            ].set(
                jnp.roll(
                    jnp.roll(
                        discrete_velocities_post_collision[:, :, i],
                        Lattice.VELOCITIES[0, i],
                        axis=0,
                    ),
                    Lattice.VELOCITIES[1, i],
                    axis=1,
                )
            )

        return discrete_velocities_streamed

    discrete_velocities_prev = get_equilibrium_discrete_velocities(
        velocity_profile,
        jnp.ones((domain_width, domain_height)),
    )

    for iteration_index in tqdm(range(iterations)):
        discrete_velocities_next = update(discrete_velocities_prev)

        discrete_velocities_prev = discrete_velocities_next

        if (
            iteration_index % output_interval == 0
            and visualize
            and iteration_index > delay
        ):
            density = get_density(discrete_velocities_next)
            macroscopic_velocities = get_macroscopic_velocities(
                discrete_velocities_next,
                density,
            )
            velocity_magnitude = jnp.linalg.norm(
                macroscopic_velocities,
                axis=-1,
                ord=2,
            )
            _, d_u__d_y = jnp.gradient(macroscopic_velocities[..., 0])
            d_v__d_x, _ = jnp.gradient(macroscopic_velocities[..., 1])
            curl = d_u__d_y - d_v__d_x

            output: list[list[float]] = velocity_magnitude.tolist()
            store.put(output)
