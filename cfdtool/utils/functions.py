import jax
import jax.numpy as jnp

# TODO: pull from github
# TODO: implement enums
# TODO: finish algorithn
# TODO: debug

def euclidean_norm(to_be_normed: jnp.array[float], norm_axis = -1) -> jnp.array[float]:
    return jnp.linalg.norm(
        to_be_normed,
        axis = norm_axis,
        ord = 2
    )

def get_density(discrete_velocities: jnp.array[float]) -> jnp.array[float]:

    density = jnp.sum(discrete_velocities, axis = -1)
    return density

def get_macroscopic_velocities(discrete_velocities: jnp.array[float],
                                density: jnp.array[float]) -> jnp.array[float]:
    macroscopic_velocities = jnp.einsum(
        "NMQ,dQ->NM",
        discrete_velocities,
        LATTICE_VELOCITES, # TODO: replace
    ) / density[...,jnp.newaxis]
    return macroscopic_velocities

def get_equilibrium_discrete_velocites(macroscopic_velocites: jnp.array[float],
                                       density: jnp.array[float]) -> jnp.array[float]:
    
    projected_discrete_velocites = jnp.einsum(
        "dQ,NMd -> NMQ",
        LATTICE_VELOCITES,
        macroscopic_velocites
    )
    macroscopic_velocites_magnitude = euclidean_norm(projected_discrete_velocites)
    equilibrium_discrete_velocites = (
        density[..., jnp.newaxis]
        *
        LATTICE_WEIGHTS[jnp.newaxis, jnp.newaxis, :]
        *
        (
            1
            +
            3 * projected_discrete_velocites
            + 
            9/2 * projected_discrete_velocites ** 2
            - 
            3/2 * macroscopic_velocites_magnitude[..., jnp.newaxis]**2
        )
    )
    return equilibrium_discrete_velocites

def calculate_kinematic_viscocity(velocity: float, 
                                  length_scale: float, 
                                  reynolds_number: float) -> float:
    """
    Implicit value defined by the Reynolds Number.
    """
    return (velocity*length_scale)/reynolds_number

def relaxation_omega(kinematic_viscocity: float) -> float:
    return 1.0/(3.0*kinematic_viscocity*0.5)

def generate_mesh(x_points: int, y_points: int) -> tuple[jnp.array, jnp.array]:
    x = jnp.arrange(x_points)
    y = jnp.arrange(y_points)
    X, Y = jnp.meshgrid(x, y, 'ij')
    return (X, Y)

def initialize_velocity(inlet_velocity: float, x_points: int, y_points: int) -> jnp.array:
    velocity_profile = jnp.zeros((x_points, y_points, 2))
    velocity_profile = velocity_profile.at[:,:,0].set(inlet_velocity)
    return velocity_profile

def lbm_update(discrete_velocity_prev: jnp.array, 
               velocity_profile: jnp.array, 
               kinematic_viscocity: float,
               obstacle_mask: jnp.array[jnp.int32]) -> jnp.array:
    """
    The Algorithm
    """
    # 1. Perscribe outflow BC at right boundary
    discrete_velocity_prev = discrete_velocity_prev.at[-1,:, LEFT_VELOCITES].set(discrete_velocity_prev[-2,:, LEFT_VELOCITES])

    # 2. Compute macroscopic velocities
    density_prev = get_density(discrete_velocity_prev)
    macroscopic_velocities_prev = get_macroscopic_velocities(discrete_velocity_prev, density_prev)

    # 3. Inflow Dirichlet with Zou/He scheme
    macroscopic_velocities_prev = macroscopic_velocities_prev.at[0,1:-1, :].set(
        velocity_profile[0, 1:-1, :]
    )
    density_prev = density_prev[0, :].set(
        (
            get_density(discrete_velocity_prev[0, :, PURE_VERT_VELOCItieS]. T)
            +
            2 * 
            get_density(discrete_velocity_prev[0, :, LEFT_VEL].T)
        ) / (
            1 - macroscopic_velocities_prev[0, :, 0]
        )
    )

    # 4. Compute discrete Equilibria velocites
    equilibrium_discrete_velocites = get_equilibrium_discrete_velocites(
        macroscopic_velocities_prev, 
        density_prev
    )
    discrete_velocity_prev = discrete_velocity_prev.at[0, :, RIGHT_VEL].set(
        equilibrium_discrete_velocites.at[0, :, RIGHT_VEL]
    )

    # 5. Collision Step

    discrete_velocit_post_collision = (
        discrete_velocity_prev
        -
        relaxation_omega(kinematic_viscocity)
        *
        (
            discrete_velocity_prev
            -
            equilibrium_discrete_velocites
        )
    )
    # Obstacle mask BC
    for i,_ in enumerate(N_DISCRETE_VELOCITiES):
        discrete_velocit_post_collision  = discrete_velocit_post_collision.at[obstacle_mask, LATTICE_INDICIES[i]].set(
            discrete_velocity_prev[obstacle_mask, OPPOSITE_LATTICE[i]]
        )