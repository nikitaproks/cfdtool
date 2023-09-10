from utils.classes import Circle, Mesh
from utils.schemas import Node
from utils.functions import (
    run_simulation,
    calculate_kinematic_viscocity,
    initialize_velocity,
)


def simulation() -> None:
    domain_height = 100
    domain_width = 1000
    circle_radius = 15
    inlet_velocity = 0.04

    center_node = Node(x=50, y=350)
    circle = Circle(circle_radius)
    mesh = Mesh(domain_width, domain_height)
    mesh.place(circle, center_node)

    kinematic_viscocity = calculate_kinematic_viscocity(
        inlet_velocity, circle_radius, reynolds_number=20000
    )
    velocity_profile = initialize_velocity(
        inlet_velocity, domain_height, domain_width
    )
    run_simulation(
        15000,
        velocity_profile,
        kinematic_viscocity,
        mesh.get_grid(),
        n_x_points=domain_height,
        n_y_ponts=domain_width,
    )
