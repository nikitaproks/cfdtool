import logging

from classes.shapes import Circle
from classes.mesh import Mesh
from classes.store import Store
from utils.schemas import Node
from utils.functions import run_simulation

from utils.schemas import Settings

logger = logging.getLogger(__name__)


def simulation(store: Store, settings: Settings):
    logger.info("Simulation started running")
    domain_height = 50
    domain_width = 300
    circle_radius = domain_height // 9
    inlet_velocity = 0.04
    reynolds_number = 80

    center_node = Node(x=domain_width // 5, y=domain_height // 2)
    circle = Circle(circle_radius)
    mesh = Mesh(domain_width, domain_height)
    mesh.place(circle, center_node)
    print(mesh.get_grid())

    run_simulation(
        store,
        mesh.get_grid().astype(bool),
        domain_height,
        domain_width,
        circle_radius,
        inlet_velocity,
        reynolds_number,
        15000,
        True,
        1000,
    )
