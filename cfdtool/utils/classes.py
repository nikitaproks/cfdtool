from queue import Queue

import jax.numpy as jnp
from jax.lax import dynamic_update_slice

from utils.schemas import Node


class Shape:
    def __init__(self):
        self.grid = self.create_grid()

    def area(self):
        raise NotImplementedError(
            "This method should be implemented by the subclasses."
        )

    def create_grid(self):
        raise NotImplementedError(
            "This method should be implemented by the subclasses."
        )


class Rectangle(Shape):
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        super().__init__()

    def create_grid(self) -> jnp.ndarray:
        grid = jnp.ones((self.height, self.width))
        return grid.astype(jnp.int32)


class Circle(Shape):
    def __init__(self, radius: int):
        self.radius = radius
        super().__init__()

    def create_grid(self) -> jnp.ndarray:
        diameter = 2 * self.radius

        grid_points = int(diameter)

        # Create a meshgrid
        x, y = jnp.meshgrid(
            jnp.linspace(-self.radius, self.radius, grid_points),
            jnp.linspace(-self.radius, self.radius, grid_points),
        )

        # Calculate the distance from the center (0,0)
        distance_from_center = jnp.sqrt(x**2 + y**2)

        # Create a boolean mask for the circle
        circle_mask = distance_from_center <= self.radius
        return circle_mask.astype(jnp.int32)


class Mesh:
    def __init__(self, width: int, height: int) -> None:
        self.grid = jnp.zeros((width, height)).astype(jnp.int32)

    def place(self, shape: Shape, center: Node):
        start_x = center.x - shape.grid.shape[0] // 2
        start_y = center.y - shape.grid.shape[1] // 2

        self.grid = dynamic_update_slice(
            self.grid,
            shape.grid,
            (jnp.int32(start_x), jnp.int32(start_y)),
        )

    def get_grid(self):
        return self.grid


class Store:
    def __init__(self) -> None:
        self.queue: Queue = Queue()
        self.simulation_running: bool = False
        self.simulation_paused: bool = False
        self.reset: bool = False

    def put(self, data):
        self.queue.put(data)

    def get(self):
        return self.queue.get()

    def empty(self):
        return self.queue.empty()

    def set_simulation_running(self, simulation_running):
        self.simulation_running = simulation_running

    def is_simulation_running(self):
        return self.simulation_running

    def is_simulation_paused(self):
        return self.simulation_paused

    def set_simulation_paused(self, simulation_paused):
        self.simulation_paused = simulation_paused

    def set_simulation_reset(self, reset):
        self.reset = reset

    def is_simulation_reset(self):
        return self.reset
