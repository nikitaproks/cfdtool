import jax.numpy as jnp
from jax.lax import dynamic_update_slice

from classes.shapes import Shape
from utils.schemas import Node


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
