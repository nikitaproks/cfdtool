from enum import Enum

from pydantic import BaseModel

import jax.numpy as jnp


class Lattice(Enum):
    VELOCITIES = jnp.array(
        [[0, 1, 0, -1, 0, 1, -1, -1, 1], [0, 0, 1, 0, -1, 1, 1, -1, -1]]
    )
    INDICES = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    WEIGHTS = jnp.array(
        [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]
    )
    OPOSITE_INDECES = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])


class NodeVelocity(Enum):
    RIGHT = jnp.array([1, 5, 8])
    UP = jnp.array([2, 5, 6])
    LEFT = jnp.array([3, 6, 7])
    DOWN = jnp.array([4, 7, 8])
    PURE_VERTICAL = jnp.array([0, 2, 4])
    PURE_HORIZONTAL = jnp.array([0, 1, 3])


class Node(BaseModel):
    x: int
    y: int
