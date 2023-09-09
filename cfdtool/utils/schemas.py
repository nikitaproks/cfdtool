from dataclasses import dataclass

from pydantic import BaseModel
import jax.numpy as jnp


class Settings(BaseModel):
    width: int
    height: int
    iterations: int


class Node(BaseModel):
    x: int
    y: int


@dataclass
class SimulationOutput:
    iteration: int
    x_velocity: jnp.array
    y_velocity: jnp.array
    velocity_magnitude: jnp.array
    curl: jnp.array
