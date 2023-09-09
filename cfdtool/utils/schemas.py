from enum import Enum

from pydantic import BaseModel


class Lattice(Enum):
    VELOCITIES = [
        [0, 1, 0, -1, 0, 1, -1, -1, 1],
        [0, 0, 1, 0, -1, 1, 1, -1, -1],
    ]
    INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    WEIGHTS = [
        4 / 9,
        1 / 9,
        1 / 9,
        1 / 9,
        1 / 9,
        1 / 36,
        1 / 36,
        1 / 36,
        1 / 36,
    ]
    OPOSITE_INDECES = [0, 3, 4, 1, 2, 7, 8, 5, 6]


class NodeVelocity(Enum):
    RIGHT = [1, 5, 8]
    UP = [2, 5, 6]
    LEFT = [3, 6, 7]
    DOWN = [4, 7, 8]
    PURE_VERTICAL = [0, 2, 4]
    PURE_HORIZONTAL = [0, 1, 3]


class Settings(BaseModel):
    width: int
    height: int
    iterations: int


class Node(BaseModel):
    x: int
    y: int
