import jax.numpy as jnp
from jax.lax import dynamic_update_index_in_dim

from tqdm import tqdm

from cfdtool.utils.constants import AirfoilEnum


class Shape:
    grid = None

    def __init__(self):
        self.grid = self.empty_grid()

    def empty_grid(self):
        raise NotImplementedError(
            "This method should be implemented by the subclasses."
        )

    def construct(self):
        raise NotImplementedError(
            "This method should be implemented by the subclasses."
        )

    def rotate(self, angle: float):
        # Angle in degrees
        raise NotImplementedError(
            "This method should be implemented by the subclasses."
        )

    def get_grid(self):
        if self.grid is None:
            raise ValueError("The grid is empty. Please construct it first.")
        return self.grid


class Rectangle(Shape):
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        super().__init__()

    def empty_grid(self) -> jnp.ndarray:
        grid = jnp.zeros((self.height, self.width))
        return grid.astype(jnp.int32)

    def construct(self) -> jnp.ndarray:
        self.grid = jnp.ones((self.height, self.width)).astype(jnp.int32)
        return self.grid


class Circle(Shape):
    def __init__(self, radius: int):
        self.radius = radius
        super().__init__()

    def empty_grid(self) -> jnp.ndarray:
        grid = jnp.zeros((self.radius * 2, self.radius * 2))
        return grid.astype(jnp.int32)

    def construct(self) -> jnp.ndarray:
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
        self.grid = circle_mask.astype(jnp.int32)
        return self.grid


class Airfoil(Shape):
    path = "cfdtool/data/airfoils/"

    def __init__(self, airfoil_type: AirfoilEnum, chord_length_nodes: int):
        self.chord_length_nodes = chord_length_nodes
        self.original_points: jnp.ndarray = self._read_airfoil(airfoil_type)
        self.contour_points: jnp.ndarray = self.get_contour(
            self.original_points, chord_length_nodes
        )
        super().__init__()

    def get_contour(self, points: jnp.ndarray, chord_length_nodes: int):
        scaled_points = jnp.round(points * chord_length_nodes)
        contour_points = dynamic_update_index_in_dim(
            scaled_points,
            (scaled_points[:, 1] - scaled_points[:, 1].min()),
            1,
            axis=1,
        ).astype(jnp.int32)
        return contour_points

    def _read_airfoil(self, airfoil_type: AirfoilEnum):
        data_raw = ""
        # Reading the airfoil data from the file
        with open(f"{self.path}/{airfoil_type.name.lower()}.dat", "r") as file:
            data_raw = file.read()
        data_list = data_raw.split("\n")
        points = jnp.array(
            [
                list(map(float, line.strip().replace("  ", " ").split(" ")))
                for line in data_list[1:]
            ]
        )
        return points

    def empty_grid(self):
        width = (
            self.contour_points[:, 0].max() - self.contour_points[:, 0].min()
        ) + 1
        height = (
            self.contour_points[:, 1].max() - self.contour_points[:, 1].min()
        ) + 1
        grid = jnp.zeros((height, width))
        return grid.astype(jnp.int32)

    def rotate(self, degrees: float):
        midpoint_x = (self.contour_points[:, 0].max() + 1) // 2
        midpoint_y = (self.contour_points[:, 1].max() + 1) // 2
        radians = degrees * jnp.pi / 180

        output_points = []
        self.or_cont = self.contour_points
        for point in self.contour_points:
            x_dash = point[0] - midpoint_x
            y_dash = point[1] - midpoint_y

            x_dash_dash = x_dash * jnp.cos(radians) - y_dash * jnp.sin(radians)
            y_dash_dash = x_dash * jnp.sin(radians) + y_dash * jnp.cos(radians)

            x = midpoint_x + x_dash_dash
            y = midpoint_y + y_dash_dash
            output_points.append([round(x), round(y)])

        output_points = jnp.array(output_points)
        # Adjust for negatives
        min_x = output_points[:, 0].min()
        min_y = output_points[:, 1].min()

        output_points = output_points - jnp.array([min_x, min_y])

        self.contour_points = jnp.array(output_points).astype(jnp.int32)
        self.grid = self.empty_grid()

    def construct(self):
        points = self.contour_points
        for i in range(points.shape[0]):
            point = points[i]
            next_point = (
                points[i + 1] if i + 1 < points.shape[0] else points[0]
            )
            self.grid = self.grid.at[point[1], point[0]].set(1)
            intermediate_points = self._bresenham_line(
                point[0], point[1], next_point[0], next_point[1]
            )
            for inter_point in intermediate_points:
                self.grid = self.grid.at[
                    inter_point[1],
                    inter_point[0],
                ].set(1)
        self.grid = self._flood_fill(
            self.grid, self.grid.shape[1] // 2, self.grid.shape[0] // 2, 0, 1
        ).astype(jnp.int32)
        return self.grid

    def _bresenham_line(self, x0, y0, x1, y1):
        """Generate points on a line using Bresenham's algorithm."""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return points

    def _flood_fill(self, grid, start_x, start_y, old_val, new_val):
        """Iterative flood fill using a queue."""
        if grid[start_y][start_x] != old_val:
            return

        queue = [(start_x, start_y)]

        pbar = tqdm(
            total=grid.shape[0] * grid.shape[1],
            desc="Flood filling",
            position=0,
        )
        visited = set()
        while queue:
            x, y = queue.pop(0)
            if (int(x), int(y)) in visited:
                continue
            visited.add((int(x), int(y)))
            pbar.update(1)
            if x < 0 or y < 0 or x >= len(grid[0]) or y >= len(grid):
                continue
            if grid[y][x] != old_val:
                continue

            grid = grid.at[y, x].set(new_val)

            queue.append((x + 1, y))
            queue.append((x - 1, y))
            queue.append((x, y + 1))
            queue.append((x, y - 1))
        pbar.close()
        return grid
