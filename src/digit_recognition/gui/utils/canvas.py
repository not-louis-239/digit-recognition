import pygame as pg
from pygame import Surface
import numpy as np

from digit_recognition.utils.constants import NOISE_EPSILON


class Canvas:
    def __init__(self, width: int, height: int) -> None:
        self.cells: list[list[float]] = [[0.0 for _ in range(width)] for _ in range(height)]
        self.width = width
        self.height = height

    def __getitem__(self, i: int | tuple[int, int]) -> float:
        if isinstance(i, tuple) and len(i) == 2 and all(isinstance(coord, int) for coord in i):
            x, y = i
            return self.cells[y][x]

        raise IndexError(f"Canvas.__getitem__ takes a tuple of ints (x, y), not {i}")

    def __setitem__(self, i: int | tuple[int, int], value: float) -> None:
        if isinstance(i, tuple) and len(i) == 2 and all(isinstance(coord, int) for coord in i):
            x, y = i
            self.cells[y][x] = value
            return

        raise IndexError(f"Canvas.__setitem__ takes a tuple of ints (x, y), not {i}")

    def is_empty(self) -> bool:
        sum_pixels = sum(cell for row in self.cells for cell in row)
        return sum_pixels < NOISE_EPSILON

    def clear(self) -> None:
        for y in range(self.height):
            for x in range(self.width):
                self[x, y] = 0.0

    def handle_mouse_input(self, dt_s: float, mouse_pos: tuple[float, float], brush_size: float, brush_strength: float) -> None:
        """
        Accepts mouse_pos in its own grid space, ([0..self.width], [0..self.height])

        brush_strength is intensity increase per second, e.g. 20.0 means the pixel will reach full intensity after 1/20th of a second of being under the brush.
        This allows for consistent drawing regardless of frame rate.
        """

        raw_inc = brush_strength * dt_s

        mx, my = mouse_pos
        rad = brush_size / 2
        rad_sq = rad * rad

        x_min = int(max(0, mx - rad))
        x_max = int(min(self.width - 1, mx + rad))
        y_min = int(max(0, my - rad))
        y_max = int(min(self.height - 1, my + rad))

        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                if 0 <= x < self.width and 0 <= y < self.height:
                    dx, dy = mx - (x + 0.5), my - (y + 0.5)  # using 0.5 to target the pixel centre
                    dist_sq = dx ** 2 + dy ** 2
                    if dist_sq <= rad_sq:
                        self[x, y] = min(1.0, self[x, y] + raw_inc * (1.0 - (dist_sq / rad_sq)))

    def to_one_hot(self, correct_digit: int) -> tuple[np.ndarray, int, np.ndarray]:
        image: np.ndarray = np.array(self.cells, dtype=np.float32)
        label: np.ndarray = np.zeros(10, dtype=np.float32)
        label[correct_digit] = 1.0
        return (image, correct_digit, label)

    def as_array(self) -> np.ndarray:
        return np.array(self.cells, dtype=np.float32)

    def draw(self, surface: Surface, start_pos: tuple[int, int], tile_size_px: int) -> None:
        """Draws the canvas onto the given surface. Assumes the surface is the same size as the canvas."""
        for y in range(self.height):
            for x in range(self.width):
                intensity = int(self[x, y] * 255)
                pg.draw.rect(surface, (intensity, intensity, intensity), (start_pos[0] + x * tile_size_px, start_pos[1] + y * tile_size_px, tile_size_px, tile_size_px))
