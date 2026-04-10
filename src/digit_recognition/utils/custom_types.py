from typing import TypeAlias, NamedTuple
from pathlib import Path

import numpy as np

Colour: TypeAlias = tuple[int, int, int]
AColour: TypeAlias = tuple[int, int, int, int]

Coord2: TypeAlias = tuple[float, float]
IntCoord2: TypeAlias = tuple[int, int]
FontProfile: TypeAlias = tuple[Path | None, int]  # (family, size)

# arrays must be of size (IMAGE_SIZE, IMAGE_SIZE), with an attached label for the correct digit
RawImagesType: TypeAlias = list[tuple[np.ndarray, int]]

class _OneHotTuple(NamedTuple):
    image: np.ndarray
    correct_digit: int
    one_hot: np.ndarray

OneHotType: TypeAlias = list[_OneHotTuple]  # (image, correct_digit, one_hot_array)
