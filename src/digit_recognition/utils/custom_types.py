from typing import TypeAlias
from pathlib import Path

# XXX: ImageArray is deprecated as it is performance-poor and should be replaced with np.ndarray
ImageArray: TypeAlias = list[list[float]]

Colour: TypeAlias = tuple[int, int, int]
AColour: TypeAlias = tuple[int, int, int, int]

Coord2: TypeAlias = tuple[float, float]
IntCoord2: TypeAlias = tuple[int, int]
FontProfile: TypeAlias = tuple[Path | None, int]  # (family, size)
