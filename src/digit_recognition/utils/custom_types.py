from typing import TypeAlias
from pathlib import Path

ImageArray: TypeAlias = list[list[float]]

Colour: TypeAlias = tuple[int, int, int]
AColour: TypeAlias = tuple[int, int, int, int]

Coord2: TypeAlias = tuple[float, float]
IntCoord2: TypeAlias = tuple[int, int]
FontProfile: TypeAlias = tuple[Path | None, int]  # (family, size)
