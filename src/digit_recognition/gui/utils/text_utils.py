from typing import Literal
from pathlib import Path
from functools import lru_cache

import pygame as pg
from pygame import Surface

from ...utils.custom_types import Colour, AColour

_FONT_OBJECT_CACHE: dict[tuple[Path | None, int], pg.font.Font] = {}
def _get_font(path: Path | None, size: int) -> pg.font.Font:
    if (path, size) not in _FONT_OBJECT_CACHE:
        if path is None:
            font_obj = pg.font.Font(None, size)
        else:
            font_obj = pg.font.Font(str(path), size)
        _FONT_OBJECT_CACHE[(path, size)] = font_obj
    return _FONT_OBJECT_CACHE[(path, size)]

@lru_cache(maxsize=500)
def _get_rendered_text(text: str, colour: tuple[int, int, int], path: Path, size: int, rotation: int) -> pg.Surface:
    font_obj = _get_font(path, size)
    img = font_obj.render(text, True, colour)
    if rotation != 0:
        img = pg.transform.rotate(img, rotation)
    return img

def draw_text(
        surface: Surface, pos: tuple[int, int],
        horiz_align: Literal['left', 'centre', 'right'],
        vert_align: Literal['top', 'centre', 'bottom'],
        text: str, colour: Colour | AColour,
        font_profile: tuple[Path | None, int],
        rotation: int = 0
    ) -> None:
    # We explicitly ban float values for rotation as it hinders the ability of the cache
    # to improve performance.

    img = _get_rendered_text(text, colour, *font_profile, rotation)
    rect = img.get_rect()

    # Horizontal
    if horiz_align == "left":
        rect.left = pos[0]
    elif horiz_align == "centre":
        rect.centerx = pos[0]
    elif horiz_align == "right":
        rect.right = pos[0]
    else:
        raise ValueError(f"Invalid horiz_align: {horiz_align}")

    # Vertical
    if vert_align == "top":
        rect.top = pos[1]
    elif vert_align == "centre":
        rect.centery = pos[1]
    elif vert_align == "bottom":
        rect.bottom = pos[1]
    else:
        raise ValueError(f"Invalid vert_align: {vert_align}")

    surface.blit(img, rect)
