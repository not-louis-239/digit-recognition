import pygame as pg
from pygame import Surface
from typing import Literal
from pathlib import Path
from ...utils.custom_types import Colour, AColour

_FONT_OBJECT_CACHE: dict[tuple[str | None, int], pg.font.Font] = {}

def draw_text(
        surface: Surface, pos: tuple[int, int],
        horiz_align: Literal['left', 'centre', 'right'],
        vert_align: Literal['top', 'centre', 'bottom'],
        text: str, colour: Colour | AColour,
        font_size: int, font_family: pg.font.Font | Path | str | None = None,
        rotation: float = 0
    ) -> None:
    if isinstance(font_family, pg.font.Font):
        font_obj = font_family
    else:
        font_obj_profile = (str(font_family) if font_family is not None else None, font_size)

        # Caching font objects avoid expensive recreation of font objects
        # every time we want to draw some text
        if font_obj_profile not in _FONT_OBJECT_CACHE:
            font_obj = pg.font.Font(font_obj_profile[0], font_obj_profile[1])
            _FONT_OBJECT_CACHE[font_obj_profile] = font_obj
        else:
            font_obj = _FONT_OBJECT_CACHE[font_obj_profile]

    img = font_obj.render(text, True, colour)
    if rotation != 0:
        img = pg.transform.rotate(img, rotation)

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
