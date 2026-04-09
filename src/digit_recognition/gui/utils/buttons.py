import pygame as pg
from pygame import Surface

from ...utils.custom_types import Colour
from .text_utils import draw_text

class Button:
    def __init__(
            self, x: int, y: int, w: float, h: float,
            bg_colour: Colour, fg_colour: Colour,
            text: str, font_size: int, font: pg.font.Font
        ) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        self.bg_colour = bg_colour
        self.fg_colour = fg_colour

        self.text = text
        self.font_size = font_size
        self.font = font

    def contains(self, point: tuple[float, float]) -> bool:
        px, py = point
        return (
            self.x <= px <= self.x + self.w and
            self.y <= py <= self.y + self.h
        )

    def check_click(self, events: list[pg.event.Event]) -> bool:
        for event in events:
            if event.type == pg.MOUSEBUTTONDOWN:
                mouse_pos = pg.mouse.get_pos()
                if self.contains(mouse_pos):
                    return True
        return False

    def draw(self, surface: Surface) -> None:
        pg.draw.rect(surface, self.bg_colour, (self.x, self.y, self.w, self.h))
        draw_text(
            surface=surface, pos=(self.x, self.y),
            horiz_align='centre', vert_align='centre',
            text=self.text, colour=self.fg_colour,
            font_size=self.font_size, font_family=self.font
        )
