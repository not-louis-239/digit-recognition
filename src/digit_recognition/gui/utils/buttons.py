import pygame as pg
from pygame import Surface

from ...utils.custom_types import Colour, FontProfile
from .text_utils import draw_text
from ...utils.dirs import DIRS

DEFAULT_BUTTON_FG_COLOUR = (140, 140, 255)
DEFAULT_BUTTON_BG_COLOUR = (75, 75, 125)
DEFAULT_BUTTON_FONT_SIZE = 36
DEFAULT_BUTTON_FONT_FAMILY = (DIRS.assets.fonts / "SourceCodePro-Medium.ttf").path()

class Button:
    def __init__(
            self, pos: tuple[int, int], size: tuple[int, int], text: str,
            bg_colour: Colour = DEFAULT_BUTTON_BG_COLOUR, fg_colour: Colour = DEFAULT_BUTTON_FG_COLOUR,
            font_profile: FontProfile = (DEFAULT_BUTTON_FONT_FAMILY, DEFAULT_BUTTON_FONT_SIZE)
        ) -> None:
        self.x, self.y = size
        self.w, self.h = size

        self.bg_colour = bg_colour
        self.fg_colour = fg_colour

        self.text = text
        self.font_profile = font_profile

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
            surface=surface, pos=(int(self.x + self.w / 2), int(self.y + self.h / 2)),
            horiz_align='centre', vert_align='centre',
            text=self.text, colour=self.fg_colour,
            font_profile=self.font_profile
        )
