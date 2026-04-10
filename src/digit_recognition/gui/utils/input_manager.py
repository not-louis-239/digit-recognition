from enum import IntEnum

import pygame as pg
from pygame.key import ScancodeWrapper

class MouseButton(IntEnum):
    LMB = 0
    MMB = 1
    RMB = 2

class InputManager:
    def __init__(self) -> None:
        self.dt_s: float = 0
        self.events: list[pg.event.Event] = pg.event.get()
        self.prev_keys: ScancodeWrapper = pg.key.get_pressed()
        self.cur_keys: ScancodeWrapper = pg.key.get_pressed()

    def update_keys(self, new_keys: ScancodeWrapper, events: list[pg.event.Event], dt_s: float) -> None:
        """Run this once at the start of each event loop"""

        self.dt_s = dt_s
        self.events = events
        self.prev_keys, self.cur_keys = self.cur_keys, new_keys

    def is_down(self, key: int) -> bool:
        """Returns True as long as the key is held"""
        return self.cur_keys[key]

    def went_down(self, key: int) -> bool:
        """Returns True on the first frame the key is held down"""
        return self.cur_keys[key] and not self.prev_keys[key]

    def went_up(self, key: int) -> bool:
        """Returns True on the first frame the key is released"""
        return not self.cur_keys[key] and self.prev_keys[key]

    def mouse_is_down(self, button: int) -> bool:
        """Returns True as long as the mouse button is held"""
        return pg.mouse.get_pressed()[button]

    def mouse_went_down(self, button: int) -> bool:
        """Returns True on the first frame the mouse button is held down"""
        return any(event.type == pg.MOUSEBUTTONDOWN and event.button == button for event in self.events)

    def mouse_went_up(self, button: int) -> bool:
        """Returns True on the first frame the mouse button is released"""
        return any(event.type == pg.MOUSEBUTTONUP and event.button == button for event in self.events)

    def get_events(self) -> list[pg.event.Event]:
        """Returns the list of events from the most recent event loop"""
        return self.events
