import pygame as pg
from pygame.key import ScancodeWrapper

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

    def pressed(self, key: int) -> bool:
        return self.cur_keys[key] and not self.prev_keys[key]
