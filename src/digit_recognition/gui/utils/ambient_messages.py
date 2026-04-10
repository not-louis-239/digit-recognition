from dataclasses import dataclass

from ...utils.custom_types import Colour

@dataclass
class AmbientMessage:
    text: str = ""
    colour: Colour = (0, 0, 0)
    lifetime_s: float = 0

    def __repr__(self) -> str:
        return f"AmbientMessage(text={self.text}, colour={self.colour})"

    @property
    def active(self) -> bool:
        return self.lifetime_s > 0

    def set_msg(self, *, text: str, colour: Colour, lifetime_s: float) -> None:
        self.text = text
        self.colour = colour
        self.lifetime_s = lifetime_s

    def clear(self):
        self.set_msg(text="", colour=(0, 0, 0), lifetime_s=0)

    def update(self, dt_s: float) -> None:
        if not self.active:
            return

        self.lifetime_s -= dt_s
        if self.lifetime_s <= 0:
            self.clear()
