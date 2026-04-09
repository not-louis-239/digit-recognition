from typing import TYPE_CHECKING

from pygame import Surface

from ..states import State
from ..utils.input_manager import InputManager
from ..utils.buttons import Button
from ..utils.text_utils import draw_text
from ...utils.constants import WN_W, WN_H

if TYPE_CHECKING:
    from ...gui.utils.asset_manager import Assets

class MainState(State):
    def __init__(self, assets: Assets) -> None:
        self.assets = assets

    def reset(self) -> None:
        pass

    def update(self) -> None:
        pass

    def take_input(self, input_manager: InputManager) -> None:
        ...

    def draw(self, wn: Surface) -> None:
        wn.fill((30, 30, 30))

        draw_text(
            surface=wn, pos=(WN_W // 2, WN_H // 2), horiz_align='centre', vert_align='centre',
            text="Digit Recognition Evolution Simulator", colour=(255, 255, 255), font_profile=(self.assets.font_monospaced, 48)
        )


