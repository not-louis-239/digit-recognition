from typing import TYPE_CHECKING

from pygame import Surface

from . import State, StateID, StateChangeRequest
from ..utils.input_manager import InputManager
from ..utils.buttons import Button
from ..utils.text_utils import draw_text
from ...utils.constants import WN_W, WN_H

if TYPE_CHECKING:
    from ..utils.asset_manager import Assets

class TitleState(State):
    def __init__(self, assets: Assets) -> None:
        self.assets = assets

        self.start_button = Button(
            WN_W // 2 - 100, WN_H // 2 - 50, 200, 100,
            (75, 75, 125), (140, 140, 255), "Start", (self.assets.font_monospaced, 36)
        )

    def reset(self) -> None:
        pass

    def update(self, dt_s: float) -> None:
        pass

    def take_input(self, input_manager: InputManager) -> StateChangeRequest:
        if self.start_button.check_click(input_manager.events):
            return StateChangeRequest(new=StateID.SIM)
        return StateChangeRequest()

    def draw(self, wn: Surface) -> None:
        wn.fill((30, 30, 30))

        draw_text(
            surface=wn, pos=(WN_W // 2, 200), horiz_align='centre', vert_align='centre',
            text="Digit Recognition Evolution Simulator", colour=(255, 255, 255), font_profile=(self.assets.font_monospaced, 48)
        )

        self.start_button.draw(wn)


