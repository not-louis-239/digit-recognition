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

        button_gap = 110
        start_x, start_y = (WN_W // 2 - 200, WN_H // 2 - 50)
        self.start_button = Button((start_x, start_y), (400, 70), "Start Simulation")
        self.gallery_button = Button((start_x, start_y + button_gap), (400, 70), "Gallery")

    def reset(self) -> None:
        pass

    def update(self, dt_s: float) -> None:
        pass

    def take_input(self, input_manager: InputManager) -> StateChangeRequest:
        if self.start_button.check_click(input_manager.events):
            return StateChangeRequest(new=StateID.SIM)
        if self.gallery_button.check_click(input_manager.events):
            return StateChangeRequest(new=StateID.GALLERY)
        return StateChangeRequest()

    def draw(self, wn: Surface) -> None:
        wn.fill((30, 30, 30))

        draw_text(
            surface=wn, pos=(WN_W // 2, 200), horiz_align='centre', vert_align='centre',
            text="Digit Recognition Evolution Simulator", colour=(255, 255, 255), font_profile=(self.assets.monospaced_light, 48)
        )

        self.start_button.draw(wn)
        self.gallery_button.draw(wn)
