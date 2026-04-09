from pygame import Surface
from typing import TYPE_CHECKING

from digit_recognition.utils.constants import WN_W, WN_H
from digit_recognition.gui.utils.text_utils import draw_text
from digit_recognition.gui.utils.asset_manager import Assets
from digit_recognition.gui.utils.buttons import Button
from digit_recognition.gui.utils.input_manager import InputManager
from digit_recognition.gui.states import State, StateChangeRequest
from digit_recognition.utils.seasons import format_year

if TYPE_CHECKING:
    from digit_recognition.digit_recogniser.simulation import Simulation

class SimState(State):
    def __init__(self, assets: Assets, sim: Simulation):
        super().__init__(assets)
        self.run_button = Button((WN_W - 250, 50), (200, 100), "Run")
        self.sim_running: bool = False
        self.sim = sim

    def reset(self) -> None:
        ...

    def update(self, dt_s: float) -> None:
        if self.sim_running:
            self.sim.run_generation(self.assets.training_data)

    def take_input(self, input_manager: InputManager) -> StateChangeRequest:
        if self.run_button.check_click(input_manager.events):
            self.sim_running = not self.sim_running

        return StateChangeRequest()

    def draw(self, wn: Surface) -> None:
        wn.fill((30, 30, 30))

        self.run_button.draw(wn)

        if self.sim_running:
            text = "Sim: Running"
            colour = (100, 255, 100)
        else:
            text = "Sim: Paused"
            colour = (255, 100, 100)

        draw_text(
            surface=wn, pos=(WN_W - 250, 300), horiz_align='left', vert_align='centre',
            text=text, colour=colour, font_profile=(self.assets.monospaced_reg, 36)
        )

        year, season = format_year(self.sim.epoch)

        draw_text(
            surface=wn, pos=(50, 150), horiz_align='left', vert_align='centre',
            text=f"{season.name}", colour=season.colour, font_profile=(self.assets.monospaced_reg, 36)
        )
        draw_text(
            surface=wn, pos=(50, 250), horiz_align='left', vert_align='centre',
            text=f"Year {self.sim.epoch}", colour=(255, 255, 255), font_profile=(self.assets.monospaced_reg, 24)
        )
