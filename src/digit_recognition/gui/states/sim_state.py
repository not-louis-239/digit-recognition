from pygame import Surface
from typing import TYPE_CHECKING

from digit_recognition.utils.constants import WN_W, WN_H
from digit_recognition.gui.utils.text_utils import draw_text
from digit_recognition.gui.utils.asset_manager import Assets
from digit_recognition.gui.utils.buttons import Button
from digit_recognition.gui.utils.input_manager import InputManager
from digit_recognition.gui.states import State, StateChangeRequest, StateID
from digit_recognition.utils.seasons import format_year

if TYPE_CHECKING:
    from digit_recognition.digit_recogniser.simulation import Simulation

class SimState(State):
    def __init__(self, assets: Assets, sim: Simulation):
        super().__init__(assets)
        self.padding = 30
        self.run_button = Button((WN_W - 200 - self.padding, self.padding), (200, 80), "Run")
        self.sim_running: bool = False
        self.sim = sim

        self.return_button = Button((self.padding, WN_H - 100 - self.padding), (200, 100), "Return")

    def reset(self) -> None:
        ...

    def update(self, dt_s: float) -> None:
        if self.sim_running:
            self.sim.run_generation(self.assets.training_data)

    def take_input(self, input_manager: InputManager) -> StateChangeRequest:
        if self.return_button.check_click(input_manager.events):
            return StateChangeRequest(new=StateID.TITLE)

        if self.run_button.check_click(input_manager.events):
            self.sim_running = not self.sim_running

        return StateChangeRequest()

    def draw(self, wn: Surface) -> None:
        wn.fill((30, 30, 30))

        self.run_button.draw(wn)
        self.return_button.draw(wn)

        if self.sim_running:
            text = "Sim: Running"
            colour = (100, 255, 100)
        else:
            text = "Sim: Paused"
            colour = (255, 100, 100)

        draw_text(
            surface=wn, pos=(WN_W - self.padding, self.padding + 110), horiz_align='right', vert_align='top',
            text=text, colour=colour, font_profile=(self.assets.monospaced_reg, 36)
        )

        year, season = format_year(self.sim.epoch)

        draw_text(
            surface=wn, pos=(self.padding, self.padding), horiz_align='left', vert_align='top',
            text=f"{season.name}", colour=season.colour, font_profile=(self.assets.monospaced_reg, 36)
        )
        draw_text(
            surface=wn, pos=(self.padding, self.padding + 60), horiz_align='left', vert_align='top',
            text=f"Year {year}", colour=(255, 255, 255), font_profile=(self.assets.monospaced_reg, 24)
        )
