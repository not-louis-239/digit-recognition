from pygame import Surface

from digit_recognition.gui.utils.asset_manager import Assets
from digit_recognition.gui.utils.input_manager import InputManager
from digit_recognition.gui.states import State, StateChangeRequest

class SimState(State):
    def __init__(self, assets: Assets):
        super().__init__(assets)

    def reset(self) -> None:
        ...

    def update(self, dt_s: float) -> None:
        ...

    def take_input(self, input_manager: InputManager) -> StateChangeRequest:
        return StateChangeRequest()

    def draw(self, wn: Surface) -> None:
        wn.fill((30, 30, 30))
