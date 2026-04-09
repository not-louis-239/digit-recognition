from pygame import Surface

from digit_recognition.gui.utils.asset_manager import Assets
from digit_recognition.gui.utils.input_manager import InputManager
from digit_recognition.gui.states import State

class SimState(State):
    def __init__(self, assets: Assets):
        super().__init__(assets)

    def reset(self) -> None:
        ...

    def update(self) -> None:
        ...

    def take_input(self, input_manager: InputManager) -> None:
        ...

    def draw(self, wn: Surface) -> None:
        ...
