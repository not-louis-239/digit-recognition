from typing import TYPE_CHECKING

from pygame import Surface

from digit_recognition.gui.utils.input_manager import InputManager
from digit_recognition.gui.states import State, StateChangeRequest

if TYPE_CHECKING:
    from digit_recognition.gui.utils.asset_manager import Assets

class GalleryState(State):
    """Draw a digit and see what the model thinks it is.
    Optionally, enter the correct digit for your image and add it
    to the training set. This will allow you to train the model.
    Additionally, browse the training data."""

    def __init__(self, assets: Assets) -> None:
        self.assets = assets

    def reset(self) -> None:
        ...

    def update(self, dt_s: float) -> None:
        ...

    def take_input(self, input_manager: InputManager) -> StateChangeRequest:
        return StateChangeRequest()

    def draw(self, wn: Surface) -> None:
        wn.fill((30, 30, 30))
