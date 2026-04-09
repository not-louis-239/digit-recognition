from abc import ABC, abstractmethod
from enum import StrEnum, auto
from typing import TYPE_CHECKING

from pygame import Surface

from ...gui.utils.input_manager import InputManager

if TYPE_CHECKING:
    from ...gui.utils.asset_manager import Assets

class StateID(StrEnum):
    MAIN = auto()
    SIM = auto()
    GALLERY = auto()

class State(ABC):
    def __init__(self, assets: Assets):
        self.assets = assets

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def take_input(self, input_manager: InputManager) -> None:
        raise NotImplementedError

    @abstractmethod
    def draw(self, wn: Surface):
        raise NotImplementedError
