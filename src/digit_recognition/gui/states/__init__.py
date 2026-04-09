from abc import ABC, abstractmethod
from enum import StrEnum, auto
from pygame import Surface

from ...gui.utils.input_manager import InputManager

class StateID(StrEnum):
    MAIN = auto()

class State(ABC):
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
