from abc import ABC, abstractmethod
from enum import StrEnum, auto
from typing import TYPE_CHECKING
from dataclasses import dataclass

from pygame import Surface

from ...gui.utils.input_manager import InputManager

if TYPE_CHECKING:
    from ...gui.utils.asset_manager import Assets

class StateID(StrEnum):
    MAIN = auto()
    SIM = auto()
    GALLERY = auto()

@dataclass(kw_only=True, frozen=True)
class StateChangeRequest:
    new: StateID | None = None

class State(ABC):
    def __init__(self, assets: Assets):
        self.assets = assets

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self, dt_s: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def take_input(self, input_manager: InputManager) -> StateChangeRequest:
        raise NotImplementedError

    @abstractmethod
    def draw(self, wn: Surface):
        raise NotImplementedError
