from pathlib import Path

import pygame as pg
import numpy as np

from ...utils.dirs import DIRS
from ...digit_recogniser.image_manager import load_imgs_from_npy
from ...utils.custom_types import TrainingDataType

class Assets:
    def __init__(self) -> None:
        self.monospaced_light: Path = (DIRS.assets.fonts / "SourceCodePro-ExtraLight.ttf").path()
        self.monospaced_reg: Path = (DIRS.assets.fonts / "SourceCodePro-Medium.ttf").path()
        self.training_data: TrainingDataType = load_imgs_from_npy((DIRS.assets.training_data / "digits.npy").path())
