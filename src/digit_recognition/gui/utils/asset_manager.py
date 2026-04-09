from pathlib import Path

import pygame as pg

from ...utils.custom_types import ImageArray
from ...utils.dirs import DIRS
from ...digit_recogniser.image_manager import load_imgs_from_csv

class Assets:
    def __init__(self) -> None:
        self.font_monospaced: Path = (DIRS.assets.fonts / "SourceCodePro-VariableFont_wght.ttf").path()
        self.training_data: list[tuple[ImageArray, int]] = load_imgs_from_csv((DIRS.assets.training_data / "digits.csv").path())
