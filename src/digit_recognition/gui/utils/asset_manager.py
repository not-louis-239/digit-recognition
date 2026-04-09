from pathlib import Path

import pygame as pg
import numpy as np

from ...utils.dirs import DIRS
from ...digit_recogniser.image_manager import load_imgs_from_npy

class Assets:
    def __init__(self) -> None:
        self.font_monospaced: Path = (DIRS.assets.fonts / "SourceCodePro-VariableFont_wght.ttf").path()
        self.training_data: list[tuple[np.ndarray, int]] = load_imgs_from_npy((DIRS.assets.training_data / "digits.npy").path())
