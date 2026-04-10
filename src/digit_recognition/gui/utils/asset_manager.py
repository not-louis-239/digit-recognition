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

        training_data: TrainingDataType = load_imgs_from_npy((DIRS.assets.training_data / "digits.npy").path())

        # Cache one_hots to improve performance. Tuples are (image, correct_digit, one_hot_array)
        self.one_hots: list[tuple[np.ndarray, int, np.ndarray]] = []
        for image, correct_digit in training_data:
            one_hot = np.zeros((10, 1))
            one_hot[correct_digit] = 1.0
            self.one_hots.append((image, int(correct_digit), one_hot))
