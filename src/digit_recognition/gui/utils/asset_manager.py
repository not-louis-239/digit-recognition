from pathlib import Path

import pygame as pg
import numpy as np

from ...utils.dirs import DIRS
from ...digit_recogniser.image_manager import load_imgs_from_npy
from ...utils.custom_types import RawImagesType, OneHotType

class Assets:
    def __init__(self) -> None:
        self.monospaced_light: Path = (DIRS.assets.fonts / "SourceCodePro-ExtraLight.ttf").path()
        self.monospaced_reg: Path = (DIRS.assets.fonts / "SourceCodePro-Medium.ttf").path()

        # Cache one_hots to improve performance. Tuples are (image, correct_digit, one_hot_array)
        self.training_data: OneHotType = self._training_data_to_one_hots(load_imgs_from_npy((DIRS.assets.training_data / "digits_training.npy").path()))
        self.test_data: OneHotType = self._training_data_to_one_hots(load_imgs_from_npy((DIRS.assets.training_data / "digits_test.npy").path()))

    def _training_data_to_one_hots(self, data: RawImagesType) -> OneHotType:
        one_hots = []
        for image, correct_digit in data:
            one_hot = np.zeros((10, 1))
            one_hot[correct_digit] = 1.0
            one_hots.append((image, int(correct_digit), one_hot))
        return one_hots
