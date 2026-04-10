from pathlib import Path
from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np

from ...utils.dirs import DIRS
from ...digit_recogniser.image_manager import load_imgs_from_npy
from ...utils.custom_types import RawImagesType, OneHotType
from ...digit_recogniser.simulation import Evaluation

if TYPE_CHECKING:
    from ...digit_recogniser.digit_recogniser import DigitRecogniser

@dataclass
class DigitRecogniserWrapper:
    name: str
    common_name: str
    perf: Evaluation

    @property
    def model(self) -> DigitRecogniser:
        return self.perf.model

    @property
    def loss(self) -> float:
        return self.perf.loss

    @property
    def accuracy_rate(self) -> float:
        return self.perf.accuracy_rate

class Assets:
    def __init__(self) -> None:
        self.monospaced_light: Path = (DIRS.assets.fonts / "SourceCodePro-ExtraLight.ttf").path()
        self.monospaced_reg: Path = (DIRS.assets.fonts / "SourceCodePro-Medium.ttf").path()

        # Cache one_hots to improve performance. Tuples are (image, correct_digit, one_hot_array)
        self.training_data: OneHotType = self._training_data_to_one_hots(load_imgs_from_npy((DIRS.assets.training_data / "digits_training.npy").path()))
        self.test_data: OneHotType = self._training_data_to_one_hots(load_imgs_from_npy((DIRS.assets.training_data / "digits_test.npy").path()))

        # Get models
        self.model_wrappers: list[DigitRecogniserWrapper] = []
        models_dir = (DIRS.assets.display_models).path()

        for model_file in models_dir.glob("*"):
            ...

    def _training_data_to_one_hots(self, data: RawImagesType) -> OneHotType:
        one_hots = []
        for image, correct_digit in data:
            one_hot = np.zeros((10, 1))
            one_hot[correct_digit] = 1.0
            one_hots.append((image, int(correct_digit), one_hot))
        return one_hots
