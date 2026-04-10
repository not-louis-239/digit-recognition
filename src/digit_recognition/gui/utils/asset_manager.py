import json
from pathlib import Path
from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np

from ...utils.dirs import DIRS
from ...digit_recogniser.image_manager import load_imgs_from_npy
from ...utils.custom_types import RawImagesType, OneHotType
from ...digit_recogniser.simulation import Evaluation, Simulation

if TYPE_CHECKING:
    from ...digit_recogniser.digit_recogniser import DigitRecogniser

@dataclass
class DigitRecogniserWrapper:
    name: str
    common_name: str
    model: DigitRecogniser
    perf: Evaluation | None  # lazy-load models and evaluate at runtime

    @property
    def loss(self) -> float | None:
        if self.perf is not None:
            return self.perf.loss
        return None

    @property
    def accuracy_rate(self) -> float | None:
        if self.perf is not None:
            return self.perf.accuracy_rate
        return None

def assign_evals(wrappers_list: list[DigitRecogniserWrapper], sim: Simulation, data: OneHotType) -> None:
    """Needs a Simulation instance to evaluate models to avoid a potential circular reference."""
    for wrapper in wrappers_list:
        if wrapper.perf is None:
            wrapper.perf = sim.evaluate_model(wrapper.model, data)

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

        for model_dir in models_dir.glob("*"):
            manifest = model_dir / "manifest.json"
            if not manifest.exists():
                print("Warning: skipping loading model with missing manifest:", model_dir)
                continue

            with open(manifest, "r") as mani:
                manifest_data = json.load(mani)

            files = manifest_data.get("files", [])
            if not files:
                print("Warning: skipping loading model with no associated files in manifest:", model_dir)
                continue
            if (len_files := len(files)) > 1:
                print(f"Warning: model has {len_files} associated files (expected 1). Only the first file will be loaded:", model_dir)

            name = manifest_data.get("name", "unknown_model")
            common_name = manifest_data.get("common_name", name)
            model = DigitRecogniser.from_json(files[0])
            perf = None

            wrapper = DigitRecogniserWrapper(name=name, common_name=common_name, model=model, perf=perf)
            self.model_wrappers.append(wrapper)

    def _training_data_to_one_hots(self, data: RawImagesType) -> OneHotType:
        one_hots = []
        for image, correct_digit in data:
            one_hot = np.zeros((10, 1))
            one_hot[correct_digit] = 1.0
            one_hots.append((image, int(correct_digit), one_hot))
        return one_hots
