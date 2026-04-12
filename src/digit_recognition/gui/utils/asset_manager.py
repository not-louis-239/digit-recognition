import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np

from ...utils.dirs import DIRS
from ...digit_recogniser.image_manager import load_imgs_from_npy
from ...utils.custom_types import RawImagesType, OneHotType
from ...digit_recogniser.simulation import Evaluation, Simulation
from ...digit_recogniser.digit_recogniser import DigitRecogniser
from ...utils.diagnostic_helpers import print_warn, print_info, print_err

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
        self.dev_data: OneHotType = self._training_data_to_one_hots(load_imgs_from_npy((DIRS.assets.training_data / "digits_dev.npy").path()))
        self.test_data: OneHotType = self._training_data_to_one_hots(load_imgs_from_npy((DIRS.assets.training_data / "digits_test.npy").path()))

        # Get models
        self.model_wrappers: list[DigitRecogniserWrapper] = []
        models_dir = (DIRS.assets.display_models).path()

        def should_ignore(filepath: Path) -> tuple[bool, str]:
            """Return ignore, msg"""
            if filepath.name.startswith("."):
                return True, f"skipping hidden file or directory in models dir: {filepath}"
            if filepath.name.startswith("__") and filepath.name.endswith("__"):
                return True, f"{filepath} is ignored due to \"__\" prefix and suffix"
            if filepath.name in [
                "manifest.schema.json"
            ]:
                return True, f"ignoring {filepath} (as intended)"
            return False, ""

        for model_dir in models_dir.iterdir():
            ignore, msg = should_ignore(model_dir)
            if ignore:
                print_info(msg)
                continue

            # model_dir: example: "victor__1_0_0__13k"
            print_info(f"found file or directory: {model_dir}")

            if not model_dir.is_dir():
                print_warn(f"skipping non-directory in models directory: {model_dir}")
                continue

            manifest = model_dir / "manifest.json"
            if not manifest.exists():
                print_warn(f"skipping loading model with missing manifest: {model_dir}")
                continue

            with open(manifest, "r") as mani:
                manifest_data = json.load(mani)

            files = manifest_data.get("files", [])
            if not files:
                print_warn(f"skipping loading model with no associated files in manifest: {model_dir}")
                continue
            if (len_files := len(files)) > 1:
                print_warn(f"model has {len_files} associated files (expected 1). Only the first file will be loaded: {model_dir}")

            filepath = model_dir / files[0]
            if not filepath.exists():
                print_warn(f"skipping loading model with missing file: {filepath}")
                continue

            try:
                with open(filepath, "r") as f:
                    model_data_raw = json.load(f)  # Check if JSON is valid
                    assert isinstance(model_data_raw, dict), "Model JSON should be a dictionary at the top level"
            except json.JSONDecodeError:  # Checking for FileNotFoundError isn't necessary as we already check if the file exists, but we still need to catch JSONDecodeError to avoid crashing on corrupted files
                print_warn(f"skipping loading model with corrupted JSON file: {filepath}")
                continue

            name = manifest_data.get("name", "unknown_model")
            common_name = manifest_data.get("common_name", name)

            try:
                model = DigitRecogniser.from_json(model_data_raw)
            except Exception as e:
                print_err(f"failed to load model from JSON file: {filepath}")
                print_err(f"({type(e).__name__}: {e})")
                continue

            perf = None

            wrapper = DigitRecogniserWrapper(name=name, common_name=common_name, model=model, perf=perf)
            self.model_wrappers.append(wrapper)
            print_info(f"successfully loaded model: {name} (common name: {common_name})")

    def _training_data_to_one_hots(self, data: RawImagesType) -> OneHotType:
        one_hots = []
        for image, correct_digit in data:
            one_hot = np.zeros((10, 1))
            one_hot[correct_digit] = 1.0
            one_hots.append((image, int(correct_digit), one_hot))
        return one_hots
