from pathlib import Path
from typing import Any
import json

from ..utils import lerp
from ..utils.custom_types import JSONType
from ..utils.diagnostic_helpers import print_warn, print_info
from digit_recognition.utils.dirs import DIRS

def load_config(path: Path = (DIRS.data / "config.json").path()) -> JSONType:
    try:
        with open(path, "r", encoding="utf-8") as f:
            conf = json.load(f)
        print_info(f"Successfully loaded config file '{path}':")
        print_info("\n" + conf)
    except FileNotFoundError:
        print_info("Config file not found. Using defaults.")
    except json.JSONDecodeError:
        print_warn("Config.json is corrupted. Using defaults.")
    except Exception as e:
        print_warn(f"Failed to load config: {e}. Using defaults.")

    print_warn(f"Could not load config, using defaults")
    return {}

_config = load_config()
def _get(key: str, *, default: Any):
    val = _config.get(key, None)
    if val is None:
        print_warn(f"Config key '{key}' not found.")
        print_warn(f"Using default: {default}")
        return default
    return val

BASE_SELECTION_PRESSURE = _get("base_selection_pressure", default=10)
LOGIT_GAIN = _get("logit_gain", default=0.5)
IMAGE_SIZE = _get("image_size", default=28)
NEW_PARAM_RANGE = _get("new_param_range", default=0.25)
NEURONS_PER_HIDDEN_LAYER = _get("neurons_per_hidden_layer", default=16)
NUM_HIDDEN_LAYERS = _get("num_hidden_layers", default=2)
POPULATION_SIZE = _get("population_size", default=20)
MUTATION_MILESTONES = _get(
    "mutation_milestones",
    default=[
        [0, 0.03],
        [500, 0.02],
        [1000, 0.01],
        [1500, 0.005],
        [2000, 0.002],
        [2500, 0.00125],
        [3000, 0.00008]
    ]
)


def calc_mutation_rate(epoch: int) -> float:
    """Return a smart mutation rate (higher at start, lower as time passes)"""

    # Define milestones: (epoch, rate)
    milestones = MUTATION_MILESTONES
    assert len(milestones) > 0

    # Handle bounds
    if epoch <= milestones[0][0]: return milestones[0][1]
    if epoch >= milestones[-1][0]: return milestones[-1][1]

    # Find the current segment
    for i in range(len(milestones) - 1):
        start_e, start_r = milestones[i]
        end_e, end_r = milestones[i+1]

        if epoch < end_e:
            t = (epoch - start_e) / (end_e - start_e)
            return lerp(start_r, end_r, t)

    return milestones[-1][1]

def _test():
    for i in range(0, 20200, 100):
        print(f"At epoch {i}, mutation rate is: {calc_mutation_rate(i):.4f}")

if __name__ == "__main__":
    _test()
