from __future__ import annotations

import json

from ..utils import lerp
from ..utils.dirs import DIRS

def _load_config() -> dict:
    config_path = (DIRS.assets / "config.json").path()
    try:
        with open(config_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}
    return {}

_CONFIG = _load_config()

def _get(key: str, default):
    return _CONFIG.get(key, default)

# --- Generic Config ---
__version__ = "0.1.0"

# --- Softmax/Prediction Config ---
LOGIT_GAIN = float(_get("LOGIT_GAIN", 1.1))

# --- Loss Calculation Config ---
CONFIDENCE_PENALTY_FACTOR = float(_get("CONFIDENCE_PENALTY_FACTOR", 0.05))
SMALL_MARGIN_PENALTY_FACTOR = float(_get("SMALL_MARGIN_PENALTY_FACTOR", 0.05))
TARGET_MARGIN = float(_get("TARGET_MARGIN", 0.3))  # top guess - 2nd top guess

# --- Simulator Config ---
IMAGE_SIZE = int(_get("IMAGE_SIZE", 28))

NEW_CONFIG_RANGE = float(_get("NEW_CONFIG_RANGE", 0.5))
NEURONS_PER_HIDDEN_LAYER = int(_get("NEURONS_PER_HIDDEN_LAYER", 16))

NUM_HIDDEN_LAYERS = int(_get("NUM_HIDDEN_LAYERS", 2))
POPULATION_SIZE = int(_get("POPULATION_SIZE", 50))

SCALE_MUTATION_FACTOR = float(_get("SCALE_MUTATION_FACTOR", 1.05))
SCALE_MUTATION_CHANCE = float(_get("SCALE_MUTATION_CHANCE", 0.1))

# Past this, there will be no new immigrants or hypermutants
HARDENING_EPOCH = int(_get("HARDENING_EPOCH", 4000))

# Only keep the best 1 / selection_pressure models each generation
# Base selection pressure is modified based on season and in future, possibly other factors
BASE_SELECTION_PRESSURE = float(_get("BASE_SELECTION_PRESSURE", 12))

IMMIGRATION_RATE = float(_get("IMMIGRATION_RATE", 0.0))
HYPERMUTATION_RATE = float(_get("HYPERMUTATION_RATE", 0.0))

# --- GUI Config ---
FPS = int(_get("FPS", 60))
WN_W = int(_get("WN_W", 1250))
WN_H = int(_get("WN_H", 820))

def calc_mutation_rate(epoch: int) -> float:
    """Return a smart mutation rate (higher at start, lower as time passes)"""

    # Define milestones: (epoch, rate)
    milestones = _get(
        "MUTATION_MILESTONES",
        [
            (0, 0.03),
            (3000, 0.02),
            (6000, 0.01),
            (10000, 0.005),
            (20000, 0.002),
            (40000, 0.00125),
        ],
    )

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
