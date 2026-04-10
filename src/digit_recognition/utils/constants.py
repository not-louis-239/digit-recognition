# --- Generic Config ---
__version__ = "0.1.0"

# --- Simulator Config ---
IMAGE_SIZE = 28

NEW_CONFIG_RANGE = 0.1
STARTING_MUTATION_RATE = 0.05

NEURONS_PER_HIDDEN_LAYER = 16

NUM_HIDDEN_LAYERS = 2
POPULATION_SIZE = 50

# Only keep the best 1 / SELECTION_PRESSURE models each generation
SELECTION_PRESSURE = 20

# --- GUI Config ---
FPS = 60
WN_W, WN_H = 1250, 750

def calc_mutation_rate(gen: int) -> float:
    """Return a smart mutation rate (higher at start, lower as time passes)"""

    factor = 1 / (gen ** 0.35)
    return STARTING_MUTATION_RATE * factor
