from ..utils import lerp

# --- Generic Config ---
__version__ = "0.1.0"

# --- Loss Calculation Config ---
CONFIDENCE_PENALTY_FACTOR = 0.5
SMALL_MARGIN_PENALTY_FACTOR = 0.2

# --- Simulator Config ---
IMAGE_SIZE = 28

NEW_CONFIG_RANGE = 0.1
NEURONS_PER_HIDDEN_LAYER = 16

NUM_HIDDEN_LAYERS = 2
POPULATION_SIZE = 25

# Only keep the best 1 / selection_pressure models each generation
# Base selection pressure is modified based on season and in future, possibly other factors
BASE_SELECTION_PRESSURE = 15

IMMIGRATION_RATE = 0.007
HYPERMUTATION_RATE = 0.007

# --- GUI Config ---
FPS = 60
WN_W, WN_H = 1250, 820

def calc_mutation_rate(epoch: int) -> float:
    """Return a smart mutation rate (higher at start, lower as time passes)"""

    # Define milestones: (epoch, rate)
    milestones = [
        (0, 0.05),
        (3000, 0.02),
        (6000, 0.01),
        (10000, 0.005),
        (20000, 0.002)
    ]

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
