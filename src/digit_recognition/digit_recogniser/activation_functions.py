import numpy as np
from typing import Callable

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def leaky_relu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, 0.01 * x)

def sigmoid(x: np.ndarray) -> np.ndarray:
    """This would be deprecated... if not for the fact that some old models use it."""
    # NumPy's exp handles entire arrays at once
    return 1 / (1 + np.exp(-x))

# Use the FUNCMAP to determine the correct activation function for a model based on its
# listed activation function in its manifest.json.
# This allows for more flexibility in the future if we want to add more activation functions.
# The strings in this function must match exactly the strings in the manifest.json files of the models,
# otherwise it will raise a KeyError when trying to load a model with an unrecognized activation function.
_FUNCMAP: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "relu": relu,
    "leaky_relu": leaky_relu,
    "sigmoid": sigmoid
}

def get_activation_func(s: str) -> Callable[[np.ndarray], np.ndarray]:
    """Get the activation function corresponding to the given string."""
    if s not in _FUNCMAP:
        raise ValueError(f"Unrecognized activation function: {s}")
    return _FUNCMAP[s]
