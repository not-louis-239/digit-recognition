# Copyright 2026 Louis Masarei-Boulton

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TypeAlias, Any
from pathlib import Path
import numpy as np
import json

from ..utils.constants import STARTING_MUTATION_RATE
from ..utils.custom_types import ImageArray

def sigmoid(x: np.ndarray) -> np.ndarray:
    # NumPy's exp handles entire arrays at once
    return 1 / (1 + np.exp(-x))

def calculate_loss(correct: np.ndarray, actual: np.ndarray) -> float:
    """Uses NumPy to calculate the Mean Squared Error (Sum of Squares)."""
    # (correct - actual)**2 handles the subtraction and squaring for the whole array
    return np.sum((correct - actual) ** 2)

class Layer:
    def __init__(self, input_size, output_size):
        # We initialize all weights for the whole layer at once
        # Using "He Initialization" (scaling by sqrt of input size) helps stability
        self.weights = np.random.uniform(-1, 1, (output_size, input_size))
        self.bias = np.random.uniform(-1, 1, (output_size, 1))

    def forward(self, inputs):
        # inputs: (input_size, 1)
        # weight matrix multiplication + bias
        return sigmoid(np.dot(self.weights, inputs) + self.bias)

    def mutate(self, rate=STARTING_MUTATION_RATE):
        # Mutate the entire matrix at once using a mask
        self.weights += np.random.uniform(-rate, rate, self.weights.shape)
        self.bias += np.random.uniform(-rate, rate, self.bias.shape)

class DigitRecogniser:
    def __init__(self):
        # 784 -> 16 -> 16 -> 10
        self.layers = [
            Layer(784, 16),
            Layer(16, 16),
            Layer(16, 10)
        ]

    def to_json(self) -> dict[str, Any]:
        """
        Convert the model's layers into a JSON-serializable dictionary.
        """
        serializable_layers = []
        for layer in self.layers:
            serializable_layers.append({
                "weights": layer.weights.tolist(), # Convert NumPy array to list
                "bias": layer.bias.tolist()
            })

        return {
            "layers": serializable_layers,
            "metadata": {
                "num_hidden_layers": len(self.layers) - 1,
                "neurons_per_layer": len(self.layers[0].weights)
            }
        }

    @classmethod
    def from_json(cls, path: Path) -> DigitRecogniser:
        """
        Load a model from a JSON file.
        """
        with open(path, "r") as f:
            data = json.load(f)

        # Create instance without calling __init__ (prevents random weight gen)
        model = cls.__new__(cls)
        model.layers = []

        for layer_data in data["layers"]:
            # Create a blank Layer object
            # Note: We assume the Layer class is defined or we create a dummy
            new_layer = Layer.__new__(Layer)
            new_layer.weights = np.array(layer_data["weights"])
            new_layer.bias = np.array(layer_data["bias"])
            model.layers.append(new_layer)

        return model

    def copy(self) -> DigitRecogniser:
        """
        Creates a brand new DigitRecogniser instance with the exact
        same weights and biases as this one.
        """
        # Create a "blank" instance
        # Using __new__ avoids running the random initialization in __init__
        new_model = DigitRecogniser.__new__(DigitRecogniser)

        # Deep copy the layers list
        new_model.layers = []
        for layer in self.layers:
            # Create a new Layer object
            new_layer = Layer.__new__(Layer)

            # Use NumPy's .copy() to duplicate the actual weight data
            new_layer.weights = layer.weights.copy()
            new_layer.bias = layer.bias.copy()

            new_model.layers.append(new_layer)

        return new_model

    def predict(self, image_array: np.ndarray | ImageArray):
        # Ensure input is a column vector (784, 1)
        out = np.array(image_array).flatten().reshape(-1, 1)
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def mutate(self):
        for layer in self.layers:
            layer.mutate()
