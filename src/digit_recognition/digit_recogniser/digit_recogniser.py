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

from typing import Any, TypedDict
import numpy as np

from ..utils.constants import STARTING_MUTATION_RATE, NEW_CONFIG_RANGE, IMAGE_SIZE

def sigmoid(x: np.ndarray) -> np.ndarray:
    # NumPy's exp handles entire arrays at once
    return 1 / (1 + np.exp(-x))

def softmax(x: np.ndarray) -> np.ndarray:
    # Stable softmax for vectors or batched matrices.
    # Works with shape (N, 1) or (C, N) where classes are along axis 0.
    x_max = np.max(x, axis=0, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

class Layer:
    def __init__(self, input_size: int, output_size: int) -> None:
        """Initialise a Layer with a random configuration."""
        # We initialize all weights for the whole layer at once
        # Using "He Initialization" (scaling by sqrt of input size) helps stability
        self.weights = np.random.uniform(-NEW_CONFIG_RANGE, NEW_CONFIG_RANGE, (output_size, input_size))
        self.bias = np.random.uniform(-NEW_CONFIG_RANGE, NEW_CONFIG_RANGE, (output_size, 1))

    def shape(self) -> tuple[int, int]:
        """Returns the layer's shape in terms of (in, out)"""
        return self.weights.shape

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Process inputs and return outputs dependent on oneself's contents."""
        # inputs: (input_size, 1)
        # weight matrix multiplication + bias
        return sigmoid(np.dot(self.weights, inputs) + self.bias)

    def mutate(self, rate=STARTING_MUTATION_RATE):
        """Generate a slightly different version of oneself."""
        # Mutate the entire matrix at once using a mask
        self.weights += np.random.uniform(-rate, rate, self.weights.shape)
        self.bias += np.random.uniform(-rate, rate, self.bias.shape)

    def copy(self) -> Layer:
        """Create a new Layer instance that is exactly the same as this one."""

        # Create a blank instance without running __init__
        new = Layer.__new__(Layer)

        # Use NumPy's .copy() to ensure we aren't just pointing to the same memory
        # If we didn't use .copy(), mutating the child would also change the parent!
        new.weights = self.weights.copy()
        new.bias = self.bias.copy()

        return new

class FirstLayerVisual(TypedDict):
    images: list[list[list[float]]]
    biases: list[float]

class DigitRecogniserVisual(TypedDict):
    layer_sizes: list[int]
    first_layer: FirstLayerVisual

class DigitRecogniser:
    def __init__(self, epoch: int = 0):
        """Creates a new DigitRecogniser with a random configuration"""
        # 784 -> 16 -> 16 -> 10
        self.layers = [
            Layer(784, 16),
            Layer(16, 16),
            Layer(16, 10)
        ]

        self.epoch = epoch  # purely cosmetic, but nice to keep track of

    def to_json(self) -> dict[str, Any]:
        """Convert the model's layers into a JSON-serializable dictionary."""
        serializable_layers = []
        for layer in self.layers:
            serializable_layers.append({
                "weights": layer.weights.tolist(), # Convert NumPy array to list
                "bias": layer.bias.tolist()
            })

        return {
            "layers": serializable_layers,
            "metadata": {
                "architecture": [l.shape()[0] for l in self.layers] + [self.layers[-1].shape()[1]],
                "epoch": self.epoch
            }
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> DigitRecogniser:
        """Load a model from a JSON-style object."""

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

        model.epoch = data["metadata"]["epoch"]

        return model

    def copy(self) -> DigitRecogniser:
        """
        Creates a brand new DigitRecogniser instance with the exact
        same weights, biases and metadata as this one.
        """
        # Create a "blank" instance
        # Using __new__ avoids running the random initialisation in __init__
        new_model = DigitRecogniser.__new__(DigitRecogniser)

        # Deep copy the layers list
        new_model.layers = [layer.copy() for layer in self.layers]

        # Copy metadata
        new_model.epoch = self.epoch

        return new_model

    def predict(self, image_array: np.ndarray) -> np.ndarray:
        """Predict for a single image. Not deprecated as it can still be used
        for the test mode, where you can give a model an image and have it predict it."""

        # Ensure input is a column vector (784, 1)
        out = image_array.flatten().reshape(-1, 1)

        # Pass input through each one of the layers
        for i, layer in enumerate(self.layers):
            out = layer.forward(out)
            if i == len(self.layers) - 1:
                out = softmax(out)
        return out

    def predict_batch(self, image_arrays: np.ndarray) -> np.ndarray:
        """Predict multiple images at once. This improves performance as
        prediction is vectorised."""

        # images shape: (N, 28, 28) or (N, 784)
        X = image_arrays.reshape(image_arrays.shape[0], -1).T  # (784, N)
        out = X
        for i, layer in enumerate(self.layers):
            out = sigmoid(layer.weights @ out + layer.bias)  # (out, N)
            if i == len(self.layers) - 1:
                out = softmax(out)
        return out  # (10, N)

    def mutate(self, rate=STARTING_MUTATION_RATE) -> None:
        """Change one's configuration slightly"""
        for layer in self.layers:
            layer.mutate(rate)

    def spawn_child(self, current_epoch: int, mutation_rate: float) -> DigitRecogniser:
        """Return a slightly mutated version of oneself.
        Basically "asexual reproduction" in a sense."""

        child = self.copy()
        child.mutate(mutation_rate)
        child.epoch = current_epoch

        return child

    def visualise(self) -> DigitRecogniserVisual:
        """
        Returns visual data of oneself for the GUI to render.
        Arrays are normalized to 0..1 so the UI can colour map easily.
        """
        layer0 = self.layers[0]
        weights = layer0.weights  # shape (n, IMAGE_SIZE * IMAGE_SIZE)
        imgs = weights.reshape(weights.shape[0], IMAGE_SIZE, IMAGE_SIZE)

        # normalize each image independently to 0..1 for display
        norm_imgs: list[list[list[float]]] = []
        for img in imgs:
            mn, mx = img.min(), img.max()
            if mx - mn < 1e-6:
                norm_imgs.append((img * 0).tolist())
            else:
                norm_imgs.append(((img - mn) / (mx - mn)).tolist())

        return {
            "layer_sizes": [l.shape()[0] for l in self.layers] + [self.layers[-1].shape()[1]],
            "first_layer": {
                "images": norm_imgs,   # list[28][28] floats in 0..1
                "biases": layer0.bias.flatten().tolist(),
            },
        }
