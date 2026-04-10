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
from warnings import deprecated
import numpy as np

from ..utils import chance
from ..utils.constants import IMAGE_SIZE, LOGIT_GAIN, SCALE_MUTATION_FACTOR, SCALE_MUTATION_CHANCE, NEURONS_PER_HIDDEN_LAYER

def leaky_relu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, 0.01 * x)

def sigmoid(x: np.ndarray) -> np.ndarray:
    """This would be deprecated... if not for the fact that some old models use it."""
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
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((output_size, 1), dtype=np.float32)

    def shape(self) -> tuple[int, int]:
        """Returns the layer's shape in terms of (out, in)"""
        return self.weights.shape

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Process inputs and return outputs dependent on oneself's contents."""
        # inputs: (input_size, batch_size)
        return np.matmul(self.weights, inputs) + self.bias

    def intensify(self, scalar: float) -> None:
        """Scale all weights by a scalar. This can be used for hypermutants,
        immigrants or to give a layer more "oomph" (prediction confidence)."""
        self.weights *= scalar
        self.bias *= scalar

    def mutate(self, rate: float):
        """Generate a slightly different version of oneself."""
        self.weights += np.random.randn(*self.weights.shape).astype(np.float32) * rate
        self.bias += np.random.randn(*self.bias.shape).astype(np.float32) * rate

    def copy(self) -> Layer:
        """Create a new Layer instance that is exactly the same as this one."""

        new = Layer.__new__(Layer)
        new.weights = np.copy(self.weights)
        new.bias = np.copy(self.bias)

        return new

class FirstLayerVisual(TypedDict):
    images: list[list[list[float]]]
    biases: list[float]

class DigitRecogniserVisual(TypedDict):
    layer_sizes: list[int]
    first_layer: FirstLayerVisual

class DigitRecogniser:
    def __init__(self, epoch: int = 0, grace: int = 0):
        """Creates a new DigitRecogniser with a random configuration"""
        # 784 -> 16 -> 16 -> 10
        self.layers: list[Layer] = [
            Layer(784, NEURONS_PER_HIDDEN_LAYER),
            Layer(NEURONS_PER_HIDDEN_LAYER, NEURONS_PER_HIDDEN_LAYER),
            Layer(NEURONS_PER_HIDDEN_LAYER, 10)
        ]

        self.epoch = epoch  # purely cosmetic, but nice to keep track of
        self.grace = grace  # determines number of "safety" generations where it can't be culled

    def to_json(self) -> dict[str, Any]:
        """Convert the model's layers into a JSON-serializable dictionary."""
        serializable_layers = []
        for layer in self.layers:
            serializable_layers.append({
                "weights": layer.weights.tolist(), # Convert torch tensor to list
                "bias": layer.bias.tolist()
            })

        return {
            "layers": serializable_layers,
            "metadata": {
                "architecture": list(self.shape()),
                "epoch": self.epoch
            }
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> DigitRecogniser:
        """Load a model from a JSON-style object."""

        # Create instance without calling __init__ (prevents random weight gen)
        model = cls.__new__(cls)

        # Load layers
        model.layers = []
        for layer_data in data["layers"]:
            new_layer = Layer.__new__(Layer)
            new_layer.weights = np.array(layer_data["weights"], dtype=np.float32)
            new_layer.bias = np.array(layer_data["bias"], dtype=np.float32)
            model.layers.append(new_layer)

        # Metadata
        model.epoch = data["metadata"]["epoch"]
        model.grace = 0  # models aren't automatically invincible upon reload

        return model

    def copy(self, preserve_grace: bool = False) -> DigitRecogniser:
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
        new_model.grace = self.grace if preserve_grace else 0

        return new_model

    def predict(self, image_array: np.ndarray) -> np.ndarray:
        """Predict for a single image. Not deprecated as it can still be used
        for the test mode, where you can give a model an image and have it predict it."""

        out = image_array.flatten().astype(np.float32).reshape(-1, 1)  # (784, 1)

        for i, layer in enumerate(self.layers):
            out = layer.forward(out)
            if i < len(self.layers) - 1:
                out = np.where(out > 0, out, 0.01 * out)
            else:
                out = softmax(out * LOGIT_GAIN)

        return out.squeeze(1)

    def predict_batch(self, image_arrays: np.ndarray) -> np.ndarray:
        """Predict multiple images at once. This improves performance as
        prediction is vectorised."""

        X = image_arrays.reshape(image_arrays.shape[0], -1).T.astype(np.float32)  # (784, N)
        out = X
        for i, layer in enumerate(self.layers):
            out = layer.forward(out)
            if i < len(self.layers) - 1:
                out = np.where(out > 0, out, 0.01 * out)

        out = softmax(out * LOGIT_GAIN)
        return out

    def mutate(self, rate: float) -> None:
        """Change one's configuration slightly"""
        for layer in self.layers:
            # First, additive mutation
            layer.mutate(rate)
            # Then, multiplicative mutation
            if chance(SCALE_MUTATION_CHANCE):
                layer.intensify(SCALE_MUTATION_FACTOR)

    def spawn_child(self, current_epoch: int, mutation_rate: float) -> DigitRecogniser:
        """Return a slightly mutated version of oneself.
        Basically "asexual reproduction" in a sense."""

        child = self.copy()
        child.mutate(mutation_rate)
        child.epoch = current_epoch

        return child

    def spawn_child_with_mate(self, mate: DigitRecogniser, current_epoch: int, mutation_rate: float) -> DigitRecogniser:
        """Sexual reproduction implementation using layer masks to mix-and-match weights and biases"""

        if mate is self:
            # When attempting to mate with oneself, it collapses back to asexual reproduction
            return self.spawn_child(current_epoch, mutation_rate)

        child = self.copy()

        for i in range(len(child.layers)):
            b = mate.layers[i]
            c = child.layers[i]

            mask = np.random.randint(0, 2, size=c.weights.shape).astype(bool)
            c.weights[~mask] = b.weights[~mask]

            bias_mask = np.random.randint(0, 2, size=c.bias.shape).astype(bool)
            c.bias[~bias_mask] = b.bias[~bias_mask]

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

    def shape(self) -> tuple[int, ...]:
        """Returns the architecture of the model as a tuple, e.g. (784, 16, 16, 10)"""
        return tuple(layer.shape()[1] for layer in self.layers) + (self.layers[-1].shape()[0],)
