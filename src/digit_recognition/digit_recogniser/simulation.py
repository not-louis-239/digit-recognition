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

from pathlib import Path
import random
import json
import csv

import numpy as np
from .digit_recogniser import DigitRecogniser, calculate_loss
from ..utils.custom_types import ImageArray
from ..utils.constants import POPULATION_SIZE, IMAGE_SIZE

class Simulation:
    def __init__(self, seed: Path | None = None) -> None:
        self.population: list[DigitRecogniser] = []
        if not seed:
            for _ in range(POPULATION_SIZE):
                self.population.append(DigitRecogniser())
        else:
            new = DigitRecogniser.from_json(seed)
            for _ in range(POPULATION_SIZE):
                new_model = new.copy()
                new_model.mutate()  # inject some initial genetic diversity
                self.population.append(new_model)

    def evaluate_model(self, model: DigitRecogniser, data: list[tuple[ImageArray, int]]) -> float:
        total_loss = 0.0
        for image, label in data:
            # Create a 10x1 column of zeros
            correct = np.zeros((10, 1))
            # Set the correct index to 1.0
            correct[label] = 1.0

            prediction = model.predict(image)

            # Now both are NumPy arrays of shape (10, 1)
            total_loss += calculate_loss(correct, prediction)

        return total_loss / len(data)

    def run_generation(self, training_data: list[tuple[ImageArray, int]]) -> None:
        """
        Run a generation.
        Eliminate the worst individuals.
        Mutate the best individuals.

        The strong shall eat the weak, as it is said.
        """

        # Calculate fitness for everyone
        # We store them as (loss, model) tuples so we can sort them
        scored_population = []
        for model in self.population:
            loss = self.evaluate_model(model, training_data)
            scored_population.append((loss, model))

        # Sort by loss (lowest is best!)
        scored_population.sort(key=lambda x: x[0])

        print(f"Generation Best Loss: {scored_population[0][0]:.4f}")

        # Selection: Keep the top 10%. The rest? Goodbye.
        num_elites = max(1, POPULATION_SIZE // 10)
        elites = [pair[1] for pair in scored_population[:num_elites]]

        # Repopulation: Fill the rest of the slots with mutated clones
        new_generation = [e for e in elites]

        while len(new_generation) < POPULATION_SIZE:
            # Pick a random winner from the elites
            parent = random.choice(elites)
            child = parent.copy()
            child.mutate()
            new_generation.append(child)

        self.population = new_generation

    def save_best_model(self, path: Path = Path(__file__).parent / "best_model.json"):
        """Save the best model to a JSON file."""
        # Assuming run_generation was just called, population[0] is the best
        best_model = self.population[0]

        with open(path, "w") as f:
            json.dump(best_model.to_json(), f, indent=4)

        print(f"\033[95m[System]: The Great Library has archived the model to '{path}'\033[0m")

def load_images(file: Path) -> list[tuple[ImageArray, int]]:
    """Loads all images from a text file into an iterable of arrays."""
    dataset: list[tuple[ImageArray, int]] = []

    if not file.exists():
        print(f"Warning: {file} not found.")
        return dataset

    with open(file, "r") as f:
        # Using csv reader is more robust than manual string splitting
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            # label is the first item, pixels are the rest
            label = int(row[0])

            # Convert strings to floats
            # If your Pygame script saves 0/1, these are already normalized.
            # If using MNIST CSV (0-255), use: [float(p) / 255.0 for p in row[1:]]
            pixels = [float(p) for p in row[1:]]

            # Reshape 1D list into 2D ImageArray (28x28)
            image_2d = []
            for i in range(0, len(pixels), IMAGE_SIZE):
                image_2d.append(pixels[i : i + IMAGE_SIZE])

            dataset.append((image_2d, label))

    return dataset
