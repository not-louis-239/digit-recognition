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

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import random
import json

import numpy as np
from .digit_recogniser import DigitRecogniser
from ..utils.constants import POPULATION_SIZE
from ..utils.custom_types import TrainingDataType

def calculate_loss(correct: np.ndarray, actual: np.ndarray) -> float:
    """Calculate the Mean Squared Error"""
    assert correct.shape == actual.shape
    return np.mean((correct - actual) ** 2, dtype=np.float64)

@dataclass
class Evaluation:
    """Associative object to store evaluation results and tie them to a model."""
    loss: float
    accuracy_rate: float
    model: DigitRecogniser

class Simulation:
    def __init__(self, seed: list[dict[str, Any]] | None = None) -> None:
        """If called with no seed, initialises a fresh population.
        If called with a seed (list of raw dictionaries for models),
        it treats the seed as the starting population."""

        self.population: list[DigitRecogniser] = []
        self.last_evals: list[Evaluation] = []  # makes results accessible to GUI elements

        if seed is not None:
            # Load the population from the seed
            seed_population: list[DigitRecogniser] = []
            for model_data in seed:
                seed_population.append(DigitRecogniser.from_json(model_data))

            # Initialise the population as containing all members of the seed
            self.population = seed_population[:]

            # For any empty spots, choose a random model from the seed population to fill it up
            self.epoch: int = self.population[0].epoch
            while len(self.population) < POPULATION_SIZE:
                model = random.choice(seed_population)
                self.population.append(model)

                # Assume the epoch of the 'latest' descendant
                if model.epoch > self.epoch:
                    self.epoch = model.epoch

            return

        for _ in range(POPULATION_SIZE):
            self.population.append(DigitRecogniser())
            self.epoch: int = 0

    def evaluate_model(self, model: DigitRecogniser, data: TrainingDataType) -> tuple[float, float]:
        """Returns tuple of (average_loss, accuracy_rate), where 0 <= accuracy_rate <= 1."""

        total_loss: float = 0.0
        correct_guesses: int = 0
        total_samples: int = len(data)

        for image, correct_num in data:
            # Prepare Ground Truth (One-Hot Encoding)
            correct_one_hot = np.zeros((10, 1))
            correct_one_hot[correct_num] = 1.0

            # Get Prediction
            prediction = model.predict(image) # Returns (10, 1) array of probabilities

            # Calculate Loss for this specific image
            total_loss += calculate_loss(correct_one_hot, prediction)

            # Check Accuracy
            # np.argmax finds the index of the highest value (the model's "choice")
            if np.argmax(prediction) == correct_num:
                correct_guesses += 1

        average_loss = total_loss / total_samples
        accuracy_rate = correct_guesses / total_samples

        return (average_loss, accuracy_rate)

    def run_generation(self, training_data: TrainingDataType) -> None:
        """
        Run a generation.
        Eliminate all but the best individuals, then have the best
        individuals make offspring for the next generation.

        The strong shall eat the weak, as it is said.
        """

        # Calculate fitness for everyone
        # We store them as (loss, model) tuples so we can sort them
        results: list[Evaluation] = []
        for model in self.population:
            loss, accuracy_rate = self.evaluate_model(model, training_data)
            results.append(Evaluation(loss=loss, accuracy_rate=accuracy_rate, model=model))

        # Sort by loss (lowest is best!)
        results.sort(key=lambda entry: entry.loss)
        self.last_evals = results

        best_eval: Evaluation = results[0]
        print(f"Generation Best Loss: {best_eval.loss:.4f} | Best Acc: {best_eval.accuracy_rate:.4%}")

        # Selection: Keep the top 10%. The rest? Goodbye.
        num_elites = max(1, POPULATION_SIZE // 10)
        elites: list[DigitRecogniser] = [e.model for e in results[:num_elites]]

        # Repopulation: Fill the rest of the slots with mutated clones
        new_generation = elites.copy()

        while len(new_generation) < POPULATION_SIZE:
            # Pick a random winner from the elites and have it make an offspring
            parent = random.choice(elites)
            new_generation.append(parent.spawn_child(self.epoch + 1))

        self.population = new_generation

        self.epoch += 1

    def get_best_models(self, n: int, /) -> list[DigitRecogniser]:
        """Get the `n` best model from the simulation."""
        # Assuming run_generation was just called, population[0] is the best (lowest loss)
        return self.population[:n]

def load_from_dir(dir_path: Path) -> list[DigitRecogniser]:
    models = []

    if not dir_path.exists():
        print(f"No such directory: {dir_path}")
        return models
    if not dir_path.is_dir():
        print(f"Path is not a directory: {dir_path}")
        return models

    for file in dir_path.rglob("*.json"):
        try:
            with open(file, "r") as f:
                model_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: skipping corrupted JSON in file: '{file}'")
            continue

        model = DigitRecogniser.from_json(model_data)
        models.append(model)

    return models

def save_to_dir(dir_path: Path, data: list[Evaluation]) -> None:
    # Using list[Evaluation] so that can generate filenames based on model performance

    epoch = max(ev.model.epoch for ev in data)
    data.sort(key=lambda ev: ev.loss)  # lowest loss first

    for rank, ev in enumerate(data, start=1):
        filename = f"epoch_{epoch}_rank_{rank}_loss_{ev.loss:.4f}.json"
        model_data = ev.model.to_json()

        with open(dir_path / filename, "w") as f:
            json.dump(model_data, f, indent=4)
