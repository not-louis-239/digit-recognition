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
from typing import Any
import random

import numpy as np
from .digit_recogniser import DigitRecogniser, calculate_loss
from ..utils.custom_types import ImageArray
from ..utils.constants import POPULATION_SIZE

@dataclass
class Evaluation:
    loss: float
    accuracy_rate: float
    model: DigitRecogniser

class Simulation:
    def __init__(self, seed: dict[str, Any] | None = None) -> None:
        self.population: list[DigitRecogniser] = []

        if seed is not None:
            new = DigitRecogniser.from_json(seed)
            for _ in range(POPULATION_SIZE):
                new_model = new.copy()
                new_model.mutate()  # inject some initial genetic diversity
                self.population.append(new_model)
            self.epoch = new_model.epoch
            return

        for _ in range(POPULATION_SIZE):
            self.population.append(DigitRecogniser())
            self.epoch = 0

    def evaluate_model(self, model: DigitRecogniser, data: list[tuple[ImageArray, int]]) -> tuple[float, float]:
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

    def run_generation(self, training_data: list[tuple[ImageArray, int]]) -> None:
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

    def get_best_model(self) -> DigitRecogniser:
        """Get the best model from the simulation."""
        # Assuming run_generation was just called, population[0] is the best (lowest loss)
        return self.population[0]
