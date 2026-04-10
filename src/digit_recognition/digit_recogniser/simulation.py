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
from ..utils.dirs import DIRS
from ..utils.constants import POPULATION_SIZE, BASE_SELECTION_PRESSURE, STARTING_MUTATION_RATE, calc_mutation_rate
from ..utils.seasons import format_year

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
        else:
            for _ in range(POPULATION_SIZE):
                self.population.append(DigitRecogniser())
                self.epoch: int = 0

        self.year, self.season = format_year(self.epoch)

    def evaluate_model(self, model: DigitRecogniser, data: list[tuple[np.ndarray, int, np.ndarray]]) -> tuple[float, float]:
        """Returns tuple of (average_loss, accuracy_rate), where 0 <= accuracy_rate <= 1. data is tuple[img, label, one_hot]"""

        images = np.array([img for img, *_ in data], dtype=np.float32)  # (N, 28, 28)
        labels = np.array([label for _, label, _ in data], dtype=np.int64)  # (N,)

        preds = model.predict_batch(images)  # (10, N)
        print(f"Min: {preds.min()}, Mean: {preds.mean()}, Max: {preds.max():.4f}")


        # one-hot targets (10, N)
        targets = np.zeros_like(preds)
        targets[labels, np.arange(labels.size)] = 1.0

        loss = -np.sum(targets * np.log(np.clip(preds, 1e-15, 1 - 1e-15))) / labels.size
        accuracy = np.mean(np.argmax(preds, axis=0) == labels)

        return loss, accuracy

    def run_generation(self, one_hots: list[tuple[np.ndarray, int, np.ndarray]]) -> None:
        """
        Run a generation. Takes training data in the form of list[tuple[image, correct_digit, one_hot]]
        Eliminate all but the best individuals, then have the best
        individuals make offspring for the next generation.

        The strong shall eat the weak, as it is said.
        """

        # Calculate parameters like mutation rate and selection pressure
        mutation_rate = calc_mutation_rate(self.epoch) * self.season.mutation_modifier
        selection_pressure = BASE_SELECTION_PRESSURE * self.season.selection_pressure_modifier

        # Calculate fitness for everyone
        # We store them as (loss, model) tuples so we can sort them
        results: list[Evaluation] = []
        for model in self.population:
            loss, accuracy_rate = self.evaluate_model(model, one_hots)
            results.append(Evaluation(loss=loss, accuracy_rate=accuracy_rate, model=model))

        # Sort by loss (lowest is best!)
        results.sort(key=lambda entry: entry.loss)
        self.last_evals = results

        best_eval: Evaluation = results[0]
        print(f"Generation Best Loss: {best_eval.loss:.4f} | Best Acc: {best_eval.accuracy_rate:.4%}")

        # Selection: Keep the top 10%. The rest? Goodbye.
        num_elites: int = int(max(1, POPULATION_SIZE // selection_pressure))
        elites: list[DigitRecogniser] = [e.model for e in results[:num_elites]]

        # Repopulation: Fill the rest of the slots with mutated clones
        new_generation = elites.copy()

        while len(new_generation) < POPULATION_SIZE:
            # Pick a random winner from the elites and have it make an offspring
            parent = random.choice(elites)
            new_generation.append(parent.spawn_child(self.epoch + 1, mutation_rate))

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

def save_to_dir(data: list[Evaluation], dir_path: Path = DIRS.incubator.path()) -> None:
    # Using list[Evaluation] so that can generate filenames based on model performance

    epoch = max(ev.model.epoch for ev in data)

    BASE_DIR = dir_path / f"epoch_{epoch:08d}"
    BASE_DIR.mkdir(parents=True, exist_ok=True)  # I'm damn traumatised by FileNotFoundErrors, I forgot for a second that this is required

    data.sort(key=lambda ev: ev.loss)  # lowest loss first

    for rank, ev in enumerate(data, start=1):
        filename = f"epoch_{epoch}_rank_{rank}_loss_{ev.loss:.4f}.json"
        model_data = ev.model.to_json()

        with open(BASE_DIR / filename, "w") as f:
            json.dump(model_data, f, indent=4)
