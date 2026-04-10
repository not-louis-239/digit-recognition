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
import torch
import random
import json

import numpy as np
from .digit_recogniser import DigitRecogniser, device
from ..utils import chance, clamp
from ..utils.dirs import DIRS
from ..utils.constants import (
    POPULATION_SIZE,
    BASE_SELECTION_PRESSURE,
    IMMIGRATION_RATE,
    HYPERMUTATION_RATE,
    CONFIDENCE_PENALTY_FACTOR,
    SMALL_MARGIN_PENALTY_FACTOR,
    TARGET_MARGIN,
    HARDENING_EPOCH,
    LOGIT_GAIN,
    USE_GPU_ACCEL,
    calc_mutation_rate
)
from ..utils.seasons import get_year_and_season

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

        self._cached_training_data_id: int | None = None
        self._cached_images: torch.Tensor | None = None
        self._cached_labels: torch.Tensor | None = None

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

        self.year, self.season = get_year_and_season(self.epoch)

    def evaluate_model(self, model: DigitRecogniser, data: list[tuple[np.ndarray, int, np.ndarray]]) -> tuple[float, float]:
        """Returns tuple of (average_loss, accuracy_rate), where 0 <= accuracy_rate <= 1. data is tuple[img, label, one_hot]
        Not deprecated as it can still be used to evaluate one model. But for evaluating many models, use evaluate_models_batch() instead."""

        # Subsample data for faster evaluation (use only 20% of data for speed)
        subsample_size = max(100, len(data) // 5)  # At least 100 samples, or 20% of data
        data_subset = random.sample(data, subsample_size)

        images = np.array([img for img, *_ in data_subset], dtype=np.float32)  # (N, 28, 28)
        labels = np.array([label for _, label, _ in data_subset], dtype=np.int64)  # (N,)

        preds = model.predict_batch(images)  # (10, N)

        # Enable this debug print to check for output saturation
        # print(f"Min: {preds.min():.4f}, Mean: {preds.mean():.4f}, Max: {preds.max():.4f}")

        # one-hot targets (10, N)
        targets = np.zeros_like(preds)
        targets[labels, np.arange(labels.size)] = 1.0

        loss = -np.sum(targets * np.log(np.clip(preds, 1e-15, 1 - 1e-15))) / labels.size

        # Confidence penalty: reward higher probability on the correct class
        correct_probs = preds[labels, np.arange(labels.size)]
        confidence_penalty = np.mean(1.0 - correct_probs)

        # Margin penalty: reward a larger gap between top-1 and top-2 predictions
        top2 = np.partition(preds, -2, axis=0)
        top1 = top2[-1]
        second = top2[-2]
        margins = top1 - second
        margin_penalty = np.mean(np.clip(TARGET_MARGIN - margins, 0.0, 1.0))

        loss += CONFIDENCE_PENALTY_FACTOR * confidence_penalty
        loss += SMALL_MARGIN_PENALTY_FACTOR * margin_penalty
        accuracy = np.mean(np.argmax(preds, axis=0) == labels)

        return loss, accuracy

    def _prepare_cached_data(self, data: list[tuple[np.ndarray, int, np.ndarray]]) -> None:
        """Cache the training dataset as torch tensors, optionally on the selected device."""
        data_id = id(data)
        if self._cached_training_data_id == data_id:
            return

        images = np.stack([img for img, *_ in data], dtype=np.float32)
        labels = np.array([label for _, label, _ in data], dtype=np.int64)

        tensor_images = torch.from_numpy(images.reshape(images.shape[0], -1).T).float()
        tensor_labels = torch.from_numpy(labels)

        if USE_GPU_ACCEL:
            tensor_images = tensor_images.to(device)
            tensor_labels = tensor_labels.to(device)

        self._cached_images = tensor_images
        self._cached_labels = tensor_labels
        self._cached_training_data_id = data_id

    def evaluate_models_batch(self, models: list[DigitRecogniser], data: list[tuple[np.ndarray, int, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
        """Batched evaluation of all models for speed. Returns (losses, accuracies) arrays."""
        subsample_size = max(100, len(data) // 5)
        self._prepare_cached_data(data)

        assert self._cached_images is not None and self._cached_labels is not None

        num_models = len(models)
        num_samples = self._cached_labels.shape[0]
        indices = random.sample(range(num_samples), subsample_size)

        images = self._cached_images[:, indices]  # (784, N)
        labels = self._cached_labels[indices]     # (N,)

        # Stack weights and biases for all models
        layer_weights = []
        layer_biases = []
        for i in range(len(models[0].layers)):
            w = torch.stack([m.layers[i].weights for m in models])  # (num_models, out, in)
            b = torch.stack([m.layers[i].bias for m in models])    # (num_models, out, 1)
            layer_weights.append(w.to(device if USE_GPU_ACCEL else w.device))
            layer_biases.append(b.to(device if USE_GPU_ACCEL else b.device))

        # Prepare input: expand to (num_models, 784, N)
        out = images.unsqueeze(0).expand(num_models, -1, -1)

        # Forward pass for all models
        for i, (w, b) in enumerate(zip(layer_weights, layer_biases)):
            out = torch.matmul(w, out) + b
            if i < len(layer_weights) - 1:
                out = torch.where(out > 0, out, 0.01 * out)

        # Softmax on output layer
        out = torch.softmax(out * LOGIT_GAIN, dim=1)  # (num_models, 10, N)

        # Compute one-hot targets on the same device
        targets = torch.zeros((num_models, 10, subsample_size), device=out.device)
        target_indices = labels.view(1, 1, -1).expand(num_models, -1, -1)
        targets.scatter_(1, target_indices, 1.0)

        # Loss
        loss = -torch.sum(targets * torch.log(torch.clamp(out, 1e-15, 1 - 1e-15)), dim=[1, 2]) / subsample_size

        # Accuracy
        preds = torch.argmax(out, dim=1)
        acc = torch.mean((preds == labels.unsqueeze(0)).float(), dim=1)

        return loss.detach().cpu().numpy(), acc.detach().cpu().numpy()

    def run_generation(self, one_hots: list[tuple[np.ndarray, int, np.ndarray]]) -> None:
        """
        Run a generation. Takes training data in the form of list[tuple[image, correct_digit, one_hot]]
        Eliminate all but the best individuals, then have the best
        individuals make offspring for the next generation.

        The strong shall eat the weak, as it is said.
        """

        # First, update year and season. This is required for seasonal effects to update correctly.
        self.year, self.season = get_year_and_season(self.epoch)

        # Calculate parameters like mutation rate and selection pressure
        mutation_rate = calc_mutation_rate(self.epoch) * self.season.mutation_modifier
        selection_pressure = BASE_SELECTION_PRESSURE * self.season.selection_pressure_modifier
        len_population = len(self.population)
        print(f"Generation {self.epoch}. Mutation Rate: {mutation_rate:.4f}, Selection Pressure: {selection_pressure:.1f}, Population: {len_population}")

        # Calculate fitness for everyone
        # Batched evaluation for speed
        losses, accuracies = self.evaluate_models_batch(self.population, one_hots)
        results: list[Evaluation] = [Evaluation(loss=losses[i], accuracy_rate=accuracies[i], model=model) for i, model in enumerate(self.population)]

        # Sort by loss (lowest is best!)
        results.sort(key=lambda entry: entry.loss)
        self.last_evals = results

        best_eval: Evaluation = results[0]
        print(f"Generation Best Loss: {best_eval.loss:.4f} | Best Acc: {best_eval.accuracy_rate:.4%}")

        # Selection
        num_survivors: int = int(clamp(POPULATION_SIZE // selection_pressure, (1, len_population)))

        elite = [e.model for e in results[:num_survivors]]
        protected = [e.model for e in results if e.model.grace > 0]
        survivors = list({*elite, *protected})

        print(f"Found {len(survivors)} survivors.")

        # Repopulation: Fill the rest of the slots with mutated clones
        new_generation = survivors.copy()

        while len(new_generation) < POPULATION_SIZE:
            if self.epoch < HARDENING_EPOCH and chance(IMMIGRATION_RATE):
                # Immigration: add an entirely new model
                new = DigitRecogniser(epoch=self.epoch + 1, grace=20)
            elif self.epoch < HARDENING_EPOCH and chance(HYPERMUTATION_RATE):
                # Hypermutation: mutate an existing model by a lot more than usual
                new = random.choice(survivors).copy()
                new.grace = 20
                new.mutate(rate=mutation_rate * 20)
            elif len(survivors) >= 2 and chance(0.6):
                # Sexual reproduction: pick two survivors and mate them
                parent_a, parent_b = random.sample(survivors, 2)
                new = parent_a.spawn_child_with_mate(parent_b, self.epoch + 1, mutation_rate)
            else:
                # Asexual reproduction: pick a random survivor and mutate them
                parent = random.choice(survivors)
                new = parent.spawn_child(self.epoch + 1, mutation_rate)
            new_generation.append(new)

        self.population = new_generation

        # Decrement grace counters for all survivors
        for individual in self.population:
            individual.grace = max(0, individual.grace - 1)

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
