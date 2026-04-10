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

from digit_recognition.utils.colours import col, RESET, BOLD, FAINT
from digit_recognition.utils.custom_types import Colour

@dataclass(kw_only=True)
class Season:
    name: str
    colour: Colour
    mutation_modifier: float
    selection_pressure_modifier: float

# Spring, Summer = higher mutation, "easier" environment (less selection pressure)
# Autumn, Winter = lower mutation (genetic hardening), "harsher" environment (more selection pressure)
SEASONS: list[Season] = [
    Season(
        name="Spring",
        colour=(100, 255, 100),
        mutation_modifier=1.35,
        selection_pressure_modifier=0.9
    ),
    Season(
        name="Summer",
        colour=(255, 220, 100),
        mutation_modifier=1.5,
        selection_pressure_modifier=0.6
    ),
    Season(
        name="Autumn",
        colour=(255, 140, 100),
        mutation_modifier=0.9,
        selection_pressure_modifier=1.1
    ),
    Season(
        name="Winter",
        colour=(200, 200, 255),
        mutation_modifier=0.6,
        selection_pressure_modifier=1.6
    )
]

def get_season(generation: int) -> Season:
    """Returns the season for a given year"""
    season_idx = generation % len(SEASONS)
    return SEASONS[season_idx]

def format_year(generation: int) -> tuple[int, Season]:
    """Returns (year, season)"""

    season = get_season(generation)
    year = generation // len(SEASONS)

    return (year, season)
