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

@dataclass
class Season:
    name: str
    colour: Colour

SEASONS: list[Season] = [
    Season("Spring", (100, 255, 100)),
    Season("Summer", (255, 220, 100)),
    Season("Autumn", (255, 140, 100)),
    Season("Winter", (200, 200, 255))
]

def format_year(generation: int) -> tuple[int, Season]:
    """Returns (year, season)"""

    season_idx = generation % len(SEASONS)

    season = SEASONS[season_idx]
    year = generation // len(SEASONS)

    return (year, season)
