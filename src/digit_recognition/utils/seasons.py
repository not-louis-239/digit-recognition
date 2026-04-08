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

from digit_recognition.utils.colours import col, RESET, BOLD

@dataclass
class Season:
    name: str
    colour: str

SEASONS: list[Season] = [
    Season("Spring", col(118)),
    Season("Summer", col(220)),
    Season("Autumn", col(209)),
    Season("Winter", col(153))
]

def format_year(generation: int) -> str:
    season_idx = generation % len(SEASONS)

    season = SEASONS[season_idx]
    year = generation // len(SEASONS)

    return f"{BOLD}{season.colour}{season.name}{RESET}, Year {BOLD}{year}{RESET}"

def _test():
    for gen in range(8):
        print(format_year(gen))

if __name__ == "__main__":
    _test()
