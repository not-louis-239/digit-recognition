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


import sys
from pathlib import Path

# allow Python to resolve imports from the src/ directory
# this is a sort of "hacky" solution, but it works, so who cares?
# must be before any `from digit_recognition.*` imports can take place
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

def main():
    pass

if __name__ == "__main__":
    main()
