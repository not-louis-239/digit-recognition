from typing import NoReturn
from pathlib import Path

import pygame as pg
import sys
import json

from digit_recognition.digit_recogniser.digit_recogniser import DigitRecogniser
from digit_recognition.digit_recogniser.simulation import Simulation
from digit_recognition.digit_recogniser.load_images import load_images
from digit_recognition.utils.dirs import DIRS

class App:
    def __init__(self) -> None:
        DIRS.generated_models.path().mkdir(parents=True, exist_ok=True)

        print("Enter a seed model (JSON) from which to start the simulation (or Enter to start fresh)")

        while True:
            seed_path_str = input("> ").strip()

            if not seed_path_str:
                self.sim = Simulation()
                break

            seed_path = Path(seed_path_str)

            if not seed_path.exists():
                print(f"Error: No such file.")
                continue

            if not seed_path.is_file():
                print(f"Error: Not a file.")
                continue

            try:
                with open(seed_path, "r") as f:
                    seed = json.load(f)
                    self.sim = Simulation(seed)
                    break
            except json.JSONDecodeError:
                print(f"Error: JSON file is corrupted.")
            except Exception as e:
                print(f"Error: {e} ({type(e).__name__})")

        pg.init()
        self.images = load_images((DIRS.assets.training_data / "digits.csv").path())

    def run(self):
        running = True

        while running:
            ...

        self.quit()

    def quit(self) -> NoReturn:
        pg.quit()
        sys.exit(0)

def main():
    try:
        App().run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)

if __name__ == "__main__":
    main()
