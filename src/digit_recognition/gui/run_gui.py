from typing import NoReturn
from pathlib import Path

import pygame as pg
import sys
import json

from digit_recognition.digit_recogniser.digit_recogniser import DigitRecogniser
from digit_recognition.digit_recogniser.simulation import Simulation
from digit_recognition.digit_recogniser.image_manager import load_imgs_from_csv
from digit_recognition.gui.states import StateID, State
from digit_recognition.gui.states.main_state import MainState
from digit_recognition.gui.utils.input_manager import InputManager
from digit_recognition.utils.dirs import DIRS

FPS = 60

class App:
    def __init__(self) -> None:
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

        pg.init()  # MUST be before any pygame steps or else they will fail
        self.input_manager = InputManager()
        self.images = load_imgs_from_csv((DIRS.assets.training_data / "digits.csv").path())

        self.states: dict[StateID, State] = {
            StateID.MAIN: MainState()
        }

    def update(self) -> None:
        ...

    def take_input(self) -> None:
        ...

    def draw(self, wn) -> None:
        wn.fill((255, 255, 255))

    def run(self):
        wn = pg.display.set_mode((1250, 750))
        pg.display.set_caption("Digit Recognition Evolution Simulator")

        pg.mixer.set_num_channels(32)
        clock = pg.time.Clock()

        running = True
        while running:
            # Quit events
            for event in self.input_manager.events:
                if event.type == pg.QUIT:
                    running = False

            # Update
            self.update()
            self.take_input()
            self.draw(wn)

            # Update input
            dt_s = clock.tick(FPS) / 1000
            events = pg.event.get()
            keys = pg.key.get_pressed()
            self.input_manager.update_keys(keys, events, dt_s)

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
