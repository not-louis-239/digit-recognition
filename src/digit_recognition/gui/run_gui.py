from typing import NoReturn, Any
from pathlib import Path

import pygame as pg
import sys
import json
from pygame import Surface

from digit_recognition.digit_recogniser.simulation import Simulation
from digit_recognition.gui.states import StateID, State
from digit_recognition.gui.states.title_state import TitleState
from digit_recognition.gui.states.sim_state import SimState
from digit_recognition.gui.states.gallery_state import GalleryState
from digit_recognition.gui.utils.input_manager import InputManager
from digit_recognition.gui.utils.asset_manager import Assets, assign_evals
from digit_recognition.utils.constants import WN_W, WN_H, FPS
from digit_recognition.utils.dirs import DIRS

class App:
    def __init__(self) -> None:
        print("Enter a seed JSON file or directory to start the simulation.")
        print("Press Enter to resume from the latest incubator run, or type START_FRESH to start fresh.")

        while True:
            seed_path_str = input("> ").strip()

            if seed_path_str == "START_FRESH":
                self.sim = Simulation()
                break

            if not seed_path_str:
                incubator_path = DIRS.incubator.path()
                best_dir: Path | None = None
                best_epoch = -1

                if incubator_path.exists() and incubator_path.is_dir():
                    for child in incubator_path.iterdir():
                        if not child.is_dir():
                            continue
                        if not child.name.startswith("epoch_"):
                            continue
                        json_files = list(child.rglob("*.json"))
                        if not json_files:
                            continue
                        max_epoch_in_dir = -1
                        for file in json_files:
                            try:
                                with open(file, "r") as f:
                                    data = json.load(f)
                                if isinstance(data, dict):
                                    epoch = int(data.get("metadata", {}).get("epoch", -1))
                                    if epoch > max_epoch_in_dir:
                                        max_epoch_in_dir = epoch
                                elif isinstance(data, list):
                                    for item in data:
                                        if isinstance(item, dict):
                                            epoch = int(item.get("metadata", {}).get("epoch", -1))
                                            if epoch > max_epoch_in_dir:
                                                max_epoch_in_dir = epoch
                            except json.JSONDecodeError:
                                continue
                        if max_epoch_in_dir > best_epoch:
                            best_epoch = max_epoch_in_dir
                            best_dir = child

                if best_dir is None:
                    self.sim = Simulation()
                    break

                seed: list[dict[str, Any]] = []
                json_files = sorted(best_dir.rglob("*.json"))
                for file in json_files:
                    try:
                        with open(file, "r") as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            seed.extend(data)
                        else:
                            seed.append(data)
                    except json.JSONDecodeError:
                        print(f"Warning: skipping corrupted JSON in file: '{file}'")
                if not seed:
                    self.sim = Simulation()
                    break

                print(f"Auto-resuming from incubator: {best_dir}")
                self.sim = Simulation(seed)
                break

            seed_path = Path(seed_path_str)

            if not seed_path.exists():
                print("Error: No such file or directory.")
                continue

            try:
                if seed_path.is_dir():
                    seed: list[dict[str, Any]] = []
                    json_files = sorted(seed_path.rglob("*.json"))
                    if not json_files:
                        print("Error: Directory contains no JSON files.")
                        continue

                    for file in json_files:
                        try:
                            with open(file, "r") as f:
                                data = json.load(f)
                            if isinstance(data, list):
                                seed.extend(data)
                            else:
                                seed.append(data)
                        except json.JSONDecodeError:
                            print(f"Warning: skipping corrupted JSON file: '{file}'")
                    if not seed:
                        print("Error: No valid JSON models found in directory.")
                        continue

                    self.sim = Simulation(seed)
                    break

                if not seed_path.is_file():
                    print("Error: Path is not a file.")
                    continue

                with open(seed_path, "r") as f:
                    data = json.load(f)
                    seed = data if isinstance(data, list) else [data]
                    self.sim = Simulation(seed)
                    break
            except json.JSONDecodeError:
                print("Error: JSON file is corrupted.")
            except Exception as e:
                print(f"Error: {e} ({type(e).__name__})")

        pg.init()  # MUST be before any pygame steps or else such steps will fail
        self.input_manager = InputManager()
        self.assets = Assets()

        self.states: dict[StateID, State] = {
            StateID.TITLE: TitleState(self.assets),
            StateID.SIM: SimState(self.assets, self.sim),
            StateID.GALLERY: GalleryState(self.assets)
        }

        self.state = StateID.TITLE

        # Can only do this after self.sim is initialised
        assign_evals(sim=self.sim, wrappers_list=self.assets.model_wrappers, data=self.assets.test_data)

    def enter_state(self, state: StateID) -> None:
        self.state = state
        self.states[self.state].reset()

    def update(self, dt_s: float) -> None:
        self.states[self.state].update(dt_s)

    def take_input(self) -> None:
        state_change_request = self.states[self.state].take_input(self.input_manager)
        if state_change_request.new is not None:
            self.enter_state(state_change_request.new)

    def draw(self, wn: Surface) -> None:
        self.states[self.state].draw(wn)

    def run(self):
        wn = pg.display.set_mode((WN_W, WN_H))
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
            self.update(self.input_manager.dt_s)
            self.take_input()
            self.draw(wn)
            pg.display.flip()

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
