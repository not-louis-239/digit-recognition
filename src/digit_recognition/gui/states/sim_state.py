from pygame import Surface
import pygame as pg
import math

from digit_recognition.utils.constants import WN_W, WN_H
from digit_recognition.gui.utils.text_utils import draw_text
from digit_recognition.gui.utils.asset_manager import Assets
from digit_recognition.gui.utils.buttons import Button
from digit_recognition.gui.utils.input_manager import InputManager
from digit_recognition.gui.states import State, StateChangeRequest, StateID
from digit_recognition.utils.seasons import format_year
from digit_recognition.digit_recogniser.simulation import Simulation, save_to_dir
from digit_recognition.gui.utils.ambient_messages import AmbientMessage

class SimState(State):
    def __init__(self, assets: Assets, sim: Simulation):
        super().__init__(assets)
        self.padding = 30
        self.sim_running: bool = False
        self.autosave: bool = True
        self.autosave_interval: int = 500  # autosave when epoch % self.autosave_interval = 0
        self.sim = sim

        self.run_button = Button((WN_W - 200 - self.padding, self.padding), (200, 80), "Run")
        self.return_button = Button((self.padding, WN_H - 100 - self.padding), (200, 100), "Return")
        self.save_button = Button((2 * self.padding + 200, WN_H - 100 - self.padding), (300, 100), "Save Models")

        self.autosave_button = Button((3 * self.padding + 500, WN_H - 100 - self.padding), (300, 100), "Autosave: Off")
        self.autosave_dec_button = Button((3 * self.padding + 500, WN_H - 170 - self.padding), (145, 60), "-10")
        self.autosave_inc_button = Button((3 * self.padding + 655, WN_H - 170 - self.padding), (145, 60), "+10")

        self.notifs = AmbientMessage()

    def reset(self) -> None:
        self.sim_running = False

    def update(self, dt_s: float) -> None:
        self.notifs.update(dt_s)

        if self.sim_running:
            self.sim.run_generation(self.assets.one_hots)

            if (self.autosave) and (self.sim.last_evals) and (self.sim.epoch % self.autosave_interval == 0):
                self.notifs.set_msg(text=f"Epoch {self.sim.epoch:,} saved (Autosave)", colour=(100, 255, 100), lifetime_s=1.5)
                save_to_dir(self.sim.last_evals[:10])

    def take_input(self, input_manager: InputManager) -> StateChangeRequest:
        if self.return_button.check_click(input_manager.events):
            return StateChangeRequest(new=StateID.TITLE)

        if self.run_button.check_click(input_manager.events):
            self.sim_running = not self.sim_running

        if self.save_button.check_click(input_manager.events):
            if self.sim.last_evals:
                save_to_dir(self.sim.last_evals[:10])  # save the 10 best models
                self.notifs.set_msg(text="Models saved successfully!", colour=(100, 255, 100), lifetime_s=3)
            else:
                # No data to save
                self.notifs.set_msg(text="No data to save. Press Run to start simulation first!", colour=(255, 100, 100), lifetime_s=2)

        if self.autosave_button.check_click(input_manager.events):
            self.autosave = not self.autosave
            self.autosave_button.set_appearance(text=f"Autosave: {'On' if self.autosave else 'Off'}")
            if self.autosave:
                self.notifs.set_msg(text=f"Autosave enabled (every {self.autosave_interval:,} epochs).", colour=(100, 255, 100), lifetime_s=2)
            else:
                self.notifs.set_msg(text="Autosave disabled.", colour=(255, 200, 100), lifetime_s=2)

        if self.autosave_dec_button.check_click(input_manager.events):
            self.autosave_interval = max(100, self.autosave_interval - 100)
            self.notifs.set_msg(text=f"Autosave interval: {self.autosave_interval:,} epochs.", colour=(200, 200, 200), lifetime_s=2)

        if self.autosave_inc_button.check_click(input_manager.events):
            self.autosave_interval += 100
            self.notifs.set_msg(text=f"Autosave interval: {self.autosave_interval:,} epochs.", colour=(200, 200, 200), lifetime_s=2)

        return StateChangeRequest()

    def draw(self, wn: Surface) -> None:
        wn.fill((30, 30, 30))

        # Draw buttons
        self.run_button.draw(wn)
        self.return_button.draw(wn)
        self.save_button.draw(wn)
        self.autosave_button.draw(wn)
        self.autosave_dec_button.draw(wn)
        self.autosave_inc_button.draw(wn)

        # Show running status
        if self.sim_running:
            text = "Sim: Running"
            colour = (100, 255, 100)
        else:
            text = "Sim: Paused"
            colour = (255, 100, 100)

        draw_text(
            surface=wn, pos=(WN_W - self.padding, self.padding + 110), horiz_align='right', vert_align='top',
            text=text, colour=colour, font_profile=(self.assets.monospaced_reg, 36)
        )

        # Show results from last generation
        if self.sim.last_evals:
            best = self.sim.last_evals[0]
            draw_text(
                surface=wn, pos=(WN_W - self.padding, self.padding + 185), horiz_align='right', vert_align='top',
                text=f"Best Loss: {best.loss:.4f}", colour=(220, 220, 220),
                font_profile=(self.assets.monospaced_reg, 24)
            )
            draw_text(
                surface=wn, pos=(WN_W - self.padding, self.padding + 220), horiz_align='right', vert_align='top',
                text=f"Best Acc: {best.accuracy_rate:.2%}", colour=(220, 220, 220),
                font_profile=(self.assets.monospaced_reg, 24)
            )

            draw_text(
                surface=wn, pos=(WN_W - self.padding, self.padding + 260), horiz_align='right', vert_align='top',
                text="Best Model (visualised):", colour=(220, 220, 220),
                font_profile=(self.assets.monospaced_reg, 22)
            )

            visual = best.model.visualise()
            images = visual["first_layer"]["images"]
            if images:
                tile_scale = 2
                tile_size = 28 * tile_scale
                tile_gap = 6
                cols = max(1, int(math.ceil(math.sqrt(len(images)))))
                rows = int(math.ceil(len(images) / cols))

                total_w = cols * tile_size + (cols - 1) * tile_gap
                start_x = WN_W - self.padding - total_w
                start_y = self.padding + 295

                for idx, img in enumerate(images):
                    col = idx % cols
                    row = idx // cols
                    x0 = start_x + col * (tile_size + tile_gap)
                    y0 = start_y + row * (tile_size + tile_gap)

                    for r, row_vals in enumerate(img):
                        for c, val in enumerate(row_vals):
                            v = max(0, min(255, int(val * 255)))
                            pg.draw.rect(
                                wn, (v, v, v),
                                (x0 + c * tile_scale, y0 + r * tile_scale, tile_scale, tile_scale)
                            )
        else:
            draw_text(
                surface=wn, pos=(WN_W - self.padding, self.padding + 185), horiz_align='right', vert_align='top',
                text="Press Run to start the simulation.", colour=(220, 220, 220),
                font_profile=(self.assets.monospaced_reg, 24)
            )

        year, season = format_year(self.sim.epoch)

        # Show season, year and generation
        draw_text(
            surface=wn, pos=(self.padding, self.padding), horiz_align='left', vert_align='top',
            text=f"{season.name}", colour=season.colour, font_profile=(self.assets.monospaced_reg, 36)
        )
        draw_text(
            surface=wn, pos=(self.padding, self.padding + 60), horiz_align='left', vert_align='top',
            text=f"Year {year:,}", colour=(255, 255, 255), font_profile=(self.assets.monospaced_reg, 24)
        )
        draw_text(
            surface=wn, pos=(self.padding, self.padding + 90), horiz_align='left', vert_align='top',
            text=f"Generation {self.sim.epoch:,}", colour=(150, 150, 150), font_profile=(self.assets.monospaced_reg, 24)
        )

        autosave_colour = (100, 255, 100) if self.autosave else (255, 100, 100)
        draw_text(
            surface=wn, pos=(self.padding, self.padding + 140), horiz_align='left', vert_align='top',
            text=f"Autosave: {f"Every {self.autosave_interval:,} epochs" if self.autosave else "Off"}", colour=autosave_colour, font_profile=(self.assets.monospaced_reg, 24)
        )

        # Show notification popups
        draw_text(
            surface=wn, pos=(self.padding, WN_H - self.padding * 2 - 100), horiz_align='left', vert_align='bottom',
            text=f"{self.notifs.text}", colour=self.notifs.colour, font_profile=(self.assets.monospaced_reg, 24)
        )
