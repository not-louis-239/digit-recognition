import math

import pygame as pg
import numpy as np
from pygame import Surface

from digit_recognition.utils.constants import WN_W, WN_H, BASE_SELECTION_PRESSURE, calc_mutation_rate, IMAGE_SIZE
from digit_recognition.gui.utils.text_utils import draw_text
from digit_recognition.gui.utils.asset_manager import Assets
from digit_recognition.gui.utils.buttons import Button, DEFAULT_BUTTON_FONT_FAMILY
from digit_recognition.gui.utils.input_manager import InputManager
from digit_recognition.gui.states import State, StateChangeRequest, StateID
from digit_recognition.digit_recogniser.simulation import Simulation, save_to_dir
from digit_recognition.gui.utils.ambient_messages import AmbientMessage
from digit_recognition.utils import clamp

class SimState(State):
    def __init__(self, assets: Assets, sim: Simulation):
        super().__init__(assets)
        self.padding = 30
        self.sim_running: bool = False
        self.autosave: bool = True
        self.autosave_interval: int = 500  # autosave when epoch % self.autosave_interval = 0
        self.autosave_increment: int = 100
        self.sim = sim
        self.minimal_ui: bool = False

        self.run_button = Button((WN_W - 150 - self.padding, self.padding), (150, 60), "Run", font_profile=(DEFAULT_BUTTON_FONT_FAMILY, 28))
        self.return_button = Button((self.padding, WN_H - 75 - self.padding), (150, 75), "Return", font_profile=(DEFAULT_BUTTON_FONT_FAMILY, 30))
        self.save_button = Button((2 * self.padding + 150, WN_H - 75 - self.padding), (225, 75), "Save Models", font_profile=(DEFAULT_BUTTON_FONT_FAMILY, 28))

        self.autosave_button = Button((3 * self.padding + 375, WN_H - 75 - self.padding), (250, 75), "Autosave: On", font_profile=(DEFAULT_BUTTON_FONT_FAMILY, 28))
        self.autosave_dec_button = Button((3 * self.padding + 375, WN_H - 120 - self.padding), (120, 35), f"-{self.autosave_increment}", font_profile=(DEFAULT_BUTTON_FONT_FAMILY, 24))
        self.autosave_inc_button = Button((3 * self.padding + 505, WN_H - 120 - self.padding), (120, 35), f"+{self.autosave_increment}", font_profile=(DEFAULT_BUTTON_FONT_FAMILY, 24))

        self.toggle_ui_button = Button((WN_W - 150 - self.padding, WN_H - 75 - self.padding), (150, 75), "Minimal UI", font_profile=(DEFAULT_BUTTON_FONT_FAMILY, 24))

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
        if input_manager.went_down(pg.K_SPACE):
            self.sim_running = not self.sim_running

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
            self.autosave_interval = max(self.autosave_increment, self.autosave_interval - self.autosave_increment)
            self.notifs.set_msg(text=f"Autosave interval: {self.autosave_interval:,} epochs.", colour=(200, 200, 200), lifetime_s=2)

        if self.autosave_inc_button.check_click(input_manager.events):
            self.autosave_interval += self.autosave_increment
            self.notifs.set_msg(text=f"Autosave interval: {self.autosave_interval:,} epochs.", colour=(200, 200, 200), lifetime_s=2)

        if self.toggle_ui_button.check_click(input_manager.events):
            self.minimal_ui = not self.minimal_ui
            self.toggle_ui_button.set_appearance(text="Full UI" if self.minimal_ui else "Minimal UI")

        return StateChangeRequest()

    def draw(self, wn: Surface) -> None:
        wn.fill((30, 30, 30))

        left_items_start_y = self.padding + 170

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

            if not self.minimal_ui:
                # Show raw outputs on a sample image for confidence diagnostics
                if self.assets.one_hots:
                    sample_idx = self.sim.epoch % len(self.assets.one_hots)
                    sample_img, sample_label, _ = self.assets.one_hots[sample_idx]
                    preds = best.model.predict(sample_img).flatten()
                    model_prediction: int = int(np.argmax(preds))

                    diagnostic_start_y = self.padding + 290

                    draw_text(
                        surface=wn, pos=(WN_W - self.padding, diagnostic_start_y), horiz_align='right', vert_align='top',
                        text=f"Raw outputs (i={sample_idx}, ✓={sample_label})", colour=(200, 200, 200),
                        font_profile=(self.assets.monospaced_reg, 20)
                    )

                    for i in range(10):
                        colour = (100, 220, 250) if i == model_prediction else (100, 255, 100) if i == sample_label else (115, 115, 115)

                        # Draw bar graph to the left of the text
                        diagnostic_margin_x = 300
                        text_to_bar_padding = 15
                        bar_graph_size = diagnostic_margin_x - self.padding - text_to_bar_padding
                        entry_y = diagnostic_start_y + 35 + i * 18
                        gap = 3

                        pg.draw.rect(surface=wn, color=(0, 0, 0), rect=(WN_W - diagnostic_margin_x + text_to_bar_padding, entry_y + gap, bar_graph_size, 18 - gap))
                        bar_width = int(clamp(preds[i], (0.0, 1.0)) * bar_graph_size)
                        pg.draw.rect(surface=wn, color=colour, rect=(WN_W - diagnostic_margin_x + text_to_bar_padding, entry_y + gap, bar_width, 18 - gap))

                        # Draw text
                        draw_text(
                            surface=wn, pos=(WN_W - diagnostic_margin_x, entry_y),
                            horiz_align='right', vert_align='top',
                            text=f"{i}: {preds[i]:.4f}", colour=colour,
                            font_profile=(self.assets.monospaced_reg, 18)
                        )

                    last_y = entry_y
                    draw_text(
                        surface=wn, pos=(WN_W - self.padding, last_y + 30), horiz_align='right', vert_align='top',
                        text=f"Top Guess Confidence: {preds[model_prediction]:.4f}", colour=(200, 200, 200),
                        font_profile=(self.assets.monospaced_reg, 20)
                    )

                    # Render the sample image so it's clear what the model "saw"
                    sample_scale = 5
                    sample_size = IMAGE_SIZE * sample_scale
                    sample_x = WN_W - self.padding - sample_size
                    sample_y = last_y + 70
                    for r, row_vals in enumerate(sample_img):
                        for c, val in enumerate(row_vals):
                            v = max(0, min(255, int(val * 255)))
                            pg.draw.rect(
                                wn, (v, v, v),
                                (sample_x + c * sample_scale, sample_y + r * sample_scale, sample_scale, sample_scale)
                            )

                visual = best.model.visualise()
                images = visual["first_layer"]["images"]
                if images:
                    tile_scale = 2
                    tile_size = 28 * tile_scale
                    tile_gap = 6
                    cols = max(1, int(math.ceil(math.sqrt(len(images)))))
                    total_w = cols * tile_size + (cols - 1) * tile_gap
                    if total_w > 400:
                        tile_scale = 1
                        tile_size = 28 * tile_scale
                        total_w = cols * tile_size + (cols - 1) * tile_gap

                    x_offset = 20
                    start_x = WN_W // 2 - total_w // 2 + x_offset
                    start_y = left_items_start_y + 80

                    draw_text(
                        surface=wn, pos=(WN_W // 2 + x_offset, left_items_start_y + 30), horiz_align='centre', vert_align='top',
                        text="Best Model (visualised):", colour=(220, 220, 220),
                        font_profile=(self.assets.monospaced_reg, 22)
                    )

                    for idx, img in enumerate(images):
                        col = idx % cols
                        row = idx // cols
                        x0 = start_x + col * (tile_size + tile_gap)
                        y0 = start_y + row * (tile_size + tile_gap)

                        # Create surface from image array for efficient rendering
                        img_array = np.array(img)
                        pixel_array = (img_array * 255).astype(np.uint8)
                        rgb_array = np.stack([pixel_array] * 3, axis=-1)
                        surf = pg.surfarray.make_surface(rgb_array)
                        if tile_scale != 1:
                            surf = pg.transform.scale(surf, (tile_size, tile_size))
                        wn.blit(surf, (x0, y0))

                # Show loss distribution as bar graph with lowest loss on the left
                losses = [ev.loss for ev in self.sim.last_evals]
                min_loss = min(losses)
                max_loss = max(losses)
                spread = max(max_loss - min_loss, 1e-8)

                graph_w = 520
                graph_h = 120
                graph_x = (WN_W - graph_w) // 2 + 40
                graph_y = self.padding + 20

                pg.draw.rect(wn, (45, 45, 60), (graph_x, graph_y, graph_w, graph_h))

                bar_w = max(1, graph_w / max(1, len(losses)))
                for i, ev in enumerate(self.sim.last_evals):
                    norm = (ev.loss - min_loss) / spread
                    bar_h = int(norm * graph_h)
                    x = graph_x + i * bar_w
                    y = graph_y + (graph_h - bar_h)
                    colour = (100, 255, 100) if ev.model.grace > 0 else (140, 200, 255)
                    pg.draw.rect(wn, colour, (x, y, bar_w + 1, bar_h))  # +1 removes gaps

                draw_text(
                    surface=wn, pos=(graph_x, graph_y - 6), horiz_align='left', vert_align='bottom',
                    text="Loss Distribution (best → worst)", colour=(180, 180, 180),
                    font_profile=(self.assets.monospaced_reg, 18)
                )

                draw_text(
                    surface=wn, pos=(graph_x - 10, graph_y), horiz_align='right', vert_align='centre',
                    text=f"{max_loss:.4f}", colour=(180, 180, 180),
                    font_profile=(self.assets.monospaced_reg, 18)
                )

                draw_text(
                    surface=wn, pos=(graph_x - 10, graph_y + graph_h), horiz_align='right', vert_align='centre',
                    text=f"{min_loss:.4f}", colour=(180, 180, 180),
                    font_profile=(self.assets.monospaced_reg, 18)
                )

                draw_text(
                    surface=wn, pos=(graph_x - 10, graph_y + graph_h // 2 - 12), horiz_align='right', vert_align='centre',
                    text=f"r={max_loss - min_loss:.4f}", colour=(255, 255, 255),
                    font_profile=(self.assets.monospaced_reg, 18)
                )
                draw_text(
                    surface=wn, pos=(graph_x - 10, graph_y + graph_h // 2 + 12), horiz_align='right', vert_align='centre',
                    text=f"{(max_loss - min_loss) / max_loss:.2%}", colour=(255, 255, 255),
                    font_profile=(self.assets.monospaced_reg, 18)
                )
        elif not self.minimal_ui:
            draw_text(
                surface=wn, pos=(WN_W - self.padding, self.padding + 185), horiz_align='right', vert_align='top',
                text="Press Run to start the simulation.", colour=(220, 220, 220),
                font_profile=(self.assets.monospaced_reg, 24)
            )

        # Draw buttons
        if not self.minimal_ui:
            self.run_button.draw(wn)
            self.return_button.draw(wn)
            self.save_button.draw(wn)
            self.autosave_button.draw(wn)
            self.autosave_dec_button.draw(wn)
            self.autosave_inc_button.draw(wn)

        # Always draw toggle button
        self.toggle_ui_button.draw(wn)

        if not self.minimal_ui:
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

        # Show season, year and generation
        draw_text(
            surface=wn, pos=(self.padding, self.padding), horiz_align='left', vert_align='top',
            text=f"{self.sim.season.name}", colour=self.sim.season.colour, font_profile=(self.assets.monospaced_reg, 36)
        )
        draw_text(
            surface=wn, pos=(self.padding, self.padding + 60), horiz_align='left', vert_align='top',
            text=f"Year {self.sim.year:,}", colour=(255, 255, 255), font_profile=(self.assets.monospaced_reg, 22)
        )
        draw_text(
            surface=wn, pos=(self.padding, self.padding + 90), horiz_align='left', vert_align='top',
            text=f"Generation {self.sim.epoch:,}", colour=(150, 150, 150), font_profile=(self.assets.monospaced_reg, 22)
        )

        if not self.minimal_ui:
            # Show autosave status
            autosave_colour = (100, 255, 100) if self.autosave else (255, 100, 100)
            autosave_text = f"Every {self.autosave_interval:,} epochs" if self.autosave else "Off"
            draw_text(
                surface=wn, pos=(self.padding, left_items_start_y), horiz_align='left', vert_align='top',
                text=f"Autosave: {autosave_text}", colour=autosave_colour, font_profile=(self.assets.monospaced_reg, 22)
            )

            # Show shape of the current models as (l0, l1...ln) where ln is the number of neurons in each layer
            shape = self.sim.population[0].shape()

            draw_text(
                surface=wn, pos=(self.padding, left_items_start_y + 40), horiz_align='left', vert_align='top',
                text=f"Model Shape: {shape}", colour=(220, 220, 220), font_profile=(self.assets.monospaced_reg, 22)
            )

            # Show population, mutation rate, selection pressure
            mutation_rate = calc_mutation_rate(self.sim.epoch) * self.sim.season.mutation_modifier
            selection_pressure = BASE_SELECTION_PRESSURE * self.sim.season.selection_pressure_modifier
            population_size = len(self.sim.population)
            draw_text(
                surface=wn, pos=(self.padding, left_items_start_y + 70), horiz_align='left', vert_align='top',
                text=f"Population: {population_size}", colour=(220, 220, 220), font_profile=(self.assets.monospaced_reg, 22)
            )
            draw_text(
                surface=wn, pos=(self.padding, left_items_start_y + 100), horiz_align='left', vert_align='top',
                text=f"Mutation Rate: {mutation_rate:.4f}", colour=(220, 220, 220), font_profile=(self.assets.monospaced_reg, 22)
            )
            draw_text(
                surface=wn, pos=(self.padding, left_items_start_y + 130), horiz_align='left', vert_align='top',
                text=f"Selection Pressure: {selection_pressure:.2f}", colour=(220, 220, 220), font_profile=(self.assets.monospaced_reg, 22)
            )

            # Show population composition: elites, protected
            # TODO: eventually we'd want this to be retrieved, not computed, but it is not a major problem as it is relatively cheap on the CPU
            elites_count = clamp(int(population_size // max(1e-12, selection_pressure)), (1, population_size))  # using epsilon minimum to prevent ZeroDivisionError
            protected_count = sum(1 for model in self.sim.population if model.grace > 0)

            draw_text(
                surface=wn, pos=(self.padding, left_items_start_y + 160), horiz_align='left', vert_align='top',
                text=f"Elites: {elites_count} | Protected: {protected_count}", colour=(200, 200, 200),
                font_profile=(self.assets.monospaced_reg, 22)
            )

            # Show notification popups
            draw_text(
                surface=wn, pos=(self.padding, WN_H - self.padding * 2 - 100), horiz_align='left', vert_align='bottom',
                text=f"{self.notifs.text}", colour=self.notifs.colour, font_profile=(self.assets.monospaced_reg, 24)
            )
