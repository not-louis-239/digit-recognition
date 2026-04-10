from pygame import Surface
import pygame as pg
import math

from digit_recognition.utils.constants import WN_W, WN_H, BASE_SELECTION_PRESSURE, calc_mutation_rate
from digit_recognition.gui.utils.text_utils import draw_text
from digit_recognition.gui.utils.asset_manager import Assets
from digit_recognition.gui.utils.buttons import Button
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

            # Show raw outputs on a sample image for confidence diagnostics
            if self.assets.one_hots:
                sample_idx = self.sim.epoch % len(self.assets.one_hots)
                sample_img, sample_label, _ = self.assets.one_hots[sample_idx]
                preds = best.model.predict(sample_img).flatten()

                draw_text(
                    surface=wn, pos=(WN_W - self.padding, self.padding + 255), horiz_align='right', vert_align='top',
                    text=f"Raw outputs (sample {sample_idx}, label {sample_label})", colour=(200, 200, 200),
                    font_profile=(self.assets.monospaced_reg, 20)
                )

                for i in range(10):
                    colour = (100, 255, 100) if i == sample_label else (180, 180, 180)

                    # Draw bar graph to the left of the text
                    diagnostic_margin_x = 300
                    text_to_bar_padding = 15
                    bar_graph_size = diagnostic_margin_x - self.padding - text_to_bar_padding
                    entry_y = self.padding + 280 + i * 18
                    gap = 3

                    pg.draw.rect(surface=wn, color=(0, 0, 0), rect=(WN_W - diagnostic_margin_x + text_to_bar_padding, entry_y + gap, bar_graph_size, 18 - gap))
                    pg.draw.rect(surface=wn, color=colour, rect=(WN_W - diagnostic_margin_x + text_to_bar_padding, entry_y + gap, bar_graph_size * preds[i], 18 - gap))

                    # Draw text
                    draw_text(
                        surface=wn, pos=(WN_W - diagnostic_margin_x, entry_y),
                        horiz_align='right', vert_align='top',
                        text=f"{i}: {preds[i]:.4f}", colour=colour,
                        font_profile=(self.assets.monospaced_reg, 18)
                    )

            draw_text(
                surface=wn, pos=(WN_W - self.padding, self.padding + 470), horiz_align='right', vert_align='top',
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
                start_y = self.padding + 505

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

        # Show autosave status
        autosave_colour = (100, 255, 100) if self.autosave else (255, 100, 100)
        autosave_text = f"Every {self.autosave_interval:,} epochs" if self.autosave else "Off"
        draw_text(
            surface=wn, pos=(self.padding, self.padding + 140), horiz_align='left', vert_align='top',
            text=f"Autosave: {autosave_text}", colour=autosave_colour, font_profile=(self.assets.monospaced_reg, 22)
        )

        # Show population, mutation rate, selection pressure
        mutation_rate = calc_mutation_rate(self.sim.epoch) * self.sim.season.mutation_modifier
        selection_pressure = BASE_SELECTION_PRESSURE * self.sim.season.selection_pressure_modifier
        population_size = len(self.sim.population)
        draw_text(
            surface=wn, pos=(self.padding, self.padding + 180), horiz_align='left', vert_align='top',
            text=f"Population: {population_size}", colour=(220, 220, 220), font_profile=(self.assets.monospaced_reg, 22)
        )
        draw_text(
            surface=wn, pos=(self.padding, self.padding + 210), horiz_align='left', vert_align='top',
            text=f"Mutation Rate: {mutation_rate:.4f}", colour=(220, 220, 220), font_profile=(self.assets.monospaced_reg, 22)
        )
        draw_text(
            surface=wn, pos=(self.padding, self.padding + 240), horiz_align='left', vert_align='top',
            text=f"Selection Pressure: {selection_pressure:.2f}", colour=(220, 220, 220), font_profile=(self.assets.monospaced_reg, 22)
        )


        # Show population composition: elites, protected
        # TODO: eventually we'd want this to be retrieved, not computed, but it is not a major problem as it is relatively cheap on the CPU
        elites_count = 0
        protected_count = 0
        if self.sim.last_evals:
            elites_count = clamp(int(population_size // max(1e-12, selection_pressure)), (1, population_size))  # using epsilon minimum to prevent ZeroDivisionError
            protected_count = sum(1 for model in self.sim.population if model.grace > 0)

        draw_text(
            surface=wn, pos=(self.padding, self.padding + 320), horiz_align='left', vert_align='top',
            text=f"Elites: {elites_count} | Protected: {protected_count}", colour=(200, 200, 200),
            font_profile=(self.assets.monospaced_reg, 22)
        )


        # Show loss distribution as bar graph with lowest loss on the left
        if self.sim.last_evals:
            losses = [ev.loss for ev in self.sim.last_evals]
            min_loss = min(losses)
            max_loss = max(losses)
            spread = max(max_loss - min_loss, 1e-8)

            graph_w = 520
            graph_h = 120
            graph_x = (WN_W - graph_w) // 2 + 40
            graph_y = self.padding + 20

            pg.draw.rect(wn, (45, 45, 60), (graph_x, graph_y, graph_w, graph_h))

            bar_w = max(1, graph_w // max(1, len(losses)))
            for i, ev in enumerate(self.sim.last_evals):
                loss = ev.loss
                norm = (loss - min_loss) / spread
                bar_h = int(norm * graph_h)
                x = graph_x + i * bar_w
                y = graph_y + (graph_h - bar_h)
                colour = (100, 255, 100) if ev.model.grace > 0 else (140, 200, 255)
                pg.draw.rect(wn, colour, (x, y, bar_w, bar_h))

            draw_text(
                surface=wn, pos=(graph_x, graph_y - 6), horiz_align='left', vert_align='bottom',
                text="Loss Distribution (best → worst)", colour=(180, 180, 180),
                font_profile=(self.assets.monospaced_reg, 18)
            )


        # Show notification popups
        draw_text(
            surface=wn, pos=(self.padding, WN_H - self.padding * 2 - 100), horiz_align='left', vert_align='bottom',
            text=f"{self.notifs.text}", colour=self.notifs.colour, font_profile=(self.assets.monospaced_reg, 24)
        )
