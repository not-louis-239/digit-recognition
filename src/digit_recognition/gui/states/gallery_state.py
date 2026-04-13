from typing import TYPE_CHECKING

import pygame as pg
import numpy as np
from pygame import Surface

from digit_recognition.gui.utils.canvas import Canvas
from digit_recognition.gui.utils.ambient_messages import AmbientMessage
from digit_recognition.gui.utils.input_manager import MouseButton, InputManager
from digit_recognition.gui.utils.buttons import Button
from digit_recognition.gui.utils.text_utils import draw_text
from digit_recognition.gui.states import State, StateChangeRequest, StateID
from digit_recognition.utils import lerp, invlerp
from digit_recognition.digit_recogniser.simulation import Simulation
from digit_recognition.utils.constants import BRUSH_SIZE, BRUSH_STRENGTH, WN_H
from digit_recognition.utils.config import IMAGE_SIZE

if TYPE_CHECKING:
    from digit_recognition.gui.utils.asset_manager import Assets

def map_coordinate(
        coord_in: tuple[float, float],
        topleft_in: tuple[float, float],
        botright_in: tuple[float, float],
        topleft_out: tuple[float, float],
        botright_out: tuple[float, float]
    ) -> tuple[float, float]:

    # Helper functions for clarity
    x_in, y_in = coord_in

    # 1. Find the relative position (0.0 to 1.0) within the input rect
    tx = invlerp(topleft_in[0], botright_in[0], x_in)
    ty = invlerp(topleft_in[1], botright_in[1], y_in)

    # 2. Map that position to the output rect
    x_out = lerp(topleft_out[0], botright_out[0], tx)
    y_out = lerp(topleft_out[1], botright_out[1], ty)

    return (x_out, y_out)

class GalleryState(State):
    """Draw a digit and see what the model thinks it is.
    Optionally, enter the correct digit for your image and add it
    to the training set. This will allow you to train the model.
    Additionally, browse the training data."""

    def __init__(self, assets: Assets, sim: Simulation) -> None:
        self.assets = assets
        self.ui_padding = 30

        self.canvas = Canvas(IMAGE_SIZE, IMAGE_SIZE)
        self.return_button = Button(pos=(self.ui_padding, WN_H - self.ui_padding - 75), size=(150, 75), text="Back")
        self.notif = AmbientMessage()

        self.datasets = [
            assets.training_data,
            assets.dev_data,
            assets.test_data
        ]

        self.last_model_pred: np.ndarray | None = None
        self.model_idx = 0
        self.sim = sim

        self.reset()

    def reset(self) -> None:
        self.canvas.clear()
        self.last_model_pred = None
        self.notif.clear()

    def update(self, dt_s: float) -> None:
        pass

    def take_input(self, input_manager: InputManager) -> StateChangeRequest:
        if self.return_button.check_click(input_manager.events):
            return StateChangeRequest(new=StateID.TITLE)

        # Clear canvas
        if input_manager.went_down(pg.K_BACKSPACE):
            self.last_model_pred = None
            self.canvas.clear()

        # Change model
        if input_manager.went_down(pg.K_e):
            self.last_model_pred = None
            self.model_idx = (self.model_idx + 1) % len(self.assets.model_wrappers)

        # Predict
        if input_manager.went_down(pg.K_SPACE):
            if self.canvas.is_empty():
                self.notif.set_msg(text="Canvas is insufficiently filled.", colour=(255, 200, 100), lifetime_s=2)
            else:
                self.last_model_pred = self.assets.model_wrappers[self.model_idx].model.predict(self.canvas.as_array())

        if input_manager.mouse_is_down(MouseButton.LMB):
            mouse_pos = pg.mouse.get_pos()
            canvas_topleft = (self.ui_padding, self.ui_padding)
            canvas_botright = (self.ui_padding + IMAGE_SIZE * 20, self.ui_padding + IMAGE_SIZE * 20)

            if (canvas_topleft[0] <= mouse_pos[0] <= canvas_botright[0] and
                canvas_topleft[1] <= mouse_pos[1] <= canvas_botright[1]):
                self.last_model_pred = None  # doesn't apply once canvas is edited

                # Map mouse position to canvas coordinates
                canvas_coord = map_coordinate(
                    coord_in=mouse_pos,
                    topleft_in=canvas_topleft,
                    botright_in=canvas_botright,
                    topleft_out=(0, 0),
                    botright_out=(IMAGE_SIZE, IMAGE_SIZE)
                )
                self.canvas.handle_mouse_input(
                    dt_s=input_manager.dt_s, mouse_pos=canvas_coord, brush_size=BRUSH_SIZE, brush_strength=BRUSH_STRENGTH
                )

        return StateChangeRequest()

    def draw(self, wn: Surface) -> None:
        CANVAS_SIZE = 560
        CANVAS_PIXEL_SIZE = CANVAS_SIZE / IMAGE_SIZE

        wn.fill((30, 30, 30))
        self.canvas.draw(wn, tile_size_px=CANVAS_PIXEL_SIZE, start_pos=(self.ui_padding, self.ui_padding))
        self.return_button.draw(wn)

        # Show the model being used
        text_start_x = 2 * self.ui_padding + CANVAS_SIZE
        text_start_y = self.ui_padding

        draw_text(
            surface=wn, pos=(text_start_x, text_start_y), horiz_align='left', vert_align='top',
            text=f"Using Model: {self.assets.model_wrappers[self.model_idx].common_name} ({self.model_idx + 1} / {len(self.assets.model_wrappers)})",
            colour=(220, 220, 220), font_profile=(self.assets.monospaced_reg, 22)
        )

        # Show last prediction
        if self.last_model_pred is not None:
            last_pred = np.argmax(self.last_model_pred)

            draw_text(
                surface=wn, pos=(text_start_x, text_start_y + 30), horiz_align='left', vert_align='top',
                text=f"Predicted label: {last_pred}", colour=(220, 220, 220), font_profile=(self.assets.monospaced_reg, 22)
            )

            # Draw bar graph
            text_start_y += 30
            bar_graph_left_x = text_start_x
            bar_graph_top_y = text_start_y + self.ui_padding + 30
            bar_graph_height = 240
            bar_width = 120

            half_bar_gap = 4
            bar_height = (bar_graph_height / 10)

            for i in range(10):
                colour = (100, 220, 255) if i == last_pred else (130, 130, 130)

                entry_top_y = int(bar_graph_top_y + i * bar_height)
                pred = self.last_model_pred[i]

                draw_text(
                    surface=wn, pos=(bar_graph_left_x, entry_top_y + int(bar_height / 2 - half_bar_gap)),
                    horiz_align='left', vert_align='centre', text=f"{i}: {pred:.4f}",
                    colour=colour, font_profile=(self.assets.monospaced_reg, 22)
                )

                pg.draw.rect(wn, (0, 0, 0), (bar_graph_left_x + 135, entry_top_y + half_bar_gap, bar_width, bar_height - 2 * half_bar_gap))
                pg.draw.rect(wn, colour, (bar_graph_left_x + 135, entry_top_y + half_bar_gap, int(bar_width * pred), bar_height - 2 * half_bar_gap))

        # Show notifications
        draw_text(
            wn, (2 * self.ui_padding + CANVAS_SIZE, WN_H - self.ui_padding - 120), 'left', 'bottom',
            self.notif.text, colour=self.notif.colour, font_profile=(self.assets.monospaced_reg, 22)
        )

        # Show instructions
        draw_text(
            wn, (2 * self.ui_padding + CANVAS_SIZE, WN_H - self.ui_padding - 90), 'left', 'bottom',
            "Backspace/Delete - Clear", colour=(255, 255, 255), font_profile=(self.assets.monospaced_reg, 22)
        )
        draw_text(
            wn, (2 * self.ui_padding + CANVAS_SIZE, WN_H - self.ui_padding - 60), 'left', 'bottom',
            "Left-Click - Draw", colour=(255, 255, 255), font_profile=(self.assets.monospaced_reg, 22)
        )
        draw_text(
            wn, (2 * self.ui_padding + CANVAS_SIZE, WN_H - self.ui_padding - 30), 'left', 'bottom',
            "Space - See Prediction", colour=(255, 255, 255), font_profile=(self.assets.monospaced_reg, 22)
        )
        draw_text(
            wn, (2 * self.ui_padding + CANVAS_SIZE, WN_H - self.ui_padding), 'left', 'bottom',
            "E - Change Model", colour=(255, 255, 255), font_profile=(self.assets.monospaced_reg, 22)
        )
