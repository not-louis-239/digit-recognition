from typing import TYPE_CHECKING

import pygame as pg
import numpy as np
from pygame import Surface

from digit_recognition.gui.utils.canvas import Canvas
from digit_recognition.gui.utils.ambient_messages import AmbientMessage
from digit_recognition.gui.utils.input_manager import MouseButton, InputManager
from digit_recognition.gui.utils.buttons import Button
from digit_recognition.gui.states import State, StateChangeRequest, StateID
from digit_recognition.utils import lerp, invlerp
from digit_recognition.digit_recogniser.simulation import Simulation
from digit_recognition.utils.constants import BRUSH_SIZE, BRUSH_STRENGTH, IMAGE_SIZE, WN_H

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

        self.view_idx = 0
        self.is_drawing_mode: bool = False
        self.show_number_balance: bool = False
        self.active_digit: int = 0
        self.sim = sim

        self.reset()

    def reset(self) -> None:
        self.canvas.clear()

        # Update view of best model (assuming Simulation.run_generation() was called)
        best = self.sim.get_best_models(1)[0]
        best = self.assets.model_wrappers[0].model
        self.best_model = best

    def update(self, dt_s: float) -> None:
        pass

    def take_input(self, input_manager: InputManager) -> StateChangeRequest:
        if self.return_button.check_click(input_manager.events):
            return StateChangeRequest(new=StateID.TITLE)

        if input_manager.went_down(pg.K_b):
            self.show_number_balance = not self.show_number_balance
        if input_manager.went_down(pg.K_BACKSPACE):
            if self.is_drawing_mode:
                self.canvas.clear()
        if input_manager.went_down(pg.K_SPACE):
            self.model_prediction = self.best_model.predict(self.canvas.as_array())
            print(f"Model prediction: {np.argmax(self.model_prediction)}")

        if input_manager.mouse_is_down(MouseButton.LMB):
            mouse_pos = pg.mouse.get_pos()
            canvas_topleft = (self.ui_padding, self.ui_padding)
            canvas_botright = (self.ui_padding + IMAGE_SIZE * 20, self.ui_padding + IMAGE_SIZE * 20)

            if (canvas_topleft[0] <= mouse_pos[0] <= canvas_botright[0] and
                canvas_topleft[1] <= mouse_pos[1] <= canvas_botright[1]):
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
        wn.fill((30, 30, 30))
        self.canvas.draw(wn, tile_size_px=20, start_pos=(self.ui_padding, self.ui_padding))
        self.return_button.draw(wn)
