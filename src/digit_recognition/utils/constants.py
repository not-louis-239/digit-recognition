from __future__ import annotations

# --- Generic Config ---
__version__: tuple[int, int, int] = (0, 1, 0)

# --- Canvas Config ---
BRUSH_SIZE = 3.2
BRUSH_STRENGTH = 20  # intensity increase per second of drawing on the same pixel
NOISE_EPSILON = 5  # number of fully bright pixels equivalent below which the canvas is considered empty

# --- GUI Config ---
FPS = 60
WN_W = 1250
WN_H = 850
