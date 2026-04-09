"""
all_purpose_gallery.py

Helper tool to manage training data images

Features:
* loads all data from CSV when the script starts
* hotkey to "quick clean" images
    * sort images by correct digit (0-9)
    * delete empty images
* hotkey to write images to disk
    * for each row:
        * first number = correct digit (0..9)
        * all other numbers = pixels ([0.0..1.0])
* make new images directly in the gallery
"""

import pygame as pg
import csv
import sys
import math
import time
from pathlib import Path
from typing import TypeAlias

# --- Configuration ---
CSV_PATH = Path(__file__).resolve().parent.parent / "assets" / "training_data" / "digits.csv"

WIDTH, HEIGHT = 900, 750
GRID_SIZE = 28
PIXEL_SIZE = 560 // GRID_SIZE
BRUSH_SIZE = 3.5
BRUSH_STRENGTH = 0.3
FPS = 120

BG_COLOR = (30, 30, 30)
CANVAS_BG = (0, 0, 0)
TEXT_COLOR = (255, 255, 255)
ACCENT_COLOR = (255, 204, 0) # Stardew Gold

ImageArray: TypeAlias = list[list[float]]

def load_imgs_from_csv(path: Path) -> list[list]:
    data = []
    if not path.exists(): return data
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            label = int(row[0])
            pixels = [float(x) for x in row[1:]]
            grid = [pixels[i:i+GRID_SIZE] for i in range(0, len(pixels), GRID_SIZE)]
            data.append([label, grid])
    return data

def save_imgs_to_csv(data: list, path: Path) -> None:
    if not path.parent.exists(): path.parent.mkdir(parents=True)
    with open(path, "w") as f:
        for label, grid in data:
            flattened = [f"{pixel:.2f}" for row in grid for pixel in row]
            f.write(f"{label}," + ",".join(flattened) + "\n")

class GalleryApp:
    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption("Digit Civilization: Data IDE")
        self.font = pg.font.SysFont("Helvetica", 20)
        self.header_font = pg.font.SysFont("Helvetica", 24, bold=True)

        self.data = load_imgs_from_csv(CSV_PATH)
        self.idx = 0
        self.is_drawing_mode = False
        self.show_balance = True
        self.new_label = 0
        self.current_grid = [[0.0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

        self.last_scroll_time = 0

        # Notification System
        self.notif_text = ""
        self.notif_expiry = 0

    def set_notif(self, text: str, duration: float = 3.0):
        self.notif_text = text
        self.notif_expiry = time.time() + duration

    def get_counts(self) -> dict[int, int]:
        counts = {i: 0 for i in range(10)}
        for label, _ in self.data:
            counts[label] += 1
        return counts

    def quick_clean(self):
        NOISE_EPSILON = 2.0
        old_count = len(self.data)
        self.data = [item for item in self.data if sum(sum(row) for row in item[1]) > NOISE_EPSILON]
        self.data.sort(key=lambda x: x[0])
        removed = old_count - len(self.data)
        self.idx = min(self.idx, max(0, len(self.data)-1))
        self.set_notif(f"Cleaned: {removed} empty images removed. Sorted by digit.")

    def draw_balance_sheet(self):
        if not self.show_balance: return
        start_x = 600
        counts = self.get_counts()
        title = self.header_font.render("Number Balance", True, ACCENT_COLOR)
        self.screen.blit(title, (start_x, 20))

        total = len(self.data)
        for i in range(10):
            count = counts[i]
            color = (100, 255, 100)
            if total > 0:
                avg = total / 10

                # Far below average: Red (not enough)
                if count < avg * 0.75:
                    color = (255, 100, 100)
                # Far above average: Yellow (too much)
                elif count > avg * 1.25:
                    color = (255, 255, 100)

            txt = self.font.render(f"Digit {i}: {count}", True, color)
            self.screen.blit(txt, (start_x, 60 + i * 30))

        total_txt = self.font.render(f"Total: {total} | Avg: {avg:.2f}", True, ACCENT_COLOR)
        self.screen.blit(total_txt, (start_x, 370))

        self.screen.blit(self.font.render("Red = not enough", True, (255, 100, 100)), (start_x, 450))
        self.screen.blit(self.font.render("Yellow = too many", True, (255, 255, 100)), (start_x, 480))
        self.screen.blit(self.font.render("Green = enough", True, (100, 255, 100)), (start_x, 510))

    def draw_ui(self):
        self.screen.fill(BG_COLOR)
        pg.draw.rect(self.screen, CANVAS_BG, (20, 20, 560, 560))

        display_grid = self.current_grid if self.is_drawing_mode else (self.data[self.idx][1] if self.data else None)
        label = self.new_label if self.is_drawing_mode else (self.data[self.idx][0] if self.data else "N/A")

        if display_grid:
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    val = int(display_grid[r][c] * 255)
                    if val > 0:
                        pg.draw.rect(self.screen, (val, val, val), (20 + c*PIXEL_SIZE, 20 + r*PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))

        if time.time() < self.notif_expiry:
            notif_surf = self.font.render(self.notif_text, True, ACCENT_COLOR)
            self.screen.blit(notif_surf, (20, 585))

        mode_str = "[DRAWING]" if self.is_drawing_mode else f"[VIEWING: {self.idx+1}/{len(self.data)}]"
        texts = [
            f"Mode: {mode_str} | Active Digit: {label}",
            "SPACE: Toggle Draw | B: Toggle Balance | C: Clean | S: Save",
            "ARROWS: Nav | 0-9: Set Label | ENTER: Commit Drawing | D: Delete"
        ]
        for i, t in enumerate(texts):
            self.screen.blit(self.font.render(t, True, TEXT_COLOR), (20, 620 + i*25))

        self.draw_balance_sheet()

    def run(self):
        clock = pg.time.Clock()
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT: self.exit_app()

                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        self.exit_app()
                    if event.key == pg.K_b:
                        self.show_balance = not self.show_balance

                    if event.key == pg.K_SPACE:
                        self.is_drawing_mode = not self.is_drawing_mode
                        self.current_grid = [[0.0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
                        self.set_notif("Mode Switched")

                    # RESTORED: Number key detection for the active digit label
                    if pg.K_0 <= event.key <= pg.K_9:
                        self.new_label = event.key - pg.K_0
                        if not self.is_drawing_mode and self.data:
                            # In viewing mode, hot-swap the label of the current item
                            self.data[self.idx][0] = self.new_label
                            self.set_notif(f"Label changed to {self.new_label} for current image")
                        elif not self.is_drawing_mode:
                            self.set_notif(f"Label set to {self.new_label} (Enter Draw Mode to use)")

                    if self.is_drawing_mode and event.key == pg.K_RETURN:
                        self.data.append([self.new_label, [row[:] for row in self.current_grid]])
                        self.current_grid = [[0.0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
                        self.set_notif(f"Recorded new Digit {self.new_label}!")

                    if event.key in (pg.K_d, pg.K_BACKSPACE) and self.data:
                        if self.is_drawing_mode:
                            # If drawing, just wipe the current canvas
                            self.current_grid = [[0.0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
                            self.set_notif("Canvas Cleared")
                        else:
                            # If viewing, delete the entry from the dataset
                            self.data.pop(self.idx)
                            self.idx = max(0, self.idx - 1)
                            self.set_notif("Entry Deleted from Dataset")

                    if event.key == pg.K_c:
                        self.quick_clean()

                    if event.key == pg.K_s:
                        save_imgs_to_csv(self.data, CSV_PATH)
                        self.set_notif("Progress Saved to Disk!")

            if not self.is_drawing_mode and self.data:
                keys = pg.key.get_pressed()
                current_time = time.time()
                if current_time - self.last_scroll_time > 0.033:
                    if keys[pg.K_RIGHT]:
                        self.idx = (self.idx + 1) % len(self.data)
                        self.last_scroll_time = current_time
                    elif keys[pg.K_LEFT]:
                        self.idx = (self.idx - 1) % len(self.data)
                        self.last_scroll_time = current_time

            if self.is_drawing_mode and pg.mouse.get_pressed()[0]:
                self.handle_drawing()

            self.draw_ui()
            pg.display.flip()
            clock.tick(FPS)

    def handle_drawing(self):
        mx, my = pg.mouse.get_pos()
        m_col, m_row = (mx - 20) / PIXEL_SIZE, (my - 20) / PIXEL_SIZE
        radius = BRUSH_SIZE / 2
        for r in range(int(m_row - radius), int(m_row + radius + 1)):
            for c in range(int(m_col - radius), int(m_col + radius + 1)):
                if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                    dist = math.sqrt((m_row - (r+0.5))**2 + (m_col - (c+0.5))**2)
                    if dist / radius < 1.0:
                        self.current_grid[r][c] = min(1.0, self.current_grid[r][c] + BRUSH_STRENGTH * (1.0 - (dist/radius)**2))

    def exit_app(self):
        save_imgs_to_csv(self.data, CSV_PATH)
        pg.quit()
        sys.exit(0)

def main():
    GalleryApp().run()

if __name__ == "__main__":
    main()
