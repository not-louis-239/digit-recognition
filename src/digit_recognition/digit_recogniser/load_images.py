import csv
from pathlib import Path

from ..utils.custom_types import ImageArray
from ..utils.constants import IMAGE_SIZE

def load_images(csv_file: Path) -> list[tuple[ImageArray, int]]:
    """Loads all images from a CSV file into an iterable of arrays."""
    if not csv_file.exists():
        print(f"Warning: {csv_file} not found.")
        return []

    images: list[tuple[ImageArray, int]] = []

    with open(csv_file, "r") as f:
        # Using csv reader is more robust than manual string splitting
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            # label is the first item, pixels are the rest
            label = int(row[0])

            # Convert strings to floats
            pixels = [float(p) for p in row[1:]]

            # Reshape 1D list into 2D ImageArray (28x28)
            image_2d = []
            for i in range(0, len(pixels), IMAGE_SIZE):
                image_2d.append(pixels[i : i + IMAGE_SIZE])

            images.append((image_2d, label))

    return images
