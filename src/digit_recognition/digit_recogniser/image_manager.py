import numpy as np
from pathlib import Path

from digit_recognition.utils.dirs import DIRS
from digit_recognition.utils.custom_types import TrainingDataType
from ..utils.constants import IMAGE_SIZE

def load_imgs_from_npy(data_path: Path) -> TrainingDataType:
    if not data_path.exists():
        return []

    raw = np.load(data_path)

    expected_cols = 1 + IMAGE_SIZE * IMAGE_SIZE
    if raw.ndim != 2 or raw.shape[1] != expected_cols:
        raise ValueError(f"Invalid data shape: {raw.shape}")

    labels = raw[:, 0].astype(int)
    images = raw[:, 1:].reshape(-1, IMAGE_SIZE, IMAGE_SIZE)

    return list(zip(images, labels))

def save_imgs_to_npy(data_path: Path, data: TrainingDataType) -> None:
    if not data:
        return

    images = np.array([img for img, _ in data], dtype=np.float32)
    labels = np.array([label for _, label in data], dtype=np.float32)

    if images.ndim != 3 or images.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Invalid image shape: {images.shape}")

    flat_images = images.reshape(len(images), -1)
    final_array = np.concatenate((labels[:, None], flat_images), axis=1)

    data_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(data_path, final_array)

def _test():
    import matplotlib.pyplot as _plt
    import random as _r

    data = load_imgs_from_npy((DIRS.assets.training_data / "digits.npy").path())

    if not data:
        print("No data found")
        return

    _plt.ion()  # interactive mode ON

    fig, ax = _plt.subplots()
    img_artist = None

    try:
        while True:
            img, label = data[_r.randint(0, len(data) - 1)]

            ax.set_title(str(label))

            if img_artist is None:
                img_artist = ax.imshow(img, cmap="gray")
            else:
                img_artist.set_data(img)

            _plt.pause(0.2)

    except KeyboardInterrupt:
        _plt.close(fig)

if __name__ == "__main__":
    _test()
