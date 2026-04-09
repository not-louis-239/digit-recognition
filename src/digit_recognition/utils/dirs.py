from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

ROOT_DIR = Path(__file__).resolve().parents[3]

class _Dirs:
    # Top-level, stable project directories
    if TYPE_CHECKING:
        assets: _Dirs
        lib: _Dirs
        screensavers: _Dirs

    def __init__(self, root: Path) -> None:
        self._root = root

    def __getattr__(self, name: str) -> _Dirs:
        if name.startswith("_"):
            raise AttributeError(f"'{name}'")
        return _Dirs(self._root / name)

    def __truediv__(self, name: str) -> _Dirs:
        return _Dirs(self._root / name)

    def path(self) -> Path:
        return self._root

    # Intentionally excluded __fspath__ to ensure consistency of calling .path() to retrieve path
    # def __fspath__(self) -> str:
    #     return str(self._root)

DIRS = _Dirs(ROOT_DIR)

def _test():
    print(DIRS.path())
    print(DIRS.assets.path())
    print(DIRS.assets.images.path())
    print((DIRS.assets.images / "sprite.png").path())

    try:
        DIRS.assets._images
    except AttributeError as e:
        print(f"{type(e).__name__}: {e}")

if __name__ == "__main__":
    _test()
