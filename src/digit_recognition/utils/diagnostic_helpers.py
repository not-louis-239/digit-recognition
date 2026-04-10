import sys


COL_RESET = "\033[0m"

COL_BOLD = "\033[1m"
COL_FAINT = "\033[2m"
COL_ITALIC = "\033[3m"
COL_UNDERLINE = "\033[4m"
COL_BLINK = "\033[5m"


def col(code: int) -> str:
    return f"\033[38;5;{code}m"

COL_ERROR = col(9)  # red
COL_WARN = col(11)  # yellow
COL_INFO = col(12)  # blue

def print_info(s: str) -> None:
    print(f"{COL_INFO}info: {COL_RESET}{s}")
def print_warn(s: str) -> None:
    print(f"{COL_WARN}warn: {COL_RESET}{s}")
def print_err(s: str) -> None:
    print(f"{COL_ERROR}error: {COL_RESET}{s}")
def print_fatal(s: str) -> None:
    print(f"{COL_ERROR}fatal: {COL_RESET}{s}")
    sys.exit(1)

def _show_palette():
    for i in range(16):
        for j in range(16):
            code = i * 16 + j
            print(f"{col(code)}{code:03d}", end=" ")
        print()

if __name__ == "__main__":
    _show_palette()
