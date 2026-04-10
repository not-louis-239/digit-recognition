import random as _r

def chance(p: float, /) -> bool:
    """Usage:

    >>> if chance(0.3):
    ...     print("30% chance")
    >>> else:
    ...    print("70% chance")
    """
    return _r.random() < p

def clamp(x: float, r: tuple[float, float], /) -> float:
    min_, max_ = r
    return max(min_, min(max_, x))

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t
