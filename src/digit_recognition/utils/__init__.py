import random as _r

def chance(p: float, /) -> bool:
    return _r.random() < p
