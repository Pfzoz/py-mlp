import numpy as np
from typing import Callable


def L2(reg_term: float) -> Callable[[np.ndarray], float]:
    return lambda w: (reg_term / 2) * np.sum(w)


def L2_prime(reg_term: float) -> Callable[[np.ndarray], float]:
    return lambda w: reg_term * np.sum(w)


REGS_DICT = {"L2": [L2, L2_prime]}
