import numpy as np
from typing import Callable


def L2(
    loss: Callable[[np.ndarray, np.ndarray], float], reg_term: float
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], float]:
    def regularized_loss(yt, y, weights):
        cost = loss(yt, y) + (reg_term / 2) * np.sum(weights)
        return cost

    return regularized_loss


def L2_prime(
    loss_prime: Callable[[np.ndarray, list[np.ndarray]], np.ndarray], reg_term: float
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], float]:
    def regularized_loss(yt, y, weights):
        cost = loss_prime(yt, y) + reg_term * np.sum(weights)
        return cost

    return regularized_loss
