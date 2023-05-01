import numpy as np
from math import e as E
from typing import Callable

# Activation Functions


def relu(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30, 30)
    return np.maximum(0, x)


def relu_prime(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return (x > 0) * 1


def soft_max(x: np.ndarray) -> np.ndarray:
    return np.log(1+E**x)


def soft_max_prime(x: np.ndarray) -> np.ndarray:
    return (E**x)/(1+E**x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30, 30)
    return 1/(1+(E**(-x)))


def sigmoid_prime(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30, 30)
    return x*(1-x)

# Losses


def mean_squared_error(estimated_y: np.ndarray,
                       y: np.ndarray) -> np.ndarray:
    return np.mean(np.power(y-estimated_y, 2))


def mean_squared_error_prime(estimated_y: np.ndarray,
                             y: np.ndarray) -> np.ndarray:
    return (-2*(y-estimated_y))/estimated_y.size


def error(estimated_y: np.ndarray,
          y: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(y-estimated_y))


def error_prime(estimated_y: np.ndarray,
                y: np.ndarray) -> np.ndarray:
    return np.sign(y-estimated_y)/estimated_y.size

# Regularizations


def _regularizer_l1(self, loss: Callable[[any, any]],
                    l1_term: float) -> Callable[[any, any, list[np.ndarray]]]:
    def regularized_loss(estimated_y, y, weights_list: list[np.ndarray]):
        cost = loss(estimated_y, y) + (l1_term/2) * \
            np.sum([np.sum(i**2) for i in weights_list])
        return cost
    return regularized_loss


def _regularizer_l1_prime(self, loss_prime: Callable[[any, any]],
                          l1_term: float) -> Callable[[any, any, list[np.ndarray]]]:
    def regularized_loss(estimated_y, y, weights_list: list[np.ndarray]):
        cost = loss_prime(estimated_y, y) + l1_term * \
            np.sum([np.sum(i) for i in weights_list])
        return cost
    return regularized_loss
