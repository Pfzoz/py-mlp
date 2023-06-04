import numpy as np
from math import e as E


def none(x):
    return x


def none_prime(x):
    return 1


def no_function(x: np.ndarray) -> np.ndarray:
    return x


def no_function_prime(x: np.ndarray) -> np.ndarray:
    return x


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_prime(x: np.ndarray) -> np.ndarray:
    return (x > 0) * 1


def soft_max(x: np.ndarray) -> np.ndarray:
    return np.log(1 + E**x)


def soft_max_prime(x: np.ndarray) -> np.ndarray:
    return (E**x) / (1 + E**x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30, 30)
    return 1 / (1 + (E ** (-x)))


def sigmoid_prime(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30, 30)
    return x * (1 - x)


AFUNCS_DICT = {
    "sigmoid": [sigmoid, sigmoid_prime],
    "None": [none, none_prime],
    "none": [none, none_prime],
}
