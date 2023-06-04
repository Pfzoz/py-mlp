import numpy as np

def mean_squared_error(estimated_y: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.mean(np.power(y - estimated_y, 2))


def mean_squared_error_prime(estimated_y: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (-2 * (y - estimated_y)) / estimated_y.size


def error(estimated_y: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(y - estimated_y))


def error_prime(estimated_y: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sign(y - estimated_y) / estimated_y.size

LOSSES_DICT = {
    "mse": [mean_squared_error, mean_squared_error_prime],
    "mean-squared-error": [mean_squared_error, mean_squared_error_prime],
}