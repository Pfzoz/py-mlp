import numpy as np
from typing import Callable
from .calculations.afuncs import *

AFUNCS_DICT = {
    "sigmoid": [sigmoid, sigmoid_prime],
}

class Layer:
    
    def __init__(self,
                 neurons: int,
                 activation_function: Callable[[np.ndarray], np.ndarray] | str,
                 activation_prime: Callable[[np.ndarray], np.ndarray] = None) -> None:
        self.neurons = neurons
        if not (type(activation_function) is str):
            self.activation_function = activation_function
            self.activation_prime = activation_prime
        else:
            self.activation_function = AFUNCS_DICT[activation_function][0]
            self.activation_prime = AFUNCS_DICT[activation_function][1]

    def compile(self) -> list:
        shape = (self.neurons, 1)
        activations = np.ndarray(shape, dtype=np.float128)
        biases = np.full(shape, 0, dtype=np.float128)
        return activations, biases
    
    def compile_weights(self,
                        __o: "Layer") -> np.ndarray:
        shape = (__o.neurons, self.neurons)
        weights = np.ndarray(shape, dtype=np.float128)
        for i in range(shape[0]):
            for j in range(shape[1]):
                weights[i][j] = np.random.normal()
        return weights
