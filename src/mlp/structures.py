import numpy as np
from typing import Callable

class Layer:
    
    def __init__(self,
                 neurons: int,
                 activation_function: Callable[[np.ndarray], np.ndarray],
                 activation_prime: Callable[[np.ndarray], np.ndarray]) -> None:
        self.neurons = neurons
        self.activation_function = activation_function
        self.activation_prime = activation_prime

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
