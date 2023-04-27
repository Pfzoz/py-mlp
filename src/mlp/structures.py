import numpy as np
from .calculations import relu, relu_prime
from typing import Callable


class Neuron:

    def __init__(self,
                 activation_function: Callable,
                 activation_prime: Callable) -> None:
        self.activation_function = activation_function
        self.activation_prime = activation_prime


class Layer:

    def __init__(self,
                 neurons: int,
                 activation_function: Callable,
                 activation_prime: Callable) -> None:
        def n(n): return Neuron(activation_function, activation_prime)
        self.neurons = list(map(n, range(neurons)))

    def compile(self) -> np.ndarray:
        """
        Returns 3 matrixes corresponding to activations,
        activation functions and activation functions primes.
        """
        shape = (len(self.neurons), 1)
        a = np.ndarray(shape, dtype=np.float128)
        af = np.ndarray(shape, dtype=object)
        ap = np.ndarray(shape, dtype=object)
        b = np.full(shape, 0, dtype=np.float128)
        for i in range(shape[0]):
            af[i, 0] = self.neurons[i].activation_function
            ap[i, 0] = self.neurons[i].activation_prime
        return a, af, ap, b

    def add_neurons(self, *args: Neuron) -> None:
        for neuron in args:
            self.neurons.append(neuron)

    def connection(self, __o: "Layer") -> np.ndarray:
        """
        Parameters: Another Layer object "foward" to this one.

        Returns a matrix corresponding to the weights between
        the layer. Values are taken from Gaussian normal distribution.
        """
        shape = (len(__o.neurons), len(self.neurons))
        w = np.ndarray(shape, dtype=np.float128)
        for i in range(shape[0]):
            for j in range(shape[1]):
                w[i, j] = np.random.normal()
        return w
