import numpy as np
from typing import Callable
from .structures import Layer
from .calculations.regularization import L2, L2_prime


class Model:
    def __init__(self, loss: Callable, loss_prime: Callable) -> None:
        self.activation_functions = []
        self.activation_primes = []
        self.weight_matrixes = []
        self.bias_matrixes = []
        self.activations = []
        self.loss = loss
        self.loss_prime = loss_prime

    def __repr__(self) -> str:
        return_str = "==Activations==\n"
        for a in self.activations:
            return_str += str(a) + "\n"
        return_str += "\n==Weights==\n"
        for w in self.weight_matrixes:
            return_str += str(w) + "\n"
        return_str += "\n==Biases==\n"
        for b in self.bias_matrixes:
            return_str += str(b) + "\n"
        return return_str

    def feed_foward(self, x: np.ndarray) -> None:
        self.activations[0] = x
        self.zs = []
        self.zs.append(x)
        for i in range(len(self.activations) - 1):
            z = (
                np.matmul(self.weight_matrixes[i], self.activations[i])
                + self.bias_matrixes[i]
            )
            a = self.activation_functions[i](z)
            self.activations[i + 1] = a
            self.zs.append(z)

    def back_propagate(self, y: np.ndarray, learning_rate: float) -> None:
        delta = self.loss_prime(
            self.activations[-1], y, self.weight_matrixes[-1]
        ) * self.activation_primes[-1](self.activations[-1])
        for i in range(len(self.weight_matrixes) - 1, -1, -1):
            self.bias_matrixes[i] -= learning_rate * delta
            # 1
            self.weight_matrixes[i] -= learning_rate * np.matmul(delta, self.activations[i].transpose())
            z = self.zs[i]
            sp = self.activation_primes[i](z)
            delta = np.matmul(self.weight_matrixes[i].transpose(), delta) * sp

    def fit(
        self,
        x: list[np.ndarray],
        y: list[np.ndarray],
        epochs: int = 100,
        learning_rate: float = 0.0001,
    ) -> None:
        for epoch in range(epochs):
            main_error = 0
            for x_data, y_data in zip(x, y):
                self.feed_foward(x_data)
                # print(self.zs)
                main_error += np.mean(np.power(self.activations[-1] - y_data, 2))
                self.back_propagate(y_data, learning_rate)
            print(
                f"EPOCH {epoch} - MSE:", main_error / len(y), f"({main_error}/{len(y)})"
            )


class MLP:
    def __init__(
        self,
        loss: Callable[[np.ndarray, np.ndarray], float],
        loss_prime: Callable[[np.ndarray, np.ndarray], np.ndarray],
        regularization: float,
    ) -> None:
        self.layers = []
        self.loss = loss
        self.loss_prime = loss_prime
        self.regularization = regularization

    # def compile(self) -> Model:

    def compile(self) -> Model:
        model = Model(L2(self.loss, self.regularization), L2_prime(self.loss_prime, self.regularization))
        for i, layer in enumerate(self.layers):
            activations, biases = layer.compile()
            model.activations.append(activations)
            if i != 0:
                model.activation_functions.append(layer.activation_function)
                model.activation_primes.append(layer.activation_prime)
                model.bias_matrixes.append(biases)
            if i != len(self.layers) - 1:
                w = layer.compile_weights(self.layers[i + 1])
                model.weight_matrixes.append(w)
        return model

    def push_layer(self, layer: Layer) -> None:
        self.layers.append(layer)
