import numpy as np
from math import e as E
from .structures import Layer
from typing import Callable
from .calculations import _regularizer_l1, _regularizer_l1_prime


class Model:

    def __init__(self,
                 loss: Callable,
                 loss_prime: Callable) -> None:
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
        for i in range(len(self.activations)-1):
            z = np.matmul(
                self.weight_matrixes[i], self.activations[i])+self.bias_matrixes[i]
            a = self.activation_functions[i][0][0](z)
            self.activations[i+1] = a
            self.zs.append(z)

    def back_propagate(self,
                       y: np.ndarray,
                       learning_rate: float) -> None:
        delta = self.loss_prime(self.activations[-1], y, np.array(
            self.weight_matrixes)) * self.activation_primes[-1][0][0](self.activations[-1])
        for i in range(len(self.weight_matrixes)-1, -1, -1):
            self.bias_matrixes[i] -= learning_rate*delta
            # 1
            self.weight_matrixes[i] -= np.matmul(delta,
                                                 self.activations[i].transpose())
            z = self.zs[i-1]
            sp = self.activation_primes[i][0][0](z)
            delta = np.matmul(self.weight_matrixes[i].transpose(), delta) * sp

    def fit(self,
            x: list[np.ndarray],
            y: list[np.ndarray],
            epochs: int = 100,
            learning_rate: float = 0.0001) -> None:
        for epoch in range(epochs):
            main_error = 0
            for x_data, y_data in zip(x, y):
                self.feed_foward(x_data)
                main_error += np.mean(
                    np.power(self.activations[-1] - y_data, 2))
                self.back_propagate(y_data, learning_rate)
            print(f"EPOCH {epoch} - MSE:", main_error /
                  len(y), f"({main_error}/{len(y)})")


class MLP:

    def __init__(self,
                 loss: Callable,
                 loss_prime: Callable,
                 regularization: float) -> None:
        self.layers = []
        self.loss = loss
        self.loss_prime = loss_prime
        self.regularization = regularization

    def compile(self) -> Model:
        regularized_loss = _regularizer_l1(self.loss, self.regularization)
        regularized_loss_prime = _regularizer_l1_prime(
            self.loss_prime, self.regularization)
        model = Model(regularized_loss, regularized_loss_prime)
        for i, layer in enumerate(self.layers):
            a, af, ap, b = layer.compile()
            model.activations.append(a)
            if i != 0:
                model.activation_functions.append(af)
                model.activation_primes.append(ap)
                model.bias_matrixes.append(b)
            if i != len(self.layers)-1:
                w = layer.connection(self.layers[i+1])
                model.weight_matrixes.append(w)
        return model

    def add_layer(self, layer: Layer) -> None:
        self.layers.append(layer)
