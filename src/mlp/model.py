import numpy as np
from math import e as E
from .structures import Layer
from typing import Callable


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

    def  __repr__(self) -> str:
        return_str = "==Activations==\n"
        for a in self.activations: return_str += str(a) + "\n"
        return_str += "\n==Weights==\n"
        for w in self.weight_matrixes: return_str += str(w) + "\n"
        return_str += "\n==Biases==\n"
        for b in self.bias_matrixes: return_str += str(b) + "\n"
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
        delta = self.loss_prime(self.activations[-1], y, np.array(self.weight_matrixes), 0.000001) * self.activation_primes[-1][0][0](self.activations[-1])
        for i in range(len(self.weight_matrixes)-1, -1, -1):
            self.bias_matrixes[i] -= learning_rate*delta
            self.weight_matrixes[i] -= np.matmul(delta, self.activations[i].transpose()) # 1
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
                main_error += np.mean(np.power(self.activations[-1] - y_data, 2))
                # main_error += self.loss(self.activations[-1], y_data, np.array(self.weight_matrixes), 0.000001)
                self.back_propagate(y_data, learning_rate)
            print("MSE:",main_error/len(y), f"({main_error}/{len(y)})")

class MLP:

    def __init__(self,
                 loss: Callable,
                 loss_prime: Callable) -> None:
        self.layers = []
        self.loss = loss
        self.loss_prime = loss_prime

    def compile(self) -> Model:
        regularized_loss = lambda estimated_y, y, w, l : self.loss(estimated_y, y) + (l/2)*np.sum([np.sum(i**2) for i in w])
        regularized_loss_prime = lambda estimated_y, y, w, l : self.loss_prime(estimated_y, y) + l*np.sum([np.sum(i) for i in w])
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

# class MLP:
#     def __init__(self,
#                  layers: list[Layer] | list[int],
#                  loss = mse,
#                  dloss = dmse,

#                  **kwargs) -> None:
#         """
#         ==Desc==
#             Constructor for MLP model.

#         ==Params==
#             layers : specifies the layers of the model. A list of layers can be given, or a list of int,
#             assuming it's a MLP; the activation functions must then be specified in "activations" kwarg,
#             and the derivatives in "dactivations" kwarg.
#         """
#         self.weights_matrixes = []
#         self.biases_matrixes = []
#         self.layers = layers
#         self._initialize_weights()
#         self._initialize_biases()
#         self.activations = [np.zeros((n, 1), dtype=np.float128) for n in layers]
#         self.activation_functions = [func for func in kwargs["activation"]]
#         self.dactivation_functions = [func for func in kwargs["dactivation"]]
#         self.loss = loss
#         self.dloss = dloss

#     def _initialize_weights(self) -> None:
#         for i, n in enumerate(self.layers[1:]):
#             weights_matrix = np.ndarray((n, self.layers[i]), dtype=np.float128)
#             for j in range(n):
#                 for k in range(self.layers[i]):
#                     weights_matrix[j][k] = np.random.normal()
#             self.weights_matrixes.append(weights_matrix)

#     def _initialize_biases(self) -> None:
#         for n in self.layers[1:]:
#             biases_matrix = np.zeros((n, 1), dtype=np.float128)
#             self.biases_matrixes.append(biases_matrix)

#     def __repr__(self) -> str:
#         return_str = "== Weights ==\n"
#         for weight_matrix in self.weights_matrixes:
#             return_str += str(weight_matrix) + "\n"
#         return_str += "== Biases ==\n"
#         for bias_matrix in self.biases_matrixes:
#             return_str += str(bias_matrix) + "\n"
#         return_str += "== Activation Functions ==\n"
#         for activationf_matrix in self.activation_functions:
#             return_str += str(activationf_matrix) + "\n"
#         return return_str

#     def feed_foward(self, x: np.ndarray) -> np.ndarray:
#         """
#         ==Desc==
#         "Feedfowards" the input(s) throughout the model.

#         ==Params==
#         """
#         self.activations[0] = x
#         for i in range(len(self.layers)-1):
#             # print(self.activations[i], self.weights_matrixes[i])
#             z = np.matmul(self.weights_matrixes[i], self.activations[i])+self.biases_matrixes[i]
#             a = self.activation_functions[i](z)
#             self.activations[i+1] = a
#         return self.activations[-1]

#     def back_propagate(self, x, y) -> None:
#         pass

#     def fit(self, x: np.ndarray,
#             y: np.ndarray,
#             epochs: int = 100,
#             learning_rate: float = 0.01) -> None:

#         pass
