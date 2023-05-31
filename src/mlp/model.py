import numpy as np
from typing import Callable
from .structures import Layer
from .calculations.regularization import L2, L2_prime
from .calculations.afuncs import *
from .calculations.losses import *

def none(x):
    return x

def none_prime(x):
    return 1

AFUNCS_DICT = {
    "sigmoid": [sigmoid, sigmoid_prime],
    "None": [none, none_prime],
    "none": [none, none_prime]
}

LOSSES_DICT = {
    "mse": [mean_squared_error, mean_squared_error_prime],
    "mean-squared-error": [mean_squared_error, mean_squared_error_prime],
}


class Model:
    """
        Modelo funcional de uma MLP. Não há abstrações, todos os valores são mantidos em matrizes
        e calculados algebricamente.

        Recomenda-se obter uma instância de Model a partir da classe MLP.
    """
    def __init__(self, loss: Callable, loss_prime: Callable, learning_rate = 0.1, epochs = 100) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
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
        """
            Alimenta a rede com uma entrada correspondente ao formato da primeira camada (input layer).
            Importante: como as ativações são representadas em matrizes de uma só coluna. É necessário
            usar uma matriz de uma coluna no argumento 'x'.

            Args:
                x -> Matriz correspodente aos valores a serem alimentados à rede.
        """
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
        """
            Retro-propaga um valor com base em uma matriz de valores "verdadeiros" 'y'.
            Importante: como as ativações são representadas em matrizes de uma só coluna. É necessário
            usar uma matriz de uma coluna no argumento 'y'.

            Args:
                y -> Matriz de valores correspondentes a camada final da rede.
                learning_rate -> Taxa de aprendizado utilizado para alterar os pesos e vieses.
        """
        delta = self.loss_prime(
            self.activations[-1], y, self.weight_matrixes[-1]
        ) * self.activation_primes[-1](self.activations[-1])
        for i in range(len(self.weight_matrixes) - 1, -1, -1):
            self.bias_matrixes[i] -= learning_rate * delta
            # 1
            self.weight_matrixes[i] -= learning_rate * np.matmul(
                delta, self.activations[i].transpose()
            )
            z = self.zs[i]
            sp = self.activation_primes[i](z)
            delta = np.matmul(self.weight_matrixes[i].transpose(), delta) * sp

    def fit(
        self,
        x: list[np.ndarray],
        y: list[np.ndarray],
        epochs: int = None,
        learning_rate: float = None,
    ) -> None:
        """
            Otimiza a rede com base no gradiente descendente stocástico. Para isso utiliza listas de 
            matrizes de valores de entradas 'x' e valores verdadeiros 'y', por determinadas 'épocas'.

            Args:
                x -> lista de matrizes de entrada da rede.
                y -> lista de matrizes 'verdadeiras' de saída da rede.
                epochs -> 'épocas', quantidades de ciclos de retro-propagação da rede.
                learning_rate -> Taxa de aprendizado a ser utilizada na retro-propagação e alteração dos
                    parâmetros da rede.
        """
        self.epochs = epochs if epochs else self.epochs
        self.learning_rate = learning_rate if learning_rate else self.learning_rate
        for epoch in range(self.epochs):
            main_error = 0
            for x_data, y_data in zip(x, y):
                self.feed_foward(x_data)
                # print(self.zs)
                main_error += np.mean(np.power(self.activations[-1] - y_data, 2))
                self.back_propagate(y_data, self.learning_rate)
            print(
                f"EPOCH {epoch} - MSE:", main_error / len(y), f"({main_error}/{len(y)})"
            )

class MLP:
    """
        Classe "casca" para criação de uma MLP. É uma abstração para ser convertida na classe
        funcional final 'Model'

        Args:
            loss -> Função de custo, pode ser uma função ou string reconhecível em 'LOSSES_DICT'
            loss_prime -> Função derivada da função de custo, não necessária se uma string reconhecível foi
                passada em 'loss'.
            regularization -> Fator da regularização, por default é usado a L2 e o único jeito de "desativa-la"
                é utilizar um valor nulo (0).
    """
    def __init__(
        self,
        loss: Callable[[np.ndarray, np.ndarray], float] | str,
        loss_prime: Callable[[np.ndarray, np.ndarray], np.ndarray] | str = None,
        regularization: float = 0.0,
    ) -> None:
        self.layers = []
        if not (type(loss) is str):
            self.loss = loss
            self.loss_prime = loss_prime
        else:
            self.loss = LOSSES_DICT[loss][0]
            self.loss_prime = LOSSES_DICT[loss][1]
        self.regularization = regularization

    # def compile(self) -> Model:

    def compile(self) -> Model:
        """
            Compila a abstração (as 'Layers') para um modelo funcional (classe 'Model')
        """
        model = Model(
            L2(self.loss, self.regularization),
            L2_prime(self.loss_prime, self.regularization),
        )
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
            model.base_loss = self.loss
        return model

    def push_layer(self, layer: Layer) -> None:
        """
            Adiciona uma nova camada a lista de camadas.

            Args:
                layer -> Objeto do tipo 'Layer'.
        """
        self.layers.append(layer)