import numpy as np
from typing import Callable
from .structures import Layer
from .calculations.losses import LOSSES_DICT
from .calculations.regularization import REGS_DICT


def adjust_shape(x: np.ndarray | list, target_shape: tuple | list) -> np.ndarray | None:
    print("Here:", x)
    if not type(x) is list:
        if x.shape == target_shape:
            return x
        elif len(x.shape) == 1:
            x = np.array([x]).transpose()
            if x.shape != tuple(target_shape):
                return None
            else:
                return x
        elif len(x.shape) == 2:
            x = x.transpose()
            if x.shape != tuple(target_shape):
                return None
            else:
                return x
        else:
            return None
    else:
        for i in range(len(x)):
            x[i] = adjust_shape(x[i], target_shape=target_shape)


class Model:
    """
    Modelo funcional de uma MLP. Não há abstrações, todos os valores são mantidos em matrizes
    e calculados algebricamente.

    Recomenda-se obter uma instância de Model a partir da classe MLP.
    """

    def __init__(
        self, loss: Callable, loss_prime: Callable, learning_rate: float, epochs: int
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation_functions = []
        self.activation_primes = []
        self.weight_matrixes = []
        self.bias_matrixes = []
        self.activations = []
        self.loss = loss
        self.loss_prime = loss_prime

    def feed_foward(self, x: np.ndarray) -> None:
        """
        Alimenta a rede com uma entrada correspondente ao formato da primeira camada (input layer).
        Importante: como as ativações são representadas em matrizes de uma só coluna. É necessário
        usar uma matriz de uma coluna no argumento 'x'.

        Args:
            x -> Matriz correspodente aos valores a serem alimentados à rede.
        """
        self.activations[0] = x
        self.zs = [
            x,
        ]
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
        adjust_shape(x, self.activations[0].shape)
        adjust_shape(y, self.activations[-1].shape)
        self.epochs = epochs if epochs else self.epochs
        self.learning_rate = learning_rate if learning_rate else self.learning_rate
        for epoch in range(self.epochs):
            main_error = 0
            for x_data, y_data in zip(x, y):
                self.feed_foward(x_data)
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

    def compile(
        self,
        regularization: Callable[[np.ndarray], float] | str = None,
        regularization_prime: Callable[[np.ndarray], float] = None,
        learning_rate: float = 0.1,
        epochs: int = 100,
        regularization_factor: float = 0.0001,
    ) -> Model:
        """
        Compila a abstração (as 'Layers') para um modelo funcional (classe 'Model')
        """
        if type(regularization) is str:
            regularization_prime = REGS_DICT[regularization][1](regularization_factor)
            reg_func = REGS_DICT[regularization][0](regularization_factor)
            model = Model(
                lambda yt, y, weights: self.loss(yt, y) + reg_func(weights),
                lambda yt, y, weights: self.loss_prime(yt, y)
                + regularization_prime(weights),
                learning_rate=learning_rate,
                epochs=epochs,
            )
        elif regularization is None:
            model = Model(
                lambda yt, y, w: self.loss(yt, y) + 0,
                lambda yt, y, w: self.loss_prime(yt, y) + 0,
                learning_rate=learning_rate,
                epochs=epochs,
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
            model.compile_log = {
                "base_loss": self.loss.__name__,
                "regularization": regularization.__name__
                if regularization and not type(regularization) is str
                else str(regularization),
                "regularization_factor": regularization_factor,
            }
        return model

    def push_layer(self, layer: Layer) -> None:
        """
        Adiciona uma nova camada a lista de camadas.

        Args:
            layer -> Objeto do tipo 'Layer'.
        """
        self.layers.append(layer)
