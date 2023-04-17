import numpy as np
from math import e as E

class Neuron:
    def __init__(self) -> None:
        pass


class Layer:
    def __init__(self) -> None:
        pass


class MLP:
    def __init__(self,
                 layers: list[Layer] | list[int], **kwargs) -> None:
        """
        ==Desc==
            Constructor for MLP model.

        ==Params==
            layers : specifies the layers of the model. A list of layers can be given, or a list of int,
            assuming it's a MLP; the activation functions must then be specified in "activations" kwarg,
            and the derivatives in "dactivations" kwarg.
        """
        if type(layers) == list:
            self.weights_matrixes = []
            self.biases_matrixes = []
            self.activation_functions = []
            self.activations = []
            self.d_activation_functions = []
            self.layers = layers
            if len(self.layers) <= 1:
                print("Raised exception: Single-layer MLP is its own input?")
                raise Exception
            for i, neuron_amount in enumerate(layers[1:]):
                weights_matrix = np.ndarray((neuron_amount, layers[i]))
                for j in range(neuron_amount):
                    for k in range(layers[i]):
                        weights_matrix[j][k] = np.random.normal()
                self.weights_matrixes.append(weights_matrix)
                biases_matrix = np.zeros((neuron_amount, 1))
                for j in range(neuron_amount):
                    biases_matrix[j][0] = 0
                self.biases_matrixes.append(biases_matrix)
            if "activations" in kwargs.keys():
                if type(kwargs["activations"]) == list and type(kwargs["activations"][0]) == np.ndarray:
                    self.activation_functions = kwargs["activations"]
                    self.d_activation_functions = kwargs["dactivations"]
                elif type(kwargs["activations"]) == list and str(type(kwargs["activations"][0])) == "<class 'function'>":
                    if len(kwargs["activations"]) != len(self.weights_matrixes):
                        print("Raised exception: Missing Activation Function Specfication.")
                        raise Exception
                    for i, neuron_amount in enumerate(layers[1:]):
                        functions_matrix = np.full((neuron_amount, 1), kwargs["activations"][i])
                        self.activation_functions.append(functions_matrix)
                        dfunctions_matrix = np.full((neuron_amount, 1), kwargs["dactivations"][i])
                        self.d_activation_functions.append(dfunctions_matrix)
                elif str(type(kwargs["activations"])) == "<class 'function'>":
                    for i, neuron_amount in enumerate(layers[1:]):
                        functions_matrix = np.full((neuron_amount, 1), kwargs["activations"])
                        self.activation_functions.append(functions_matrix)
                        dfunctions_matrix = np.full((neuron_amount, 1), kwargs["dactivations"])
                        self.d_activation_functions.append(dfunctions_matrix)
                
            self.activations = []
            for i, neuron_amount in enumerate(layers[:]):
                activations_matrix = np.zeros((neuron_amount, 1))
                self.activations.append(activations_matrix)
        elif type(layers) == list:
            pass

    def __repr__(self) -> str:
        return_str = "== Weights ==\n"
        for weight_matrix in self.weights_matrixes:
            return_str += str(weight_matrix) + "\n"
        return_str += "== Biases ==\n"
        for bias_matrix in self.biases_matrixes:
            return_str += str(bias_matrix) + "\n"
        return_str += "== Activation Functions ==\n"
        for activationf_matrix in self.activation_functions:
            return_str += str(activationf_matrix) + "\n"
        return return_str

    def feed_foward(self, x: np.ndarray) -> np.ndarray:
        """
        ==Desc==
        "Feedfowards" the input(s) throughout the model.

        ==Params==
        """
        self.activations[0] = x
        for i in range(len(self.layers)-1):
            print(self.activations[i], self.weights_matrixes[i])
            z = np.matmul(self.weights_matrixes[i], self.activations[i])+self.biases_matrixes[i]
            a = self.activation_functions[i][0][0](z) # Lazy Fix
            self.activations[i+1] = a
        return self.activations[-1]

    def fit(self, x: np.ndarray,
            y: np.ndarray,
            epochs: int = 0,
            learning_rate: float = 0.01) -> None:
        pass

    def back_propagate(self) -> None:
        pass

def sigmoid(x):
    return 1/(1+(E**(-x)))

def dsigmoid(x):
    return x*(1-x)

if __name__ == "__main__":
    e = MLP([2, 3, 2], activations=sigmoid, dactivations=dsigmoid)
    print(e)
    print(e.feed_foward(np.array([[1], [2]])))