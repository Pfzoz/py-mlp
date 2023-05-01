import numpy as np
from mlp.model import MLP
from mlp.structures import Layer
from mlp.calculations import sigmoid, sigmoid_prime, relu, relu_prime, error, error_prime
from mlp.calculations import mean_squared_error as mse, mean_squared_error_prime as msep



if __name__ == "__main__":
    mlp = MLP(mse, msep, 0.0001)
    mlp.add_layer(Layer(2, sigmoid, sigmoid_prime))
    mlp.add_layer(Layer(3, sigmoid, sigmoid_prime))
    mlp.add_layer(Layer(2, sigmoid, sigmoid_prime))
    model = mlp.compile()
    print(model)
    x = [np.array([[0], [0]]), np.array([[1], [0]]), np.array([[0], [1]]), np.array([[1], [1]])]
    y = [np.array([[1], [1]]), np.array([[0], [1]]), np.array([[1], [0]]), np.array([[0], [0]])]
    model.fit(x, y, epochs=200, learning_rate=0.001)
    model.feed_foward(np.array([[0], [0]]))
    print(model.activations[-1][:, 0])
    model.feed_foward(np.array([[0], [1]]))
    print(model.activations[-1][:, 0])
    model.feed_foward(np.array([[1], [0]]))
    print(model.activations[-1][:, 0])
    model.feed_foward(np.array([[1], [1]]))
    print(model.activations[-1][:, 0])