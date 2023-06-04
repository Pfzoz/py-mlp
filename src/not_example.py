import numpy as np
from mlp.model import MLP
from mlp.structures import Layer
from mlp.calculations.afuncs import sigmoid, sigmoid_prime
from mlp.calculations.losses import mean_squared_error as mse, mean_squared_error_prime as msep



if __name__ == "__main__":
    mlp = MLP(mse, msep, 0)
    mlp.push_layer(Layer(2, sigmoid, sigmoid_prime))
    mlp.push_layer(Layer(3, sigmoid, sigmoid_prime))
    mlp.push_layer(Layer(2, sigmoid, sigmoid_prime))
    model = mlp.compile("L2")
    print(model)
    x = [np.array([0,0]),np.array([1,0]), np.array([0,1]), np.array([1,1])]
    y = [np.array([1,1]),np.array([0,1]), np.array([1,0]), np.array([0,0])]
    model.fit(x, y, epochs=200, learning_rate=1)
    model.feed_foward(np.array([[0], [0]]))
    print(model.activations[-1][:, 0])
    model.feed_foward(np.array([[0], [1]]))
    print(model.activations[-1][:, 0])
    model.feed_foward(np.array([[1], [0]]))
    print(model.activations[-1][:, 0])
    model.feed_foward(np.array([[1], [1]]))
    print(model.activations[-1][:, 0])