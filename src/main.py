import numpy as np
from mlp.model import MLP
from mlp.structures import Layer
from mlp.calculations import sigmoid, sigmoid_prime, relu, relu_prime, error, error_prime
from mlp.calculations import mean_squared_error as mse, mean_squared_error_prime as msep

def truth(x):
    return x

def truth_prime(x):
    return 1

if __name__ == "__main__":
    mlp = MLP(mse, msep)
    mlp.add_layer(Layer(1, relu, relu_prime))
    mlp.add_layer(Layer(8, relu, relu_prime))
    mlp.add_layer(Layer(8, relu, relu_prime))
    mlp.add_layer(Layer(1, relu, relu_prime))
    model = mlp.compile()
    # model.weight_matrixes[0] = np.array([[-0.424, 0.358], [-0.740, -0.577], [-0.961, -0.469]])
    # model.weight_matrixes[1] = np.array([[-0.017, -0.893, 0.148]])
    
    print(model)
    x = [np.array([[0], [0]]), np.array([[1], [0]]), np.array([[0], [1]]), np.array([[1], [1]])]
    y = [np.array([[0]]), np.array([[1]]), np.array([[1]]), np.array([[0]])]
    # # x = [np.array([[i]]) for i in list(np.random.choice(list(range(-500, 500)), 400))]
    # # y = [i**2 for i in x]
    # # max_value = np.max(x + y)
    # # x = [i/max_value for i in x]
    # # y = [i/max_value for i in y]
    # model.fit(x, y, epochs=200, learning_rate=0.00001)
    # test = x[np.random.choice(len(x))]
    # model.feed_foward(test)
    # print(f"==Output for {test*max_value} ==")
    # # print(model.weight_matrixes)
    # print(model.activations[-1]*max_value)
    # print(f"==Output for {2} ==")
    # model.feed_foward(np.array([[2/max_value]]))
    # print(model.activations[-1]*max_value)