import pandas as pd
import numpy as np
from mlp.model import MLP
from mlp.structures import Layer
from mlp.calculations import sigmoid, sigmoid_prime, relu, relu_prime, error, error_prime
from mlp.calculations import mean_squared_error as mse, mean_squared_error_prime as msep

def train_test_split(x_data : list, y_data, x_ratio : float) -> tuple[list, list, list, list]:
    indexes = list(range(len(x_data)))
    
    x_indexes = list(np.random.choice(indexes, len(x_data)))
    train_x = []
    train_y = []
    remove_list = []
    for i in range(int(len(x_data)*x_ratio)):
        train_x.append(x_data[x_indexes[i]])
        train_y.append(y_data[x_indexes[i]])
        remove_list.append(x_indexes[i])
    [x_indexes.remove(i) for i in remove_list]
    test_x = []
    test_y = []
    for i in x_indexes:
        test_x.append(x_data[i])
        test_y.append(y_data[i])
    return train_x, train_y, test_x, test_y

def encode(data : list) -> list:
    encoder = {}
    for x in data:
        if type(x) == str and not x in encoder.keys():
            encoder[x] = len(encoder.keys())
    return_data = [encoder[x] for x in data]
    return return_data

def normalize(data : np.ndarray) -> np.ndarray:
    min_value, max_value = (np.amin(data), np.amax(data))
    normalized_data = (data-min_value)/(max_value-min_value)
    return normalized_data

if __name__ == "__main__":
    iris_set = pd.read_csv("/home/pedrozoz/repositories/py-repo/py-mlp/assets/iris.data", sep=',')
    data = []
    for i, row in iris_set.iterrows():
        data.append(np.array(row))
    x = normalize(np.array([i[:-1] for i in data]))
    x = [np.array([[i[0]], [i[1]], [i[2]], [i[3]]]) for i in x]
    y = normalize(np.array(encode([i[-1] for i in data])))
    y = [np.array([[i]]) for i in y]
    x, y, test_x, test_y = train_test_split(x, y, 0.75)
    print(test_x)
    #
    mlp = MLP(mse, msep, 0.000001)
    mlp.add_layer(Layer(4, sigmoid, sigmoid_prime))
    mlp.add_layer(Layer(4, sigmoid, sigmoid_prime))
    mlp.add_layer(Layer(1, sigmoid, sigmoid_prime))
    model = mlp.compile()
    model.fit(x, y, epochs=20, learning_rate=0.00001)
    for x, y in zip(test_x, test_y):
        model.feed_foward(x)
        print("true:", y)
        print("pred:", model.activations[-1][0])

#Setosa=0 Versicolor=0.5 Virginica=1