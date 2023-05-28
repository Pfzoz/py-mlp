import pandas as pd
import numpy as np
from mlp.model import MLP
from mlp.structures import Layer
from mlp.utils.treats import train_test_split, encode, normalize

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
    #
    mlp = MLP("mse")
    mlp.push_layer(Layer(4, "sigmoid"))
    mlp.push_layer(Layer(4, "sigmoid"))
    mlp.push_layer(Layer(1, "sigmoid"))
    model = mlp.compile()
    model.fit(x, y, epochs=100, learning_rate=0.1)
    for x, y in zip(test_x, test_y):
        model.feed_foward(x)
        print("true:", y)
        print("pred:", model.activations[-1][0])