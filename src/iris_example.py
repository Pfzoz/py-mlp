import pandas as pd
import numpy as np
from mlp.model import MLP
from mlp.structures import Layer
from mlp.utils.treats import train_test_split, encode, normalize
from mlp.backup import save_model

if __name__ == "__main__":
    # Leitura de Dados
    iris_set = pd.read_csv("/home/pedrozoz/repositories/py-repo/py-mlp/assets/iris.data", sep=',')
    data = []
    for i, row in iris_set.iterrows():
        data.append(np.array(row))
    # Normalização, Encodificação e Separamento de Treino e Teste
    x = normalize(np.array([i[:-1] for i in data]))
    x = [np.array([[i[0]], [i[1]], [i[2]], [i[3]]]) for i in x]
    y = normalize(np.array(encode([i[-1] for i in data])))
    y = [np.array([[i]]) for i in y]
    x, y, test_x, test_y = train_test_split(x, y, 0.75)
    # Modelo
    mlp = MLP("mse")
    mlp.push_layer(Layer(4, "sigmoid"))
    mlp.push_layer(Layer(4, "sigmoid"))
    mlp.push_layer(Layer(1, "sigmoid"))
    model = mlp.compile("L2", epochs=100, learning_rate=0.1)
    # Treinamento
    model.fit(x, y)
    # Validação
    for x, y in zip(test_x, test_y):
        model.feed_foward(x)
        print("true:", y)
        print("pred:", model.activations[-1][0])
    
    # -- EXEMPLO DE ARQUIVOS DE MODELOS '.json' SALVOS E CARREGADOS --
    
    save_model(model, "iris_model.json")
    # loaded_model = load_model("iris_model.json")
    # for x, y in zip(test_x, test_y):
    #     loaded_model.feed_foward(x)
    #     print("true:", y)
    #     print("pred:", loaded_model.activations[-1][0])