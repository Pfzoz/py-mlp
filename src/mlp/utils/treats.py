import numpy as np

def train_test_split(x_data : list, y_data, x_ratio : float) -> tuple[list, list, list, list]:
    """
        Função para separar uma lista de matrizes de entradas e saídas em dados de treino e teste.
        Aleatório.
    """
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
    # print("MIN:", min_value, "MAX:", max_value)
    normalized_data = (data-min_value)/(max_value-min_value)
    return normalized_data