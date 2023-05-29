from .model import *
from json import dump, load

REVERSE_DICT = {
    sigmoid : "sigmoid",
    mean_squared_error : "mse",
}

def save_model(model : Model | MLP, path : str = "", save_compilation : bool = True) -> dict:
    result = {
        "architecture": {
            "layers": [],
            "loss": str,
            "regularization": {
                "factor": float,
                "type": str
            },
        },
        "details": {
            "model": 0,
            "weights": [],
            "biases": [],
            "lr": float,
            "epochs": int,
        }
    }
    for i, layer in enumerate(model.activations):
        if i == 0:
            layer_dict = {
                "n": layer.shape[0],
                "af": "None",
            }
        else:
            layer_dict = {
                "n": layer.shape[0],
                "af": REVERSE_DICT[model.activation_functions[i-1]],
            }
        result["architecture"]["layers"].append(layer_dict)
    print("RESULT:", result)
    result["architecture"]["loss"] = REVERSE_DICT[model.base_loss]
    result["architecture"]["regularization"]["type"] = "None"
    result["architecture"]["regularization"]["factor"] = 0.0
    if type(model) is Model and save_compilation:
        result["details"]["model"] = 1
        for weights in model.weight_matrixes:
            result["details"]["weights"].append(weights.astype(float).tolist())
        for biases in model.bias_matrixes:
            result["details"]["biases"].append(biases.astype(float).tolist())
        result["details"]["lr"] = model.learning_rate
        result["details"]["epochs"] = model.epochs
    with open(path, "w+") as json_file:
        dump(result, json_file, indent=4)

def load_model(path : str) -> Model | MLP:
    model_json = ""
    with open(path) as json_file:
        model_json = load(json_file)
    print(model_json)
    new_mlp = MLP(model_json["architecture"]["loss"],
                  regularization=float(model_json["architecture"]["regularization"]["factor"]))
    for layer in model_json["architecture"]["layers"]:
        new_mlp.push_layer(Layer(layer["n"], layer["af"]))
    if model_json["details"]["model"]:
        result = new_mlp.compile()
        load_weights = model_json["details"]["weights"]
        load_biases = model_json["details"]["biases"]
        for i in range(len(load_weights)):
            load_weights[i] = np.array(load_weights[i])
        for i in range(len(load_biases)):
            load_biases[i] = np.array(load_biases[i])
        result.weight_matrixes = load_weights
        result.bias_matrixes = load_biases
        result.epochs = model_json["details"]["epochs"]
        result.learning_rate = model_json["details"]["lr"]
        return result
    else:
        return new_mlp
    
