from .model import *
from json import dump, load


def save_model(
    model: Model | MLP, path: str = "", save_compilation: bool = True
) -> dict:
    """Salva o modelo, compilado ou não.

    Args:
        model -> Modelo a ser salvo.
        path -> Caminho do .json.
        save_compilation -> flag opcional para desativar a salvação dos parâmetros.
    """
    result = {
        "architecture": {
            "layers": [],
            "loss": str,
            "regularization": {"factor": float, "type": str},
        },
        "details": {
            "model": 0,
            "weights": [],
            "biases": [],
            "lr": float,
            "epochs": int,
        },
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
                "af": model.activation_functions[i - 1].__name__,
            }
        result["architecture"]["layers"].append(layer_dict)
    result["architecture"]["loss"] = model.compile_log["base_loss"]
    result["architecture"]["regularization"]["type"] = model.compile_log[
        "regularization"
    ]
    result["architecture"]["regularization"]["factor"] = model.compile_log[
        "regularization_factor"
    ]
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


def load_model(path: str) -> Model | MLP:
    """Carrega o modelo, compilado ou não.

    Args:
        path -> Arquivo .json a ser carregado.
    """
    model_json = ""
    with open(path) as json_file:
        model_json = load(json_file)
    print(model_json)
    new_mlp = MLP(
        model_json["architecture"]["loss"],
        regularization=float(model_json["architecture"]["regularization"]["factor"]),
    )
    for layer in model_json["architecture"]["layers"]:
        new_mlp.push_layer(Layer(layer["n"], layer["af"]))
    if model_json["details"]["model"]:
        result = new_mlp.compile(
            model_json["regularization"]["type"],
            regularization_factor=model_json["regularization"]["factor"],
        )
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
