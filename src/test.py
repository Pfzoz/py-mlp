from mlp.model import MLP
from mlp.structures import Layer
from mlp.calculations.losses import mean_squared_error as mse
from mlp.calculations.losses import mean_squared_error_prime as msep
from mlp.calculations.afuncs import sigmoid, sigmoid_prime


myMlp = MLP(mse, msep, 0.0001)

myMlp.push_layer(Layer(3, sigmoid, sigmoid_prime))
myMlp.push_layer(Layer(3, sigmoid, sigmoid_prime))
myMlp.push_layer(Layer(3, sigmoid, sigmoid_prime))
myMlp = myMlp.compile()
print(myMlp.activation_functions)