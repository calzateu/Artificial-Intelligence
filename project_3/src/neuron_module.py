import cost_and_activation_functions as ca
import numpy as np


class Neuron:
    def __init__(self, n_inputs):
        self.weights = np.random.rand(n_inputs, 1)
        self.activation_derivative_inputs = None

    def __linear_combination(self, inputs):
        return np.dot(inputs, self.weights)

    def forward(self, inputs):
        linear_combination = self.__linear_combination(inputs)
        self.activation_derivative_inputs = ca.derivative_activation(linear_combination)
        forwarded_inputs = ca.activation(linear_combination)
        return forwarded_inputs

    def update_weights(self, inputs, local_gradient, learning_rate):
        weight_deltas = learning_rate * local_gradient * inputs
        weight_deltas = weight_deltas.reshape(len(weight_deltas), 1)
        self.weights -= weight_deltas
