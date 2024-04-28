import numpy as np


class Neuron:
    def __init__(self, n_inputs):
        self.weights = np.random.rand(n_inputs, 1)
        self.activation_derivative_inputs = None

    def __linear_combination(self, inputs):
        return np.dot(inputs, self.weights)

    def __activation(self, z):
        return 1 / (1 + np.exp(-z))

    def __activation_derivative(self, z):
        return z * (1 - z)

    def cost_function(self, targets, outputs):
        return

    def forward(self, inputs):
        linear_combination = self.__linear_combination(inputs)
        self.activation_derivative_inputs = self.__activation_derivative(linear_combination)
        forwarded_inputs = self.__activation(linear_combination)
        return forwarded_inputs

    def update_weights(self, inputs, error, learning_rate):
        delta = error * self.activation_derivative_inputs
        weight_deltas = learning_rate * inputs * delta
        weight_deltas = weight_deltas.reshape(len(weight_deltas), 1)
        self.weights -= weight_deltas
