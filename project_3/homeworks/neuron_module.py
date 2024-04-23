import numpy as np


class Neuron:
    def __init__(self, n_inputs, n_outputs):
        self.weights = np.random.rand(n_inputs)

    def __linear_combination(self, inputs):
        return np.dot(inputs, self.weights)

    def __activation(self, z):
        return 1 / (1 + np.exp(-z))

    def __activation_derivative(self, z):
        return z * (1 - z)

    def cost_function(self, targets, outputs):
        return

    def forward(self, inputs):
        return self.__activation(self.__linear_combination(inputs))

    def update_weights(self, inputs, error, learning_rate):
        v = self.__linear_combination(inputs)
        delta = error * self.__activation_derivative(v)
        weight_deltas = learning_rate * np.dot(delta, inputs)
        self.weights -= weight_deltas
