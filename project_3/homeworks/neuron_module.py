import numpy as np


class Neuron:
    def __init__(self, n_inputs):
        self.weights = np.random.rand(n_inputs)
        self.forwarded_inputs = None

    def __linear_combination(self, inputs):
        return np.dot(inputs, self.weights)

    def __activation(self, z):
        return 1 / (1 + np.exp(-z))

    def __activation_derivative(self, z):
        return z * (1 - z)

    def cost_function(self, targets, outputs):
        return

    def forward(self, inputs):
        self.forwarded_inputs = self.__activation(self.__linear_combination(inputs))
        return self.forwarded_inputs

    def update_weights(self, inputs, error, learning_rate):
        delta = error * self.forwarded_inputs
        weight_deltas = learning_rate * np.dot(delta, inputs)
        self.weights -= weight_deltas
