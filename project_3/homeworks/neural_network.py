import neuron_module as nm
import numpy as np


def cost_mse(targets, outputs):
    return 1/2*(outputs - targets)**2

class Layer:
    def __init__(self, n_neurons=1, n_inputs=1):
        self.neurons = [nm.Neuron(n_inputs) for _ in range(n_neurons)]
        self.forwarded_inputs = None
        self.inputs = None
        self.local_gradients = None
        self.weights = np.zeros((n_inputs, n_neurons))

        for i, neuron in enumerate(self.neurons):
            self.weights[:, i:i + 1] = neuron.weights

    def forward(self, inputs):
        self.inputs = inputs
        self.forwarded_inputs = np.zeros(len(self.neurons))
        for i, neuron in enumerate(self.neurons):
            self.forwarded_inputs[i] = neuron.forward(inputs)
        return self.forwarded_inputs

    def calculate_output_error_cross_entropy(self, targets):
        # Binary cross entropy loss
        return -(targets * np.log(self.forwarded_inputs) + (1 - targets) * np.log(1 - self.forwarded_inputs))

    def calculate_output_error_mse(self, targets):
        # Mean squared error
        return 1/2*(self.forwarded_inputs - targets)**2

    def derivate_output_error_cross_entropy(self, targets):
        # If dimensions don't match then throw an error
        if targets.shape != self.forwarded_inputs.shape:
            raise ValueError(f"The dimensions of the targets and the output of the layer don't match {targets.shape} != {self.forwarded_inputs.shape}.")
        # Derivative of binary cross entropy loss
        return - targets / self.forwarded_inputs + (1 - targets) / (1 - self.forwarded_inputs)

    def derivate_output_error_mse(self, targets):
        # If dimensions don't match then throw an error
        if targets.shape != self.forwarded_inputs.shape:
            raise ValueError(f"The dimensions of the targets and the output of the layer don't match {targets.shape} != {self.forwarded_inputs.shape}.")
        # Derivative of mean squared error
        return self.forwarded_inputs - targets

    def derivative_activation(self, z):
        return z * (1 - z)

    def local_gradient_last_layer(self, targets):
        # Calculate local gradient
        cost_derivative = self.derivate_output_error_mse(targets)
        self.local_gradients = np.zeros(len(self.neurons))
        for i, neuron in enumerate(self.neurons):
            self.local_gradients[i] = cost_derivative[i] * self.derivative_activation(self.forwarded_inputs[i])

        return self.local_gradients

    def local_gradient_hidden_layer(self, w_l, local_gradient_l):
        # Calculate local gradient
        self.local_gradients = np.dot(w_l, local_gradient_l)
        for i, neuron in enumerate(self.neurons):
            self.local_gradients[i] *= self.derivative_activation(self.forwarded_inputs[i])

        # self.local_gradients = self.local_gradients.reshape(len(self.local_gradients), 1)
        return self.local_gradients

    def update_neurons(self, local_gradients_l, learning_rate):
        for i, neuron in enumerate(self.neurons):
            neuron.update_weights(self.inputs, local_gradients_l[i], learning_rate)

        for i, neuron in enumerate(self.neurons):
            self.weights[:, i:i + 1] = neuron.weights


class NeuralNetwork:
    def __init__(self, n_inputs=1, hidden_layers=None, n_outputs=1, learning_rate=0.01):
        # Initialize input layer
        self.layers = [Layer(n_neurons=n_inputs, n_inputs=n_inputs)]

        if hidden_layers is None:
            hidden_layers = []
        hidden_layers = [n_inputs] + hidden_layers

        # Initialize hidden layers
        self.layers += [Layer(n_neurons=hidden_layers[i+1], n_inputs=hidden_layers[i])
                        for i in range(len(hidden_layers) - 1)]

        # Initialize output layer
        self.layers += [Layer(n_neurons=n_outputs, n_inputs=hidden_layers[-1])]

        self.learning_rate = learning_rate

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            # We pass the output of the previous layer as the input to the current layer
            output = layer.forward(output)

        return output

    def backward(self, targets):
        local_gradients = self.layers[-1].local_gradient_last_layer(targets)
        for i in reversed(range(len(self.layers))):
            if i != len(self.layers) - 1:
                w = self.layers[i+1].weights
                local_gradients = self.layers[i].local_gradient_hidden_layer(w, local_gradients)

            layer = self.layers[i]
            layer.update_neurons(local_gradients, self.learning_rate)

        return self.layers[-1].calculate_output_error_mse(targets)

    def score(self, inputs, targets):
        error = 0
        for i in range(len(inputs)):
            self.forward(inputs[i])
            error += cost_mse(targets[i], self.layers[-1].forwarded_inputs)
        return error
