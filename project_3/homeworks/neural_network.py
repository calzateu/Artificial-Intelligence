import neuron_module as nm
import numpy as np


class Layer:
    def __init__(self, n_neurons=1, n_inputs=1):
        self.neurons = [nm.Neuron(n_inputs) for _ in range(n_neurons)]
        self.forwarded_inputs = None
        self.inputs = None

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

    def update_neurons(self, output_error, learning_rate):
        # for neuron, error in zip(self.neurons, output_error):
        #     neuron.update_weights(input_data, error, learning_rate)
        for i, neuron in enumerate(self.neurons):
            neuron.update_weights(self.inputs, output_error[i], learning_rate)


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
        # derivative_output_error = self.layers[-1].derivate_output_error_cross_entropy(targets)
        derivative_output_error = self.layers[-1].derivate_output_error_mse(targets)

        # for i in reversed(range(len(self.layers) - 1)):
        for i in [len(self.layers) - 1]:
            layer = self.layers[i]
            layer.update_neurons(derivative_output_error, self.learning_rate)

        return self.layers[-1].calculate_output_error_mse(targets)
