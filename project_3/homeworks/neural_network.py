import neuron_module as nm
import numpy as np


class Layer:
    def __init__(self, n_neurons=1, n_inputs=1, n_outputs=1):
        self.neurons = [nm.Neuron(n_inputs, n_outputs) for _ in range(n_neurons)]
        self.forwarded_inputs = None

    def forward(self, inputs):
        self.forwarded_inputs = np.array([neuron.forward(inputs) for neuron in self.neurons]).T
        return self.forwarded_inputs

    def calculate_output_error(self, targets):
        # Binary cross entropy loss
        return -(targets * np.log(self.forwarded_inputs) + (1 - targets) * np.log(1 - self.forwarded_inputs))

    def update_neurons(self, input_data, output_error, learning_rate):
        # for neuron, error in zip(self.neurons, output_error):
        #     neuron.update_weights(input_data, error, learning_rate)
        error = sum(output_error)
        for neuron in self.neurons:
            neuron.update_weights(input_data, error, learning_rate)


class NeuralNetwork:
    def __init__(self, n_inputs=1, hidden_layers=None, n_outputs=1, learning_rate=0.01):
        # Initialize input layer
        self.layers = [Layer(n_neurons=n_inputs, n_inputs=n_inputs, n_outputs=1)]

        if hidden_layers is None:
            hidden_layers = []
        hidden_layers = [n_inputs] + hidden_layers

        # Initialize hidden layers
        self.layers += [Layer(n_neurons=hidden_layers[i+1], n_inputs=hidden_layers[i], n_outputs=1)
                        for i in range(len(hidden_layers) - 1)]

        # Initialize output layer
        self.layers += [Layer(n_neurons=n_outputs, n_inputs=hidden_layers[-1], n_outputs=1)]

        self.learning_rate = learning_rate

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, input_data, targets):
        output_error = self.layers[-1].calculate_output_error(targets)
        #for i in reversed(range(len(self.layers) - 1)):
        for i in [len(self.layers) - 1]:
            if i == 0:
                input_data_backward = input_data
            else:
                input_data_backward = self.layers[i - 1].forwarded_inputs
            layer = self.layers[i]
            layer.update_neurons(input_data_backward, output_error, self.learning_rate)


# neural_network = NeuralNetwork(n_inputs=1, hidden_layers=[10, 10], n_outputs=1, learning_rate=0.01)
# input = [1]

neural_network = NeuralNetwork(n_inputs=2, hidden_layers=[2], n_outputs=1, learning_rate=0.01)
input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])
# input = np.array([0, 1])
# targets = np.array([0])

print(neural_network.layers)
print(neural_network.forward(input))

for i in range(1000):
    neural_network.forward(input)
    neural_network.backward(input, targets)

print(neural_network.layers)
print(neural_network.forward(input))

