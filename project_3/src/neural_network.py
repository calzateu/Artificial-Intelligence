import cost_and_activation_functions as ca
import neuron_module as nm
import numpy as np


class Layer:
    def __init__(self, n_neurons=1, n_inputs=1):
        self.n_neurons = n_neurons
        self.neurons = [nm.Neuron(n_inputs) for _ in range(n_neurons)]
        self.forwarded_inputs = None
        self.inputs = None
        self.local_gradients = None
        self.weights = np.zeros((n_inputs, n_neurons))

        for i, neuron in enumerate(self.neurons):
            self.weights[:, i:i + 1] = neuron.weights

        # If we want to store the local gradients over the course of training
        self.local_gradients_matrix = None

    def forward(self, inputs):
        self.inputs = inputs
        self.forwarded_inputs = np.zeros(self.n_neurons)
        for i, neuron in enumerate(self.neurons):
            self.forwarded_inputs[i] = neuron.forward(inputs)
        return self.forwarded_inputs

    def local_gradient_last_layer(self, targets):
        # Calculate local gradient
        cost_derivative = ca.derivate_output_error_mse(targets, self.forwarded_inputs)
        self.local_gradients = np.zeros(self.n_neurons)
        for i, neuron in enumerate(self.neurons):
            self.local_gradients[i] = cost_derivative[i] * ca.derivative_activation(self.forwarded_inputs[i])

        return self.local_gradients

    def local_gradient_hidden_layer(self, w_l, local_gradient_l):
        # Calculate local gradient
        self.local_gradients = np.dot(w_l, local_gradient_l)
        for i, neuron in enumerate(self.neurons):
            self.local_gradients[i] *= ca.derivative_activation(self.forwarded_inputs[i])

        # self.local_gradients = self.local_gradients.reshape(len(self.local_gradients), 1)
        return self.local_gradients

    def update_neurons(self, local_gradients_l, learning_rate):
        for i, neuron in enumerate(self.neurons):
            neuron.update_weights(self.inputs, local_gradients_l[i], learning_rate)

        for i, neuron in enumerate(self.neurons):
            self.weights[:, i:i + 1] = neuron.weights


class NeuralNetwork:
    def __init__(self, n_inputs=1, hidden_layers=None, n_outputs=1, learning_rate=0.01):
        self.n_outputs = n_outputs
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
        self.n_layers = len(self.layers)

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            # We pass the output of the previous layer as the input to the current layer
            output = layer.forward(output)

        return output

    def predict_dataset(self, inputs):
        predictions = np.zeros((len(inputs), self.n_outputs))
        for i in range(len(inputs)):
            self.forward(inputs[i])
            predictions[i] = self.layers[-1].forwarded_inputs
        return predictions

    def backward(self, targets, save_local_gradients=False, current_index=0):
        local_gradients = self.layers[-1].local_gradient_last_layer(targets)
        for i in reversed(range(self.n_layers)):
            if i != self.n_layers - 1:
                w = self.layers[i+1].weights
                local_gradients = self.layers[i].local_gradient_hidden_layer(w, local_gradients)

            layer = self.layers[i]
            layer.update_neurons(local_gradients, self.learning_rate)

            if save_local_gradients:
                self.layers[i].local_gradients_matrix[current_index] = local_gradients

        final_error = ca.calculate_output_error_mse(targets, self.layers[-1].forwarded_inputs)

        return final_error

    def score(self, inputs, targets):
        error = 0
        for i in range(len(inputs)):
            self.forward(inputs[i])
            error += ca.calculate_output_error_mse(targets[i], self.layers[-1].forwarded_inputs)
        return error

    def train(self, inputs, targets, epochs=1, save_local_gradients=False):
        indices = np.arange(len(inputs))

        # Used to store the errors for each row of each epoch
        number_of_rows = epochs * len(indices)
        errors = np.zeros((number_of_rows, self.n_outputs))

        # Initialize local gradients matrix for each layer if return_local_gradients is True
        if save_local_gradients:
            for layer in self.layers:
                layer.local_gradients_matrix = np.zeros((number_of_rows, layer.n_neurons))

        cont = 0
        for i in range(epochs):
            np.random.shuffle(indices)
            for j in indices:
                self.forward(inputs[j])

                # Backpropagation
                errors[cont] = self.backward(targets[j], save_local_gradients, cont)
                cont += 1

        return errors
