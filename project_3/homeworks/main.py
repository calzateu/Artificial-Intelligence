import numpy as np
import neural_network as nn

# neural_network = NeuralNetwork(n_inputs=1, hidden_layers=[10, 10], n_outputs=1, learning_rate=0.01)
# input = [1]

neural_network = nn.NeuralNetwork(n_inputs=2, hidden_layers=[2], n_outputs=1, learning_rate=0.01)
input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [1]])
# targets = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
# input = np.array([0, 1])
# targets = np.array([0])

print(neural_network.layers)
print(neural_network.forward(input))

for i in range(1000):
    neural_network.forward(input)
    neural_network.backward(input, targets)

print(neural_network.layers)
print(neural_network.forward(input))
