import numpy as np
import neural_network as nn

neural_network = nn.NeuralNetwork(n_inputs=2, hidden_layers=[2], n_outputs=1, learning_rate=1)

input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [1]])
# targets = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
# input = np.array([[0, 0]])
# targets = np.array([[0]])


print("Score: ", neural_network.score(input, targets))

outputs = np.zeros(len(input))
indices = np.array(range(len(input)))
for i in range(6000):
    np.random.shuffle(indices)
    for j in indices:
        outputs[j] = neural_network.forward(input[j])
        error = neural_network.backward(targets[j])

    if i % 100 == 0:
        print("Iteration: ", i, outputs)

print(outputs)

print("Score: ", neural_network.score(input, targets))
