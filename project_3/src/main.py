import graphics as gr
import data_processing as dp
import neural_network as nn
import pandas as pd

n_outputs = 2
neural_network = nn.NeuralNetwork(n_inputs=3, hidden_layers=[2], n_outputs=n_outputs, learning_rate=1)

# inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# targets = np.array([[0], [1], [1], [1]])
# targets = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
# inputs = np.array([[0, 0]])
# targets = np.array([[0]])

data = pd.read_csv("../data/data.csv")

print(data.head())

data_copy = data.copy()
data_copy = dp.preprocess_data(data_copy)

inputs = data_copy[['X0', 'X1', 'X2']].to_numpy()
targets = data_copy[['X3', 'X4']].to_numpy()

print("Score: ", neural_network.score(inputs, targets))

epochs = 10
errors = neural_network.train(inputs, targets, epochs=epochs)

print("Score: ", neural_network.score(inputs, targets))

print(data_copy.head())


outputs = neural_network.predict_dataset(inputs)

observations = [0, 1, 2, 900]
for observation in observations:
    print(f"Target at observation {observation}: {targets[observation]} and output: {outputs[observation]}")

gr.plot_errors(epochs, errors, number_of_data_points=len(inputs))
