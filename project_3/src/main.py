import graphics as gr
import data_processing as dp
import neural_network as nn
import pandas as pd


def main():
    n_outputs = 2
    neural_network = nn.NeuralNetwork(n_inputs=3, hidden_layers=[2], n_outputs=n_outputs, learning_rate=1)

    # Data for testing the neural network with the XOR function
    # inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # targets = np.array([[0], [1], [1], [1]])
    # targets = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])

    data = pd.read_csv("../data/data.csv")

    print("Original data: ")
    print(data.head())

    data_copy = data.copy()
    data_copy = dp.preprocess_data(data_copy)

    print("Preprocessed data: ")
    print(data_copy.head())

    print()

    inputs = data_copy[['X0', 'X1', 'X2']].to_numpy()
    targets = data_copy[['X3', 'X4']].to_numpy()

    print("Score before training: ", neural_network.score(inputs, targets))

    epochs = 10
    save_local_gradients = True
    errors = neural_network.train(inputs, targets, epochs=epochs, save_local_gradients=save_local_gradients)

    print("Score after training: ", neural_network.score(inputs, targets))

    print()

    outputs = neural_network.predict_dataset(inputs)

    observations = [0, 1, 2, 900]
    for observation in observations:
        print(f"Target at observation {observation}: {targets[observation]} and output: {outputs[observation]}")

    print()
    print("Plotting results...")
    print()

    gr.plot_energy(epochs, errors, number_of_data_points=len(inputs))
    gr.plot_errors(epochs, errors, number_of_data_points=len(inputs))

    if save_local_gradients:
        gr.plot_local_gradients(neural_network, epochs, number_of_data_points=len(inputs))


if __name__ == "__main__":
    main()
