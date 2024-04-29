import graphics as gr
import data_processing as dp
import neural_network as nn
import pandas as pd


def main():

    # Set experiment parameters
    n_inputs = 3
    n_outputs = 2
    hidden_layers = [2]
    epochs = 50
    learning_rate = 1
    # Tolerance for the mean of the local gradients. If the mean is below this value, the training is stopped
    tolerance = 0.001

    neural_network = nn.NeuralNetwork(n_inputs=n_inputs, hidden_layers=hidden_layers, n_outputs=n_outputs,
                                      learning_rate=learning_rate)

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

    save_local_gradients = True
    errors, epochs_trained = neural_network.train(inputs, targets, epochs=epochs,
                                                  save_local_gradients=save_local_gradients, tolerance=tolerance)

    print("Score after training: ", neural_network.score(inputs, targets))

    print()

    outputs = neural_network.predict_dataset(inputs)

    observations = [0, 1, 2, 900]
    for observation in observations:
        print(f"Target at observation {observation}: {targets[observation]} and output: {outputs[observation]}")

    print()
    print("Plotting results...")
    print()

    gr.plot_energy(epochs_trained, errors, number_of_data_points=len(inputs))
    gr.plot_errors(epochs_trained, errors, number_of_data_points=len(inputs))

    if save_local_gradients:
        gr.plot_local_gradients(neural_network, epochs_trained, number_of_data_points=len(inputs))


if __name__ == "__main__":
    main()
