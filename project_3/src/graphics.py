import matplotlib.pyplot as plt
import numpy as np


def plot_energy(epochs, errors, number_of_data_points):
    print("Plotting energy...")
    epochs_list = [i * number_of_data_points for i in range(epochs)]

    error_means = 1/2*np.mean(errors, axis=1)

    plt.plot(error_means)
    plt.vlines(x=epochs_list, ymin=min(error_means), ymax=max(error_means), colors='r', linestyles='dashed', label="Epochs")

    plt.title("Energy over epochs")
    plt.ylabel("Energy")
    plt.xlabel("Epochs")
    plt.show()


def plot_errors(epochs, errors, number_of_data_points):
    print("Plotting errors...")
    epochs_list = [i * number_of_data_points for i in range(epochs)]
    number_of_outputs = errors.shape[1]

    # Plot errors
    fig, ax = plt.subplots(number_of_outputs, 1)
    plt.suptitle(f"Error for each point over {epochs} epochs")
    for i in range(number_of_outputs):
        ax[i].set_title(f"Output {i + 1} error")
        ax[i].set_ylabel("Error")
        ax[i].set_xlabel("Data point index over epochs")

        error_output = errors[:, i]
        ax[i].plot(error_output)

        ax[i].vlines(x=epochs_list, ymin=min(error_output), ymax=max(error_output), colors='r', linestyles='dashed', label="Epochs")

    plt.legend()
    plt.show()


def plot_local_gradients(neural_network, epochs_trained, number_of_data_points):
    epochs_list = [i * number_of_data_points for i in range(epochs_trained)]
    number_of_rows = epochs_trained * number_of_data_points

    for i in range(neural_network.n_layers):
        print(f"Plotting local gradients for layer {i + 1}...")
        fig, ax = plt.subplots(neural_network.layers[i].n_neurons, 1)
        for j in range(neural_network.layers[i].n_neurons):
            ax[j].set_title(f"Local gradient for neuron {j + 1} in layer {i + 1}")
            ax[j].set_ylabel("Local gradient")
            ax[j].set_xlabel("Data point index")

            local_gradient = neural_network.layers[i].local_gradients_matrix[:number_of_rows, j]
            ax[j].plot(local_gradient)

            ax[j].vlines(x=epochs_list, ymin=min(local_gradient), ymax=max(local_gradient), colors='r',
                         linestyles='dashed', label="Epochs")

        plt.show()

    # Now plot the mean of local gradients over each layer in the same plot
    print("Plotting mean of local gradients over each layer...")
    fig, ax = plt.subplots(neural_network.n_layers, 1)
    min_lim = float('inf')
    max_lim = float('-inf')
    for i, layer in enumerate(neural_network.layers):
        ax[i].set_title(f"Mean of local gradients for layer {i + 1}")
        ax[i].set_ylabel("Local gradient")
        ax[i].set_xlabel("Data point index")

        means = np.mean(neural_network.layers[i].local_gradients_matrix[:number_of_rows], axis=1)
        ax[i].plot(means)

        min_lim = min(min_lim, min(means))
        max_lim = max(max_lim, max(means))

    for i in range(neural_network.n_layers):
        ax[i].vlines(x=epochs_list, ymin=min_lim, ymax=max_lim, colors='r', linestyles='dashed',
                     label="Epochs")

    plt.show()
