import matplotlib.pyplot as plt


def plot_errors(epochs, errors, number_of_data_points):
    epochs_list = [i * number_of_data_points for i in range(epochs)]
    number_of_outputs = errors.shape[1]

    # Plot errors
    fig, ax = plt.subplots(number_of_outputs, 1)
    plt.suptitle(f"Error for each point over {epochs} epochs")
    for i in range(number_of_outputs):
        ax[i].set_title(f"Output {i + 1} error")
        ax[i].set_ylabel("Error")
        ax[i].set_xlabel("Data point index over epochs")
        ax[i].plot(errors[:, i])

        ax[i].vlines(x=epochs_list, ymin=0, ymax=1, colors='r', linestyles='dashed', label="Epochs")

    plt.legend()
    plt.show()


def plot_local_gradients(neural_network, epochs, number_of_data_points):
    epochs_list = [i * number_of_data_points for i in range(epochs)]

    for i in range(neural_network.n_layers):
        fig, ax = plt.subplots(neural_network.layers[i].n_neurons, 1)
        for j in range(neural_network.layers[i].n_neurons):
            ax[j].set_title(f"Local gradient for neuron {j + 1} in layer {i + 1}")
            ax[j].set_ylabel("Local gradient")
            ax[j].set_xlabel("Data point index")
            ax[j].plot(neural_network.layers[i].local_gradients_matrix[:, j])

            ax[j].vlines(x=epochs_list, ymin=min(neural_network.layers[i].local_gradients_matrix[:, j]),
                         ymax=max(neural_network.layers[i].local_gradients_matrix[:, j]), colors='r', linestyles='dashed', label="Epochs")

        plt.show()
