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
