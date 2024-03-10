import data_processing as dp
import input_output as io
import matplotlib.pyplot as plt
import norms
import os
import seaborn as sns
import synthetic_data


if __name__ == '__main__':
    # Choose if you want to generate synthetic data
    generate_synthetic_data = False
    if generate_synthetic_data:
        # Specify min and max values for each variable.
        min_vals = [0, 0, 0, 0]
        max_vals = [1, 24, 49, 49]

        # Generate 10000 samples
        num_samples = 10000

        synthetic_data_prbs = synthetic_data.generate_synthetic_data_prbs(
            num_samples, min_vals, max_vals, graph=False, save_results=False
            )

        # Get results
        synthetic_data_uniform = synthetic_data.generate_synthetic_data(min_vals, max_vals, num_samples,
                                                                        save_results=False)

    current_path = os.getcwd()

    # Read synthetic data
    is_default_data = True
    if is_default_data:
        data = io.read_data(filename="output_centroid.csv", current_path=current_path)
    else:
        data = io.read_data(path_to_data="your_path/data.csv")

    # Get subsample of 100 rows
    subsample = dp.get_subsample(data, 100)

    # Normalize data
    subsample = dp.normalize(subsample)

    distances_euclidean = dp.compute_distances(subsample, norms.euclidean_norm)
    # distances_manhattan = dp.compute_distances(subsample, norms.manhattan_norm)
    # distances_chebyshev = dp.compute_distances(subsample, norms.p_norm, 2)

    print(distances_euclidean)
    print(len(distances_euclidean))

    # Heatmap of the distances matrix using Seaborn and Matplotlib.
    plt.figure(figsize=(8, 6))
    sns.heatmap(distances_euclidean, cmap="YlGnBu", fmt=".1f", linewidths=.5)

    # Add labels
    plt.xlabel("Index")
    plt.ylabel("Index")
    plt.title("Distances matrix")

    # Show the plot
    plt.show()
