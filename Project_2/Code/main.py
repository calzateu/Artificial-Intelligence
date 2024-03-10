import data_processing as dp
import input_output as io
import matplotlib.pyplot as plt
import norms
import os
import seaborn as sns
import synthetic_data


if __name__ == '__main__':

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
    data = io.read_synthetic_data("output_centroid_.csv", current_path)

    # Get subsample of 100 rows
    subsample = dp.get_subsample(data, 100)

    # Normalize data
    subsample = dp.normalize(subsample)

    distances_euclidean = dp.compute_distances(subsample, norms.euclidean_norm)
    # distances_manhattan = dp.compute_distances(subsample, norms.manhattan_norm)
    # distances_chebyshev = dp.compute_distances(subsample, norms.p_norm, 2)

    print(distances_euclidean)
    print(len(distances_euclidean))

    # Crear un mapa de calor con matplotlib y seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(distances_euclidean, cmap="YlGnBu", fmt=".1f", linewidths=.5)

    # Añadir etiquetas a los ejes
    plt.xlabel("Índices de la matriz")
    plt.ylabel("Índices de la matriz")
    plt.title("Matriz de Distancias")

    # Mostrar la gráfica
    plt.show()
