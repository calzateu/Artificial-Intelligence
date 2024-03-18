import clustering
import data_processing as dp
import dimensionality_reduction as dr
import fuzzy_functions
import graphics as gr
import input_output as io
import itertools
import pandas as pd
import norms
import numpy as np
import tabulate
from typing import Callable
import sklearn.metrics.cluster as cluster_metrics
import synthetic_data as sd


def run_system_all_chases(inputs: dict, t_norms: list[Callable], s_norms: list[Callable], defuzz_methods: list[str],
                          save_graphs: bool = False):
    """
    A function to run a fuzzy logic system with multiple t-norms, s-norms, defuzzification methods, and the
    option to save graphs.

    Args:
        inputs (dict): The input variables for the fuzzy logic system.
        t_norms (list[Callable]): List of t-norm functions.
        s_norms (list[Callable]): List of s-norm functions.
        defuzz_methods (list[str]): List of defuzzification methods.
        save_graphs (bool, optional): Option to save graphs. Defaults to False.
    """
    universe_of_variables = fuzzy_functions.build_universe_of_variables()
    membership_functions = fuzzy_functions.build_membership_functions(universe_of_variables)

    # Run system
    data_dict = fuzzy_functions.build_data_dict(universe_of_variables, membership_functions)
    for t_norm in t_norms:
        for s_norm in s_norms:
            fuzzy_functions.run_system(data_dict, inputs, universe_of_variables, membership_functions,
                                       defuzz_methods=defuzz_methods, t_norm=t_norm, s_norm=s_norm, graphics=True,
                                       save_graphs=save_graphs)


def call_graph_response_surface(inputs: dict, variable_x: str, variable_y: str, animation: str = None,
                                color_map: str = 'rainbow', defuzz_methods: list[str] = None,
                                t_norm: Callable = np.fmin, s_norm: Callable = np.fmax, animation_velocity=1,
                                save_graphs=False, save_animation=False):
    """
    Generates the response surface graph for the given input variables and options. It is a 3d surface graph,
    which can be animated.

    Args:
        inputs (dict): The input variables for the fuzzy logic system.
        variable_x (str): The variable to use for the x-axis.
        variable_y (str): The variable to use for the y-axis.
        animation (str, optional): The variable to animate the graph with. Defaults to None.
        color_map (str, optional): The color map to use for the graph. Defaults to 'rainbow'.
        defuzz_methods (list[str], optional): The defuzzification methods to use. Defaults to ['centroid'].
        t_norm (Callable, optional): The t-norm function to use. Defaults to np.fmin.
        s_norm (Callable, optional): The s-norm function to use. Defaults to np.fmax.
        animation_velocity (int, optional): The velocity of the animation. Defaults to 1.
        save_graphs (bool, optional): Whether to save the graphs. Defaults to False.
        save_animation (bool, optional): Whether to save the animation. Defaults to False.

    Returns:
        None
    """
    if defuzz_methods is None:
        defuzz_methods = ['centroid']

    universe_of_variables = fuzzy_functions.build_universe_of_variables()
    membership_functions = fuzzy_functions.build_membership_functions(universe_of_variables)
    data_dict = fuzzy_functions.build_data_dict(universe_of_variables, membership_functions)

    variables = {'x_withdrawal_percentage': inputs['withdrawal_percentage'], 'x_hour': inputs['hour'],
                 'x_transactions_per_day': inputs['transactions_per_day'],
                 'x_transactions_per_month': inputs['transactions_per_month'],
                 variable_x: universe_of_variables[variable_x], variable_y: universe_of_variables[variable_y]}

    if animation:
        variables[animation] = universe_of_variables[animation]

    gr.graph_response_surface(variable_x, variable_y, animation, variables, data_dict, universe_of_variables,
                              membership_functions, color_map, defuzz_methods=defuzz_methods, t_norm=t_norm,
                              s_norm=s_norm, animation_velocity=animation_velocity, save_graphs=save_graphs,
                              save_animation=save_animation)


def run_graph_response_surface_all_chases(inputs: dict, x_variables: list[str], y_variables: list[str],
                                          animations: list[str], t_norms: list[Callable], s_norms: list[Callable],
                                          defuzz_methods: list[str] = None, animation_velocity: int = 1,
                                          save_graphs: bool = False, save_animation: bool = False):
    """
    A function to run graph response surface for all combinations of input parameters and options.
    :param inputs: A dictionary of input values
    :param x_variables: A list of x-axis variables
    :param y_variables: A list of y-axis variables
    :param animations: A list of animation variables
    :param t_norms: A list of t-norm functions
    :param s_norms: A list of s-norm functions
    :param defuzz_methods: A list of defuzzification methods
    :param animation_velocity: An integer representing animation velocity
    :param save_graphs: A boolean indicating whether to save graphs
    :param save_animation: A boolean indicating whether to save animation
    """
    for t_norm in t_norms:
        for s_norm in s_norms:
            for axis_x in x_variables:
                for axis_y in y_variables:
                    if axis_x == axis_y:
                        continue
                    for animation in animations:
                        if animation == axis_x or animation == axis_y:
                            continue
                        print(f"Axis X: {axis_x}, Axis Y: {axis_y}, Animation: {animation}")
                        call_graph_response_surface(inputs, axis_x, axis_y,
                                                    animation=animation,
                                                    defuzz_methods=defuzz_methods,
                                                    t_norm=t_norm, s_norm=s_norm,
                                                    animation_velocity=animation_velocity,
                                                    save_graphs=save_graphs,
                                                    save_animation=save_animation)


def __run_clustering_pipeline(clustering_method: Callable, data: pd.DataFrame, graphic_clusters: bool = False,
                              num_components: int = 2, distance_matrix: np.ndarray = None,
                              return_center_points: bool = False, intra_cluster_index: Callable = None,
                              extra_cluster_index: Callable = None, true_labels: np.ndarray = None,
                              graphics: bool = False, **kwargs) -> np.ndarray | None:
    """
    Run a clustering pipeline using the specified method on the provided data.
    Args:
        clustering_method (Callable): The clustering method to use.
        data (pd.DataFrame): The input data for clustering.
        graphic_clusters (bool): Flag indicating whether to visualize the clusters (default False).
        num_components (int): The number of components for dimensionality reduction (default 2).
        return_center_points (bool): Flag indicating whether to return the cluster centers (default False).
        intra_cluster_index (Callable): The intra-cluster index function to use.
        extra_cluster_index (Callable): The extra-cluster index function to use.
        distance_matrix (np.ndarray): The distance matrix.
        graphics (bool): Flag indicating whether to visualize the clustering results (default False).
    Returns:
        np.ndarray | None: The cluster centers or None if return_center_points is False.
    """
    print(f"Running clustering pipeline with {clustering_method.__name__}...")
    if distance_matrix is None:
        distance_matrix = dp.compute_distances((data, data), norms.euclidean_norm)

    # Run clustering method.
    cluster_centers, center_points, distances_data_to_centers, labels = clustering_method(
        data=data, norm=norms.euclidean_norm, distance_matrix=distance_matrix, graphics=graphics, **kwargs
    )

    print(f"Found {len(cluster_centers)} cluster centers with {clustering_method.__name__}:")
    print(cluster_centers)

    if labels is None:
        # Label data
        print("Labeling data...")
        labels = dp.label_data(data, cluster_centers, distances_data_to_centers)

    # Run dimensionality reduction for PCA, t-SNE, and UMAP.
    if graphic_clusters:
        # Run dimensionality reduction
        print("Running dimensionality reduction to visualize clustering results...")
        # Check if number of axes is 2 or 3.
        if num_components == 2:
            axes = ["PC1", "PC2"]
        elif num_components == 3:
            axes = ["PC1", "PC2", "PC3"]
        else:
            raise ValueError("Number of axes must be 2 or 3.")

        principal_df_pca, transformed_cen_points_pca = dr.reduce_dimensionality(
            "pca", data, center_points, num_components=num_components, axes=axes
        )
        principal_df_tsne, transformed_cen_points_tsne = dr.reduce_dimensionality(
            "tsne", data, center_points, num_components=num_components, axes=axes
        )
        principal_df_umap, transformed_cen_points_umap = dr.reduce_dimensionality(
            "umap", data, center_points, num_components=num_components, axes=axes
        )
        plot_names = ["PCA", "t-SNE", "UMAP"]

        # Visualize clustering results.
        print("Visualizing clustering results...")
        gr.graph_clustering_results_for_multiple_datasets([principal_df_pca, principal_df_tsne,
                                                           principal_df_umap], cluster_centers,
                                                          [transformed_cen_points_pca,
                                                           transformed_cen_points_tsne, transformed_cen_points_umap],
                                                          labels, plot_names, axes, save_graphs=True,
                                                          method_name=clustering_method.__name__)

    # End of pipeline
    print()

    values_to_return = [None]*3
    if return_center_points:
        values_to_return[0] = center_points

    if intra_cluster_index is not None:
        print(f"Running intra-cluster index with {intra_cluster_index.__name__}...")
        if len(set(labels)) <= 1:
            values_to_return[1] = 0
        else:
            values_to_return[1] = intra_cluster_index(data, labels)

    if extra_cluster_index is not None and true_labels is not None:
        print(f"Running extra-cluster index with {extra_cluster_index.__name__}...")
        values_to_return[2] = extra_cluster_index(true_labels, labels)

    return values_to_return


def run_unsupervised_pipeline(generate_synthetic_data: bool = False, num_samples: int = 10000,
                              run_clustering: bool = False, is_in_data_folder: bool = True, name_of_dataset: str = None,
                              path_to_data: str = None, drop_axes: list = None, subsample_size: int = None,
                              clustering_methods_names: list[str] = None,
                              graphic_clusters: bool = False, num_components: int = 2, run_distances: bool = False,
                              save_graphs: bool = False, **kwargs) -> None:
    """
    A function to run an unsupervised pipeline with options to generate synthetic data and calculate distances.

    Args:
        generate_synthetic_data (bool): Whether to generate synthetic data.
        num_samples (int): The number of samples to generate with synthetic data.
        run_clustering (bool): Whether to run clustering.
        is_in_data_folder (bool): Whether the data is in the data folder. It is used to run clustering.
        name_of_dataset (str): The name of the dataset if it is in the data folder.
        path_to_data (str): The path to the data if it is not in the data folder.
        drop_axes (list): The axes to drop.
        subsample_size (int): The size of the subsample.
        clustering_methods_names (list[str]): The clustering methods to use.
        graphic_clusters (bool): Whether to graph the clusters.
        num_components (int): The number of components.
        run_distances (bool): Whether to run distance calculations.
        save_graphs (bool): Whether to save the graphs.

    Returns:
        None
    """
    # Choose if you want to generate synthetic data.
    if generate_synthetic_data:
        # Generate 10000 samples
        print(f"Generating {num_samples} samples of synthetic data...")

        # Specify min and max values for each variable.
        min_vals = [0, 0, 0, 0]
        max_vals = [1, 24, 49, 49]

        sd.generate_synthetic_data(
            "prbs", num_samples, min_vals, max_vals, graph=True, save_results=True
        )

    if run_clustering:
        print("Running clustering...")

        # Choose norm
        norm_name = kwargs["norm_name"]

        if norm_name == "euclidean":
            norm = norms.euclidean_norm
        elif norm_name == "manhattan":
            norm = norms.manhattan_norm
        elif norm_name == "p-norm":
            norm = norms.p_norm
            p = kwargs.get("p", None)
            if p is None:
                print(f"Parameter p not specified. Using p-norm with p = {kwargs.get('p', 3)}")
                kwargs["p"] = 3
        elif norm_name == "mahalanobis":
            norm = norms.mahalanobis_distance
        elif norm_name == "cosine":
            norm = norms.cosine_similarity
        else:
            raise ValueError(f"Norm {norm_name} not recognized.")

        # Read data
        if is_in_data_folder:
            data = io.read_data(filename=name_of_dataset)
        else:
            data = io.read_data(custom_path_to_data=path_to_data)

        # Drop axes
        if drop_axes is not None:
            data = data.drop(drop_axes, axis=1)

        # Get subsample
        if subsample_size:
            print(f"Getting subsample of {subsample_size} rows...")
            subsample = dp.get_subsample(data, subsample_size)
        else:
            subsample = data

        normalized_subsample = dp.preprocess_data(subsample)

        # Get the clustering methods
        if clustering_methods_names is not None:
            clustering_methods = []
            for method_name in clustering_methods_names:
                if method_name == "mountain":
                    clustering_methods.append(clustering.mountain_clustering)
                elif method_name == "subtractive":
                    clustering_methods.append(clustering.subtractive_clustering)
                elif method_name == "k-means":
                    clustering_methods.append(clustering.k_means_clustering)
                elif method_name == "fuzzy c-means":
                    clustering_methods.append(clustering.fuzzy_c_means_clustering)
                elif method_name == "db scan":
                    clustering_methods.append(clustering.dbscan_clustering)
                else:
                    raise ValueError(f"Clustering method {method_name} not recognized."
                                     f"Try 'mountain', 'subtractive', 'k-means', or 'fuzzy c-means'.")

        else:
            clustering_methods = [clustering.fuzzy_c_means_clustering]

        distance_matrix = dp.compute_distances(data=normalized_subsample, norm=norm, **kwargs)

        # Run each clustering method
        for clustering_method in clustering_methods:
            __run_clustering_pipeline(clustering_method=clustering_method, data=normalized_subsample,
                                      graphic_clusters=graphic_clusters, num_components=num_components,
                                      distance_matrix=distance_matrix, return_center_points=False, graphics=False,
                                      save_graphs=save_graphs,
                                      **kwargs)

    if run_distances:
        print("Calculating distances...")
        # Read synthetic data
        is_default_data = True
        if is_default_data:
            data = io.read_data(filename="output_centroid.csv")
        else:
            data = io.read_data(custom_path_to_data="your_path/data.csv")

        # Get subsample
        if subsample_size:
            print(f"Getting subsample of {subsample_size} rows...")
            subsample = dp.get_subsample(data, subsample_size)
        else:
            subsample = data

        # Normalize data
        subsample = dp.preprocess_data(subsample)

        distances_euclidean = dp.compute_distances(subsample, norms.euclidean_norm)
        distances_manhattan = dp.compute_distances(subsample, norms.manhattan_norm)
        distances_p_norm = dp.compute_distances(subsample, norms.p_norm, **{"p": 3})
        covariance_matrix = np.cov(subsample, rowvar=False)
        distances_mahalanobis = dp.compute_distances(subsample, norms.mahalanobis_distance,
                                                     **{"covariance_matrix": covariance_matrix})
        distances_cosine = dp.compute_distances(subsample, norms.cosine_similarity)

        gr.grap_distance_matrix(distances_euclidean, method_name="Euclidean", save_graphs=save_graphs)
        gr.grap_distance_matrix(distances_manhattan, method_name="Manhattan", save_graphs=save_graphs)
        gr.grap_distance_matrix(distances_p_norm, method_name="P-norm", save_graphs=save_graphs)
        gr.grap_distance_matrix(distances_mahalanobis, method_name="Mahalanobis", save_graphs=save_graphs)
        gr.grap_distance_matrix(distances_cosine, method_name="Cosine", save_graphs=save_graphs)


def build_matrices(dict_results, methods):
    matrices = []

    for method_name, results in dict_results.items():
        param_keys = list(methods[method_name].keys())

        if len(param_keys) == 1:
            param_values = list(methods[method_name].values())
            rows = param_values[0]
            y = results

            export_matrix = [[row] + values for row, values in zip(rows, y.tolist())]
            matrices.append((method_name, export_matrix))

        elif len(param_keys) == 2:
            param_values = list(methods[method_name].values())
            rows, columns = param_values[0], param_values[1]
            z = results

            export_matrix = [[row] + values for row, values in zip(rows, z.tolist())]

            # Add column labels
            columns.insert(0, "")
            export_matrix.insert(0, columns)

            matrices.append((method_name, export_matrix))

    return matrices


def run_clustering_algorithms_and_plot_indices(is_in_data_folder: bool = True, name_of_dataset: str = None,
                                               target: str = None,
                                               path_to_data: str = None, drop_axes: list = None,
                                               subsample_size: int = None, clustering_methods_names: list[str] = None,
                                               graphic_clusters: bool = False, num_components: int = 2,
                                               save_graphs: bool = False, **kwargs):
    print("Running clustering analysis with clustering indices...")

    # Choose norm
    norm_name = kwargs["norm_name"]

    if norm_name == "euclidean":
        norm = norms.euclidean_norm
    elif norm_name == "manhattan":
        norm = norms.manhattan_norm
    elif norm_name == "p-norm":
        norm = norms.p_norm
        p = kwargs.get("p", None)
        if p is None:
            print(f"Parameter p not specified. Using p-norm with p = {kwargs.get('p', 3)}")
            kwargs["p"] = 3
    elif norm_name == "mahalanobis":
        norm = norms.mahalanobis_distance
    elif norm_name == "cosine":
        norm = norms.cosine_similarity
    else:
        raise ValueError(f"Norm {norm_name} not recognized.")

    # Read data
    if is_in_data_folder:
        data = io.read_data(filename=name_of_dataset)
    else:
        data = io.read_data(custom_path_to_data=path_to_data)

    # Get subsample
    if subsample_size:
        print(f"Getting subsample of {subsample_size} rows...")
        subsample = dp.get_subsample(data, subsample_size)
    else:
        subsample = data

    true_labels = None
    if target is not None:
        # Get labels
        unique_labels = subsample[target].unique()

        # Create a dictionary to map labels to integers
        mapping = {label: num + 1 for num, label in enumerate(unique_labels)}

        # Apply the mapping to the labels
        true_labels = subsample[target].replace(mapping)

    # Drop axes
    if drop_axes is None:
        drop_axes = []
    subsample = subsample.drop(drop_axes, axis=1)

    normalized_subsample = dp.preprocess_data(subsample=subsample)

    # Get the clustering methods and parameters
    methods = dict()
    if clustering_methods_names is not None:
        for method_name in clustering_methods_names:
            if method_name == "mountain":
                sigmas = kwargs["sigma_list"]
                betas = kwargs["beta_list"]
                methods[clustering.mountain_clustering] = {"sigmas": sigmas, "betas": betas}
                methods[clustering.mountain_clustering.__name__] = {"sigmas": sigmas, "betas": betas}
                print(f"Loaded mountain clustering with sigmas = {sigmas} and betas = {betas}.")
            elif method_name == "subtractive":
                r_as = kwargs["r_a_list"]
                r_bs = kwargs["r_b_list"]
                methods[clustering.subtractive_clustering] = {"r_as": r_as, "r_bs": r_bs}
                methods[clustering.subtractive_clustering.__name__] = {"r_as": r_as, "r_bs": r_bs}
                print(f"Loaded subtractive clustering with r_as = {r_as} and r_bs = {r_bs}.")
            elif method_name == "k-means":
                ks = kwargs["k_list"]
                methods[clustering.k_means_clustering] = {"ks": ks}
                methods[clustering.k_means_clustering.__name__] = {"ks": ks}
                print(f"Loaded k-means clustering with ks = {ks}.")
            elif method_name == "fuzzy c-means":
                cs = kwargs["c_list"]
                ms = kwargs["m_list"]
                methods[clustering.fuzzy_c_means_clustering] = {"cs": cs, "ms": ms}
                methods[clustering.fuzzy_c_means_clustering.__name__] = {"cs": cs, "ms": ms}
                print(f"Loaded fuzzy c-means clustering with cs = {cs} and ms = {ms}.")
            else:
                raise ValueError(f"Clustering method {method_name} not recognized."
                                 f"Try 'mountain', 'subtractive', 'k-means', or 'fuzzy c-means'.")

    else:
        raise ValueError("Clustering methods and parameters not specified.")

    print()

    distance_matrix = dp.compute_distances(data=normalized_subsample, norm=norm, **kwargs)

    # Initialize dictionaries to store results
    results_intra = dict()
    results_extra = dict()

    # Go through the methods and their parameters and save the results.
    for method, params in methods.items():
        if not isinstance(method, str):
            param_combinations = list(itertools.product(*params.values()))
            size = len(param_combinations)

            # Verify the number of parameters
            if len(params.values()) == 2:
                param_values = list(params.values())
                num_rows = len(param_values[0])
                num_cols = len(param_values[1])
            else:
                num_rows = size
                num_cols = 1

            results_intra[method.__name__] = [0] * (len(param_combinations))
            results_extra[method.__name__] = [0] * (len(param_combinations))
            cont = 0
            for combination in param_combinations:
                kwargs_temp = {param[:-1]: value for param, value in zip(params.keys(), combination)}

                # Call the method with the corresponding parameters
                print(f"Running {method.__name__} with params: {kwargs_temp}")

                # Don't return the center points.
                kwargs_temp["return_center_points"] = False

                result = __run_clustering_pipeline(clustering_method=method, data=normalized_subsample,
                                                   graphic_clusters=graphic_clusters, num_components=num_components,
                                                   distance_matrix=distance_matrix,
                                                   intra_cluster_index=cluster_metrics.calinski_harabasz_score,
                                                   extra_cluster_index=cluster_metrics.fowlkes_mallows_score,
                                                   true_labels=true_labels,
                                                   graphics=False, **kwargs_temp)

                # Store results for plotting
                results_intra[method.__name__][cont] = result[1]
                results_extra[method.__name__][cont] = result[2]
                print(f"Done with {method.__name__} with params: {kwargs_temp} and \n"
                      f"got inter-cluster index: {result[1]} and extra-cluster index: {result[2]}.")

                cont += 1

            results_intra[method.__name__] = np.array(results_intra[method.__name__]).reshape((num_rows, num_cols))
            results_extra[method.__name__] = np.array(results_extra[method.__name__]).reshape((num_rows, num_cols))

    print("Done running clustering pipeline.")

    print("Plotting results...")

    # Min-max normalization of intra and extra results
    results_intra_normalized = dict()
    results_extra_normalized = dict()
    for method_name in results_intra.keys():
        # Check if all the elements are the same for normalization
        if np.all(results_intra[method_name] == results_intra[method_name][0]):
            results_intra_normalized[method_name] = results_intra[method_name] / results_intra[method_name]
        else:
            results_intra_normalized[method_name] = (results_intra[method_name] - results_intra[method_name].min()) / (
                    results_intra[method_name].max() - results_intra[method_name].min())

        if np.all(results_extra[method_name] == results_extra[method_name][0]):
            results_extra_normalized[method_name] = results_extra[method_name] / results_extra[method_name]
        else:
            results_extra_normalized[method_name] = (results_extra[method_name] - results_extra[method_name].min()) / (
                    results_extra[method_name].max() - results_extra[method_name].min())

    # Calculate weighted results
    weighted_results = {}
    for method_name in results_intra.keys():
        weighted_results[method_name] = 0.5*np.array(results_intra_normalized[method_name]) + 0.5*np.array(
            results_extra_normalized[method_name])

    # Plot results in 3d graphic with each set of parameters and as z value the inter-cluster index.
    gr.plot_indices(weighted_results, methods, save_graphs)

    for method_name, results in weighted_results.items():
        keys = list(methods[method_name].keys())
        values = list(methods[method_name].values())
        if len(values) == 2:
            gr.grap_distance_matrix(results, method_name+" weighted indices", save_graphs, x_labels=values[0],
                                    y_labels=values[1], x_name=keys[0][:-1], y_name=keys[1][:-1])
        else:
            gr.grap_distance_matrix(results, method_name+" weighted indices", save_graphs, y_labels=values[0],
                                    y_name=keys[0][:-1])

    for method_name, results in results_intra.items():
        keys = list(methods[method_name].keys())
        values = list(methods[method_name].values())
        if len(values) == 2:
            gr.grap_distance_matrix(results, method_name+" intra indices", save_graphs, x_labels=values[0],
                                    y_labels=values[1], x_name=keys[0][:-1], y_name=keys[1][:-1])
        else:
            gr.grap_distance_matrix(results, method_name+" intra indices", save_graphs, y_labels=values[0],
                                    y_name=keys[0][:-1])

    for method_name, results in results_extra.items():
        keys = list(methods[method_name].keys())
        values = list(methods[method_name].values())
        if len(values) == 2:
            gr.grap_distance_matrix(results, method_name+" extra indices", save_graphs, x_labels=values[0],
                                    y_labels=values[1], x_name=keys[0][:-1], y_name=keys[1][:-1])
        else:
            gr.grap_distance_matrix(results, method_name+" extra indices", save_graphs, y_labels=values[0],
                                    y_name=keys[0][:-1])

    # matrices_intra = build_matrices(results_intra, methods)
    # for matrix in matrices_intra:
    #     print(f"Intra cluster indices for method {matrix[0]}:")
    #     if len(matrix[0][0]) > 1:
    #         # print(tabulate.tabulate(matrix[1][1:], headers=matrix[1][0], tablefmt="fancy_grid"))
    #         print(tabulate.tabulate(matrix[1][1:], headers=matrix[1][0], tablefmt="latex"))
    #     else:
    #         # print(tabulate.tabulate(matrix[1], tablefmt="fancy_grid"))
    #         print(tabulate.tabulate(matrix[1], tablefmt="latex"))
    #
    # matrices_extra = build_matrices(results_extra, methods)
    # for matrix in matrices_extra:
    #     print(f"Extra cluster indices for method {matrix[0]}:")
    #     if len(matrix[0][0]) > 1:
    #         # print(tabulate.tabulate(matrix[1][1:], headers=matrix[1][0], tablefmt="fancy_grid"))
    #         print(tabulate.tabulate(matrix[1], headers=matrix[1][0], tablefmt="latex"))
    #     else:
    #         # print(tabulate.tabulate(matrix[1], tablefmt="fancy_grid"))
    #         print(tabulate.tabulate(matrix[1], tablefmt="latex"))

    print("Done plotting results.")

    return
