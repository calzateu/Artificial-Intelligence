import clustering
import data_processing as dp
import dimensionality_reduction as dr
import fuzzy_functions
import graphics as gr
import input_output as io
import pandas as pd
import norms
import numpy as np
from typing import Callable, Literal
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
                              return_center_points: bool = False,
                              graphics: bool = False, **kwargs) -> np.ndarray | None:
    """
    Run a clustering pipeline using the specified method on the provided data.
    Args:
        clustering_method (Callable): The clustering method to use.
        data (pd.DataFrame): The input data for clustering.
        graphic_clusters (bool): Flag indicating whether to visualize the clusters (default False).
        num_components (int): The number of components for dimensionality reduction (default 2).
        distance_matrix (np.ndarray): The distance matrix.
        graphics (bool): Flag indicating whether to visualize the clustering results (default False).
    Returns:
        np.ndarray | None: The cluster centers or None if return_center_points is False.
    """
    print(f"Running clustering pipeline with {clustering_method.__name__}...")
    if distance_matrix is None:
        distance_matrix = dp.compute_distances((data, data), norms.euclidean_norm)

    # Run clustering method.
    cluster_centers, center_points, distances_data_to_centers = clustering_method(
        data=data, norm=norms.euclidean_norm, distance_matrix=distance_matrix, graphics=graphics, **kwargs
    )

    print(f"Found {len(cluster_centers)} cluster centers with {clustering_method.__name__}:")
    print(cluster_centers)

    # Label data
    print("Labeling data...")
    labels = dp.label_data(data, cluster_centers, distances_data_to_centers)

    # Run dimensionality reduction for PCA, t-SNE, and UMAP.
    if graphic_clusters:
        # Run dimensionality reduction
        print("Running dimensionality reduction...")
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
                                                          labels, plot_names, axes)

    if return_center_points:
        return center_points


def run_unsupervised_pipeline(generate_synthetic_data: bool = False, num_samples: int = 10000,
                              run_clustering: bool = False, is_in_data_folder: bool = True, name_of_dataset: str = None,
                              path_to_data: str = None,
                              drop_axes: list = None,
                              subsample_size: int = None, graphic_clusters: bool = False,
                              num_components: int = 2, run_distances: bool = False, **kwargs) -> None:
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
        graphic_clusters (bool): Whether to graph the clusters.
        num_components (int): The number of components.
        run_distances (bool): Whether to run distance calculations.

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
        norm_name = kwargs.get("norm_name", "euclidean")
        if norm_name == "manhattan":
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
            norm = norms.euclidean_norm

        # Read data
        if is_in_data_folder:
            data = io.read_data(filename=name_of_dataset)
        else:
            data = io.read_data(custom_path_to_data=path_to_data)

        data = data.drop(drop_axes, axis=1)

        # Get subsample
        if subsample_size:
            subsample_size = len(data)
            print(f"Getting subsample of {subsample_size} rows...")
            subsample = dp.get_subsample(data, subsample_size)
        else:
            subsample = data

        # Normalize data with min-max normalization.
        normalized_subsample = dp.normalize(subsample)

        distance_matrix = dp.compute_distances(data=normalized_subsample, norm=norm, **kwargs)

        # Run mountain clustering. Select graphics=False to not display the mountain function.
        # center_points = __run_clustering_pipeline(clustering_method=clustering.mountain_clustering,
        #                                           data=normalized_subsample, num_components=num_components,
        #                                           distance_matrix=distance_matrix, return_center_points=True,
        #                                           graphics=False, **kwargs)

        # Run subtractive clustering. Select graphics=False to not display the density function.
        # __run_clustering_pipeline(clustering_method=clustering.subtractive_clustering, data=normalized_subsample,
        #                           num_components=num_components, distance_matrix=distance_matrix, graphics=False,
        #                           **kwargs)

        # Run k-means
        # __run_clustering_pipeline(clustering_method=clustering.k_means_clustering, data=normalized_subsample,
        #                           num_components=num_components, distance_matrix=distance_matrix, graphics=True,
        #                           initial_cluster_points=center_points, **kwargs)

        # Run fuzzy c-means
        __run_clustering_pipeline(clustering_method=clustering.fuzzy_c_means_clustering, data=normalized_subsample,
                                  graphic_clusters=graphic_clusters, num_components=num_components,
                                  distance_matrix=distance_matrix, graphics=True, **kwargs)

    if run_distances:
        print("Calculating distances...")
        # Read synthetic data
        is_default_data = True
        if is_default_data:
            data = io.read_data(filename="output_centroid.csv")
        else:
            data = io.read_data(custom_path_to_data="your_path/data.csv")

        # Get subsample of 100 rows
        print("Getting subsample of 100 rows...")
        subsample = dp.get_subsample(data, 100)

        # Normalize data
        subsample = dp.normalize(subsample)

        distances_euclidean = dp.compute_distances(subsample, norms.euclidean_norm)
        distances_manhattan = dp.compute_distances(subsample, norms.manhattan_norm)
        distances_chebyshev = dp.compute_distances(subsample, norms.p_norm, **{"p": 2})
        covariance_matrix = np.cov(subsample, rowvar=False)
        distances_mahalanobis = dp.compute_distances(subsample, norms.mahalanobis_distance,
                                                     **{"covariance_matrix": covariance_matrix})

        gr.grap_distance_matrix(distances_euclidean, method_name="Euclidean")
        gr.grap_distance_matrix(distances_manhattan, method_name="Manhattan")
        gr.grap_distance_matrix(distances_chebyshev, method_name="Chebyshev")
        gr.grap_distance_matrix(distances_mahalanobis, method_name="Mahalanobis")
