import run_examples
import sklearn.metrics.cluster as cluster_metrics


if __name__ == '__main__':
    # ################## Choose if you want to generate synthetic data.                     ##################
    generate_synthetic_data = False
    num_samples = 10000

    # ################## Choose if you want to run clustering.                              ##################
    run_clustering = True

    # Choose if the data is in the output folder
    # is_in_data_folder, name_of_dataset, path_to_data = False, None, "your_path/data.csv"
    is_in_data_folder, name_of_dataset, path_to_data = True, "output_centroid_poly.csv", None

    # Select axes to drop from the data
    # drop_axes = ["Id", "Species"]
    drop_axes = "label"

    # select the labels
    target = "label"

    # Select subsample size
    sub_sample_size = 100

    # Select which algorithm to use
    clustering_methods_names = ["mountain", "subtractive", "k-means", "fuzzy c-means", "db scan"]
    # clustering_methods_names = ["db scan"]

    # Select if you want to plot the clusters
    graphic_clusters = False

    # Select number of components for dimensionality reduction. Only used if graphic_clusters is True
    num_components = 3

    kwargs = dict()

    # Select the norm to use.
    kwargs["norm_name"] = "euclidean"
    # kwargs["norm_name"] = "manhattan"
    # kwargs"norm_name"] = "mahalanobis"
    # kwargs["norm_name"], kwargs["p"] = "p-norm", 3
    # kwargs["norm_name"] = "cosine"

    # Parameters for mountain clustering
    # kwargs["sigma"] = 1
    # kwargs["beta"] = 1
    kwargs["sigma"] = 2
    kwargs["beta"] = 1

    # Parameters for subtractive clustering
    kwargs["r_a"] = 0.5
    kwargs["r_b"] = 0.5

    # Parameters for k-means clustering
    kwargs["k"] = 2

    # Parameters for fuzzy c-means clustering
    kwargs["c"] = 2
    kwargs["m"] = 1.1

    # Parameters for DBSCAN clustering
    kwargs["eps"] = 0.5
    kwargs["min_pts"] = 9

    # ################## Choose if you want to run distance calculations and plot graphs.   ##################
    run_distances = False

    run_examples.run_unsupervised_pipeline(generate_synthetic_data=generate_synthetic_data, num_samples=num_samples,
                                           run_clustering=run_clustering, is_in_data_folder=is_in_data_folder,
                                           name_of_dataset=name_of_dataset, path_to_data=path_to_data,
                                           drop_axes=drop_axes, subsample_size=sub_sample_size,
                                           clustering_methods_names=clustering_methods_names,
                                           graphic_clusters=graphic_clusters, num_components=num_components,
                                           run_distances=run_distances, save_graphs=True,
                                           intra_cluster_index=cluster_metrics.calinski_harabasz_score,
                                           extra_cluster_index=cluster_metrics.fowlkes_mallows_score, target=target,
                                           **kwargs)

    # ################## Choose if you want to run the clustering indices.                  ##################
    run_indices = False

    # Choose if the data is in the output folder
    # is_in_data_folder, name_of_dataset, path_to_data = False, None, "your_path/data.csv"
    is_in_data_folder, name_of_dataset, path_to_data = True, "output_centroid_labeled.csv", None

    # Select axes to drop from the data
    # drop_axes = ["Id", "Species"]
    drop_axes = None

    # select the labels
    target = "label"

    kwargs = dict()

    # Select the norm to use.
    kwargs["norm_name"] = "euclidean"
    # kwargs["norm_name"] = "manhattan"
    # kwargs"norm_name"] = "mahalanobis"
    # kwargs["norm_name"], kwargs["p"] = "p-norm", 3
    # kwargs["norm_name"] = "cosine"

    # Select which algorithm to use
    # clustering_methods_names = ["mountain", "subtractive", "k-means", "fuzzy c-means"]
    clustering_methods_names = ["db scan"]

    # Select max iterations
    kwargs["max_iter"] = 100

    # Select if you want to plot the clusters
    graphic_clusters = False

    # Select number of components for dimensionality reduction. Only used if graphic_clusters is True
    num_components = 2

    # Parameters for mountain clustering
    kwargs["sigma_list"] = [0.5, 1, 2]
    kwargs["beta_list"] = [0.5, 1, 2]

    # Parameters for subtractive clustering
    kwargs["r_a_list"] = [0.5, 1, 2]
    kwargs["r_b_list"] = [0.5, 0.8, 1]

    # Parameters for k-means clustering
    kwargs["k_list"] = [2, 3, 4, 5, 6]

    # Parameters for fuzzy c-means clustering
    kwargs["c_list"] = [2, 3, 4, 5]
    kwargs["m_list"] = [1.1, 1.5, 2, 3]

    # Parameters for DBSCAN clustering
    # kwargs["eps_list"] = [0.2, 0.5, 0.8]
    # kwargs["min_pts_list"] = [5, 7, 9]
    kwargs["eps_list"] = [0.5, 0.9, 1.2]
    kwargs["min_pts_list"] = [5, 7, 9]

    if run_indices:
        run_examples.run_clustering_algorithms_and_plot_indices(is_in_data_folder=is_in_data_folder,
                                                                name_of_dataset=name_of_dataset,
                                                                target=target,
                                                                path_to_data=path_to_data, drop_axes=drop_axes,
                                                                subsample_size=sub_sample_size,
                                                                clustering_methods_names=clustering_methods_names,
                                                                graphic_clusters=graphic_clusters,
                                                                num_components=num_components,
                                                                save_graphs=True, **kwargs)
