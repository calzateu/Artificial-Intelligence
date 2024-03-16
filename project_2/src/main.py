import run_examples


if __name__ == '__main__':
    # ################## Choose if you want to generate synthetic data.                     ##################
    generate_synthetic_data = False
    num_samples = 10000

    # ################## Choose if you want to run clustering.                              ##################
    run_clustering = True

    # Choose if the data is in the output folder
    # is_in_data_folder, name_of_dataset, path_to_data = False, None, "your_path/data.csv"
    is_in_data_folder, name_of_dataset, path_to_data = True, "Iris.csv", None

    # Select axes to drop from the data
    drop_axes = ["Id", "Species"]

    # Select subsample size
    sub_sample_size = None

    # Select if you want to plot the clusters
    graphic_clusters = False

    # Select number of components for dimensionality reduction. Only used if graphic_clusters is True
    num_components = 2

    kwargs = dict()

    # Select the norm to use.
    kwargs["norm_name"] = "euclidean"
    # kwargs["norm_name"] = "manhattan"
    # kwargs"norm_name"] = "mahalanobis"
    # kwargs["norm_name"], kwargs["p"] = "p-norm", 3
    # kwargs["norm_name"] = "cosine"

    # Parameters for mountain clustering
    kwargs["sigma"] = 1
    kwargs["beta"] = 1

    # Parameters for subtractive clustering
    kwargs["r_a"] = 0.5
    kwargs["r_b"] = 0.8

    # Parameters for k-means clustering
    kwargs["k"] = 4

    # Parameters for fuzzy c-means clustering
    kwargs["c"] = 4
    kwargs["m"] = 2

    # ################## Choose if you want to run distance calculations and plot graphs.   ##################
    run_distances = False

    run_examples.run_unsupervised_pipeline(generate_synthetic_data=generate_synthetic_data, num_samples=num_samples,
                                           run_clustering=run_clustering, is_in_data_folder=is_in_data_folder,
                                           name_of_dataset=name_of_dataset, path_to_data=path_to_data,
                                           drop_axes=drop_axes, subsample_size=sub_sample_size,
                                           graphic_clusters=graphic_clusters, num_components=num_components,
                                           run_distances=run_distances, **kwargs)
