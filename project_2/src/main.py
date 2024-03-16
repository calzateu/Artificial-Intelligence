import run_examples


if __name__ == '__main__':
    # Choose if you want to generate synthetic data.
    generate_synthetic_data = False

    # Choose if you want to run clustering.
    run_clustering = True
    drop_axes = ["Id", "Species"]
    sub_sample_size = None
    num_components = 2

    kwargs = {}

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

    # Choose if you want to run distance calculations and plot graphs.
    run_distances = False

    run_examples.run_unsupervised_pipeline(generate_synthetic_data=generate_synthetic_data,
                                           run_clustering=run_clustering, subsample_size=sub_sample_size,
                                           num_components=num_components, drop_axes=drop_axes,
                                           run_distances=run_distances, **kwargs)
