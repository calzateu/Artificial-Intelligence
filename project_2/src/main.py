import run_examples


if __name__ == '__main__':
    # Choose if you want to generate synthetic data.
    generate_synthetic_data = False

    # Choose if you want to run clustering.
    run_clustering = True
    drop_axes = ["Id", "Species"]
    sub_sample_size = None
    # axes = ["SepalLengthCm", "SepalWidthCm"]
    axes = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm"]

    # Choose if you want to run distance calculations and plot graphs.
    run_distances = False

    run_examples.run_unsupervised_pipeline(generate_synthetic_data=generate_synthetic_data,
                                           run_clustering=run_clustering, subsample_size=sub_sample_size,
                                           drop_axes=drop_axes, axes=axes, run_distances=run_distances)
