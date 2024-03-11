import run_examples


if __name__ == '__main__':
    # Choose if you want to generate synthetic data.
    generate_synthetic_data = False

    run_clustering = True

    run_distances = False

    run_examples.run_unsupervised_pipeline(generate_synthetic_data, run_clustering, run_distances)
