import run_examples


if __name__ == '__main__':
    # Choose if you want to generate synthetic data.
    generate_synthetic_data = False

    run_distances = True

    run_examples.run_unsupervised_pipeline(generate_synthetic_data, run_distances)
