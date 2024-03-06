from fuzzy_functions import *
from graphics import *
from tests import *
from run_examples import *
import synthetic_data
import input_output as io
import data_processing as dp
import norms


if __name__ == '__main__':
    # Generate synthetic data
    # Specify min and max values for each variable
    min_vals = [0, 0, 0, 0]
    max_vals = [1, 24, 49, 49]

    # Generate 10000 samples
    num_samples = 100

    synthetic_data_prbs = synthetic_data.generate_synthetic_data_prbs(
        num_samples, min_vals, max_vals, save_results=False
        )
    print(synthetic_data_prbs)

    # # Get results
    # synthetic_data = generate_synthetic_data(min_val, max_val, num_samples,
    #                                          save_results=True)

    # # Read synthetic data
    # data = io.read_synthetic_data("Project_2/outputs/output_centroid.csv")

    # # Get subsample of 100 rows
    # subsample = dp.get_subsample(data, 100)

    # # Normalize data
    # subsample = dp.normalize(subsample)

    # distances_euclidean = dp.compute_distances(subsample, norms.euclidean_norm)
    # distances_manhattan = dp.compute_distances(subsample, norms.manhattan_norm)
    # distances_chebyshev = dp.compute_distances(subsample, norms.p_norm, 2)

    # print(distances_euclidean)
    # print(len(distances_euclidean))
