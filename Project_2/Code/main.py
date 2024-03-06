from fuzzy_functions import *
from graphics import *
from tests import *
from run_examples import *
from synthetic_data import generate_synthetic_data
import input_output as io
import data_processing as dp
import norms


if __name__ == '__main__':
    # Generate synthetic data
    # Specify min and max values for each variable
    # min_val = [0, 0, 0, 0]
    # max_val = [1, 24, 49, 49]

    # # Generate 10000 samples
    # num_samples = 10000

    # # Get results
    # synthetic_data = generate_synthetic_data(min_val, max_val, num_samples,
    #                                          save_results=True)

    # Read synthetic data
    data = io.read_synthetic_data("Project_2/outputs/output_centroid.csv")

    # Get subsample of 100 rows
    subsample = dp.get_subsample(data, 100)

    # Normalize data
    subsample = dp.normalize(subsample)

    distances_euclidean = dp.compute_distances(subsample, norms.euclidean_norm)
    distances_manhattan = dp.compute_distances(subsample, norms.manhattan_norm)
    distances_chebyshev = dp.compute_distances(subsample, norms.p_norm, 2)

    print(distances_euclidean)
    print(len(distances_euclidean))
