from fuzzy_functions import *
from graphics import *
from tests import *
from run_examples import *
from synthetic_data import generate_synthetic_data


if __name__ == '__main__':
    # Generate synthetic data
    # Specify min and max values for each variable
    min_val = [0, 0, 0, 0]
    max_val = [1, 24, 49, 49]

    # Generate 10000 samples
    num_samples = 10000

    # Get results
    synthetic_data = generate_synthetic_data(min_val, max_val, num_samples,
                                             save_results=True)
