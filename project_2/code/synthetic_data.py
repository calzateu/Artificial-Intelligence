import fuzzy_functions
import input_output as io
import numpy as np
import prbs
from typing import Callable, Literal


def generate_synthetic_data(generation_method: Literal['prbs', 'uniform'], size: int, min_vals: list, max_vals: list,
                            t_norm: Callable = None, s_norm: Callable = None, defuzz_methods: list[str] = None,
                            graph: bool = False, save_results: bool = False) -> list[dict]:
    """
    Generate synthetic data using pseudo-random binary sequence (PRBS) method.

    Args:
        generation_method (Literal['prbs', 'uniform']): The type of synthetic data to be generated.
        size (int): The size of the synthetic data to be generated.
        min_vals (list): The list of minimum values for generating synthetic data. Each value corresponds to a variable.
        max_vals (list): The list of maximum values for generating synthetic data. Each value corresponds to a variable.
        t_norm (Callable, optional): The t-norm function to be used. Defaults to None.
        s_norm (Callable, optional): The s-norm function to be used. Defaults to None.
        defuzz_methods (list[str], optional): The list of defuzzification methods to be used. Defaults to None.
        graph (bool, optional): Whether to display graphs for the generated data. Defaults to False.
        save_results (bool, optional): Whether to save the results to a file. Defaults to False.

    Returns:
        list[dict]: A list of dictionaries containing the synthetic data generated.
    """
    if generation_method == 'prbs':
        withdrawal_percentage = prbs.generate_prbs_input(size, min_vals[0], max_vals[0], graph=graph)
        hour = prbs.generate_prbs_input(size, min_vals[1], max_vals[1], graph=graph)
        transactions_per_day = prbs.generate_prbs_input(size, min_vals[2], max_vals[2], graph=graph)
        transactions_per_month = prbs.generate_prbs_input(size, min_vals[3], max_vals[3], graph=graph)
    elif generation_method == 'uniform':
        withdrawal_percentage = np.random.uniform(min_vals[0], max_vals[0], size)
        hour = np.random.uniform(min_vals[1], max_vals[1], size)
        transactions_per_day = np.random.uniform(min_vals[2], max_vals[2], size)
        transactions_per_month = np.random.uniform(min_vals[3], max_vals[3], size)
    else:
        raise ValueError('Invalid generation method')

    universe_of_variables = fuzzy_functions.build_universe_of_variables()
    membership_functions = fuzzy_functions.build_membership_functions(universe_of_variables)

    # Used to run system
    data_dict = fuzzy_functions.build_data_dict(universe_of_variables, membership_functions)

    if not defuzz_methods:
        defuzz_methods = ['centroid', 'bisector', 'mom', 'som']
    if not t_norm:
        t_norm = np.fmin
    if not s_norm:
        s_norm = np.fmax

    # Calculate z in a vectorized way
    vectorized_run_system = np.vectorize(lambda wp, h, td, tm: fuzzy_functions.run_system(
        data_dict,
        {
            'withdrawal_percentage': wp,
            'hour': h,
            'transactions_per_day': td,
            'transactions_per_month': tm
        },
        universe_of_variables, membership_functions,
        defuzz_methods=defuzz_methods, t_norm=t_norm, s_norm=s_norm,
        graphics=False))

    # Run system in vectorized way
    results = vectorized_run_system(withdrawal_percentage,
                                    hour,
                                    transactions_per_day,
                                    transactions_per_month)

    if save_results:
        io.save_fuzzy_system_results(results, withdrawal_percentage,
                                     hour, transactions_per_day,
                                     transactions_per_month)

    return results
