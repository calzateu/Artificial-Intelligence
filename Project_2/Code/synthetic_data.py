import fuzzy_functions
import input_output as io
import numpy as np
import prbs


def generate_synthetic_data_prbs(size, min_vals, max_vals, graph=False,
                                 save_results=False):
    # Parámetros PRBS
    order = 4
    taps = [3, 0]

    # Parámetros señal
    duration = 1  # segundos
    sample_rate = 1000  # Hz

    withdrawal_percentage = prbs.generate_prbs_signal(size, min_vals[0],
                                                      max_vals[0], order,
                                                      taps, duration,
                                                      sample_rate,
                                                      graph=graph)
    hour = prbs.generate_prbs_signal(size, min_vals[1], max_vals[1], order,
                                     taps, duration, sample_rate,
                                     graph=graph)
    transactions_per_day = prbs.generate_prbs_signal(size, min_vals[2],
                                                     max_vals[2], order,
                                                     taps, duration,
                                                     sample_rate,
                                                     graph=graph)
    transactions_per_month = prbs.generate_prbs_signal(size, min_vals[3],
                                                       max_vals[3], order,
                                                       taps, duration,
                                                       sample_rate,
                                                       graph=graph)

    universe_of_variables = fuzzy_functions.build_universe_of_variables()
    membership_functions = fuzzy_functions.build_membership_functions(universe_of_variables)

    # Run system
    data_dict = fuzzy_functions.build_data_dict(universe_of_variables, membership_functions)


    defuzz_methods = ['centroid', 'bisector', 'mom', 'som']
    t_norm = np.fmin
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

    results = vectorized_run_system(withdrawal_percentage,
                                    hour,
                                    transactions_per_day,
                                    transactions_per_month)


    if save_results:
        io.save_fuzzy_system_results(results, withdrawal_percentage,
                                     hour, transactions_per_day,
                                     transactions_per_month)

    return results



def generate_synthetic_data(min_val, max_val, num_samples,
                            save_results=False):
    # Generate synthetic data for each variable
    withdrawal_percentage = np.random.uniform(min_val[0], max_val[0], num_samples)
    hour = np.random.uniform(min_val[1], max_val[1], num_samples)
    transactions_per_day = np.random.uniform(min_val[2], max_val[2], num_samples)
    transactions_per_month = np.random.uniform(min_val[3], max_val[3], num_samples)


    universe_of_variables = fuzzy_functions.build_universe_of_variables()
    membership_functions = fuzzy_functions.build_membership_functions(universe_of_variables)

    # Run system
    data_dict = fuzzy_functions.build_data_dict(universe_of_variables, membership_functions)


    defuzz_methods = ['centroid', 'bisector', 'mom', 'som']
    t_norm = np.fmin
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

    results = vectorized_run_system(withdrawal_percentage,
                                    hour,
                                    transactions_per_day,
                                    transactions_per_month)


    if save_results:
        io.save_fuzzy_system_results(results, withdrawal_percentage,
                                     hour, transactions_per_day,
                                     transactions_per_month)

    return results
