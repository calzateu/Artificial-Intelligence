import fuzzy_functions
import graphics
import skfuzzy as fuzz
import numpy as np


def test_individual_functions(inputs):
    universe_of_variables = fuzzy_functions.build_universe_of_variables()
    membership_functions = fuzzy_functions.build_membership_functions(universe_of_variables)

    # Create a dictionary with the data
    data_dict = fuzzy_functions.build_data_dict(universe_of_variables, membership_functions)

    # Call the function with the dictionary as an argument
    graphics.visualize_membership_functions(data_dict)

    interpolated_inputs = fuzzy_functions.interpolate_inputs(inputs, data_dict)
    rules = fuzzy_functions.build_rules(interpolated_inputs)
    betas = fuzzy_functions.build_betas(rules)
    limits = fuzzy_functions.build_limits(betas, membership_functions)
    aggregated_membership_functions = fuzzy_functions.aggregate_membership_functions(limits)
    risk = fuzz.defuzz(universe_of_variables['y_risk'], aggregated_membership_functions, 'centroid')
    print(f"Risk: {risk}")


def test_run_system(inputs, t_norm=np.fmin, s_norm=np.fmax,
                    defuzz_methods=['centroid', 'bisector', 'mom', 'som']):
    universe_of_variables = fuzzy_functions.build_universe_of_variables()
    membership_functions = fuzzy_functions.build_membership_functions(universe_of_variables)

    # Run system
    data_dict = fuzzy_functions.build_data_dict(universe_of_variables, membership_functions)

    dict_defuzzified_risks = fuzzy_functions.run_system(data_dict, inputs,
                                                        universe_of_variables,
                                                        membership_functions,
                                                        defuzz_methods=defuzz_methods,
                                                        t_norm=t_norm, s_norm=s_norm,
                                                        graphics=True)

    print(dict_defuzzified_risks)
