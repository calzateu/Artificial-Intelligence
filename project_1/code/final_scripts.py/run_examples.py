import fuzzy_functions
import graphics
import numpy as np


def run_system_graphics(inputs, t_norm=np.fmin, s_norm=np.fmax,
                    defuzz_methods=['centroid', 'bisector', 'mom', 'som'],
                    save_graphs=False):
    universe_of_variables = fuzzy_functions.build_universe_of_variables()
    membership_functions = fuzzy_functions.build_membership_functions(universe_of_variables)

    # Run system
    data_dict = fuzzy_functions.build_data_dict(universe_of_variables, membership_functions)

    dict_defuzzified_risks = fuzzy_functions.run_system(data_dict, inputs,
                                                        universe_of_variables,
                                                        membership_functions,
                                                        defuzz_methods=defuzz_methods,
                                                        t_norm=t_norm, s_norm=s_norm,
                                                        graphics=True,
                                                        save_graphs=save_graphs)

    print(dict_defuzzified_risks)


def call_graph_response_surface(inputs, variable_x, variable_y,
                                animation=None,
                                color_map='rainbow',
                                defuzz_methods=['centroid'],
                                t_norm=np.fmin, s_norm=np.fmax,
                                animation_velocity=1,
                                save_graphs=False,
                                save_animation=False):
    universe_of_variables = fuzzy_functions.build_universe_of_variables()
    membership_functions = fuzzy_functions.build_membership_functions(universe_of_variables)
    data_dict = fuzzy_functions.build_data_dict(universe_of_variables, membership_functions)

    variables = {
        'x_withdrawal_percentage': inputs['withdrawal_percentage'],
        'x_hour': inputs['hour'],
        'x_transactions_per_day': inputs['transactions_per_day'],
        'x_transactions_per_month': inputs['transactions_per_month']
    }

    variables[variable_x] = universe_of_variables[variable_x]
    variables[variable_y] = universe_of_variables[variable_y]
    if animation:
        variables[animation] = universe_of_variables[animation]

    graphics.graph_response_surface(variable_x, variable_y, animation, variables, data_dict,
                           universe_of_variables, membership_functions,
                           color_map, defuzz_methods=defuzz_methods,
                           t_norm=t_norm, s_norm=s_norm,
                           animation_velocity=animation_velocity,
                           save_graphs=save_graphs,
                           save_animation=save_animation)
