from fuzzy_functions import *
from graphics import *
from tests import *
from run_examples import *


def run_system_all_chases(inputs, t_norms, s_norms, defuzz_methods,
                          save_graphs=False):
    for t_norm in t_norms:
        for s_norm in s_norms:
            run_system_graphics(inputs, t_norm, s_norm, defuzz_methods,
                       save_graphs=save_graphs)


def run_graph_response_surface_all_chases(inputs, x_variables, y_variables,
                                          animations, t_norms, s_norms,
                                          defuzz_methods,
                                          animation_velocity=1,
                                          save_graphs=False,
                                          save_animation=False):
    for t_norm in t_norms:
        for s_norm in s_norms:
            for axis_x in x_variables:
                for axis_y in y_variables:
                    if axis_x == axis_y:
                        continue
                    for animation in animations:
                        if animation == axis_x or animation == axis_y:
                            continue
                        print(f"Axis X: {axis_x}, Axis Y: {axis_y}, Animation: {animation}")
                        call_graph_response_surface(inputs, axis_x, axis_y,
                                                    animation=animation,
                                                    defuzz_methods=defuzz_methods,
                                                    t_norm=t_norm, s_norm=s_norm,
                                                    animation_velocity=animation_velocity,
                                                    save_graphs=save_graphs,
                                                    save_animation=save_animation)


if __name__ == '__main__':
    # # Inputs
    # inputs = {
    #     'withdrawal_percentage': 0.7,
    #     'hour': 9,
    #     'transactions_per_day': 5,
    #     'transactions_per_month': 20
    # }
    inputs = {
        'withdrawal_percentage': 0.7,
        'hour': 9,
        'transactions_per_day': 5,
        'transactions_per_month': 20
        }
    # test_individual_functions(inputs)
    # run_system_graphics(inputs)

    # call_graph_response_surface(
    #     inputs=inputs,
    #     variable_x='x_withdrawal_percentage',
    #     #variable_x='x_transactions_per_day',
    #     variable_y='x_hour',
    #     animation='x_transactions_per_day',
    #     #animation='x_withdrawal_percentage',
    #     defuzz_methods=['centroid', 'bisector', 'som'],
    #     animation_velocity=300,
    #     save_graphs=True,
    #     save_animation=True
    #     )

    # run_system_all_chases(inputs, [np.fmin, algebraic_product],
    #                       [np.fmax, algebraic_sum],
    #                       ['centroid', 'bisector', 'som'],
    #                       save_graphs=True)

    run_graph_response_surface_all_chases(inputs,
                                          ['x_withdrawal_percentage', 'x_transactions_per_day'],
                                          ['x_hour'],
                                          ['x_withdrawal_percentage', 'x_transactions_per_day'],
                                          [np.fmin, algebraic_product],
                                          [np.fmax, algebraic_sum],
                                          ['centroid', 'bisector', 'som'],
                                          animation_velocity=300,
                                          save_graphs=False,
                                          save_animation=False)
