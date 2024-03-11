import data_processing as dp
import fuzzy_functions
import graphics
import input_output as io
import norms
import numpy as np
from typing import Callable
import synthetic_data as sd


def run_system_all_chases(inputs: dict, t_norms: list[Callable], s_norms: list[Callable], defuzz_methods: list[str],
                          save_graphs: bool = False):
    """
    A function to run a fuzzy logic system with multiple t-norms, s-norms, defuzzification methods, and the
    option to save graphs.

    Args:
        inputs (dict): The input variables for the fuzzy logic system.
        t_norms (list[Callable]): List of t-norm functions.
        s_norms (list[Callable]): List of s-norm functions.
        defuzz_methods (list[str]): List of defuzzification methods.
        save_graphs (bool, optional): Option to save graphs. Defaults to False.
    """
    universe_of_variables = fuzzy_functions.build_universe_of_variables()
    membership_functions = fuzzy_functions.build_membership_functions(universe_of_variables)

    # Run system
    data_dict = fuzzy_functions.build_data_dict(universe_of_variables, membership_functions)
    for t_norm in t_norms:
        for s_norm in s_norms:
            fuzzy_functions.run_system(data_dict, inputs, universe_of_variables, membership_functions,
                                       defuzz_methods=defuzz_methods, t_norm=t_norm, s_norm=s_norm, graphics=True,
                                       save_graphs=save_graphs)


def call_graph_response_surface(inputs: dict, variable_x: str, variable_y: str, animation: str = None,
                                color_map: str = 'rainbow', defuzz_methods: list[str] = None,
                                t_norm: Callable = np.fmin, s_norm: Callable = np.fmax, animation_velocity=1,
                                save_graphs=False, save_animation=False):
    """
    Generates the response surface graph for the given input variables and options. It is a 3d surface graph,
    which can be animated.

    Args:
        inputs (dict): The input variables for the fuzzy logic system.
        variable_x (str): The variable to use for the x-axis.
        variable_y (str): The variable to use for the y-axis.
        animation (str, optional): The variable to animate the graph with. Defaults to None.
        color_map (str, optional): The color map to use for the graph. Defaults to 'rainbow'.
        defuzz_methods (list[str], optional): The defuzzification methods to use. Defaults to ['centroid'].
        t_norm (Callable, optional): The t-norm function to use. Defaults to np.fmin.
        s_norm (Callable, optional): The s-norm function to use. Defaults to np.fmax.
        animation_velocity (int, optional): The velocity of the animation. Defaults to 1.
        save_graphs (bool, optional): Whether to save the graphs. Defaults to False.
        save_animation (bool, optional): Whether to save the animation. Defaults to False.

    Returns:
        None
    """
    if defuzz_methods is None:
        defuzz_methods = ['centroid']

    universe_of_variables = fuzzy_functions.build_universe_of_variables()
    membership_functions = fuzzy_functions.build_membership_functions(universe_of_variables)
    data_dict = fuzzy_functions.build_data_dict(universe_of_variables, membership_functions)

    variables = {'x_withdrawal_percentage': inputs['withdrawal_percentage'], 'x_hour': inputs['hour'],
                 'x_transactions_per_day': inputs['transactions_per_day'],
                 'x_transactions_per_month': inputs['transactions_per_month'],
                 variable_x: universe_of_variables[variable_x], variable_y: universe_of_variables[variable_y]}

    if animation:
        variables[animation] = universe_of_variables[animation]

    graphics.graph_response_surface(variable_x, variable_y, animation, variables, data_dict, universe_of_variables,
                                    membership_functions, color_map, defuzz_methods=defuzz_methods, t_norm=t_norm,
                                    s_norm=s_norm, animation_velocity=animation_velocity, save_graphs=save_graphs,
                                    save_animation=save_animation)


def run_graph_response_surface_all_chases(inputs: dict, x_variables: list[str], y_variables: list[str],
                                          animations: list[str], t_norms: list[Callable], s_norms: list[Callable],
                                          defuzz_methods: list[str] = None, animation_velocity: int = 1,
                                          save_graphs: bool = False, save_animation: bool = False):
    """
    A function to run graph response surface for all combinations of input parameters and options.
    :param inputs: A dictionary of input values
    :param x_variables: A list of x-axis variables
    :param y_variables: A list of y-axis variables
    :param animations: A list of animation variables
    :param t_norms: A list of t-norm functions
    :param s_norms: A list of s-norm functions
    :param defuzz_methods: A list of defuzzification methods
    :param animation_velocity: An integer representing animation velocity
    :param save_graphs: A boolean indicating whether to save graphs
    :param save_animation: A boolean indicating whether to save animation
    """
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


def run_unsupervised_pipeline(generate_synthetic_data: bool = False, run_distances: bool = False):
    """
    A function to run an unsupervised pipeline with options to generate synthetic data and calculate distances.

    Args:
        generate_synthetic_data (bool): Whether to generate synthetic data.
        run_distances (bool): Whether to run distance calculations.

    Returns:
        None
    """
    # Choose if you want to generate synthetic data
    if generate_synthetic_data:
        # Generate 10000 samples
        num_samples = 10000

        print(f"Generating {num_samples} samples of synthetic data...")

        # Specify min and max values for each variable.
        min_vals = [0, 0, 0, 0]
        max_vals = [1, 24, 49, 49]

        sd.generate_synthetic_data(
            "prbs", num_samples, min_vals, max_vals, graph=True, save_results=True
        )

    if run_distances:
        print("Calculating distances...")
        # Read synthetic data
        is_default_data = True
        if is_default_data:
            data = io.read_data(filename="output_centroid.csv")
        else:
            data = io.read_data(custom_path_to_data="your_path/data.csv")

        # Get subsample of 100 rows
        print("Getting subsample of 100 rows...")
        subsample = dp.get_subsample(data, 100)

        # Normalize data
        subsample = dp.normalize(subsample)

        distances_euclidean = dp.compute_distances(subsample, norms.euclidean_norm)
        distances_manhattan = dp.compute_distances(subsample, norms.manhattan_norm)
        distances_chebyshev = dp.compute_distances(subsample, norms.p_norm, 2)

        graphics.grap_distance_matrix(distances_euclidean, method_name="Euclidean")
        graphics.grap_distance_matrix(distances_manhattan, method_name="Manhattan")
        graphics.grap_distance_matrix(distances_chebyshev, method_name="Chebyshev")
