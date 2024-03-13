import constants
from graphics import graph_decition_region
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from typing import Callable


def algebraic_product(x: float | int, y: float | int) -> float | int:
    """
    Calculate the algebraic product of two numbers.
    Args:
        x: The first number.
        y: The second number.
    Returns:
        The product of x and y.
    """
    return x*y


def algebraic_sum(x: float | int, y: float | int) -> float | int:
    """
    Calculate the algebraic sum of two numbers.
    Args:
        y: The second number.
        x: The first number.
    Returns:
        The algebraic sum of x and y.
    """
    return x + y - x*y


def build_universe_of_variables() -> dict:
    """
    Function to build the universe of variables.
    Returns:
        universe_of_variables (dict): The universe of variables.
    """
    universe_of_variables = dict()
    universe_of_variables['x_withdrawal_percentage'] = np.arange(0, 1, 0.05)
    universe_of_variables['x_hour'] = np.arange(0, 23, 0.1)
    universe_of_variables['x_transactions_per_day'] = np.arange(0, 50, 1)
    universe_of_variables['x_transactions_per_month'] = np.arange(0, 50, 1)
    universe_of_variables['y_risk'] = np.arange(0, 1, 0.01)

    return universe_of_variables


def build_membership_functions(universe_of_variables: dict) -> dict:
    """
    Build membership functions for the given universe of variables.
    Args:
        universe_of_variables (dict): A dictionary containing the universe of variables.
    Returns:
        membership_functions (dict): A dictionary containing the membership functions.
    """
    membership_functions = dict()
    # Sigmoid membership function
    membership_functions['w_per_lo'] = fuzz.sigmf(
        universe_of_variables['x_withdrawal_percentage'], 0.3, -40
        )
    # Generalized bell membership function
    membership_functions['w_per_md'] = fuzz.gbellmf(
        universe_of_variables['x_withdrawal_percentage'], 0.2, 4, 0.5
        )
    # Sigmoid membership function
    membership_functions['w_per_hi'] = fuzz.sigmf(
        universe_of_variables['x_withdrawal_percentage'], 0.7, 40
        )

    # Sigmoid membership function
    membership_functions['h_e_m'] = fuzz.sigmf(
        universe_of_variables['x_hour'], 6.5, -2
        )
    # Generalized bell membership function
    membership_functions['h_d'] = fuzz.gbellmf(
        universe_of_variables['x_hour'], 5, 5, 11.5
        )
    # Sigmoid membership function
    membership_functions['h_n'] = fuzz.sigmf(
        universe_of_variables['x_hour'], 16.5, 2
        )

    # Sigmoid membership function
    membership_functions['tran_p_day_l'] = fuzz.sigmf(
        universe_of_variables['x_transactions_per_day'], 5, -2
        )
    # Sigmoid membership function
    membership_functions['tran_p_day_h'] = fuzz.sigmf(
        universe_of_variables['x_transactions_per_day'], 5, 2
        )

    # Sigmoid membership function
    membership_functions['tran_p_month_l'] = fuzz.sigmf(
        universe_of_variables['x_transactions_per_month'], 40, -2
        )
    # Sigmoid membership function
    membership_functions['tran_p_month_h'] = fuzz.sigmf(
        universe_of_variables['x_transactions_per_month'], 40, 2
        )

    # Sigmoid membership function
    membership_functions['risk_lo'] = fuzz.sigmf(
        universe_of_variables['y_risk'], 0.3, -40
        )
    # Generalized bell membership function
    membership_functions['risk_md'] = fuzz.gbellmf(
        universe_of_variables['y_risk'], 0.2, 4, 0.5
        )
    # Sigmoid membership function
    membership_functions['risk_hi'] = fuzz.sigmf(
        universe_of_variables['y_risk'], 0.7, 40
        )

    return membership_functions


# Defining rules
# First, we need to interpolate the input to a value in the domain of the variables.
def interpolate_inputs(inputs: dict, data_dict: dict) -> dict:
    """
    Interpolates the inputs using fuzzy membership values from the data dictionary.

    Parameters:
        inputs: A dictionary of input values.
        data_dict: A dictionary containing the domain of the variables and their membership functions vectors.

    Returns:
        interpolated_inputs: A dictionary of interpolated input values.
    """
    interpolated_inputs = dict()
    # Membership value to fuzzy set withdrawal percentage low
    interpolated_inputs['w_per_lo'] = fuzz.interp_membership(
        data_dict['ax0']['x'],
        data_dict['ax0']['low'],
        inputs['withdrawal_percentage']
        )
    # Membership value to fuzzy set withdrawal percentage medium
    interpolated_inputs['w_per_md'] = fuzz.interp_membership(
        data_dict['ax0']['x'],
        data_dict['ax0']['medium'],
        inputs['withdrawal_percentage']
        )
    # Membership value to fuzzy set withdrawal percentage high
    interpolated_inputs['w_per_hi'] = fuzz.interp_membership(
        data_dict['ax0']['x'],
        data_dict['ax0']['high'],
        inputs['withdrawal_percentage']
        )

    # Membership value to fuzzy set hour early morning
    interpolated_inputs['h_e_m'] = fuzz.interp_membership(
        data_dict['ax1']['x'],
        data_dict['ax1']['early_morning'],
        inputs['hour']
        )
    # Membership value to fuzzy set hour day
    interpolated_inputs['h_d'] = fuzz.interp_membership(
        data_dict['ax1']['x'],
        data_dict['ax1']['day'],
        inputs['hour']
        )
    # Membership value to fuzzy set hour night
    interpolated_inputs['h_n'] = fuzz.interp_membership(
        data_dict['ax1']['x'],
        data_dict['ax1']['night'],
        inputs['hour']
        )

    # Membership value to fuzzy set transactions per day low
    interpolated_inputs['tran_p_day_l'] = fuzz.interp_membership(
        data_dict['ax2']['x'],
        data_dict['ax2']['low'],
        inputs['transactions_per_day']
        )
    # Membership value to fuzzy set transactions per day high
    interpolated_inputs['tran_p_day_h'] = fuzz.interp_membership(
        data_dict['ax2']['x'],
        data_dict['ax2']['high'],
        inputs['transactions_per_day']
        )

    # Membership value to fuzzy set transactions per month low
    interpolated_inputs['tran_p_month_l'] = fuzz.interp_membership(
        data_dict['ax3']['x'],
        data_dict['ax3']['low'],
        inputs['transactions_per_month']
        )
    # Membership value to fuzzy set transactions per month high
    interpolated_inputs['tran_p_month_h'] = fuzz.interp_membership(
        data_dict['ax3']['x'],
        data_dict['ax3']['high'],
        inputs['transactions_per_month']
        )

    return interpolated_inputs


# Define operands between the sets to get the antecedent
def build_rules(interpolated_inputs: dict, t_norm: Callable = np.fmin, s_norm: Callable = np.fmax) -> dict:
    """
    Function to build rules based on interpolated inputs using t_norm and s_norm functions.

    Args:
        interpolated_inputs (dict): A dictionary of interpolated inputs.
        t_norm (Callable, optional): T-norm function. Defaults to np.fmin.
        s_norm (Callable, optional): S-norm function. Defaults to np.fmax.

    Returns:
        dict: A dictionary of rules.
    """
    rules = dict()

    rules[1] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_lo'], interpolated_inputs['h_e_m']),
                             interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_l'])
    rules[2] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_lo'], interpolated_inputs['h_e_m']),
                             interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_h'])
    rules[3] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_lo'], interpolated_inputs['h_e_m']),
                             interpolated_inputs['tran_p_day_h']), interpolated_inputs['tran_p_month_h'])
    rules[4] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_lo'], interpolated_inputs['h_d']),
                             interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_l'])
    rules[5] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_lo'], interpolated_inputs['h_d']),
                             interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_h'])
    rules[6] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_lo'], interpolated_inputs['h_d']),
                             interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_h'])
    rules[7] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_lo'], interpolated_inputs['h_n']),
                             interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_l'])
    rules[8] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_lo'], interpolated_inputs['h_n']),
                             interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_h'])
    rules[9] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_lo'], interpolated_inputs['h_n']),
                             interpolated_inputs['tran_p_day_h']), interpolated_inputs['tran_p_month_h'])

    rules[10] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_md'], interpolated_inputs['h_e_m']),
                              interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_l'])
    rules[11] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_md'], interpolated_inputs['h_e_m']),
                              interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_h'])
    rules[12] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_md'], interpolated_inputs['h_e_m']),
                              interpolated_inputs['tran_p_day_h']), interpolated_inputs['tran_p_month_h'])
    rules[13] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_md'], interpolated_inputs['h_d']),
                              interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_l'])
    rules[14] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_md'], interpolated_inputs['h_d']),
                              interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_h'])
    rules[15] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_md'], interpolated_inputs['h_d']),
                              interpolated_inputs['tran_p_day_h']), interpolated_inputs['tran_p_month_h'])
    rules[16] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_md'], interpolated_inputs['h_n']),
                              interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_l'])
    rules[17] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_md'], interpolated_inputs['h_n']),
                              interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_h'])
    rules[18] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_md'], interpolated_inputs['h_n']),
                              interpolated_inputs['tran_p_day_h']), interpolated_inputs['tran_p_month_h'])

    rules[19] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_hi'], interpolated_inputs['h_d']),
                              interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_l'])
    rules[20] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_hi'], interpolated_inputs['h_d']),
                              interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_h'])
    rules[21] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_hi'], interpolated_inputs['h_d']),
                              interpolated_inputs['tran_p_day_h']), interpolated_inputs['tran_p_month_h'])

    # Rules that simplify the output
    # 1) If the withdrawal percentage is high, and the hour is early_morning
    # or night, then the risk is high
    rules[22] = t_norm(interpolated_inputs['w_per_hi'], s_norm(interpolated_inputs['h_e_m'],
                                                               interpolated_inputs['h_n']))
    # 2) If the transactions per day is high, and the transactions per month
    # is low, then the risk is high
    rules[23] = t_norm(interpolated_inputs['tran_p_day_h'], interpolated_inputs['tran_p_month_l'])

    return rules


# We join the rules associated with each set in the output.
def build_betas(rules: dict) -> dict:
    """
    Function to build betas based on the given rules dictionary.

    Parameters:
    - rules: a dictionary containing the rules

    Returns:
    - betas: a dictionary containing the joined rules with the same output.
    """
    betas = {
        'beta_risk_low': [rules[4], rules[5], rules[6], rules[7], rules[8],
                          rules[13], rules[14], rules[15], rules[17],
                          rules[19], rules[20], rules[21]],
        'beta_risk_medium': [rules[1], rules[2], rules[3], rules[9],
                             rules[10], rules[11], rules[12], rules[16],
                             rules[17]],
        'beta_risk_high': [rules[22], rules[23]]
    }

    return betas


# We aggregate the rules. We get the output limits of the system.
def build_limits(betas: dict, membership_functions: dict) -> dict:
    """
    Build limits based on betas and membership functions and return the limits as a dictionary.

    :param betas: A dictionary containing beta values.
    :param membership_functions: A dictionary containing membership functions.
    :return: A dictionary containing the calculated limits.
    """
    limits = dict()
    limits['low'] = np.fmin(max(betas['beta_risk_low']), membership_functions['risk_lo'])
    limits['medium'] = np.fmin(max(betas['beta_risk_medium']), membership_functions['risk_md'])
    limits['high'] = np.fmin(max(betas['beta_risk_high']), membership_functions['risk_hi'])

    return limits


# Aggregate all three output membership functions together
def aggregate_membership_functions(limits: dict) -> np.ndarray:
    """
    Function to aggregate membership functions based on the given limits.

    :param limits: A dictionary containing the limits for different membership functions.
    :type limits: dict

    :return: An array containing the aggregated membership functions.
    :rtype: np.ndarray
    """
    aggregated_membership_functions = np.fmax(
        np.fmax(limits['low'], limits['medium']),
        limits['high']
        )

    return aggregated_membership_functions


# Defuzzification
def defuzz(universe_of_variables: dict, aggregated_membership_functions: np.ndarray,
           method: str = 'centroid') -> float | int:
    """
    Calculate the defuzzification value using the specified method.

    :param universe_of_variables: A dictionary containing the universe of discourse for each input variable.
    :param aggregated_membership_functions: An array containing the aggregated membership functions for each output
        variable.
    :param method: The method to use for defuzzification. Defaults to 'centroid'.
    :return: The defuzzified value as a float or integer.
    """
    return fuzz.defuzz(universe_of_variables['y_risk'], aggregated_membership_functions, method)


def build_data_dict(universe_of_variables: dict, membership_functions: dict) -> dict:
    """
    Create a dictionary with the data using the provided universe_of_variables and membership_functions.
    It helps to organize the data to graph it and to process the inputs.

    :param universe_of_variables: dictionary of universe variables
    :param membership_functions: a dictionary of membership functions
    :return: a dictionary containing the data
    :rtype: dict
    """
    # Create a dictionary with the data
    data_dict = {
        'ax0': {
            'x': universe_of_variables['x_withdrawal_percentage'],
            'low': membership_functions['w_per_lo'],
            'medium': membership_functions['w_per_md'],
            'high': membership_functions['w_per_hi'],
            'title': 'Withdrawal Percentage'
            },
        'ax1': {
            'x': universe_of_variables['x_hour'],
            'early_morning': membership_functions['h_e_m'],
            'day': membership_functions['h_d'],
            'night': membership_functions['h_n'],
            'title': 'Hour of Day'
            },
        'ax2': {
            'x': universe_of_variables['x_transactions_per_day'],
            'low': membership_functions['tran_p_day_l'],
            'high': membership_functions['tran_p_day_h'],
            'title': 'Transactions per Day'
            },
        'ax3': {
            'x': universe_of_variables['x_transactions_per_month'],
            'low': membership_functions['tran_p_month_l'],
            'high': membership_functions['tran_p_month_h'],
            'title': 'Transactions per Month'
            },
        'ax4': {
            'x': universe_of_variables['y_risk'],
            'low': membership_functions['risk_lo'],
            'medium': membership_functions['risk_md'],
            'high': membership_functions['risk_hi'],
            'title': 'Risk'
            }
    }

    return data_dict


# Running all the system
def run_system(data_dict: dict, inputs: dict, universe_of_variables: dict, membership_functions: dict,
               defuzz_methods: list[str] = None, color_map: str = constants.color_map, t_norm: Callable = np.fmin,
               s_norm: Callable = np.fmax, graphics: bool = False, save_graphs: bool = False):
    if defuzz_methods is None:
        defuzz_methods = ['centroid']

    interpolated_inputs = interpolate_inputs(inputs, data_dict)
    rules = build_rules(interpolated_inputs, t_norm, s_norm)
    betas = build_betas(rules)
    limits = build_limits(betas, membership_functions)
    aggregated_membership_functions = aggregate_membership_functions(limits)

    # TODO: vectorize this
    list_defuzzified_risks = [
        defuzz(universe_of_variables, aggregated_membership_functions, defuzz_method)
        for defuzz_method in defuzz_methods
    ]

    dict_defuzzified_risks = dict(zip(defuzz_methods, list_defuzzified_risks))

    # Graphics
    if graphics:
        color_palette = plt.cm.get_cmap(color_map)
        graph_decition_region(universe_of_variables, membership_functions,
                              limits, aggregated_membership_functions,
                              color_palette=color_palette,
                              dict_defuzzified_risks=dict_defuzzified_risks,
                              t_norm_name=t_norm.__name__,
                              s_norm_name=s_norm.__name__,
                              save_graphs=save_graphs)

    return dict_defuzzified_risks
