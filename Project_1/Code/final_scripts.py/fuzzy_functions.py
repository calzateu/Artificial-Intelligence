
from constants import *
from graphics import graph_all_alphas, graph_max_alphas, graph_decition_region
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz


def algebraic_product(x, y):
    return x*y

def algebraic_sum(x, y):
    return x + y - x*y


# Generate universe variables
def build_universe_of_variables():
    universe_of_variables = dict()
    universe_of_variables['x_withdrawal_percentage'] = np.arange(0, 1, 0.05)
    universe_of_variables['x_hour'] = np.arange(0, 23, 0.1)
    universe_of_variables['x_transactions_per_day'] = np.arange(0, 50, 1)
    universe_of_variables['x_transactions_per_month'] = np.arange(0, 50, 1)
    universe_of_variables['y_risk'] = np.arange(0, 1, 0.01)

    return universe_of_variables


# Defining membership functions
def build_membership_functions(universe_of_variables):
    membership_functions = dict()
    membership_functions['w_per_lo'] = fuzz.sigmf(
        universe_of_variables['x_withdrawal_percentage'], 0.3, -40
        )
    membership_functions['w_per_md'] = fuzz.gbellmf(
        universe_of_variables['x_withdrawal_percentage'], 0.2, 4, 0.5
        )
    membership_functions['w_per_hi'] = fuzz.sigmf(
        universe_of_variables['x_withdrawal_percentage'], 0.7, 40
        )

    membership_functions['h_e_m'] = fuzz.sigmf(
        universe_of_variables['x_hour'], 6.5, -2
        )
    membership_functions['h_d'] = fuzz.gbellmf(
        universe_of_variables['x_hour'], 5, 5, 11.5
        )
    membership_functions['h_n'] = fuzz.sigmf(
        universe_of_variables['x_hour'], 16.5, 2
        )

    membership_functions['tran_p_day_l'] = fuzz.sigmf(
        universe_of_variables['x_transactions_per_day'], 5, -2
        )
    membership_functions['tran_p_day_h'] = fuzz.sigmf(
        universe_of_variables['x_transactions_per_day'], 5, 2
        )

    membership_functions['tran_p_month_l'] = fuzz.sigmf(
        universe_of_variables['x_transactions_per_month'], 40, -2
        )
    membership_functions['tran_p_month_h'] = fuzz.sigmf(
        universe_of_variables['x_transactions_per_month'], 40, 2
        )

    membership_functions['risk_lo'] = fuzz.sigmf(
        universe_of_variables['y_risk'], 0.3, -40
        )
    membership_functions['risk_md'] = fuzz.gbellmf(
        universe_of_variables['y_risk'], 0.2, 4, 0.5
        )
    membership_functions['risk_hi'] = fuzz.sigmf(
        universe_of_variables['y_risk'], 0.7, 40
        )

    return membership_functions


# Defining rules
# First, we need to interpolate the input to a value in the domain of the variables.
def interpolate_inputs(inputs, data_dict):
    interpolated_inputs = dict()
    # withdrawal_percentage
    interpolated_inputs['w_per_lo'] = fuzz.interp_membership(
        data_dict['ax0']['x'],
        data_dict['ax0']['low'],
        inputs['withdrawal_percentage']
        )
    interpolated_inputs['w_per_md'] = fuzz.interp_membership(
        data_dict['ax0']['x'],
        data_dict['ax0']['medium'],
        inputs['withdrawal_percentage']
        )
    interpolated_inputs['w_per_hi'] = fuzz.interp_membership(
        data_dict['ax0']['x'],
        data_dict['ax0']['high'],
        inputs['withdrawal_percentage']
        )

    # hour
    interpolated_inputs['h_e_m'] = fuzz.interp_membership(
        data_dict['ax1']['x'],
        data_dict['ax1']['early_morning'],
        inputs['hour']
        )
    interpolated_inputs['h_d'] = fuzz.interp_membership(
        data_dict['ax1']['x'],
        data_dict['ax1']['day'],
        inputs['hour']
        )
    interpolated_inputs['h_n'] = fuzz.interp_membership(
        data_dict['ax1']['x'],
        data_dict['ax1']['night'],
        inputs['hour']
        )

    # transactions_per_day
    interpolated_inputs['tran_p_day_l'] = fuzz.interp_membership(
        data_dict['ax2']['x'],
        data_dict['ax2']['low'],
        inputs['transactions_per_day']
        )
    interpolated_inputs['tran_p_day_h'] = fuzz.interp_membership(
        data_dict['ax2']['x'],
        data_dict['ax2']['high'],
        inputs['transactions_per_day']
        )

    # transactions_per_month
    interpolated_inputs['tran_p_month_l'] = fuzz.interp_membership(
        data_dict['ax3']['x'],
        data_dict['ax3']['low'],
        inputs['transactions_per_month']
        )
    interpolated_inputs['tran_p_month_h'] = fuzz.interp_membership(
        data_dict['ax3']['x'],
        data_dict['ax3']['high'],
        inputs['transactions_per_month']
        )

    return interpolated_inputs

#Then, we can define the operands between the sets to obtain the antecedent
def build_rules(interpolated_inputs, t_norm=np.fmin, s_norm=np.fmax):
    rules = dict()

    # # First block with low withdrawal percentage and all the other combinations
    # rules[1] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_lo'], interpolated_inputs['h_e_m']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_l'])
    # rules[2] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_lo'], interpolated_inputs['h_e_m']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_h'])
    # rules[3] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_lo'], interpolated_inputs['h_e_m']), interpolated_inputs['tran_p_day_h']), interpolated_inputs['tran_p_month_l'])
    # rules[4] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_lo'], interpolated_inputs['h_e_m']), interpolated_inputs['tran_p_day_h']), interpolated_inputs['tran_p_month_h'])
    # rules[5] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_lo'], interpolated_inputs['h_d']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_l'])
    # rules[6] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_lo'], interpolated_inputs['h_d']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_h'])
    # rules[7] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_lo'], interpolated_inputs['h_d']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_l'])
    # rules[8] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_lo'], interpolated_inputs['h_d']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_h'])
    # rules[9] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_lo'], interpolated_inputs['h_n']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_l'])
    # rules[10] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_lo'], interpolated_inputs['h_n']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_h'])
    # rules[11] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_lo'], interpolated_inputs['h_n']), interpolated_inputs['tran_p_day_h']), interpolated_inputs['tran_p_month_l'])
    # rules[12] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_lo'], interpolated_inputs['h_n']), interpolated_inputs['tran_p_day_h']), interpolated_inputs['tran_p_month_h'])

    # # Second block with medium withdrawal percentage and all the other combinations
    # rules[13] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_md'], interpolated_inputs['h_e_m']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_l'])
    # rules[14] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_md'], interpolated_inputs['h_e_m']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_h'])
    # rules[15] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_md'], interpolated_inputs['h_e_m']), interpolated_inputs['tran_p_day_h']), interpolated_inputs['tran_p_month_l'])
    # rules[16] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_md'], interpolated_inputs['h_e_m']), interpolated_inputs['tran_p_day_h']), interpolated_inputs['tran_p_month_h'])
    # rules[17] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_md'], interpolated_inputs['h_d']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_l'])
    # rules[18] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_md'], interpolated_inputs['h_d']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_h'])
    # rules[19] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_md'], interpolated_inputs['h_d']), interpolated_inputs['tran_p_day_h']), interpolated_inputs['tran_p_month_l'])
    # rules[20] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_md'], interpolated_inputs['h_d']), interpolated_inputs['tran_p_day_h']), interpolated_inputs['tran_p_month_h'])
    # rules[21] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_md'], interpolated_inputs['h_n']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_l'])
    # rules[22] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_md'], interpolated_inputs['h_n']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_h'])
    # rules[23] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_md'], interpolated_inputs['h_n']), interpolated_inputs['tran_p_day_h']), interpolated_inputs['tran_p_month_l'])
    # rules[24] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_md'], interpolated_inputs['h_n']), interpolated_inputs['tran_p_day_h']), interpolated_inputs['tran_p_month_h'])

    # # Third block with high withdrawal percentage and all the other combinations
    # rules[25] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_hi'], interpolated_inputs['h_e_m']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_l'])
    # rules[26] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_hi'], interpolated_inputs['h_e_m']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_h'])
    # rules[27] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_hi'], interpolated_inputs['h_e_m']), interpolated_inputs['tran_p_day_h']), interpolated_inputs['tran_p_month_l'])
    # rules[28] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_hi'], interpolated_inputs['h_e_m']), interpolated_inputs['tran_p_day_h']), interpolated_inputs['tran_p_month_h'])
    # rules[29] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_hi'], interpolated_inputs['h_d']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_l'])
    # rules[30] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_hi'], interpolated_inputs['h_d']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_h'])
    # rules[31] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_hi'], interpolated_inputs['h_d']), interpolated_inputs['tran_p_day_h']), interpolated_inputs['tran_p_month_l'])
    # rules[32] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_hi'], interpolated_inputs['h_d']), interpolated_inputs['tran_p_day_h']), interpolated_inputs['tran_p_month_h'])
    # rules[33] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_hi'], interpolated_inputs['h_n']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_l'])
    # rules[34] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_hi'], interpolated_inputs['h_n']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_h'])
    # rules[35] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_hi'], interpolated_inputs['h_n']), interpolated_inputs['tran_p_day_h']), interpolated_inputs['tran_p_month_l'])
    # rules[36] = np.fmin(np.fmin(np.fmin(interpolated_inputs['w_per_hi'], interpolated_inputs['h_n']), interpolated_inputs['tran_p_day_h']), interpolated_inputs['tran_p_month_h'])

    # First block with low withdrawal percentage and all the other combinations
    rules[1] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_lo'], interpolated_inputs['h_e_m']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_l'])
    rules[2] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_lo'], interpolated_inputs['h_e_m']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_h'])
    rules[3] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_lo'], interpolated_inputs['h_e_m']), interpolated_inputs['tran_p_day_h']), interpolated_inputs['tran_p_month_h'])
    rules[4] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_lo'], interpolated_inputs['h_d']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_l'])
    rules[5] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_lo'], interpolated_inputs['h_d']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_h'])
    rules[6] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_lo'], interpolated_inputs['h_d']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_h'])
    rules[7] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_lo'], interpolated_inputs['h_n']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_l'])
    rules[8] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_lo'], interpolated_inputs['h_n']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_h'])
    rules[9] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_lo'], interpolated_inputs['h_n']), interpolated_inputs['tran_p_day_h']), interpolated_inputs['tran_p_month_h'])

    # Second block with medium withdrawal percentage and all the other combinations
    rules[10] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_md'], interpolated_inputs['h_e_m']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_l'])
    rules[11] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_md'], interpolated_inputs['h_e_m']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_h'])
    rules[12] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_md'], interpolated_inputs['h_e_m']), interpolated_inputs['tran_p_day_h']), interpolated_inputs['tran_p_month_h'])
    rules[13] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_md'], interpolated_inputs['h_d']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_l'])
    rules[14] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_md'], interpolated_inputs['h_d']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_h'])
    rules[15] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_md'], interpolated_inputs['h_d']), interpolated_inputs['tran_p_day_h']), interpolated_inputs['tran_p_month_h'])
    rules[16] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_md'], interpolated_inputs['h_n']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_l'])
    rules[17] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_md'], interpolated_inputs['h_n']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_h'])
    rules[18] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_md'], interpolated_inputs['h_n']), interpolated_inputs['tran_p_day_h']), interpolated_inputs['tran_p_month_h'])

    # Third block with high withdrawal percentage and all the other combinations
    rules[19] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_hi'], interpolated_inputs['h_d']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_l'])
    rules[20] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_hi'], interpolated_inputs['h_d']), interpolated_inputs['tran_p_day_l']), interpolated_inputs['tran_p_month_h'])
    rules[21] = t_norm(t_norm(t_norm(interpolated_inputs['w_per_hi'], interpolated_inputs['h_d']), interpolated_inputs['tran_p_day_h']), interpolated_inputs['tran_p_month_h'])

    # Rules that simplify the output
    # 1) If the withdrawal percentage is high, and the hour is early_morning
    # or night, then the risk is high
    rules[22] = t_norm(interpolated_inputs['w_per_hi'], s_norm(interpolated_inputs['h_e_m'], interpolated_inputs['h_n']))
    # 2) If the transactions per day is high, and the transactions per month
    # is low, then the risk is high
    rules[23] = t_norm(interpolated_inputs['tran_p_day_h'], interpolated_inputs['tran_p_month_l'])

    return rules


# We join the rules associated with each set in the output
def build_betas(rules):
    # betas = {
    #     'beta_risk_low': [rules[5], rules[6], rules[8], rules[9],
    #                           rules[10], rules[17], rules[18], rules[20],
    #                           rules[22], rules[29], rules[30], rules[32]],
    #     'beta_risk_medium': [rules[1], rules[2], rules[4], rules[7],
    #                           rules[12], rules[13], rules[14], rules[16],
    #                           rules[20], rules[21], rules[24]],
    #     'beta_risk_high': [rules[3], rules[11], rules[15], rules[23],
    #                           rules[25], rules[26], rules[27], rules[28],
    #                           rules[31], rules[33], rules[34], rules[35],
    #                           rules[36]]
    # }

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


# We aggregate the rules
def build_limits(betas, membership_functions):
    limits = dict()
    limits['low'] = np.fmin(max(betas['beta_risk_low']), membership_functions['risk_lo'])
    limits['medium'] = np.fmin(max(betas['beta_risk_medium']), membership_functions['risk_md'])
    limits['high'] = np.fmin(max(betas['beta_risk_high']), membership_functions['risk_hi'])

    return limits


# Aggregate all three output membership functions together
def aggregate_membership_functions(limits):
    aggregated_membership_functions = np.fmax(
        np.fmax(limits['low'], limits['medium']),
        limits['high']
        )

    return aggregated_membership_functions


# Defuzzification
def defuzz(universe_of_variables, aggregated_membership_functions, method='centroid'):
    return fuzz.defuzz(universe_of_variables['y_risk'],
                   aggregated_membership_functions, method)


def build_data_dict(universe_of_variables, membership_functions):
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
def run_system(data_dict, inputs, universe_of_variables, membership_functions,
               defuzz_methods=['centroid'],
               color_map=color_map, t_norm=np.fmin, s_norm=np.fmax,
               graphics=False, save_graphs=False):
    interpolated_inputs = interpolate_inputs(inputs, data_dict)
    rules = build_rules(interpolated_inputs, t_norm, s_norm)
    betas = build_betas(rules)
    limits = build_limits(betas, membership_functions)
    aggregated_membership_functions = aggregate_membership_functions(limits)
    list_defuzzified_risks = [
        defuzz(universe_of_variables, aggregated_membership_functions,
               defuzz_method)
               for defuzz_method in defuzz_methods
               ]

    dict_defuzzified_risks = dict(zip(defuzz_methods, list_defuzzified_risks))

    # Graphics
    if graphics:
        color_palette = plt.cm.get_cmap(color_map)
        # graph_all_alphas(universe_of_variables, membership_functions, rules,
        #                  color_palette=color_palette)
        # graph_max_alphas(universe_of_variables, membership_functions, betas,
        #                  color_palette=color_palette)
        graph_decition_region(universe_of_variables, membership_functions,
                              limits, aggregated_membership_functions,
                              color_palette=color_palette,
                              dict_defuzzified_risks=dict_defuzzified_risks,
                              t_norm_name=t_norm.__name__,
                              s_norm_name=s_norm.__name__,
                              save_graphs=save_graphs)

    return dict_defuzzified_risks
