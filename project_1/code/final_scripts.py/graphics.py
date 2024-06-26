
from constants import *
import fuzzy_functions
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os

# Plot the functions
def visualize_membership_functions(data_dict, dictionary_colors=None,
                                   color_palette=color_map, set_fill_area=None,
                                   set_bold_lines=None, set_dashed_lines=None,
                                   save_graphs=False,
                                   t_norm_name=None, s_norm_name=None):
    # Define a color palette
    print(color_palette)
    color_palette = plt.cm.get_cmap(color_palette)

    # Visualize these universes and membership functions
    fig, axes = plt.subplots(nrows=len(data_dict), figsize=(6, 2 * (len(data_dict) + 1)))

    if set_bold_lines:
        alpha = 0.7
    else:
        alpha = 1

    for i, (ax_name, data) in enumerate(data_dict.items()):
        if isinstance(axes, np.ndarray):
            ax = axes[i]
        else:
            ax = axes

        for j, (category, values) in enumerate(data.items()):
            if category == 'x' or category == 'title':
                continue  # Skip 'x' and 'title' in dictionary

            if dictionary_colors and category in dictionary_colors:
                color = dictionary_colors[category]
            else:
                color = color_palette(j/(len(data) - 1))

            if set_dashed_lines and category in set_dashed_lines:
                line = '--'
            else:
                line = '-'

            if set_bold_lines and category in set_bold_lines:
                ax.plot(data['x'], values, linewidth=3, color='k')
            elif category in set_fuzzy_methods:
                print(values)
                ax.axvline(x=values, linestyle=line, label=category,
                           linewidth=1.7, color=color)
            # Check if category is a string before adding tag. If not,
            # it possibly is a rule.
            elif isinstance(category, str):
                ax.plot(data['x'], values, line, label=category, linewidth=1.5,
                        color=color, alpha=alpha)
            else:
                ax.plot(data['x'], values, '--', linewidth=1.5, color=color,
                        alpha=alpha)

            if set_fill_area and category in set_fill_area:
                ax.fill_between(data['x'], 0, values, color=color, alpha=0.5)

        ax.plot(data['x'], 0.5*np.ones_like(data['x']), '--k', label="Min value mem. func.")
        ax.set_title(data['title'])

        # Display the legend in a place that does not obstruct the data
        ax.legend(loc='upper right')

        if save_graphs:
            name = f'Risk with different defuzzification methods and t-norm: {t_norm_name} s-norm: {s_norm_name}'
            directory = 'images/'

            # Verificar si el directorio existe, si no, crearlo
            if not os.path.exists(directory):
                os.makedirs(directory)

            fig.savefig(f'{directory}{name}.png')

    plt.tight_layout()
    plt.show()


# Graphics of the α-cuts (alpha cuts)

# Graphic of all the alphas
def graph_all_alphas(universe_of_variables, membership_functions, rules,
                     color_palette=None):
    # Create a dictionary with the data
    data_dict_all_alpha_cuts = {
        'ax0': {
            'x': universe_of_variables['y_risk'],
            'low': membership_functions['risk_lo'],
            'medium': membership_functions['risk_md'],
            'high': membership_functions['risk_hi'],
            'title': 'Risk',
            }
    }


    for key, value in rules.items():
        data_dict_all_alpha_cuts['ax0'][key] = value*np.ones_like(
            data_dict_all_alpha_cuts['ax0']['x']
            )

    # Create a dictionary with colors for specific categories
    if not color_palette:
        color_palette = plt.cm.get_cmap(color_map)
    dictionary_colors = {
        'low': color_palette(0.2),
        'medium': color_palette(0.6),
        'high': color_palette(0.9),
    }


    # Call the function with the dictionary as an argument
    visualize_membership_functions(data_dict_all_alpha_cuts,
                                  dictionary_colors=dictionary_colors,
                                  color_palette=color_palette)


# Graphic of the largest alphas
def graph_max_alphas(universe_of_variables, membership_functions, betas,
                     color_palette=None):
    # Create a dictionary with the data
    data_dict_max_alpha_cuts = {
        'ax0': {
            'x': universe_of_variables['y_risk'],
            'low': membership_functions['risk_lo'],
            'medium': membership_functions['risk_md'],
            'high': membership_functions['risk_hi'],
            'title': 'Risk'
            }
    }


    for key, values in betas.items():
        max_val = np.max(values)
        data_dict_max_alpha_cuts['ax0'][key] = max_val*np.ones_like(
            data_dict_max_alpha_cuts['ax0']['x']
            )

    # Create a dictionary with colors for specific categories
    if not color_palette:
        color_palette = plt.cm.get_cmap(color_map)
    dictionary_colors = {
        'low': color_palette(0.2),
        'beta_risk_low': color_palette(0.2),
        'medium': color_palette(0.6),
        'beta_risk_medium': color_palette(0.6),
        'high': color_palette(0.9),
        'beta_risk_high': color_palette(0.9)
    }

    # Call the function with the dictionary as an argument
    visualize_membership_functions(data_dict_max_alpha_cuts,
                                  dictionary_colors=dictionary_colors,
                                  color_palette=color_palette)

# Graphic of the min between the biggest alphas and the membership function

def graph_decition_region(universe_of_variables, membership_functions,
                          limits, aggregated_membership_functions,
                          color_palette=None,
                          dict_defuzzified_risks=None,
                          t_norm_name=None, s_norm_name=None,
                          save_graphs=False):
    # Create a dictionary with the data
    data_dict_max_min_alpha_cuts = {
        'ax0': {
            'x': universe_of_variables['y_risk'],
            'low': membership_functions['risk_lo'],
            'medium': membership_functions['risk_md'],
            'high': membership_functions['risk_hi'],
            'low_alpha': limits['low'],
            'medium_alpha': limits['medium'],
            'high_alpha': limits['high'],
            'beta_risk_bold': aggregated_membership_functions,
            'title': 'Risk with dicision region'
            }
    }

    if t_norm_name and s_norm_name:
        data_dict_max_min_alpha_cuts['ax0']['title'] = f"Risk with different defuzzification methods \n and t-norm: {t_norm_name}, s-norm: {s_norm_name}"

    if not color_palette:
        color_palette = plt.cm.get_cmap(color_map)

    same_color = [('low', 'low_alpha'), ('medium', 'medium_alpha'),
                  ('high', 'high_alpha')]

    set_dashed_lines = None
    if dict_defuzzified_risks:
        data_dict_max_min_alpha_cuts['ax0'].update(dict_defuzzified_risks)
        list_defuzz_methods = list(dict_defuzzified_risks.keys())
        set_dashed_lines = set(list_defuzz_methods)

        for defuzz_method in list_defuzz_methods:
            same_color.append((defuzz_method))

    num_same_color = len(same_color)
    colors = [
        (same_color[i][j], color_palette(i / (num_same_color - 1)))
        for i in range(num_same_color)
        for j in range(len(same_color[i]))
        ]
    # dictionary_colors = {
    #     'low': color_palette(0.2),
    #     'low_alpha': color_palette(0.2),
    #     'medium': color_palette(0.6),
    #     'medium_alpha': color_palette(0.6),
    #     'high': color_palette(0.9),
    #     'high_alpha': color_palette(0.9)
    # }

    dictionary_colors = dict(colors)

    set_bold_lines = set(['beta_risk_bold'])
    set_fill_area = set(['low_alpha', 'medium_alpha', 'high_alpha'])

    # Call the function with the dictionary as an argument
    visualize_membership_functions(data_dict_max_min_alpha_cuts,
                                  dictionary_colors=dictionary_colors,
                                  color_palette=color_palette,
                                  set_bold_lines=set_bold_lines,
                                  set_fill_area=set_fill_area,
                                  set_dashed_lines=set_dashed_lines,
                                  save_graphs=save_graphs,
                                  t_norm_name=t_norm_name,
                                  s_norm_name=s_norm_name)


# # Line showing the resulting decition surface"""
# # Create a dictionary with the data
# data_dict_max_alpha_cuts = {
#     'ax0': {
#         'x': universe_of_variables['y_risk'],
#         'low': membership_functions['risk_lo'],
#         'medium': membership_functions['risk_md'],
#         'high': membership_functions['risk_hi'],
#         'beta_risk_bold': aggregated_membership_functions,
#         'title': 'Risk'
#         }
# }


# for key, values in betas.items():
#     max_val = np.max(values)
#     data_dict_max_alpha_cuts['ax0'][key] = max_val*np.ones_like(
#         data_dict_max_alpha_cuts['ax0']['x']
#         )

# # Create a dictionary with colors for specific categories
# color_palette = plt.cm.get_cmap(color_map)
# colors_dict = {
#     'low': color_palette(0.2),
#     'beta_risk_low': color_palette(0.2),
#     'medium': color_palette(0.6),
#     'beta_risk_medium': color_palette(0.6),
#     'high': color_palette(0.9),
#     'beta_risk_high': color_palette(0.9)
# }

# set_bold_lines = set(['beta_risk_bold'])
# set_fill_area = set(['beta_risk_bold'])

# # Call the function with the dictionary as an argument
# visualize_membership_functions(data_dict_max_alpha_cuts,
#                                dictionary_colors=colors_dict,
#                                color_palette=color_palette,
#                                set_fill_area=set_fill_area,
#                                set_bold_lines=set_bold_lines)


def parce_run_system_result_to_matrix(results, defuzz_method):
    z = np.zeros_like(results, dtype=float)
    z = [[element[defuzz_method] for element in results[i]] for i in range(len(results))]
    return np.array(z)


def graph_response_surface(variable_x, variable_y, animation,
                           variables, data_dict,
                           universe_of_variables, membership_functions,
                           color_map=color_map,
                           defuzz_methods=['centroid'],
                           t_norm=np.fmin, s_norm=np.fmax,
                           animation_velocity=1,
                           save_graphs=False,
                           save_animation=False):

    variables[variable_x], variables[variable_y] = np.meshgrid(
        variables[variable_x], variables[variable_y])

    temp_array = []
    if animation:
        temp_array = universe_of_variables[animation]
        variables[animation] = temp_array[0]

    # Create an empty array for z with the same shape as x or y
    results = np.zeros_like(variables[variable_x])

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

    results = vectorized_run_system(variables['x_withdrawal_percentage'],
                              variables['x_hour'],
                              variables['x_transactions_per_day'],
                              variables['x_transactions_per_month'])

    for defuzz_method in defuzz_methods:
        z = parce_run_system_result_to_matrix(results, defuzz_method)

        # Create the 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Create the surface graph
        #surface = ax.plot_surface(x, y, z, cmap='rainbow')
        surface = ax.plot_surface(variables[variable_x], variables[variable_y],
                                z, cmap=color_map)

        # Add a color bar
        fig.colorbar(surface)

        # Set up labels
        ax.set_xlabel(variable_x)
        ax.set_ylabel(variable_y)
        ax.set_zlabel('Risk')

        #title = f'Response surface - defuzz_method: {defuzz_method} \n t_norm: {t_norm.__name__}, s_norm: {s_norm.__name__}'
        ax.set_title(f'Response surface - {defuzz_method} - animation: {animation}')
        ax.set_zlim(0,1)

        # Rotate the axes
        elev = 30
        azim = -135
        ax.view_init(elev=elev, azim=azim)

        if save_graphs:
            directory = 'images/'

            # Verificar si el directorio existe, si no, crearlo
            if not os.path.exists(directory):
                os.makedirs(directory)

            fig.savefig(f'{directory}Surface - defuzz_method: {defuzz_method} t_norm: {t_norm.__name__} s_norm: {s_norm.__name__}-{variable_x}-{variable_y}.png')

        # Función de actualización para la animación
        def update(frame):
            print(f"Frame: {frame}")
            ax.cla()  # Limpiar el eje para cada frame
            # Set up labels
            ax.set_xlabel(variable_x)
            ax.set_ylabel(variable_y)
            ax.set_zlabel('Risk')
            ax.set_title(f'Response surface - {defuzz_method} - animation: {animation}')
            ax.set_zlim(0,1)

            ax.view_init(elev=elev, azim=azim)

            variables[animation] = temp_array[frame]
            results_2 = vectorized_run_system(variables['x_withdrawal_percentage'],
                                variables['x_hour'],
                                variables['x_transactions_per_day'],
                                variables['x_transactions_per_month'])
            z = parce_run_system_result_to_matrix(results_2, defuzz_method)

            surface = ax.plot_surface(variables[variable_x], variables[variable_y],
                                    z, cmap=color_map)
            return surface

        if animation:
            # Crear la animación
            anim = FuncAnimation(fig, update, frames=len(temp_array),
                                interval=animation_velocity, repeat=False)

            if save_animation:
                directory = 'images/'

                # Verificar si el directorio existe, si no, crearlo
                if not os.path.exists(directory):
                    os.makedirs(directory)

                anim.save(f'{directory}An - def_meth: {defuzz_method} t_norm: {t_norm.__name__} s_norm: {s_norm.__name__}-{variable_x}-{variable_y} animation - {animation}.gif', writer='pillow')


        # Show the graph
        plt.show()
