from constants import *
import fuzzy_functions
import input_output as io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import os
import seaborn as sns


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


def grap_distance_matrix(distances: np.ndarray, method_name: str = "euclidean", save_graphs: bool = False,
                         x_labels=None, y_labels=None, x_name: str = "", y_name: str = "") -> None:
    """
    A function to generate a heatmap of the distances matrix using Seaborn and Matplotlib.

    Parameters:
        distances (np.ndarray): The input distances matrix.
        method_name (str): The name of the distance function (default is "euclidean").
        x_labels (list or None): List of labels for the x-axis.
        y_labels (list or None): List of labels for the y-axis.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    if x_labels is not None:
        sns.heatmap(distances, cmap="YlGnBu", fmt=".1f", linewidths=.5, xticklabels=x_labels, yticklabels=y_labels)
        plt.xlabel(f"Parameter {x_name}")
    else:
        sns.heatmap(distances, cmap="YlGnBu", fmt=".1f", linewidths=.5, yticklabels=y_labels)

    plt.ylabel(f"Parameter {y_name}")
    plt.title(f"Parameters matrix of {method_name} distance function")

    if save_graphs:
        output_path = io.__path_to_data()
        output_dir = os.path.join(output_path, "graphics")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_image = os.path.join(output_dir, f"distances_matrix_{method_name}.png")
        plt.savefig(output_image)

    plt.show()


def plot_clusters(data, cluster_centers, center_points, labels, axes, plot_name, ax):
    """
    Plots the clusters for a single dataset.

    Parameters:
        data: pandas DataFrame containing the data points
        cluster_centers: list of integers representing the cluster centers
        center_points: list of values representing the center points
        labels: list of integers representing the cluster labels
        axes: list of strings representing axes
        plot_name: name of the plot (dimensionality reduction method)
        ax: matplotlib axis to plot on

    Returns:
        None
    """
    ax.set_title(plot_name)

    # Go through the data and select the points that belong to each cluster and plot them
    # with a different color
    for label in cluster_centers:
        # Select data points that have this label
        indexes = np.where(labels == label)[0]
        # Extract the data points of this cluster
        cluster_data = data.iloc[indexes]
        # Plot the data points. For that, we need to transpose the data points to have the columns on the rows
        # and unroll them with the * operator. With this, we can plot the data points independently of the
        # dimensionality (2D or 3D)
        ax.scatter(*cluster_data[axes].values.T, label=f'Cluster {label}')

    # Plot the cluster centers with a different color
    ax.scatter(*center_points.T, c='black', s=200, alpha=0.5)

    # Add labels to the cluster centers
    for i, txt in enumerate(cluster_centers):
        ax.text(*center_points[i], f'{txt}', fontsize=12, color='black', ha='right')

    # Set axes labels
    ax.set_xlabel(axes[0])
    ax.set_ylabel(axes[1])
    if len(axes) == 3:
        ax.set_zlabel(axes[2])
    ax.legend()


def graph_clustering_results(data: pd.DataFrame, cluster_centers: list[int], center_points: np.ndarray,
                             labels: list[int], plot_name: str, axes: list[str] = None):
    """
    Visualizes clustering results based on the data points, cluster centers, and cluster labels.

    Parameters:
        data: pandas DataFrame containing the data points
        cluster_centers: list of integers representing the cluster centers
        center_points: list of values representing the center points
        labels: list of integers representing the cluster labels
        plot_name: name of the plot
        axes: list of two strings representing the x and y axes

    Returns:
        This function does not return anything, it only visualizes the clustering results using matplotlib.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    # Create a 3D axis if there are three axes
    if len(axes) == 3:
        ax = fig.add_subplot(111, projection='3d')
    plot_clusters(data, cluster_centers, center_points, labels, axes, plot_name, ax)
    plt.show()


def graph_clustering_results_for_multiple_datasets(datas: list[pd.DataFrame], cluster_centers: list[int],
                                                   center_points_by_method: list[np.ndarray], labels: np.array,
                                                   plot_names: list[str], axes_names: list[str] = None,
                                                   save_graphs: bool = False, method_name: str = None):
    """
    Visualizes clustering results based on the data points, cluster centers, and cluster labels for multiple datasets.

    Parameters:
        datas: list of pandas DataFrames containing the data points
        cluster_centers: list of integers representing the cluster centers
        center_points_by_method: list of numpy arrays representing the center points. Each array contains center points
                                  for a specific dimensionality reduction method
        labels: list of integers representing the cluster labels
        plot_names: list of strings representing the names of the plots (methods of dimensionality reduction)
        axes_names: list of strings representing the axes

    Returns:
        This function does not return anything, it only visualizes the clustering results using matplotlib.
    """
    num_plots = len(datas)
    fig = plt.figure(figsize=(5 * num_plots, 5))

    for idx, (data, plot_name, center_points) in enumerate(zip(datas, plot_names, center_points_by_method)):
        # If there are 3 axes, use 3D projection
        if len(axes_names) == 3:
            ax = fig.add_subplot(1, num_plots, idx + 1, projection='3d')
        # Otherwise, use the default projection (2D)
        else:
            ax = fig.add_subplot(1, num_plots, idx + 1)

        plot_clusters(data, cluster_centers, center_points, labels, axes_names, plot_name, ax)

    if save_graphs:
        output_path = io.__path_to_data()
        output_dir = os.path.join(output_path, "graphics")

        # Verify if the directory exists, if not, create it
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_image = os.path.join(output_dir, f"clustering_results_{method_name}.png")
        plt.savefig(output_image)

    plt.show()


def plot_indices(dict_results, methods, save_graphs: bool = False):
    for method_name, results in dict_results.items():
        param_keys = list(methods[method_name].keys())

        fig = plt.figure()

        if len(param_keys) == 1:
            plt.xlabel(param_keys[0])
            plt.ylabel('Weighted indices')

            param_values = list(methods[method_name].values())
            x = np.array(param_values[0])
            y = results

            # Normalize y
            # y = (y - np.min(y)) / (np.max(y) - np.min(y))

            plt.scatter(x, y, c=y, cmap='viridis', marker='o')

            if save_graphs:
                output_path = io.__path_to_data()
                output_dir = os.path.join(output_path, "graphics")

                # Verify if the directory exists, if not, create it
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                output_image = os.path.join(output_dir, f"parameter_exploration_{method_name}.png")
                plt.savefig(output_image)

        elif len(param_keys) == 2:
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel(param_keys[0])
            ax.set_ylabel(param_keys[1])
            ax.set_zlabel('Index')

            param_values = list(methods[method_name].values())
            x, y = np.meshgrid(param_values[0], param_values[1])
            z = results

            # Normalize z
            # z = (z - np.min(z)) / (np.max(z) - np.min(z))

            ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')

            if save_graphs:
                output_path = io.__path_to_data()
                output_dir = os.path.join(output_path, "graphics")

                # Verify if the directory exists, if not, create it
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                output_image = os.path.join(output_dir, f"parameter_exploration_{method_name}.png")
                plt.savefig(output_image)

        plt.title(method_name)
        plt.show()
