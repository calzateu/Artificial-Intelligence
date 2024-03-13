import matplotlib.pyplot as plt
import mesh
import numpy as np
import pandas as pd
from typing import Callable


def __calc_mountain_term(vector1: np.ndarray, vector2: np.ndarray, norm: Callable, constant: float) -> float:
    """
    Calculate the mountain term between two vectors using the provided norm function and
    the provided constant (it can be sigma or beta).

    Parameters:
        vector1: The first input vector.
        vector2 : The second input vector.
        norm: The norm function to calculate the distance between the vectors.
        constant: The constant used in the calculation.

    Returns:
        float: The calculated mountain term.
    """
    return np.exp(-((norm(vector1, vector2))**2)/(2*constant**2))


def mountain_clustering(data: pd.DataFrame, norm: Callable, sigma: float, beta: float,
                        graphics: bool = False) -> list[int]:
    """
    Perform mountain clustering on the given data.

    Args:
        data: The input data for clustering.
        norm: The normalization function.
        sigma: The sigma parameter.
        beta: The beta parameter.
        graphics: Whether to display graphics. Defaults to False.

    Returns:
        list[int]: The indices of the cluster centers.
    """
    # First step: create grid (mesh)
    array = np.linspace(0, 1, 3)
    grid = mesh.create_mesh(array, 4)

    # Second step: construct mountain function
    number_of_points = sum(grid.shape[:-1])
    m = np.zeros(number_of_points)

    for i in range(number_of_points):
        m[i] = sum([__calc_mountain_term(grid[i], data.iloc[j], norm, sigma) for j in range(len(data))])

    if graphics:
        plt.figure()
        plt.plot(m, 'o')
        plt.show()

    # Third step: select cluster centers
    last_center = np.argmax(m)
    cluster_centers = [last_center]

    stop = False
    while not stop:
        m_last_center = m[last_center]
        for i in range(number_of_points):
            m[i] = m[i] - m_last_center * __calc_mountain_term(grid[i], grid[last_center], norm, beta)

        last_center = np.argmax(m)

        if last_center == cluster_centers[-1]:
            stop = True
        else:
            cluster_centers.append(last_center)

    return cluster_centers
