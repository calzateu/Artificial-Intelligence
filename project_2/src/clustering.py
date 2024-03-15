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
    # TODO: don't call the norm function here, but pass it as a parameter or the matrix
    return np.exp(-((norm(vector1, vector2))**2)/(2*constant**2))


def mountain_clustering(data: pd.DataFrame, norm: Callable, sigma: float, beta: float,
                        graphics: bool = False) -> tuple[list[int], np.ndarray]:
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
        np.ndarray: The cluster centers (data points).
    """
    # First step: create grid (mesh)
    array = np.linspace(0, 1, 3)
    grid = mesh.create_mesh(array, 4)

    # Second step: construct mountain function
    number_of_points = sum(grid.shape[:-1])
    m = np.zeros(number_of_points)

    # TODO: don't call the norm function here, but pass it as a parameter or the matrix
    # Calculate the mountain term for each point
    for i in range(number_of_points):
        m[i] = sum([__calc_mountain_term(grid[i], data.iloc[j], norm, sigma) for j in range(len(data))])

    if graphics:
        plt.figure()
        plt.plot(m, 'o')
        plt.show()

    # Third step: select cluster centers as the point with the highest mountain term.
    last_center = np.argmax(m)
    cluster_centers = [last_center]

    stop = False
    while not stop:
        m_last_center = m[last_center]
        # TODO: don't call the norm function here, but pass it as a parameter or the matrix
        # Subtracting scaled Gaussian function centered at the last cluster center
        for i in range(number_of_points):
            m[i] = m[i] - m_last_center * __calc_mountain_term(grid[i], grid[last_center], norm, beta)

        last_center = np.argmax(m)

        # If the last center is the same as the previous one, stop.
        if last_center == cluster_centers[-1]:
            stop = True
        else:
            cluster_centers.append(last_center)

    return cluster_centers, grid[cluster_centers]


def __calc_density_measure(vector1: np.ndarray, vector2: np.ndarray, norm: Callable, constant: float) -> float:
    """
    Calculate the density measure between two vectors using the provided norm function and
    the provided constant (it can be r_a or r_b).

    Parameters:
        vector1: The first input vector.
        vector2 : The second input vector.
        norm: The norm function to calculate the distance between the vectors.
        constant: The constant used in the calculation.

    Returns:
        float: The calculated density measure.
    """
    # TODO: don't call the norm function here, but pass it as a parameter or the matrix
    return np.exp(-((norm(vector1, vector2))**2)/((constant/2)**2))


def subtractive_clustering(data: pd.DataFrame, norm: Callable, r_a: float, r_b: float,
                           graphics: bool = False) -> tuple[list[int], np.ndarray]:
    """
    Perform subtractive clustering on the given data.

    Args:
        data: The input data for clustering.
        norm: The normalization function.
        r_a: The r_a parameter.
        r_b: The r_b parameter.
        graphics: Whether to display graphics. Defaults to False.

    Returns:
        list[int]: The indices of the cluster centers.
        np.ndarray: The cluster centers (data points).
    """
    # First step: construct density function.
    number_of_points = len(data)
    d = np.zeros(number_of_points)

    # TODO: don't call the norm function here, but pass it as a parameter or the matrix
    # Calculate the density measure for each point
    for i in range(number_of_points):
        d[i] = sum([__calc_density_measure(data.iloc[i], data.iloc[j], norm, r_a) for j in range(number_of_points)])

    if graphics:
        plt.figure()
        plt.plot(d, 'o')
        plt.show()

    # Second step: select cluster centers as the point with the highest density measure.
    last_center = np.argmax(d)
    cluster_centers = [last_center]

    stop = False
    while not stop:
        d_last_center = d[last_center]
        # TODO: don't call the norm function here, but pass it as a parameter or the matrix
        # Subtracting scaled Gaussian function centered at the last cluster center
        for i in range(number_of_points):
            d[i] = d[i] - d_last_center * __calc_density_measure(data.iloc[i], data.iloc[last_center], norm, r_b)

        last_center = np.argmax(d)

        # If the last center is the same as the previous one, stop.
        if d[last_center] <= 0 or last_center == cluster_centers[-1]:
            stop = True
        else:
            cluster_centers.append(last_center)

    return cluster_centers, data.iloc[cluster_centers]


def k_means_clustering(data: pd.DataFrame, norm: Callable, k: int = 4, initial_cluster_centers: list[int] = None,
                       graphics: bool = False) -> tuple[list[int], np.ndarray]:
    """
    Perform k-means clustering on the given data.

    Args:
        data: The input data for clustering.
        norm: The normalization function.
        initial_cluster_centers: The initial cluster centers. Defaults to None. You can use subtractive_clustering to
                                  obtain an initial cluster center. NOTE: If you use mountain clustering, then it
                                  can be bad, because that will be the points of the grid, not the points of
                                  the data.
        k: The number of cluster centers.
        graphics: Whether to display graphics. Defaults to False.

    Returns:
        list[int]: The indices of the cluster centers.
        np.ndarray: The cluster centers (data points).
    """
    # Create a copy of the data
    data_copy = data.copy()

    # Step 1: Initialize cluster centers
    # If no initial cluster centers are provided, randomly select k points as cluster centers
    if initial_cluster_centers is None:
        cluster_centers = np.random.choice(len(data), k, replace=False)
    else:
        cluster_centers = initial_cluster_centers

    stop = False
    while not stop:
        # Step 2: determine the membership matrix U. It is assigning each point to the closest cluster center.
        for i in range(len(data)):
            # TODO: don't call the norm function here, but pass it as a parameter or the matrix
            data_copy.loc[i, 'cluster'] = np.argmin(
                [norm(data_copy.iloc[i], data_copy.iloc[j]) for j in cluster_centers]
            )

        # Update the cluster centers
        cluster_centers = data_copy.groupby('cluster').mean().reset_index().iloc[:, 1:].values
