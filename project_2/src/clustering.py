import data_processing as dp
import matplotlib.pyplot as plt
import mesh
import numpy as np
import pandas as pd
from typing import Callable


def __calc_mountain_term(norm_value: float, constant: float) -> float:
    """
    Calculate the mountain term between two vectors using the provided norm function and
    the provided constant (it can be sigma or beta).

    Parameters:
        norm_value: The norm value between two vectors.
        constant: The constant used in the calculation.

    Returns:
        float: The calculated mountain term.
    """
    # TODO: don't call the norm function here, but pass it as a parameter or the matrix
    return np.exp(-(norm_value**2)/(2*constant**2))


def mountain_clustering(data: pd.DataFrame, norm: Callable, sigma: float, beta: float,
                        graphics: bool = False, max_iterations: int = 100, **kwargs) -> tuple[list[int], np.ndarray, np.ndarray]:
    """
    Perform mountain clustering on the given data.

    Args:
        data: The input data for clustering.
        norm: The normalization function.
        sigma: The sigma parameter.
        beta: The beta parameter.
        graphics: Whether to display graphics. Defaults to False.
        max_iterations: The maximum number of iterations. Defaults to 100.

    Returns:
        list[int]: The indices of the cluster centers.
        np.ndarray: The cluster centers (data points).
        np.ndarray: The distance matrix between the data points and the cluster centers.
    """
    # First step: create grid (mesh)
    array = np.linspace(0, 1, 3)
    grid = mesh.create_mesh(array, n=data.shape[1])

    distances_data_grid = dp.compute_distances((data, grid), norm, **kwargs)

    # Second step: construct mountain function
    number_of_points = sum(grid.shape[:-1])
    m = np.zeros(number_of_points)

    # Calculate the mountain term for each point
    for i in range(number_of_points):
        m[i] = sum([__calc_mountain_term(distances_data_grid[j, i], sigma) for j in range(len(data))])

    if graphics:
        plt.figure()
        plt.plot(m, 'o')
        plt.show()

    # Third step: select cluster centers as the point with the highest mountain term.
    last_center = np.argmax(m)
    cluster_centers = [last_center]

    distances_grid_grid = dp.compute_distances((grid, grid), norm)

    stop = False
    cont = 0
    while not stop:
        m_last_center = m[last_center]
        # Subtracting scaled Gaussian function centered at the last cluster center
        for i in range(number_of_points):
            m[i] = m[i] - m_last_center * __calc_mountain_term(distances_grid_grid[i, last_center], beta)

        last_center = np.argmax(m)

        # If the last center is the same as the previous one, stop.
        if last_center == cluster_centers[-1]:
            stop = True
        else:
            cluster_centers.append(last_center)

        cont += 1
        if cont == max_iterations:
            break

    return cluster_centers, grid[cluster_centers], distances_data_grid[:, cluster_centers]


def __calc_density_measure(norm_value: float, constant: float) -> float:
    """
    Calculate the density measure between two vectors using the provided norm function and
    the provided constant (it can be r_a or r_b).

    Parameters:
        norm_value: The norm value between two vectors.
        constant: The constant used in the calculation.

    Returns:
        float: The calculated density measure.
    """
    return np.exp(-(norm_value**2)/((constant/2)**2))


def subtractive_clustering(data: pd.DataFrame, distance_matrix: np.ndarray[np.float64], r_a: float, r_b: float,
                           graphics: bool = False, max_iterations: int = 100, **kwargs) -> tuple[list[int], np.ndarray, np.ndarray]:
    """
    Perform subtractive clustering on the given data.

    Args:
        data: The input data for clustering.
        distance_matrix: The distance matrix between the data points.
        r_a: The r_a parameter.
        r_b: The r_b parameter.
        graphics: Whether to display graphics. Defaults to False.
        max_iterations: The maximum number of iterations. Defaults to 100.

    Returns:
        The indices of the cluster centers.
        The cluster centers (data points).
        The distance matrix between the data points and the cluster centers.
    """
    # First step: construct density function.
    number_of_points = len(data)
    d = np.zeros(number_of_points)

    # Calculate the density measure for each point
    for i in range(number_of_points):
        d[i] = sum([__calc_density_measure(distance_matrix[i, j], r_a) for j in range(number_of_points)])

    if graphics:
        plt.figure()
        plt.plot(d, 'o')
        plt.show()

    # Second step: select cluster centers as the point with the highest density measure.
    last_center = np.argmax(d)
    cluster_centers = [last_center]

    stop = False
    cont = 0
    while not stop:
        d_last_center = d[last_center]
        # Subtract scaled Gaussian function centered at the last cluster center.
        for i in range(number_of_points):
            d[i] = d[i] - d_last_center * __calc_density_measure(distance_matrix[i, last_center], r_b)

        last_center = np.argmax(d)

        # If the last center is the same as the previous one, stop.
        if d[last_center] <= 0 or last_center == cluster_centers[-1]:
            stop = True
        else:
            cluster_centers.append(last_center)

        cont += 1
        if cont == max_iterations:
            break

    return cluster_centers, data.iloc[cluster_centers], distance_matrix[:, cluster_centers]


def __calc_cost(data: np.ndarray, clusters: np.ndarray, distances: np.ndarray) -> float:
    """
    Calculate the cost of the clustering.

    Args:
        data: The input data.
        clusters: Array containing the cluster assignments for each data point.
        distances: The distance matrix between the data points and the cluster centers.

    Returns:
        float: The cost of the clustering.
    """
    # TODO: don't call the norm function here, but pass it as a parameter or the matrix
    # TODO: review the if. It can be worse that computing the distance matrix
    cost = 0
    for i in clusters:
        # Get the cluster points
        #for j in data[clusters == i]:
        for j in range(len(data)):
            # Add the distance of each point to the cluster center
            if clusters[j] == i:
                cost += distances[j, i]

    return cost


def assign_clusters(data: np.ndarray, cluster_centers: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """
    Assigns data points to clusters based on the distance from cluster centers.

    Args:
        data: A numpy array representing the data points to be assigned to clusters.
        cluster_centers: A numpy array representing the cluster centers.
        distances: A numpy array representing the distance matrix between the data points and the cluster centers.

    Returns:
        An array containing the cluster assignments for each data point.
    """
    # Go through each data point and assign it to the closest cluster.
    clusters = np.argmin(
        np.array([[distances[k, c_i] for c_i in range(len(cluster_centers))] for k in range(len(data))]),
        axis=1
    )
    return clusters


def k_means_clustering(data: pd.DataFrame, norm: Callable, k: int = 4, initial_cluster_points: np.ndarray = None,
                       graphics: bool = False, max_iterations: int = 100,
                       **kwargs) -> tuple[list[int], np.ndarray, np.ndarray]:
    """
    Perform k-means clustering on the given data.

    Args:
        data: The input data for clustering.
        norm: The normalization function.
        initial_cluster_points: The initial cluster centers. Defaults to None. You can use subtractive_clustering to
                                  obtain an initial cluster center. NOTE: If you use mountain clustering, then it
                                  can be bad, because that will be the points of the grid, not the points of
                                  the data.
        k: The number of cluster centers.
        graphics: Whether to display graphics. Defaults to False.
        max_iterations: The maximum number of iterations. Defaults to 100.

    Returns:
        list[int]: The indices of the cluster centers.
        np.ndarray: The cluster centers (data points).
        np.ndarray: The distance matrix between the data points and the cluster centers.
    """
    # Create a copy of the data
    data_copy = data.copy()
    # Convert the data to a numpy array, so we can perform operations easily.
    data_copy = data_copy.values

    # Step 1: Initialize cluster centers
    # If no initial cluster centers are provided, randomly select k points as cluster centers.
    if initial_cluster_points is None:
        index = np.random.choice(data_copy.shape[0], k, replace=False)
        center_points = data_copy[index]
    else:
        center_points = initial_cluster_points

    cost = np.inf
    cont = 0
    while True:
        distances = dp.compute_distances((data_copy, center_points), norm, **kwargs)

        print(f"Iteration: {cont}, Cost: {cost}")
        # Step 2: determine the membership matrix U. It is assigning each point to the closest cluster center.
        clusters = assign_clusters(data_copy, center_points, distances)

        # Step 3: compute the cost function
        new_cost = __calc_cost(data_copy, clusters, distances)

        # Check if the cost has decreased
        if cost - new_cost <= 0:
            break
        else:
            cost = new_cost

        # Update the cluster centers
        for i in range(len(center_points)):
            # Check if the cluster is empty
            if sum(clusters == i) == 0:
                index = np.random.choice(data_copy.shape[0], 1, replace=False)
                center_points[i] = data_copy[index]
            else:
                center_points[i] = data_copy[clusters == i].mean()

        cont += 1
        if cont == max_iterations:
            break

    # As we have new points, we assign them new indexes.
    nodes_index = [len(data) + i for i in range(len(center_points))]

    return nodes_index, center_points, distances


def fuzzy_c_means_clustering(data: pd.DataFrame, norm: Callable, c: int = 4, m: int = 2,
                             graphics: bool = False, max_iterations: int = 100, save_graphs: bool = False,
                             **kwargs) -> tuple[list[int], np.ndarray, np.ndarray]:
    """
    Perform fuzzy c-means clustering on the given data.

    Args:
        data: The input data for clustering.
        norm: The normalization function.
        c: The number of cluster centers.
        m: The weighting exponent.
        graphics: Whether to display graphics. Defaults to False.
        max_iterations: The maximum number of iterations. Defaults to 100.

    Returns:
        list[int]: The indices of the cluster centers.
        np.ndarray: The cluster centers (data points).
        np.ndarray: The distance matrix between the data points and the cluster centers.
    """
    # Create a copy of the data
    data_copy = data.copy()
    # Convert the data to a numpy array, so we can perform operations easily.
    data_copy = data_copy.values

    # Step 1: Initialize membership matrix with random values between 0 and 1 and the sum of each row is 1.
    # Also, the columns are the c cluster centers.
    membership_matrix = np.random.rand(len(data), c)
    membership_matrix = membership_matrix / np.sum(membership_matrix, axis=1).reshape(-1, 1)

    cost = np.inf
    cont = 0
    center_points = np.zeros((c, data_copy.shape[1]))
    while True:
        # Step 2: Calculate c fuzzy cluster centers.
        for i in range(c):
            center_points[i] = np.sum(
                [(membership_matrix[j, i] ** m) * data_copy[j] for j in range(len(data_copy))], axis=0
            ) / sum([(membership_matrix[j, i] ** m) for j in range(len(data_copy))])

        distance_matrix = dp.compute_distances((data_copy, center_points), norm, **kwargs)

        # Step 3: compute the cost function
        new_cost = 0
        for i in range(c):
            for j in range(len(data_copy)):
                new_cost += (membership_matrix[j, i] ** m) * (distance_matrix[j, i] ** 2)
        # new_cost = sum([sum([(membership_matrix[i] ** m) * norm(center_points[i], data_copy[j], **kwargs)])
        #                # for i in range(c)])

        print(f"Iteration: {cont}, Cost: {new_cost}")
        # Check if the cost has decreased
        if cost - new_cost <= 0:
            break
        else:
            cost = new_cost

        # Step 4: Update the membership matrix
        for i in range(c):
            for j in range(len(data_copy)):
                membership_matrix[j, i] = 1 / sum(
                    [((distance_matrix[j, i] / distance_matrix[j, k]) ** (2 / (m - 1))) for k in range(c)]
                )

        cont += 1
        if cont == max_iterations:
            break

    # As we have new points, we assign them new indexes.
    nodes_index = [len(data) + i for i in range(len(center_points))]

    # TODO: return distance or membership matrix?
    return nodes_index, center_points, distance_matrix
