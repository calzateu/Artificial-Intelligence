import numpy as np
import pandas as pd


def get_subsample(data: pd.DataFrame, num_samples: int) -> pd.DataFrame:
    """
    Get a random subsample of the input data.
    Args:
        data (pd.DataFrame): The input DataFrame.
        num_samples (int): The number of samples to be selected.
    Returns:
        pd.DataFrame: A random subsample of the input data.
    """
    return data.sample(n=num_samples)


def normalize(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the input DataFrame by scaling all values to the range [0, 1].
    It uses the min-max normalization technique.

    Parameters:
    - data: pandas DataFrame, the input data to be normalized

    Returns:
    - pandas DataFrame, the normalized data
    """
    return (data - data.min()) / (data.max() - data.min())


def compute_distances(data: pd.DataFrame | tuple[pd.DataFrame, np.ndarray], norm, *args) -> np.array:
    """
    Compute distances between elements in a subsample using a given norm function and optional additional arguments.
    Args:
        data: The data for which distances are to be computed. Can be either a single DataFrame or a tuple.
            If a tuple, the first element is the first DataFrame and the second element is a numpy array.
        norm: The function used to compute distances between elements.
        *args: Additional arguments to be passed to the norm function.
    Returns:
        np.array: An array of distances between elements in the subsample.
    """
    if isinstance(data, tuple):
        # If `data` is a tuple, calculate distances between the DataFrame and the numpy array
        distances = np.zeros((len(data[0]), len(data[1])))
        for i in range(len(data[0])):
            for j in range(len(data[1])):
                distances[i, j] = norm(data[0].iloc[i], data[1][j], *args)
    elif isinstance(data, pd.DataFrame):
        # If `data` is a single DataFrame, calculate distances between all points in the DataFrame
        distances = np.zeros((len(data), len(data)))
        for i in range(len(data)):
            for j in range(len(data)):
                distances[i, j] = norm(data.iloc[i], data.iloc[j], *args)
    else:
        raise TypeError("data must be either a tuple or a DataFrame")

    return distances


def label_data(data: pd.DataFrame, cluster_centers: list[int], center_points: np.ndarray,
               distance_matrix: np.ndarray) -> pd.DataFrame:
    """
    Assign labels to data based on the distance matrix and cluster centers.
    Args:
        data: The data to be labeled.
        cluster_centers: The cluster centers.
        center_points: The center points.
        distance_matrix: The distance matrix.
    Returns:
        pd.DataFrame: The labeled data.
    """
    labels = [0]*len(data)
    for i in range(len(data)):
        min_distance = np.inf
        min_center = -1
        for j in range(len(cluster_centers)):
            if distance_matrix[i, j] < min_distance:
                min_distance = distance_matrix[i, j]
                min_center = cluster_centers[j]
        labels[i] = min_center

    data["label"] = labels

    return data
