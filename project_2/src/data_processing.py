import numpy as np
import pandas as pd
from typing import Callable


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


def __build_distance_matrix(data1: np.ndarray, data2: np.ndarray, norm: Callable, **kwargs) -> np.ndarray:
    """
    Build a distance matrix between two data sets using a given norm function and optional additional arguments.
    Args:
        data1: The first data set.
        data2: The second data set.
        norm: The function used to compute distances between elements.
        **kwargs: Additional arguments to be passed to the norm function.
    Returns:
        np.ndarray: The distance matrix between the two data sets.
    """
    len_data1 = len(data1)
    len_data2 = len(data2)
    distance_matrix = np.zeros((len_data1, len_data2))
    for i in range(len_data1):
        if len_data1 == len_data2:
            for j in range(i, len_data2):
                distance_matrix[i, j] = norm(data1[i], data2[j], **kwargs)
                distance_matrix[j, i] = distance_matrix[i, j]
        else:
            for j in range(len_data2):
                distance_matrix[i, j] = norm(data1[i], data2[j], **kwargs)

    return distance_matrix


def compute_distances(data: pd.DataFrame | tuple[pd.DataFrame, np.ndarray] | tuple[pd.DataFrame, pd.DataFrame],
                      norm, **kwargs) -> np.array:
    """
    Compute distances between elements in a subsample using a given norm function and optional additional arguments.
    Args:
        data: The data for which distances are to be computed. Can be either a single DataFrame or a tuple.
            If a tuple, the first element is the first DataFrame and the second element is a numpy array.
        norm: The function used to compute distances between elements.
        **kwargs: Additional arguments to be passed to the norm function.
    Returns:
        np.array: An array of distances between elements in the subsample.
    """
    data1, data2 = None, None
    if isinstance(data, tuple):
        # If `data` is a tuple, calculate distances between the components.
        if isinstance(data[0], np.ndarray) and isinstance(data[1], np.ndarray):
            # If both `data` are numpy arrays, calculate distances between the two.
            data1 = data[0]
            data2 = data[1]
        elif isinstance(data[0], pd.DataFrame) and isinstance(data[1], np.ndarray):
            # If the first `data` component is a DataFrame and the second is a numpy array,
            # calculate distances between the DataFrame and the numpy array.
            data1 = data[0].values
            data2 = data[1]
        elif isinstance(data[0], np.ndarray) and isinstance(data[1], pd.DataFrame):
            # If the first `data` component is a numpy array and the second is a DataFrame,
            # calculate distances between the numpy array and the DataFrame.
            data1 = data[0]
            data2 = data[1].values
        elif isinstance(data[0], pd.DataFrame) and isinstance(data[1], pd.DataFrame):
            # If both `data` components are DataFrames, calculate distances between the two.
            data1 = data[0].values
            data2 = data[1].values
    elif isinstance(data, pd.DataFrame):
        # If `data` is a single DataFrame, calculate distances between all points in the DataFrame
        data1 = data.values
        data2 = data.values
    else:
        raise TypeError("data must be either a tuple or a DataFrame")

    return __build_distance_matrix(data1, data2, norm, **kwargs)


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
