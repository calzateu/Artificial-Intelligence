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


def compute_distances(subsample: pd.DataFrame, norm, *args) -> np.array:
    """
    Compute distances between elements in a subsample using a given norm function and optional additional arguments.
    Args:
        subsample (pd.DataFrame): The subsample for which distances are to be computed.
        norm: The function used to compute distances between elements.
        *args: Additional arguments to be passed to the norm function.
    Returns:
        np.array: An array of distances between elements in the subsample.
    """
    distances = np.zeros((len(subsample), len(subsample)))

    for i in range(len(subsample)):
        for j in range(len(subsample)):
            distances[i, j] = norm(subsample.iloc[i], subsample.iloc[j], *args)

    return distances


def label_data(data: pd.DataFrame, cluster_centers: list[int], distance_matrix: np.ndarray) -> pd.DataFrame:
    """
    Assign labels to data based on the distance matrix and cluster centers.
    Args:
        data: The data to be labeled.
        cluster_centers: The cluster centers.
        distance_matrix: The distance matrix.
    Returns:
        pd.DataFrame: The labeled data.
    """
    labels = [0]*len(data)
    for i in range(len(data)):
        min_distance = np.inf
        min_center = -1
        for j in cluster_centers:
            if distance_matrix[i, j] < min_distance:
                min_distance = distance_matrix[i, j]
                min_center = j
        labels[i] = min_center

    data["label"] = labels

    return data
