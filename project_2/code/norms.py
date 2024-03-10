import numpy as np


def euclidean_norm(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Calculate the Euclidean norm between two input vectors.

    Args:
        vector1 (np.ndarray): The first input vector.
        vector2 (np.ndarray): The second input vector.

    Returns:
        float: The Euclidean norm between the two input vectors.
    """
    return np.sqrt(np.sum(np.square(vector1 - vector2)))


def manhattan_norm(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Calculate the Manhattan norm between two input vectors.

    Args:
        vector1 (np.ndarray): The first input vector.
        vector2 (np.ndarray): The second input vector.

    Returns:
        float: The Manhattan norm of the input vectors.
    """
    return np.sum(np.abs(vector1 - vector2))


def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two input vectors.

    Args:
        vector1 (np.ndarray): The first input vector.
        vector2 (np.ndarray): The second input vector.

    Returns:
        float: The cosine similarity between the two input vectors.
    """
    dot_product = np.dot(vector1, vector2)
    norm_product = np.sqrt(np.sum(np.square(vector1))) * np.sqrt(np.sum(np.square(vector2)))
    similarity = dot_product / norm_product
    return similarity


def p_norm(vector1: np.ndarray, vector2: np.ndarray, p: int) -> float:
    """
    Calculate the p-norm of two vectors.

    Args:
        vector1: An array representing the first vector.
        vector2: An array representing the second vector.
        p: An integer representing the p value for the norm calculation.

    Returns:
        The p-norm distance between vector1 and vector2.
    """
    return np.power(np.sum(np.power(np.abs(vector1 - vector2), p)), 1/p)
