import itertools
import numpy as np


def create_mesh(array: np.ndarray, n: int = 1, reshape_to_matrix: bool = False) -> np.ndarray:
    """
    Generate a mesh, or a vector with all the points of the combination from an input array.

    Parameters:
        array (np.ndarray): The input array.
        n (int, optional): The number of dimensions of the mesh. Defaults to 1.
        reshape_to_matrix (bool, optional): Whether to reshape the mesh to a matrix. Defaults to False.

    Returns:
        np.ndarray: The generated mesh.
    """
    cartesian_product = np.array(list(itertools.product(array, repeat=n)))

    if reshape_to_matrix:
        length = len(array)
        dims = [0]*(n + 1)
        for i in range(n):
            dims[i] = length
        dims[n] = n

        cartesian_product = cartesian_product.reshape(dims)

    return cartesian_product
