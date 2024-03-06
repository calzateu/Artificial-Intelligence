import itertools
import numpy as np


def cartesian_product(*arrays):
    pass


def create_mesh_python(array, n=1):
    cartesian_product = np.array(list(itertools.product(array, repeat=n)))

    lenght = len(array)
    dims = [0]*(n + 1)
    for i in range(n):
        dims[i] = lenght
    dims[n] = n

    cartesian_product = cartesian_product.reshape(dims)

    return cartesian_product


#print(cartesian_product([1, 2, 3], [4, 5, 6]))
mesh = create_mesh_python([1, 2], 3)
print(mesh)
