import matplotlib.pyplot as plt
import mesh
import numpy as np
import pandas as pd
from typing import Callable


def __calc_mountain_term(vector1, vector2, norm, constant):
    return np.exp(-((norm(vector1, vector2))**2)/(2*constant**2))


def mountain_clustering(data: pd.DataFrame, norm: Callable, sigma: float, beta: float,
                        graphics: bool = False) -> list[int]:
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
        for i in range(number_of_points):
            m[i] = m[i] - m[last_center] * __calc_mountain_term(grid[i], grid[last_center], norm, beta)

        print(m)

        last_center = np.argmax(m)

        if last_center == cluster_centers[-1]:
            stop = True
        else:
            cluster_centers.append(last_center)

        print(cluster_centers)
        print(len(cluster_centers))

    print(len(cluster_centers))

    return cluster_centers
