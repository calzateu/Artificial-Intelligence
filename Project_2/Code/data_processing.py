import numpy as np
import pandas as pd


def saple(data, num_samples):
    pass


def get_subsample(data: pd.DataFrame, num_samples: int) -> pd.DataFrame:
    return data.sample(n=num_samples)


def normalize(data: pd.DataFrame) -> pd.DataFrame:
    return (data - data.min()) / (data.max() - data.min())


def compute_distances(subsample: pd.DataFrame, norm, *args) -> pd.DataFrame:
    distances = np.zeros((len(subsample), len(subsample)))

    for i in range(len(subsample)):
        for j in range(len(subsample)):
            distances[i, j] = norm(subsample.iloc[i], subsample.iloc[j], *args)

    return distances
