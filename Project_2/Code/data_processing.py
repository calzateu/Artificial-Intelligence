import numpy as np
import pandas as pd


def saple(data, num_samples):
    pass


def get_subsample(data: pd.DataFrame, num_samples: int) -> pd.DataFrame:
    return data.sample(n=num_samples)
