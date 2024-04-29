import pandas as pd


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


def preprocess_data(subsample: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by standardizing it and converting it to a numpy array.
    Args:
        subsample: The data to be preprocessed.
    Returns:
        The preprocessed data as a pd.DataFrame.
    """

    # Remove Nans
    subsample = subsample.dropna()

    # One hot encode labels
    subsample = pd.get_dummies(subsample)
    subsample.replace({True: 1, False: 0}, inplace=True)

    subsample.infer_objects(copy=False)

    # Normalize data with min-max normalization.
    normalized_subsample = normalize(subsample)

    return normalized_subsample
