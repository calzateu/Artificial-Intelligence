import pandas as pd
from sklearn.decomposition import PCA


def pca(data: pd.DataFrame, num_components: int = 2, axles: list[str] = None) -> pd.DataFrame:
    """
    A function that performs Principal Component Analysis on the input data.

    Parameters:
        data: The input data for PCA.
        num_components: The number of principal components to keep.
        axles: The names of the principal components

    Returns:
        A DataFrame containing the principal components with columns 'SepalLengthCm' and 'SepalWidthCm'.
    """
    pca_model = PCA(n_components=2)
    principal_components = pca_model.fit_transform(data)

    principal_df = pd.DataFrame(data=principal_components, columns=axles)

    return principal_df

