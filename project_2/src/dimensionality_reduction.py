import pandas as pd
from sklearn.decomposition import PCA


def pca(data: pd.DataFrame) -> pd.DataFrame:
    """
    A function that performs Principal Component Analysis on the input data.

    Parameters:
        data: The input data for PCA.

    Returns:
        A DataFrame containing the principal components with columns 'SepalLengthCm' and 'SepalWidthCm'.
    """
    pca_model = PCA(n_components=2)
    principal_components = pca_model.fit_transform(data)

    principal_df = pd.DataFrame(data=principal_components, columns=['SepalLengthCm', 'SepalWidthCm'])

    return principal_df

