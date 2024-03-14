import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap as umap_module


def pca(data: pd.DataFrame, num_components: int = 2, axes: list[str] = None) -> pd.DataFrame:
    """
    A function that performs Principal Component Analysis on the input data.

    Parameters:
        data: The input data for PCA.
        num_components: The number of principal components to keep.
        axes: The names of the principal components

    Returns:
        A DataFrame containing the principal components with columns 'SepalLengthCm' and 'SepalWidthCm'.
    """
    pca_model = PCA(n_components=num_components)
    principal_components = pca_model.fit_transform(data)

    principal_df = pd.DataFrame(data=principal_components, columns=axes)

    return principal_df


def tsne(data: pd.DataFrame, num_components: int = 2, axes: list[str] = None) -> pd.DataFrame:
    """
    A function that performs t-distributed Stochastic Neighbor Embedding on the input data.

    Parameters:
        data: The input data for t-SNE.
        num_components: The number of principal components to keep.
        axes: The names of the principal components

    Returns:
        A DataFrame containing the principal components with columns 'SepalLengthCm' and 'SepalWidthCm'.
    """
    tsne_model = TSNE(n_components=num_components)
    principal_components = tsne_model.fit_transform(data)

    principal_df = pd.DataFrame(data=principal_components, columns=axes)

    return principal_df


def umap(data: pd.DataFrame, num_components: int = 2, axes: list[str] = None) -> pd.DataFrame:
    """
    A function that performs Uniform Manifold Approximation and Projection on the input data.

    Parameters:
        data: The input data for UMAP.
        num_components: The number of principal components to keep.
        axes: The names of the principal components

    Returns:
        A DataFrame containing the principal components with columns 'SepalLengthCm' and 'SepalWidthCm'.
    """
    umap_model = umap_module.UMAP(n_components=num_components)
    principal_components = umap_model.fit_transform(data)

    principal_df = pd.DataFrame(data=principal_components, columns=axes)

    return principal_df
