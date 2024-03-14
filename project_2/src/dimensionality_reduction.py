import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap as umap_module


def pca(data: pd.DataFrame, center_points: np.ndarray = None, num_components: int = 2,
        axes: list[str] = None) -> tuple[pd.DataFrame, np.ndarray]:
    """
    A function that performs Principal Component Analysis on the input data.

    Parameters:
        data: The input data for PCA.
        center_points: The cluster center points.
        num_components: The number of principal components to keep.
        axes: The names of the principal components

    Returns:
        A tuple containing the principal components and the transformed center points. If center_points is None, the
        transformed center points will be None.
    """
    pca_model = PCA(n_components=num_components)
    principal_components = pca_model.fit_transform(data.drop(columns=['label'], axis=1))

    center_points_transformed = None
    if center_points is not None:
        center_points_transformed = pca_model.transform(center_points)

    principal_df = pd.DataFrame(data=principal_components, columns=axes)

    return principal_df, center_points_transformed


def tsne(data: pd.DataFrame, center_points: np.ndarray = None, num_components: int = 2,
         axes: list[str] = None) -> tuple[pd.DataFrame, np.ndarray]:
    """
    A function that performs t-distributed Stochastic Neighbor Embedding on the input data.

    Parameters:
        data: The input data for t-SNE.
        center_points: The cluster center points.
        num_components: The number of principal components to keep.
        axes: The names of the principal components

    Returns:
        A tuple containing the principal components and the transformed center points. If center_points is None, the
        transformed center points will be None.
    """
    # Store the original number of rows in data
    original_rows = data.shape[0]
    temp_data = data.drop(columns=['label'], axis=1)

    if center_points is not None:
        # Add the center points to the data DataFrame
        center_points_df = pd.DataFrame(data=center_points, columns=temp_data.columns)
        temp_data = pd.concat([temp_data, center_points_df], ignore_index=True)

    tsne_model = TSNE(n_components=num_components)
    principal_components = tsne_model.fit_transform(temp_data)

    # Get the values of the added center points
    if center_points is not None:
        # Center points are located after the original rows
        center_points_transformed = principal_components[original_rows:]
        # Remove the center points from the array of principal components
        principal_components = principal_components[:original_rows]
    else:
        center_points_transformed = None

    principal_df = pd.DataFrame(data=principal_components, columns=axes)

    return principal_df, center_points_transformed


def umap(data: pd.DataFrame, center_points: np.ndarray = None, num_components: int = 2,
         axes: list[str] = None) -> tuple[pd.DataFrame, np.ndarray]:
    """
    A function that performs Uniform Manifold Approximation and Projection on the input data.

    Parameters:
        data: The input data for UMAP.
        center_points: The cluster center points.
        num_components: The number of principal components to keep.
        axes: The names of the principal components

    Returns:
        A tuple containing the principal components and the transformed center points. If center_points is None, the
        transformed center points will be None.
    """
    umap_model = umap_module.UMAP(n_components=num_components)
    principal_components = umap_model.fit_transform(data.drop(columns=['label'], axis=1))

    center_points_transformed = None
    if center_points is not None:
        center_points_transformed = umap_model.transform(center_points)

    principal_df = pd.DataFrame(data=principal_components, columns=axes)

    return principal_df, center_points_transformed
