import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Literal
import umap as umap_module


def reduce_dimensionality(method: Literal['pca', 'tsne', 'umap'], data: pd.DataFrame, center_points: np.ndarray = None,
                          num_components: int = 2, axes: list[str] = None) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Reduce the dimensionality of the input data using the specified method.

    Parameters:
        method: The dimensionality reduction method to use (choices: 'pca', 'tsne', 'umap').
        data: The input DataFrame containing the data to reduce.
        center_points: An optional array of center points to include in the data.
        num_components: The number of components to reduce the data to (default is 2).
        axes: The axes to use for the output DataFrame (default is None).

    Returns:
        A tuple containing the reduced DataFrame and the transformed center points (if provided).
    """
    # Store the original number of rows in data.
    original_rows = data.shape[0]
    # Drop the label column
    temp_data = data.copy()

    # Add the center points to the data if they are provided.
    if center_points is not None:
        # Add the center points to the data DataFrame
        center_points_df = pd.DataFrame(data=center_points, columns=temp_data.columns)
        temp_data = pd.concat([temp_data, center_points_df], ignore_index=True)

    # Fit the dimensionality reduction model and transform the data.
    if method == 'pca':
        pca_model = PCA(n_components=num_components)
        principal_components = pca_model.fit_transform(temp_data)
    elif method == 'tsne':
        tsne_model = TSNE(n_components=num_components)
        principal_components = tsne_model.fit_transform(temp_data)
    elif method == 'umap':
        umap_model = umap_module.UMAP(n_components=num_components)
        principal_components = umap_model.fit_transform(temp_data)
    else:
        raise ValueError(f"Invalid dimensionality reduction method: {method}")

    # Get the values of the added center points
    if center_points is not None:
        # Center points are located after the original rows.
        center_points_transformed = principal_components[original_rows:]
        # Remove the center points from the array of principal components.
        principal_components = principal_components[:original_rows]
    else:
        center_points_transformed = None

    principal_df = pd.DataFrame(data=principal_components, columns=axes)

    return principal_df, center_points_transformed
