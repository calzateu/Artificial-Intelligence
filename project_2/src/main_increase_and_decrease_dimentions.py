import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from umap import UMAP

# Read the CSV file
data = pd.read_csv("../data/output_centroid_labeled.csv")

# Initialize the PolynomialFeatures object
poly = PolynomialFeatures(degree=2, include_bias=False)

# Transform the 'x' column into polynomial features
poly_features = poly.fit_transform(data[['transactions_per_day', 'transactions_per_month']])

print(poly_features)

# Convert the polynomial features into a DataFrame
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(
    ['transactions_per_day', 'transactions_per_month'])
                       )

# Show the resulting DataFrame
print(poly_df.head())
print(poly_df.columns)

data[poly_df.columns] = poly_df
data.to_csv("../data/output_centroid_poly.csv", index=False)


# Initialize PCA with desired number of components
pca = PCA(n_components=2)  # Adjust the number of components as needed

# Fit and transform the data using PCA
pca_features = pca.fit_transform(data)

# Convert PCA features into a DataFrame
pca_df = pd.DataFrame(pca_features, columns=['PCA_Component_1', 'PCA_Component_2'])

# Show the resulting DataFrame with PCA components
print(pca_df.head())

# Save the data with PCA components to a CSV file
pca_df.to_csv("../data/output_centroid_pca.csv", index=False)


# Initialize UMAP with desired number of components
umap = UMAP(n_components=2)  # Adjust the number of components as needed

# Fit and transform the data using UMAP
umap_features = umap.fit_transform(data)

# Convert UMAP features into a DataFrame
umap_df = pd.DataFrame(umap_features, columns=['PCA_Component_1', 'PCA_Component_2'])

# Show the resulting DataFrame with UMAP components
print(umap_df.head())

# Save the data with UMAP components to a CSV file
umap_df.to_csv("../data/output_centroid_umap.csv", index=False)

