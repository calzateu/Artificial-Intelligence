import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

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

