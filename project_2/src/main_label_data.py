import pandas as pd


data = pd.read_csv("../data/output_centroid.csv")

data["label"] = data["output"].apply(lambda x: 1 if x > 0.8 else 0)
data.to_csv("../data/output_centroid_labeled.csv", index=False)

print(data.head())
