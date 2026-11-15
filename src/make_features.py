import pandas as pd
from features import extract_features

df = pd.read_csv("../data/dataset.csv")
df = df.drop_duplicates(subset="URL")  # buang URL kembar

feat = pd.DataFrame([extract_features(u) for u in df["URL"]])
feat["url"] = df["URL"].values
feat["label"] = (df["label"].values == 0).astype(int)  # BALIK: 1 = phishing, 0 = aman

feat.to_csv("../data/features.csv", index=False)
print(feat["label"].value_counts())
print("Selesai:", feat.shape)