import pandas as pd

df = pd.read_csv("../data/dataset.csv")

print("Nama kolom:")
print(df.columns.tolist())
print()

print("5 baris pertama:")
print(df[["URL", "label"]].head())
print()

print("Jumlah baris:", len(df))
print()

print("Jumlah per label:")
print(df["label"].value_counts())
print()

print("URL duplikat:", df["URL"].duplicated().sum())
print("Data kosong di kolom URL:", df["URL"].isna().sum())