import pandas as pd
import numpy as  np

file_path=r"C:\Users\hashr\OneDrive\Desktop\PBL_6th\scaled_dataset_2.csv"
df = pd.read_csv(file_path ,nrows=2000000)
df.info()
print(df["Attack"].unique())
print(df.head(n=10))
print(df["Attack"].dtype)
print(df["Attack"].unique())
print('count of na null values: ')
print(df.isna().sum())
print(df.isnull().sum())
nan_rows = df[df['Attack'].isna()]
print("Number of NaN in 'Attack':", len(nan_rows))
print(nan_rows.head())




attack_counts = df['Attack'].value_counts()

print("Count of each attack type:")
print(attack_counts)

