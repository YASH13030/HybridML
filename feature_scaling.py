import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

file_path = r"C:\Users\hashr\OneDrive\Desktop\PBL_6th\preprocessed_dataset.csv"
output_file = "scaled_dataset_2.csv"
chunk_size = 100000

sample_data = pd.read_csv(file_path, nrows=500000)
feature_cols = sample_data.columns[:-1] 
target_col = "Attack"

sample_data[feature_cols] = sample_data[feature_cols].astype(np.float64)

sample_data.replace([np.inf, -np.inf], np.nan, inplace=True)
sample_data.fillna(sample_data.median(numeric_only=True), inplace=True)

clip_limit = sample_data[feature_cols].quantile(0.9999)
sample_data[feature_cols] = sample_data[feature_cols].clip(upper=clip_limit, axis=1)

scaler = RobustScaler()
scaler.fit(sample_data[feature_cols])

pd.DataFrame(columns=sample_data.columns).to_csv(output_file, index=False, mode='w')

for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
    print(f"Processing chunk {i+1}")

    chunk[feature_cols] = chunk[feature_cols].astype(np.float64)
    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
    chunk.fillna(sample_data.median(numeric_only=True), inplace=True)
    chunk[feature_cols] = chunk[feature_cols].clip(upper=clip_limit, axis=1)
    print(f"Before scaling, 'Attack' column NaN count: {chunk['Attack'].isnull().sum()}")
    print(f"Before scaling, 'Attack' column values: {chunk['Attack'].head()}")
    attack_column = chunk[target_col].reset_index(drop=True)

    scaled_features = scaler.transform(chunk[feature_cols])
    scaled_chunk = pd.DataFrame(scaled_features, columns=feature_cols)

    scaled_chunk["Attack"] = attack_column.astype(np.int8)

    print(f"After scaling, 'Attack' column NaN count: {scaled_chunk['Attack'].isnull().sum()}")
    print(f"After scaling, 'Attack' column values: {scaled_chunk['Attack'].head()}")
    if scaled_chunk["Attack"].isnull().sum() > 0:
        print(f"NaN detected in 'Attack' column after scaling in chunk {i+1}.")
        print(f"{scaled_chunk['Attack'].isnull().sum()} NaN values found.")
        break

    scaled_chunk.to_csv(output_file, index=False, mode='a', header=False)

print("Robust Scaling complete! Scaled data saved as:", output_file)
