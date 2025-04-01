import pandas as pd
import numpy as np

file_path = r"C:\Users\hashr\OneDrive\Desktop\PBL_6th\data\NF-UQ-NIDS-v2.csv"
output_file = "preprocessed_dataset.csv"
chunk_size = 100000  
selected_columns = [
    "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL",
    "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
    "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES", "Attack"
]


attack_mapping = {"Benign": 0, "DoS": 1, "DDoS": 1}
selected_attacks = ["Benign", "DoS", "DDoS"]

with open(output_file, "w", newline='') as f:
    header_written = False
    for chunk in pd.read_csv(file_path, usecols=selected_columns, chunksize=chunk_size):
        
        chunk = chunk[chunk["Attack"].isin(selected_attacks)]
        
        chunk["Attack"] = chunk["Attack"].map(attack_mapping)
        
        
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.fillna(0, inplace=True)
        
 
        chunk.to_csv(f, index=False, header=not header_written)
        header_written = True

 
df_preprocessed = pd.read_csv(output_file)
print("Number of rows in the preprocessed dataset:", df_preprocessed.shape[0])
