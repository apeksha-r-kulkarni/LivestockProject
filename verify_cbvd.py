import pandas as pd
import json

df = pd.read_csv('data/raw/cbvd-5/CBVD-5.csv', skiprows=10, header=None, 
                 names=['metadata_id', 'file_list', 'flags', 'temporal_coordinates', 'spatial_coordinates', 'metadata'])

total_rows = len(df)
images = set()
behavior_counts = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}

for _, row in df.iterrows():
    f_list = json.loads(row['file_list'])
    images.update(f_list)
    meta = json.loads(row['metadata'])
    if '1' in meta:
        classes = meta['1'].split(',')
        for c in classes:
            if c in behavior_counts:
                behavior_counts[c] += 1
            else:
                behavior_counts[c] = 1

print(f"Total Rows: {total_rows}")
print(f"Unique Source Images: {len(images)}")
print(f"Behavior Counts: {behavior_counts}")
