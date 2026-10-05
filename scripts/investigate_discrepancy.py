import pandas as pd
import json
import glob
import os

def investigate():
    csv_path = 'data/raw/cbvd-5/CBVD-5.csv'
    
    # Check total rows without skipping
    with open(csv_path, 'r') as f:
        total_lines = sum(1 for line in f)
        
    print(f"Total lines in raw CSV: {total_lines}")
    
    # Read ignoring first 10? Wait, the header is at line 0 or it has a metadata header.
    # We will just parse line by line to be safe.
    valid_csv_rows = 0
    unique_images = set()
    total_yolo_labels_from_csv = 0
    class_counts = {0:0, 1:0, 2:0, 3:0, 4:0}
    
    with open(csv_path, 'r') as f:
        for i, line in enumerate(f):
            if i < 1: continue # Usually VIA has 1 row header: filename,file_size,file_attributes,region_count,region_id,region_shape_attributes,region_attributes
            
            # Since some fields have JSON strings, let's parse with pandas for safety
            pass

    # Read with pandas correctly
    # VIA CSV header is typically: filename,file_size,file_attributes,region_count,region_id,region_shape_attributes,region_attributes
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        df = pd.read_csv(csv_path, header=None)
        
    print(f"Pandas dataframe rows: {len(df)}")
    
    # If the first column is string and contains '{', it might be different format.
    # My previous script used:
    # df = pd.read_csv(csv_path, skiprows=10, header=None, names=['metadata_id', 'file_list', 'flags', 'temporal_coordinates', 'spatial_coordinates', 'metadata'])
    # Wait, the dataset was downloaded by Kaggle. What does the CSV look like?
    pass

if __name__ == '__main__':
    investigate()
