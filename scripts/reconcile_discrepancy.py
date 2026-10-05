import json
import pandas as pd

def reconcile():
    csv_path = 'data/raw/cbvd-5/CBVD-5.csv'
    
    # Read total lines
    with open(csv_path, 'r') as f:
        total_lines = sum(1 for line in f)
        
    data_lines = total_lines - 10 # 10 header lines
    
    # Read via pandas
    df = pd.read_csv(csv_path, skiprows=10, header=None, names=['metadata_id', 'file_list', 'flags', 'temporal_coordinates', 'spatial_coordinates', 'metadata'])
    
    total_csv_rows = len(df)
    
    total_yolo_labels = 0
    multi_class_rows = 0
    
    unique_frames = set()
    
    for _, row in df.iterrows():
        try:
            f_list = json.loads(row['file_list'])
            if f_list: unique_frames.add(f_list[0])
            
            meta = json.loads(row['metadata'])
            if '1' in meta:
                classes = meta['1'].split(',')
                num_classes = len(classes)
                total_yolo_labels += num_classes
                if num_classes > 1:
                    multi_class_rows += 1
        except:
            pass

    print(f"Total lines in CSV: {total_lines}")
    print(f"Header lines: 10")
    print(f"Data rows: {total_csv_rows}")
    print(f"Unique frames: {len(unique_frames)}")
    print(f"Rows with multiple classes (comma-separated): {multi_class_rows}")
    print(f"Total projected YOLO labels (boxes x classes): {total_yolo_labels}")

if __name__ == '__main__':
    reconcile()
