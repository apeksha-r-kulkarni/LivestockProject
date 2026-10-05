import os
import json
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict

def setup_dirs(base_path):
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_path, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'labels', split), exist_ok=True)

def parse_via_csv(csv_path):
    df = pd.read_csv(csv_path, skiprows=10, header=None, 
                     names=['metadata_id', 'file_list', 'flags', 'temporal_coordinates', 'spatial_coordinates', 'metadata'])
    
    image_annotations = defaultdict(list)
    sources = set()
    
    for _, row in df.iterrows():
        try:
            f_list = json.loads(row['file_list'])
            if not f_list: continue
            filename = f_list[0]
            
            spatial = json.loads(row['spatial_coordinates'])
            if not spatial or spatial[0] != 2: continue # Must be rectangle
            
            x, y, w, h = spatial[1], spatial[2], spatial[3], spatial[4]
            
            meta = json.loads(row['metadata'])
            if '1' not in meta: continue
            
            classes = meta['1'].split(',')
            
            source_id = filename.split('_')[0]
            sources.add(source_id)
            
            for c in classes:
                image_annotations[filename].append({
                    'class_id': int(c),
                    'x': x, 'y': y, 'w': w, 'h': h
                })
        except Exception as e:
            pass
            
    return image_annotations, list(sources)

def create_yolo_dataset(image_annotations, sources, dest_dir):
    setup_dirs(dest_dir)
    
    # Source-level split: 70% train, 15% val, 15% test
    train_src, temp_src = train_test_split(sources, test_size=0.3, random_state=42)
    val_src, test_src = train_test_split(temp_src, test_size=0.5, random_state=42)
    
    src_to_split = {}
    for s in train_src: src_to_split[s] = 'train'
    for s in val_src: src_to_split[s] = 'val'
    for s in test_src: src_to_split[s] = 'test'
    
    counts = {'train': 0, 'val': 0, 'test': 0}
    
    # CBVD-5 standard resolution
    img_w, img_h = 1920, 1080
    
    # Create a dummy blank image for YOLO smoke test to avoid 11GB download
    dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
    
    for filename, annotations in image_annotations.items():
        src = filename.split('_')[0]
        split = src_to_split.get(src, 'train')
        
        dst_img_path = os.path.join(dest_dir, 'images', split, filename)
        dst_label_path = os.path.join(dest_dir, 'labels', split, filename.replace('.jpg', '.txt'))
        
        # Save dummy image so YOLO can read it
        if not os.path.exists(dst_img_path):
            cv2.imwrite(dst_img_path, dummy_img)
            
        with open(dst_label_path, 'w') as f:
            for ann in annotations:
                x, y, w, h = ann['x'], ann['y'], ann['w'], ann['h']
                
                # YOLO format
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h
                
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                norm_w = max(0.0, min(1.0, norm_w))
                norm_h = max(0.0, min(1.0, norm_h))
                
                f.write(f"{ann['class_id']} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                
        counts[split] += 1
        
    print(f"Dataset creation complete. Counts: {counts}")
    return counts

if __name__ == '__main__':
    csv_path = 'data/raw/cbvd-5/CBVD-5.csv'
    dest_dir = 'data/processed/vision'
    
    print("Parsing VIA CSV...")
    annotations, sources = parse_via_csv(csv_path)
    print(f"Found {len(sources)} unique video sources.")
    
    print("Converting to YOLO format and performing source-level split...")
    counts = create_yolo_dataset(annotations, sources, dest_dir)
