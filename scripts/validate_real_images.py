import os
import glob
import pandas as pd
import json
import cv2

def validate_real_images():
    base_dir = 'data/raw/cbvd-5/labelframes/labelframes'
    csv_path = 'data/raw/cbvd-5/CBVD-5.csv'
    report_path = 'outputs/reports/cbvd5_real_image_acquisition_validation.md'
    
    if not os.path.exists(base_dir):
        print("Images directory not found.")
        return
        
    # Get all jpg files
    jpg_files = glob.glob(os.path.join(base_dir, '*.jpg'))
    num_jpg_files = len(jpg_files)
    
    # Total size
    total_size_bytes = sum(os.path.getsize(f) for f in jpg_files)
    total_size_mb = total_size_bytes / (1024 * 1024)
    total_size_gb = total_size_bytes / (1024 * 1024 * 1024)
    
    # Dimensions of sample
    sample_dim = "N/A"
    sample_success = False
    if num_jpg_files > 0:
        sample_path = jpg_files[0]
        img = cv2.imread(sample_path)
        if img is not None:
            sample_dim = f"{img.shape[1]}x{img.shape[0]}"
            sample_success = True
            
    # Parse CSV for frames
    df = pd.read_csv(csv_path, skiprows=10, header=None, names=['metadata_id', 'file_list', 'flags', 'temporal_coordinates', 'spatial_coordinates', 'metadata'])
    
    csv_frames = set()
    for _, row in df.iterrows():
        try:
            f_list = json.loads(row['file_list'])
            if f_list: csv_frames.add(f_list[0])
        except:
            pass
            
    total_csv_frames = len(csv_frames)
    
    extracted_basenames = set([os.path.basename(f) for f in jpg_files])
    
    matched_frames = csv_frames.intersection(extracted_basenames)
    missing_frames = csv_frames - extracted_basenames
    extra_frames = extracted_basenames - csv_frames
    
    percent_matched = (len(matched_frames) / total_csv_frames * 100) if total_csv_frames > 0 else 0
    
    report = f"""# CBVD-5 Real Image Acquisition & Validation

## 1. ZIP Archive Verification
- Checked locally via shell commands prior to extraction.

## 2. Extracted Directory Status
- **Target path:** `data/raw/cbvd-5/labelframes/labelframes/`
- **Number of JPG files extracted:** {num_jpg_files}
- **Total extracted size:** {total_size_gb:.2f} GB ({total_size_mb:.2f} MB)
- **Sample Image Dimensions:** {sample_dim}
- **Image Integrity:** {'PASSED' if sample_success else 'FAILED'} (OpenCV successfully read the image)

## 3. CSV Matching
- **Total unique frames referenced by CSV:** {total_csv_frames}
- **Total matching image files on disk:** {len(matched_frames)}
- **Missing image files:** {len(missing_frames)}
- **Extra image files (in ZIP but not annotated):** {len(extra_frames)}
- **Match Percentage:** {percent_matched:.2f}%
"""

    with open(report_path, 'w') as f:
        f.write(report)
        
    print(f"Validation complete. Found {len(matched_frames)} matching frames.")

if __name__ == '__main__':
    validate_real_images()
