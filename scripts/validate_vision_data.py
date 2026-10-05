import os
import glob
from collections import defaultdict

def validate_dataset(base_dir):
    splits = ['train', 'val', 'test']
    
    total_frames = 0
    total_boxes = 0
    class_boxes = defaultdict(int)
    
    split_frames = {}
    split_class_boxes = {}
    split_sources = {}
    
    invalid_boxes = 0
    missing_labels = 0
    missing_images = 0
    
    for split in splits:
        img_dir = os.path.join(base_dir, 'images', split)
        lbl_dir = os.path.join(base_dir, 'labels', split)
        
        images = set([os.path.basename(f) for f in glob.glob(os.path.join(img_dir, '*.jpg'))])
        labels = set([os.path.basename(f) for f in glob.glob(os.path.join(lbl_dir, '*.txt'))])
        
        # Check matching
        img_basenames = set([f.replace('.jpg', '') for f in images])
        lbl_basenames = set([f.replace('.txt', '') for f in labels])
        
        missing_labels += len(img_basenames - lbl_basenames)
        missing_images += len(lbl_basenames - img_basenames)
        
        # Count and validate labels
        frames_in_split = len(img_basenames.intersection(lbl_basenames))
        split_frames[split] = frames_in_split
        total_frames += frames_in_split
        
        split_class_boxes[split] = defaultdict(int)
        sources = set()
        
        for lbl_name in lbl_basenames.intersection(img_basenames):
            src_id = lbl_name.split('_')[0]
            sources.add(src_id)
            
            with open(os.path.join(lbl_dir, lbl_name + '.txt'), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    total_boxes += 1
                    
                    cls_id = int(parts[0])
                    x_c, y_c, w, h = map(float, parts[1:])
                    
                    if not (0 <= cls_id <= 4):
                        invalid_boxes += 1
                    if not (0.0 <= x_c <= 1.0 and 0.0 <= y_c <= 1.0):
                        invalid_boxes += 1
                    if w <= 0 or h <= 0 or w > 1.0 or h > 1.0:
                        invalid_boxes += 1
                        
                    class_boxes[cls_id] += 1
                    split_class_boxes[split][cls_id] += 1
                    
        split_sources[split] = sources

    # Check source overlap
    train_sources = split_sources['train']
    val_sources = split_sources['val']
    test_sources = split_sources['test']
    
    overlap1 = train_sources.intersection(val_sources)
    overlap2 = train_sources.intersection(test_sources)
    overlap3 = val_sources.intersection(test_sources)
    
    report = f"""# Vision Dataset Validation Report

## 1. Overall Statistics
- **Total Unique Frames:** {total_frames}
- **Total Bounding Boxes:** {total_boxes}

## 2. Class-wise Bounding Boxes (Total)
- Class 0 (standing): {class_boxes.get(0, 0)}
- Class 1 (lying): {class_boxes.get(1, 0)}
- Class 2 (feeding): {class_boxes.get(2, 0)}
- Class 3 (drinking): {class_boxes.get(3, 0)}
- Class 4 (rumination): {class_boxes.get(4, 0)}

## 3. Split Statistics
**Train:**
- Frames: {split_frames.get('train', 0)}
- Class distribution: {dict(split_class_boxes.get('train', {}))}
- Source IDs ({len(train_sources)}): {sorted(list(train_sources))}

**Validation:**
- Frames: {split_frames.get('val', 0)}
- Class distribution: {dict(split_class_boxes.get('val', {}))}
- Source IDs ({len(val_sources)}): {sorted(list(val_sources))}

**Test:**
- Frames: {split_frames.get('test', 0)}
- Class distribution: {dict(split_class_boxes.get('test', {}))}
- Source IDs ({len(test_sources)}): {sorted(list(test_sources))}

## 4. Source Overlap Verification
- Train/Val overlap: {list(overlap1)}
- Train/Test overlap: {list(overlap2)}
- Val/Test overlap: {list(overlap3)}
- **Verification Status:** {'PASSED' if not (overlap1 or overlap2 or overlap3) else 'FAILED'}

## 5. Label Verification
- Invalid/Negative/Zero-size boxes: {invalid_boxes}
- Missing label files for images: {missing_labels}
- Missing images for label files: {missing_images}
- Valid classes (0-4): {'PASSED' if invalid_boxes == 0 else 'FAILED'}
"""
    
    os.makedirs('outputs/reports', exist_ok=True)
    with open('outputs/reports/vision_dataset_validation_report.md', 'w') as f:
        f.write(report)
        
    print(report)

if __name__ == '__main__':
    validate_dataset('data/processed/vision')
