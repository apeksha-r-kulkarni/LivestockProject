import os
import json
import shutil
import cv2

def create_yolo_dataset():
    # Base paths
    raw_img_dir = "data/raw/cbvd-5/images/"
    yolo_dir = "data/detection_yolo"
    
    # Create directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(yolo_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(yolo_dir, "labels", split), exist_ok=True)
        
    splits = ["train", "val", "test"]
    stats = {}
    total_clamped = 0
    
    img_dims_cache = {}
    
    for split in splits:
        with open(f"data/splits/{split}_split.json", "r") as f:
            data = json.load(f)
            
        print(f"Processing {split} split...")
        
        # Group by filename
        boxes_by_file = {}
        for d in data:
            filename = d["filename"]
            if filename not in boxes_by_file:
                boxes_by_file[filename] = []
            
            # test_split might have 5 elements, train/val 4. Safe parsing:
            bbox = d["bbox"]
            if len(bbox) == 5:
                x, y, w, h = bbox[1:]
            else:
                x, y, w, h = bbox
                
            boxes_by_file[filename].append([x, y, w, h])
            
        split_img_count = 0
        split_box_count = 0
        
        for filename, boxes in boxes_by_file.items():
            src_img = os.path.join(raw_img_dir, filename)
            dst_img = os.path.join(yolo_dir, "images", split, filename)
            
            # Copy image
            if not os.path.exists(dst_img):
                shutil.copy(src_img, dst_img)
                
            if filename not in img_dims_cache:
                img = cv2.imread(src_img)
                if img is not None:
                    img_dims_cache[filename] = img.shape[:2] # h, w
                else:
                    print(f"Warning: could not read {src_img}")
                    continue
                    
            img_h, img_w = img_dims_cache[filename]
            
            label_lines = []
            for box in boxes:
                x, y, w, h = box
                
                orig_x, orig_y, orig_w, orig_h = x, y, w, h
                clamped = False
                
                # Clamp coordinates safely
                if x < 0:
                    w += x # shrink width
                    x = 0
                    clamped = True
                if y < 0:
                    h += y # shrink height
                    y = 0
                    clamped = True
                    
                if x + w > img_w:
                    w = img_w - x
                    clamped = True
                if y + h > img_h:
                    h = img_h - y
                    clamped = True
                    
                # If width or height become <= 0 after clamping, ignore (should not happen for valid boxes)
                if w <= 0 or h <= 0:
                    continue
                    
                if clamped:
                    total_clamped += 1
                    
                # YOLO format: [class_id, x_center, y_center, width, height] normalized
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                
                # Ensure strictly in [0, 1] bounds just in case floating point inaccuracies
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                w_norm = max(0.0, min(1.0, w_norm))
                h_norm = max(0.0, min(1.0, h_norm))
                
                label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                split_box_count += 1
                
            # Write labels
            label_filename = filename.rsplit(".", 1)[0] + ".txt"
            label_path = os.path.join(yolo_dir, "labels", split, label_filename)
            with open(label_path, "w") as f:
                f.write("\n".join(label_lines) + "\n")
                
            split_img_count += 1
            
        stats[split] = {
            "images": split_img_count,
            "boxes": split_box_count
        }
        
    print(f"Total clamped boxes: {total_clamped}")
    for k, v in stats.items():
        print(f"{k.upper()} - Images: {v['images']}, Boxes: {v['boxes']}")
        
    # Write stats to file for validation script
    with open(os.path.join(yolo_dir, "generation_stats.json"), "w") as f:
        json.dump({"stats": stats, "total_clamped": total_clamped}, f)

    # Create data.yaml
    yaml_content = f"""train: {os.path.abspath(os.path.join(yolo_dir, 'images', 'train'))}
val: {os.path.abspath(os.path.join(yolo_dir, 'images', 'val'))}
test: {os.path.abspath(os.path.join(yolo_dir, 'images', 'test'))}

nc: 1
names:
  0: cow
"""
    with open(os.path.join(yolo_dir, "data.yaml"), "w") as f:
        f.write(yaml_content)
        
    print("Done")

if __name__ == "__main__":
    create_yolo_dataset()
