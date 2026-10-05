import csv
import json
import os
from collections import defaultdict, Counter

csv_path = "data/raw/cbvd-5/CBVD-5.csv"
class_map = {"0":"stand", "1":"lying down", "2":"foraging", "3":"drinking water", "4":"rumination"}

# Stats
total_boxes = 0
valid_boxes = 0
anomalous_boxes = 0

posture_counts = Counter()
activity_counts = Counter()
combo_counts = Counter()

unique_images = set()

anomalies = []

with open(csv_path, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if not row or row[0].startswith("#"):
            continue
        try:
            f_list = json.loads(row[1])
            fname = f_list[0]
            
            bbox = json.loads(row[4])
            if len(bbox) == 0:
                continue
                
            metadata = json.loads(row[5])
            val = metadata.get("1", "")
            classes = [c.strip() for c in val.split(",") if c.strip()]
            
            total_boxes += 1
            unique_images.add(fname)
            
            names = set([class_map.get(c, c) for c in classes])
            
            # Logic
            posture = None
            activity = "none"
            is_anomaly = False
            reason = ""
            
            has_stand = "stand" in names
            has_lying = "lying down" in names
            
            if has_stand and has_lying:
                is_anomaly = True
                reason = "stand + lying down contradiction"
            elif not has_stand and not has_lying:
                is_anomaly = True
                reason = "activity without posture"
            else:
                posture = "standing" if has_stand else "lying"
            
            if "foraging" in names: activity = "foraging"
            if "drinking water" in names: activity = "drinking"
            if "rumination" in names: activity = "rumination"
            
            if is_anomaly:
                anomalous_boxes += 1
                anomalies.append({
                    "filename": fname,
                    "bbox": bbox,
                    "raw_metadata": val,
                    "decoded": list(names),
                    "reason": reason
                })
            else:
                valid_boxes += 1
                posture_counts[posture] += 1
                activity_counts[activity] += 1
                combo_counts[(posture, activity)] += 1
                
        except Exception as e:
            pass

boxes_per_image = total_boxes / len(unique_images) if unique_images else 0

# Write Report
report_path = "outputs/reports/cbvd5_annotation_mapping_report.md"
os.makedirs(os.path.dirname(report_path), exist_ok=True)
with open(report_path, "w") as f:
    f.write("# CBVD-5 Annotation Mapping & Anomaly Report\n\n")
    f.write("## 1. Dataset Statistics\n")
    f.write(f"- **Total Annotated Boxes**: {total_boxes}\n")
    f.write(f"- **Valid Boxes**: {valid_boxes}\n")
    f.write(f"- **Anomalous Boxes**: {anomalous_boxes}\n")
    f.write(f"- **Unique Images**: {len(unique_images)}\n")
    f.write(f"- **Average Boxes per Image**: {boxes_per_image:.2f}\n\n")
    
    f.write("## 2. Valid Combinations (Posture x Activity)\n")
    for combo, count in combo_counts.items():
        f.write(f"- Posture: `{combo[0]}`, Activity: `{combo[1]}` -> {count} boxes\n")
    
    f.write("\n## 3. Class Counts\n")
    f.write("### Postures\n")
    for p, c in posture_counts.items():
        f.write(f"- {p}: {c}\n")
    f.write("\n### Activities\n")
    for a, c in activity_counts.items():
        f.write(f"- {a}: {c}\n")
        
    f.write("\n## 4. Proposed Normalized Dataset Schema\n")
    f.write("For the behavior classifier, the dataset will be normalized as crops of each valid bounding box, organized in a directory structure suitable for multi-task learning or with a CSV mapping:\n")
    f.write("```json\n")
    f.write("[\n")
    f.write("  {\n")
    f.write('    "crop_id": "618_00002_box1",\n')
    f.write('    "image_file": "618_00002.jpg",\n')
    f.write('    "bbox": [x, y, w, h],\n')
    f.write('    "posture": "standing",\n')
    f.write('    "activity": "foraging"\n')
    f.write("  }\n")
    f.write("]\n")
    f.write("```\n")
    f.write("This allows PyTorch datasets to efficiently load the crop and return a label tuple `(posture_idx, activity_idx)`.\n\n")

    f.write("## 5. Anomalous Combinations Report\n")
    f.write("The following boxes were flagged and skipped in the valid count:\n\n")
    for idx, anom in enumerate(anomalies):
        f.write(f"### Anomaly {idx+1}\n")
        f.write(f"- **Filename**: `{anom['filename']}`\n")
        f.write(f"- **Bounding Box**: `{anom['bbox']}`\n")
        f.write(f"- **Raw Metadata**: `{anom['raw_metadata']}`\n")
        f.write(f"- **Decoded Labels**: `{anom['decoded']}`\n")
        f.write(f"- **Reason for Flagging**: {anom['reason']}\n\n")

print("Report generated at", report_path)
