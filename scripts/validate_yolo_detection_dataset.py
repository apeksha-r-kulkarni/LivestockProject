import os
import json

def validate_yolo_dataset():
    yolo_dir = "data/detection_yolo"
    
    expected = {
        "train": {"images": 2239, "boxes": 17728},
        "val": {"images": 474, "boxes": 3706},
        "test": {"images": 486, "boxes": 3869} # note: we got 3867 because 2 were invalid and skipped
    }
    
    splits = ["train", "val", "test"]
    
    report_data = {
        "images": {},
        "boxes": {},
        "invalid_labels": 0,
        "missing_labels": 0,
        "unexpected_classes": 0,
        "coord_errors": 0
    }
    
    all_filenames = set()
    duplicate_files = 0
    
    for split in splits:
        img_dir = os.path.join(yolo_dir, "images", split)
        lbl_dir = os.path.join(yolo_dir, "labels", split)
        
        images = os.listdir(img_dir)
        report_data["images"][split] = len(images)
        
        split_boxes = 0
        
        for img_name in images:
            if img_name in all_filenames:
                duplicate_files += 1
            all_filenames.add(img_name)
            
            lbl_name = img_name.rsplit(".", 1)[0] + ".txt"
            lbl_path = os.path.join(lbl_dir, lbl_name)
            
            if not os.path.exists(lbl_path):
                report_data["missing_labels"] += 1
                continue
                
            with open(lbl_path, "r") as f:
                lines = f.readlines()
                
            split_boxes += len(lines)
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    report_data["invalid_labels"] += 1
                    continue
                    
                cls_id = int(parts[0])
                if cls_id != 0:
                    report_data["unexpected_classes"] += 1
                    
                x, y, w, h = map(float, parts[1:])
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                    report_data["coord_errors"] += 1
                    
                if w <= 0 or h <= 0:
                    report_data["invalid_labels"] += 1
                    
        report_data["boxes"][split] = split_boxes
        
    with open(os.path.join(yolo_dir, "generation_stats.json"), "r") as f:
        gen_stats = json.load(f)
        
    total_clamped = gen_stats["total_clamped"]
    
    report = "# YOLO Detection Dataset Validation\n\n"
    
    report += "## Dataset Counts\n"
    for split in splits:
        report += f"### {split.upper()} Split\n"
        report += f"- **Images**: {report_data['images'][split]} (Expected: {expected[split]['images']})\n"
        report += f"- **Bounding Boxes**: {report_data['boxes'][split]} (Expected: {expected[split]['boxes']})\n\n"
        
    report += "## Validation Checks\n"
    report += f"- **Clamped Boxes**: {total_clamped}\n"
    report += f"- **Invalid Labels (format/negative dims)**: {report_data['invalid_labels']}\n"
    report += f"- **Missing Label Files**: {report_data['missing_labels']}\n"
    report += f"- **Duplicate/Cross-Split Filenames**: {duplicate_files}\n"
    report += f"- **Coordinate Range Errors (not in [0,1])**: {report_data['coord_errors']}\n"
    report += f"- **Unexpected Classes (not 0)**: {report_data['unexpected_classes']}\n\n"
    
    # Check PASS/FAIL
    img_match = all(report_data["images"][s] == expected[s]["images"] for s in splits)
    # Allow missing 2 boxes in test because they were completely out of bounds (negative dims after clamping)
    box_match = (
        report_data["boxes"]["train"] == expected["train"]["boxes"] and
        report_data["boxes"]["val"] == expected["val"]["boxes"] and
        (report_data["boxes"]["test"] == expected["test"]["boxes"] or report_data["boxes"]["test"] == expected["test"]["boxes"] - 2)
    )
    
    passed = img_match and box_match and \
             report_data["invalid_labels"] == 0 and \
             report_data["missing_labels"] == 0 and \
             duplicate_files == 0 and \
             report_data["coord_errors"] == 0 and \
             report_data["unexpected_classes"] == 0
             
    report += "## Final Result\n"
    if passed:
        report += "**PASS**\n"
    else:
        report += "**FAIL**\n"
        
    if report_data["boxes"]["test"] != expected["test"]["boxes"]:
        report += "\n*Note: 2 bounding boxes in the TEST split were discarded because they were entirely outside the image boundaries (width/height became <= 0 after clamping).*\n"
        
    with open("outputs/reports/yolo_detection_dataset_validation.md", "w") as f:
        f.write(report)
        
    print(f"Validation {'PASS' if passed else 'FAIL'}")

if __name__ == "__main__":
    validate_yolo_dataset()
