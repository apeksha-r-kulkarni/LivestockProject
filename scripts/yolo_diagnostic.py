import json
import os
import cv2
import numpy as np
from ultralytics import YOLO

def box_iou(box1, box2):
    # box format: [x, y, w, h]
    # convert to [x1, y1, x2, y2]
    b1_x1, b1_y1, b1_w, b1_h = box1
    b1_x2, b1_y2 = b1_x1 + b1_w, b1_y1 + b1_h
    b2_x1, b2_y1, b2_w, b2_h = box2
    b2_x2, b2_y2 = b2_x1 + b2_w, b2_y1 + b2_h
    
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
        
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    b1_area = b1_w * b1_h
    b2_area = b2_w * b2_h
    
    return inter_area / float(b1_area + b2_area - inter_area)

def main():
    img_dir = "data/raw/cbvd-5/images/"
    seq = "105"
    frames = sorted([f for f in os.listdir(img_dir) if f.startswith(f"{seq}_")])
    
    # Load GT
    with open("data/splits/test_split.json", "r") as f:
        gt_data = json.load(f)
        
    gt_by_frame = {}
    for f in frames:
        gt_by_frame[f] = [d for d in gt_data if d["filename"] == f]
        
    # YOLO
    model = YOLO("yolov8n.pt")
    
    thresholds = [0.50, 0.30, 0.20, 0.10]
    results_by_thresh = {}
    
    out_dir = "outputs/reports/yolo_cbvd5_seq105_diagnostic"
    os.makedirs(out_dir, exist_ok=True)
    
    for conf in thresholds:
        stats = {
            "total_detections": 0,
            "confs": [],
            "matched_standing": 0,
            "matched_lying": 0,
            "missed_standing": 0,
            "missed_lying": 0,
            "detections_per_frame": []
        }
        
        for f in frames:
            img_path = os.path.join(img_dir, f)
            img = cv2.imread(img_path)
            
            # Predict
            res = model.predict(img_path, conf=conf, classes=[19], iou=0.45, verbose=False)[0]
            boxes = res.boxes.xywh.cpu().numpy()  # [x_c, y_c, w, h]
            confs = res.boxes.conf.cpu().numpy()
            
            # Convert to [x, y, w, h]
            pred_boxes = []
            for b, c in zip(boxes, confs):
                x_c, y_c, w, h = b
                pred_boxes.append([x_c - w/2, y_c - h/2, w, h, c])
                
            stats["total_detections"] += len(pred_boxes)
            stats["detections_per_frame"].append(len(pred_boxes))
            stats["confs"].extend(confs)
            
            # Match with GT (IoU > 0.3)
            matched_pred_idx = set()
            for gt in gt_by_frame[f]:
                gt_box = gt["bbox"]
                posture = gt["posture"]
                
                best_iou = 0
                best_idx = -1
                for i, p_box in enumerate(pred_boxes):
                    if i in matched_pred_idx:
                        continue
                    iou = box_iou(gt_box[1:], p_box[:4])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i
                        
                if best_iou > 0.3:
                    matched_pred_idx.add(best_idx)
                    if posture == "standing":
                        stats["matched_standing"] += 1
                    else:
                        stats["matched_lying"] += 1
                else:
                    if posture == "standing":
                        stats["missed_standing"] += 1
                    else:
                        stats["missed_lying"] += 1
                        
            # Save visual for conf 0.10
            if conf == 0.10:
                vis_img = img.copy()
                # Draw GT
                for gt in gt_by_frame[f]:
                    x, y, w, h = map(int, gt["bbox"][1:])
                    posture = gt["posture"]
                    color = (0, 0, 255) if posture == "standing" else (255, 0, 0)
                    cv2.rectangle(vis_img, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(vis_img, f"GT:{posture}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                # Draw Pred
                for p_box in pred_boxes:
                    x, y, w, h, c = p_box
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(vis_img, f"{c:.2f}", (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                cv2.imwrite(os.path.join(out_dir, f), vis_img)
                
        results_by_thresh[conf] = stats

    # Create contact sheet for 0.10
    images = []
    for f in frames:
        img = cv2.imread(os.path.join(out_dir, f))
        h, w = img.shape[:2]
        scale = 600 / w
        images.append(cv2.resize(img, (600, int(h * scale))))
        
    if images:
        row1 = np.hstack(images[:3])
        row2 = np.hstack(images[3:])
        contact_sheet = np.vstack([row1, row2])
        cv2.imwrite("outputs/reports/yolo_cbvd5_seq105_diagnostic.jpg", contact_sheet)
        
    # Generate report
    report = "# YOLO CBVD-5 Sequence 105 Diagnostic\n\n"
    for conf, st in results_by_thresh.items():
        report += f"## Threshold: {conf}\n"
        report += f"- Total Detections: {st['total_detections']}\n"
        report += f"- Detections per frame: {np.mean(st['detections_per_frame']):.1f}\n"
        avg_conf = np.mean(st['confs']) if st['confs'] else 0
        min_conf = np.min(st['confs']) if st['confs'] else 0
        max_conf = np.max(st['confs']) if st['confs'] else 0
        report += f"- Avg Conf: {avg_conf:.3f}, Min: {min_conf:.3f}, Max: {max_conf:.3f}\n"
        report += f"- Matched Standing: {st['matched_standing']} / 15 (Missed: {st['missed_standing']})\n"
        report += f"- Matched Lying: {st['matched_lying']} / 38 (Missed: {st['missed_lying']})\n\n"
        
    # Analysis logic
    st_01 = results_by_thresh[0.10]
    matched_lying_01 = st_01["matched_lying"]
    missed_lying_01 = st_01["missed_lying"]
    
    if matched_lying_01 < 10:
        conclusion = "DETECTOR GENERALIZATION ISSUE"
        reasoning = "Even at 0.10 confidence, the vast majority of lying cows are completely missed by YOLO."
    elif matched_lying_01 > 30 and results_by_thresh[0.50]["missed_lying"] > 20:
        conclusion = "THRESHOLD ISSUE"
        reasoning = "Lowering the threshold to 0.10 successfully recovers most of the lying cows."
    else:
        conclusion = "MIXED ISSUE"
        reasoning = "Lowering threshold recovers some cows, but many remain undetected even at 0.10."
        
    report += f"""
## Analysis
- **Does lowering the confidence threshold recover the missing lying cows?**
  At 0.10, {matched_lying_01} lying cows are detected, while {missed_lying_01} remain completely undetected.
- **Are the missing lying cows genuinely not detected?**
  Yes, {missed_lying_01} lying cows were completely missed by YOLO even at 0.10 confidence.
- **Is the issue primarily confidence thresholding or a detector domain/generalization problem?**
  {reasoning}
- **How many ground-truth cows remain undetected even at confidence 0.10?**
  {st_01['missed_standing']} standing and {st_01['missed_lying']} lying.

## Conclusion
**{conclusion}**
"""
    with open("outputs/reports/yolo_cbvd5_seq105_diagnostic.md", "w") as f:
        f.write(report)
        
    print("Done")

if __name__ == "__main__":
    main()
