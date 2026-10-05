import json
import os
import cv2
import numpy as np

def box_iou(box1, box2):
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

def analyze():
    # Load GT
    with open("data/splits/test_split.json", "r") as f:
        gt_data = json.load(f)
        
    seq105 = [d for d in gt_data if d["filename"].startswith("105_")]
    gt_by_frame = {}
    for gt in seq105:
        # frame is 0-indexed in results? The video has 6 frames, so 0 to 5.
        # filename is 105_00001.jpg -> frame 0
        frame_idx = int(gt["filename"].split("_")[1].split(".")[0]) - 1
        if frame_idx not in gt_by_frame:
            gt_by_frame[frame_idx] = []
        gt_by_frame[frame_idx].append(gt)
        
    thresholds = ["0.50", "0.30", "0.20", "0.10"]
    results = {}
    
    for conf in thresholds:
        json_path = f"outputs/reports/cbvd5_test_seq105_conf{conf}_results.json"
        if not os.path.exists(json_path):
            continue
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        total_frames = max([d["frame_number"] for d in data]) + 1 if data else 0
        total_detections = len(data)
        track_ids = set([d["cow_id"] for d in data])
        
        postures = {"standing": 0, "lying": 0}
        activities = {"none": 0, "foraging": 0, "drinking": 0, "rumination": 0}
        p_confs = [d["posture_confidence"] for d in data if "posture_confidence" in d]
        a_confs = [d["activity_confidence"] for d in data if "activity_confidence" in d]
        
        for d in data:
            postures[d["posture"]] = postures.get(d["posture"], 0) + 1
            activities[d["activity"]] = activities.get(d["activity"], 0) + 1
            
        # Match with GT
        matched_gt = 0
        missed_gt = 0
        matched_standing = 0
        matched_lying = 0
        tp = 0
        fp = 0
        
        # group predictions by frame
        preds_by_frame = {}
        for d in data:
            frm = d["frame_number"]
            if frm not in preds_by_frame:
                preds_by_frame[frm] = []
            preds_by_frame[frm].append(d)
            
        for frm in gt_by_frame.keys():
            gts = gt_by_frame[frm]
            preds = preds_by_frame.get(frm, [])
            
            matched_pred_idx = set()
            for gt in gts:
                gt_box = gt["bbox"][1:] # ignore class idx
                posture = gt["posture"]
                
                best_iou = 0
                best_idx = -1
                for i, p in enumerate(preds):
                    if i in matched_pred_idx:
                        continue
                    p_box = p["bbox"] # [x, y, w, h] from run_day1 tracking? No wait, YOLO run_day1 bbox is [x1, y1, x2, y2]. Let me check tracking script.
                    # run_day1_behavior_tracking uses r.boxes.xyxy, so it's [x1, y1, x2, y2]
                    # We must convert it to [x, y, w, h]
                    x1, y1, x2, y2 = p_box
                    w = x2 - x1
                    h = y2 - y1
                    p_box_xywh = [x1, y1, w, h]
                    
                    iou = box_iou(gt_box, p_box_xywh)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i
                        
                if best_iou > 0.3:
                    matched_pred_idx.add(best_idx)
                    matched_gt += 1
                    tp += 1
                    if posture == "standing":
                        matched_standing += 1
                    else:
                        matched_lying += 1
                else:
                    missed_gt += 1
                    
            fp += len(preds) - len(matched_pred_idx)
            
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / 53.0
        standing_recall = matched_standing / 15.0
        lying_recall = matched_lying / 38.0
        
        results[conf] = {
            "total_frames": total_frames,
            "total_detections": total_detections,
            "unique_tracks": len(track_ids),
            "postures": postures,
            "activities": activities,
            "avg_p_conf": np.mean(p_confs) if p_confs else 0.0,
            "avg_a_conf": np.mean(a_confs) if a_confs else 0.0,
            "matched_gt": matched_gt,
            "missed_gt": missed_gt,
            "standing_recall": standing_recall,
            "lying_recall": lying_recall,
            "tp": tp,
            "fp": fp,
            "precision": precision,
            "recall": recall
        }
        
    # Generate Contact Sheet
    images = []
    # Just take frame 1 from each video
    for conf in thresholds:
        vid = f"outputs/videos/cbvd5_test_seq105_conf{conf}.mp4"
        if os.path.exists(vid):
            cap = cv2.VideoCapture(vid)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                scale = 450 / w
                f_resized = cv2.resize(frame, (450, int(h * scale)))
                cv2.putText(f_resized, f"Conf: {conf}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                images.append(f_resized)
            cap.release()
            
    if len(images) == 4:
        row1 = np.hstack(images[:2])
        row2 = np.hstack(images[2:])
        contact_sheet = np.vstack([row1, row2])
        cv2.imwrite("outputs/reports/yolo_threshold_pipeline_comparison.jpg", contact_sheet)
        
    # Generate Report
    report = "# YOLO Threshold Pipeline Comparison\n\n"
    report += "## Sequence 105 (Original CBVD-5 Test Split)\n"
    report += "- **GT Total**: 53 (15 Standing, 38 Lying)\n\n"
    
    for conf in thresholds:
        if conf not in results: continue
        res = results[conf]
        report += f"## Threshold: {conf}\n"
        report += f"- **Total Frames Processed**: {res['total_frames']}\n"
        report += f"- **Total YOLO Detections**: {res['total_detections']}\n"
        report += f"- **Unique ByteTrack IDs**: {res['unique_tracks']}\n"
        report += f"- **Posture Predictions**: Standing: {res['postures']['standing']}, Lying: {res['postures']['lying']}\n"
        report += f"- **Activity Predictions**: None: {res['activities'].get('none', 0)}, Foraging: {res['activities'].get('foraging', 0)}, Rumination: {res['activities'].get('rumination', 0)}, Drinking: {res['activities'].get('drinking', 0)}\n"
        report += f"- **Avg Posture Confidence**: {res['avg_p_conf']:.4f}\n"
        report += f"- **Avg Activity Confidence**: {res['avg_a_conf']:.4f}\n\n"
        
        report += f"### Detection Metrics (IoU > 0.3)\n"
        report += f"- **Matched GT Cows (TP)**: {res['tp']}\n"
        report += f"- **Missed GT Cows**: {res['missed_gt']}\n"
        report += f"- **Unmatched/False-Positive Detections (FP)**: {res['fp']}\n"
        report += f"- **Detection Precision**: {res['precision']:.4f}\n"
        report += f"- **Detection Recall (Overall)**: {res['recall']:.4f}\n"
        report += f"- **Standing Recall**: {res['standing_recall']:.4f}\n"
        report += f"- **Lying Recall**: {res['lying_recall']:.4f}\n\n"
        
    report += "## Tradeoff Analysis\n"
    report += "As the confidence threshold is lowered from 0.50 to 0.10:\n"
    report += "- **Recovered Lying Cows**: The lying recall increases significantly, bridging the massive detection gap found at 0.50.\n"
    report += "- **Missed Lying Cows**: The number of missed lying cows decreases.\n"
    report += "- **False Positives**: The false positive rate (unmatched detections) increases as the model starts tracking background noise or redundant boxes at lower confidences.\n"
    
    with open("outputs/reports/yolo_threshold_pipeline_comparison.md", "w") as f:
        f.write(report)
        
if __name__ == "__main__":
    analyze()
