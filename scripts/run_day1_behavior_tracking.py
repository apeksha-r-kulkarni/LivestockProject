import cv2
import json
import torch
import os
import time
import subprocess
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from scripts.train_behavior_classifier import MultiTaskCowBehaviorModel, SquarePadResize

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run YOLO -> ByteTrack -> Behavior Classifier pipeline")
    parser.add_argument("--input_video", type=str, default="data/test_100.mp4", help="Path to the input video")
    parser.add_argument("--output_video", type=str, default="outputs/videos/day1_behavior_tracking_test.mp4", help="Path to save annotated video")
    parser.add_argument("--output_json", type=str, default="outputs/reports/day1_behavior_tracking_results.json", help="Path to save tracking JSON")
    parser.add_argument("--output_report", type=str, default="outputs/reports/day1_behavior_tracking_report.md", help="Path to save summary report")
    parser.add_argument("--v2_preprocessing", action="store_true", help="Use V2 aspect-ratio-preserving square padding instead of squashing")
    parser.add_argument("--behavior_ckpt", type=str, default="outputs/models/best_behavior_classifier.pth", help="Path to behavior classifier weights")
    parser.add_argument("--conf", type=float, default=0.50, help="YOLO confidence threshold")
    parser.add_argument("--yolo_ckpt", type=str, default="yolov8n.pt", help="Path to YOLO detector weights")
    args = parser.parse_args()

    # Paths
    yolo_ckpt = args.yolo_ckpt
    behavior_ckpt = args.behavior_ckpt
    video_path = args.input_video
    output_video_path = args.output_video
    results_json_path = args.output_json
    report_md_path = args.output_report

    # Make dirs
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    os.makedirs(os.path.dirname(results_json_path), exist_ok=True)

    # Device
    device = torch.device('cpu')

    # Load YOLO
    print("Loading YOLO model...")
    detector = YOLO(yolo_ckpt)

    # Load Behavior Model
    print("Loading behavior model...")
    ckpt = torch.load(behavior_ckpt, map_location=device)
    behavior_model = MultiTaskCowBehaviorModel(backbone_name=ckpt.get('backbone', 'mobilenet_v3_large'))
    behavior_model.load_state_dict(ckpt['model_state_dict'])
    behavior_model.eval()
    behavior_model.to(device)

    # Transforms
    if args.v2_preprocessing:
        print("Using V2 Preprocessing: SquarePadResize")
        base_resize = SquarePadResize(224)
    else:
        print("Using V1 Preprocessing: Direct Resize (Squash)")
        base_resize = transforms.Resize((224, 224))
        
    transform = transforms.Compose([
        base_resize,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Video setup
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Robust FPS calculation
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    effective_fps = cap.get(cv2.CAP_PROP_FPS) # default fallback
    
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=duration,nb_frames", 
            "-of", "json", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        probe = json.loads(result.stdout)
        stream = probe.get("streams", [{}])[0]
        
        dur_str = stream.get("duration")
        frames_str = stream.get("nb_frames")
        
        actual_duration = float(dur_str) if dur_str else None
        actual_frames = float(frames_str) if frames_str else frame_count
        
        if actual_duration and actual_duration > 0 and actual_frames:
            effective_fps = float(actual_frames) / actual_duration
    except Exception as e:
        print(f"Warning: ffprobe failed to get true FPS, falling back to cv2. {e}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, effective_fps, (width, height))

    # Tracking iterator
    track_classes = [0] if "cbvd5" in yolo_ckpt else [19]
    results_iter = detector.track(source=video_path, tracker="bytetrack.yaml", classes=track_classes, persist=True, stream=True, verbose=False, conf=args.conf)

    tracking_results = []
    
    total_frames = 0
    unique_ids = set()
    total_detections = 0
    total_predictions = 0

    print("Starting processing loop...")
    for r in results_iter:
        frame_number = total_frames
        timestamp = frame_number / effective_fps if effective_fps > 0 else 0
        total_frames += 1
        
        orig_img = r.orig_img
        annotated_frame = orig_img.copy()
        
        if r.boxes is not None and len(r.boxes) > 0 and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            track_ids = r.boxes.id.int().cpu().tolist()
            confs = r.boxes.conf.cpu().numpy()
            
            for box, track_id, yolo_conf in zip(boxes, track_ids, confs):
                unique_ids.add(track_id)
                total_detections += 1
                
                # Bounding box
                x1, y1, x2, y2 = [int(v) for v in box]
                # Clamp coordinates
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue  # Too small
                    
                # Crop
                crop = orig_img[y1:y2, x1:x2]
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(crop_rgb)
                
                # Preprocess and infer
                input_tensor = transform(pil_img).to(device)
                
                with torch.no_grad():
                    b_out = behavior_model.predict_state(input_tensor, cow_id=track_id, timestamp=timestamp)
                
                total_predictions += 1
                
                # Record
                tracking_results.append({
                    "frame_number": frame_number,
                    "timestamp": timestamp,
                    "cow_id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "posture": b_out['posture'],
                    "posture_confidence": b_out['posture_confidence'],
                    "activity": b_out['activity'],
                    "activity_confidence": b_out['activity_confidence']
                })
                
                # Overlay
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label1 = f"ID: {track_id}"
                label2 = f"{b_out['posture']} ({b_out['posture_confidence']:.2f})"
                label3 = f"{b_out['activity']} ({b_out['activity_confidence']:.2f})"
                
                cv2.putText(annotated_frame, label1, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(annotated_frame, label2, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(annotated_frame, label3, (x1, y1 - 5),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
        out_video.write(annotated_frame)

    cap.release()
    out_video.release()

    # Save JSON
    with open(results_json_path, 'w') as f:
        json.dump(tracking_results, f, indent=4)
        
    # Generate Report
    with open(report_md_path, 'w') as f:
        f.write("# Day 1 Behavior Tracking Report\n\n")
        f.write(f"- **Total Frames Processed**: {total_frames}\n")
        f.write(f"- **Unique Track IDs**: {len(unique_ids)} ({list(unique_ids)})\n")
        f.write(f"- **Cow Detections**: {total_detections}\n")
        f.write(f"- **Behavior Predictions**: {total_predictions}\n")
        f.write(f"- **Output Video**: `{output_video_path}`\n")
        f.write(f"- **Output JSON**: `{results_json_path}`\n")

    print(f"Pipeline complete! Processed {total_frames} frames.")

if __name__ == '__main__':
    main()
