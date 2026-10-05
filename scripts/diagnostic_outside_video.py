import cv2
import torch
import os
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from scripts.train_behavior_classifier import MultiTaskCowBehaviorModel

def main():
    video_path = os.path.expanduser("~/Downloads/WhatsApp Video 2026-09-30 at 8.17.15 PM.mp4")
    if not os.path.exists(video_path):
        video_path = "outputs/videos/outside_cattle_test.mp4"
    
    out_dir = "outputs/reports/outside_cattle_crops"
    report_md_path = "outputs/reports/outside_cattle_crop_diagnostic.md"
    
    os.makedirs(out_dir, exist_ok=True)
    
    yolo_ckpt = "yolov8n.pt"
    behavior_ckpt = "outputs/models/best_behavior_classifier.pth"
    
    device = torch.device('cpu')
    detector = YOLO(yolo_ckpt)
    
    ckpt = torch.load(behavior_ckpt, map_location=device)
    behavior_model = MultiTaskCowBehaviorModel(backbone_name=ckpt.get('backbone', 'mobilenet_v3_large'))
    behavior_model.load_state_dict(ckpt['model_state_dict'])
    behavior_model.eval()
    behavior_model.to(device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return
        
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    results_iter = detector.track(source=video_path, tracker="bytetrack.yaml", classes=[19], persist=True, stream=True, verbose=False)
    
    saved_crops_per_id = {}
    target_max = 5
    
    diagnostic_logs = []
    total_frames = 0
    
    for r in results_iter:
        frame_number = total_frames
        timestamp = frame_number / fps if fps > 0 else 0
        total_frames += 1
        
        orig_img = r.orig_img
        
        if r.boxes is not None and len(r.boxes) > 0 and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            track_ids = r.boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                if track_id not in saved_crops_per_id:
                    saved_crops_per_id[track_id] = 0
                    
                if saved_crops_per_id[track_id] >= target_max:
                    continue
                    
                # We want to save this crop
                x1, y1, x2, y2 = [int(v) for v in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue
                    
                crop = orig_img[y1:y2, x1:x2]
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(crop_rgb)
                
                input_tensor = transform(pil_img).to(device)
                
                with torch.no_grad():
                    b_out = behavior_model.predict_state(input_tensor, cow_id=track_id, timestamp=timestamp)
                    
                saved_crops_per_id[track_id] += 1
                
                # Save crop
                crop_fname = f"crop_id{track_id}_frame{frame_number}.jpg"
                crop_path = os.path.join(out_dir, crop_fname)
                cv2.imwrite(crop_path, crop)
                
                # Save annotated full frame
                frame_fname = f"frame_id{track_id}_frame{frame_number}.jpg"
                frame_path = os.path.join(out_dir, frame_fname)
                annotated_frame = orig_img.copy()
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"ID: {track_id}", (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.imwrite(frame_path, annotated_frame)
                
                log = {
                    "frame": frame_number,
                    "timestamp": timestamp,
                    "cow_id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "posture": b_out['posture'],
                    "posture_conf": b_out['posture_confidence'],
                    "activity": b_out['activity'],
                    "activity_conf": b_out['activity_confidence']
                }
                diagnostic_logs.append(log)
                print(f"Captured Frame {frame_number}, ID {track_id}, Posture: {log['posture']} ({log['posture_conf']:.2f})")
                
    cap.release()
    
    with open(report_md_path, 'w') as f:
        f.write("# Diagnostic Report: Outside Cattle Video\n\n")
        f.write("This report validates the exact crops produced by YOLO/ByteTrack for the Behavior Classifier.\n\n")
        
        for log in diagnostic_logs:
            f.write(f"### Track ID: {log['cow_id']} | Frame: {log['frame']} | Timestamp: {log['timestamp']:.2f}s\n")
            f.write(f"- **Bounding Box**: `{log['bbox']}`\n")
            f.write(f"- **Predicted Posture**: {log['posture']} (Conf: {log['posture_conf']:.2f})\n")
            f.write(f"- **Predicted Activity**: {log['activity']} (Conf: {log['activity_conf']:.2f})\n")
            f.write("\n")
            
    print(f"Diagnostic complete. Report saved to {report_md_path}")

if __name__ == '__main__':
    main()
