import json
import cv2
import numpy as np
import os

def generate_contact_sheet():
    json_path = "outputs/reports/outside_cattle_test_v2_results.json"
    video_path = "outputs/videos/outside_cattle_test_v2.mp4"
    out_path = "outputs/reports/v2_external_video_visual_check.jpg"
    
    if not os.path.exists(json_path) or not os.path.exists(video_path):
        print("Missing json or video")
        return
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video")
        return
        
    lying_frames = []
    standing_frames = []
    
    for frame_data in data:
        # data is a list of frame dicts, each with cow_id, posture
        # Wait, the JSON format was a flat list of dicts:
        # {"frame_number": 2, "timestamp": ..., "posture": "standing", ...}
        
        posture = frame_data.get("posture")
        frame_idx = frame_data.get("frame_number")
        
        if posture == "lying" and len(lying_frames) < 10:
            lying_frames.append(frame_idx)
        elif posture == "standing" and len(standing_frames) < 10:
            standing_frames.append(frame_idx)
            
        if len(lying_frames) >= 10 and len(standing_frames) >= 10:
            break
            
    images = []
    
    def get_frame(f_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if ret:
            # resize for contact sheet
            h, w = frame.shape[:2]
            scale = 300 / w
            return cv2.resize(frame, (300, int(h * scale)))
        return None
        
    for f_idx in lying_frames:
        f = get_frame(f_idx)
        if f is not None:
            # add label
            cv2.putText(f, f"Lying - F:{f_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            images.append(f)
            
    for f_idx in standing_frames:
        f = get_frame(f_idx)
        if f is not None:
            cv2.putText(f, f"Standing - F:{f_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            images.append(f)
            
    cap.release()
    
    if not images:
        print("No images extracted")
        return
        
    # Build grid
    cols = 5
    rows = int(np.ceil(len(images) / cols))
    
    img_h, img_w = images[0].shape[:2]
    
    contact_sheet = np.zeros((rows * img_h, cols * img_w, 3), dtype=np.uint8)
    
    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        contact_sheet[r*img_h:(r+1)*img_h, c*img_w:(c+1)*img_w] = img
        
    cv2.imwrite(out_path, contact_sheet)
    print(f"Contact sheet saved to {out_path}")

if __name__ == "__main__":
    generate_contact_sheet()
