import cv2
import numpy as np
import os
import re

def create_contact_sheet():
    report_path = "outputs/reports/outside_cattle_crop_diagnostic.md"
    img_dir = "outputs/reports/outside_cattle_crops"
    out_path = os.path.join(img_dir, "diagnostic_contact_sheet.jpg")
    
    # Parse the markdown report to get the labels
    data = []
    with open(report_path, 'r') as f:
        content = f.read()
        
    # Split by ### Track ID
    blocks = content.split("### Track ID: ")
    for block in blocks[1:]:
        lines = block.strip().split('\n')
        header_parts = lines[0].split('|')
        track_id = header_parts[0].strip()
        frame = header_parts[1].split(':')[1].strip()
        
        posture, p_conf = "", ""
        activity, a_conf = "", ""
        for line in lines:
            if "Predicted Posture" in line:
                m = re.search(r"Predicted Posture\*\*: (.*?) \(Conf: (.*?)\)", line)
                if m:
                    posture = m.group(1)
                    p_conf = m.group(2)
            elif "Predicted Activity" in line:
                m = re.search(r"Predicted Activity\*\*: (.*?) \(Conf: (.*?)\)", line)
                if m:
                    activity = m.group(1)
                    a_conf = m.group(2)
                    
        data.append({
            'track_id': track_id,
            'frame': frame,
            'posture': posture,
            'p_conf': p_conf,
            'activity': activity,
            'a_conf': a_conf
        })

    # Prepare images
    # We want a grid. Let's arrange each pair vertically or horizontally.
    # [Frame Image] [Crop Image]
    # Text below the pair.
    
    row_images = []
    
    for item in data:
        frame_img_path = os.path.join(img_dir, f"frame_id{item['track_id']}_frame{item['frame']}.jpg")
        crop_img_path = os.path.join(img_dir, f"crop_id{item['track_id']}_frame{item['frame']}.jpg")
        
        if not os.path.exists(frame_img_path) or not os.path.exists(crop_img_path):
            continue
            
        frame_img = cv2.imread(frame_img_path)
        crop_img = cv2.imread(crop_img_path)
        
        # Resize to standard height for concatenation
        target_h = 300
        
        frame_h, frame_w = frame_img.shape[:2]
        new_frame_w = int(frame_w * (target_h / frame_h))
        frame_img = cv2.resize(frame_img, (new_frame_w, target_h))
        
        crop_h, crop_w = crop_img.shape[:2]
        new_crop_w = int(crop_w * (target_h / crop_h))
        crop_img = cv2.resize(crop_img, (new_crop_w, target_h))
        
        # Add a separator
        separator = np.zeros((target_h, 20, 3), dtype=np.uint8)
        
        # Combine horizontally
        pair_img = np.hstack([frame_img, separator, crop_img])
        
        # Create text panel
        text_h = 80
        text_panel = np.ones((text_h, pair_img.shape[1], 3), dtype=np.uint8) * 255
        
        label1 = f"Track: {item['track_id']} | Frame: {item['frame']}"
        label2 = f"Posture: {item['posture']} ({item['p_conf']})"
        label3 = f"Activity: {item['activity']} ({item['a_conf']})"
        
        cv2.putText(text_panel, label1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(text_panel, label2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(text_panel, label3, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        final_pair = np.vstack([pair_img, text_panel])
        row_images.append(final_pair)
        
    # Find max width
    max_w = max(img.shape[1] for img in row_images)
    
    padded_rows = []
    for img in row_images:
        if img.shape[1] < max_w:
            padding = np.ones((img.shape[0], max_w - img.shape[1], 3), dtype=np.uint8) * 255
            img = np.hstack([img, padding])
        padded_rows.append(img)
        
    # Vertical spacer
    spacer = np.zeros((20, max_w, 3), dtype=np.uint8)
    
    final_sheet = []
    for i, img in enumerate(padded_rows):
        final_sheet.append(img)
        if i < len(padded_rows) - 1:
            final_sheet.append(spacer)
            
    contact_sheet = np.vstack(final_sheet)
    
    cv2.imwrite(out_path, contact_sheet)
    print(f"Contact sheet saved to: {out_path}")

if __name__ == "__main__":
    create_contact_sheet()
