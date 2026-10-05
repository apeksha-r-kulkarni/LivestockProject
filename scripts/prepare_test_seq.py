import json
import os
import cv2
import subprocess
import shutil

def main():
    test_split = "data/splits/test_split.json"
    with open(test_split, 'r') as f:
        data = json.load(f)
        
    sequences = {}
    for d in data:
        filename = d["filename"]
        prefix = filename.split('_')[0]
        frame = int(filename.split('_')[1].split('.')[0])
        if prefix not in sequences:
            sequences[prefix] = {"frames": set(), "boxes": []}
        sequences[prefix]["frames"].add(frame)
        sequences[prefix]["boxes"].append(d)
        
    # Find sequence with > 30 sequential frames
    best_seq = None
    best_len = 0
    for seq, info in sequences.items():
        frames = sorted(list(info["frames"]))
        if len(frames) > best_len:
            best_len = len(frames)
            best_seq = seq
            
    print(f"Selected Sequence: {best_seq} with {best_len} frames.")
    
    # Generate MP4 from raw images
    frames = sorted(list(sequences[best_seq]["frames"]))
    img_dir = "data/raw/cbvd-5/images/"
    
    temp_dir = "outputs/videos/temp_seq"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Check if we have the images
    img_path_first = os.path.join(img_dir, f"{best_seq}_{frames[0]:05d}.jpg")
    img = cv2.imread(img_path_first)
    h, w = img.shape[:2]
    print(f"Resolution: {w}x{h}")
    
    # Instead of ffmpeg which might be tricky with gaps, we will use OpenCV VideoWriter
    out_mp4 = f"outputs/videos/cbvd5_test_{best_seq}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0
    out = cv2.VideoWriter(out_mp4, fourcc, fps, (w, h))
    
    valid_frames = []
    for f_idx in frames:
        p = os.path.join(img_dir, f"{best_seq}_{f_idx:05d}.jpg")
        if os.path.exists(p):
            frame_img = cv2.imread(p)
            out.write(frame_img)
            valid_frames.append(f_idx)
    out.release()
    print(f"Saved {len(valid_frames)} frames to {out_mp4}")
    
    # Save the sequence stats
    with open("outputs/reports/temp_seq_info.json", "w") as f:
        json.dump({
            "sequence": best_seq,
            "num_frames": len(valid_frames),
            "width": w,
            "height": h,
            "fps": fps,
            "raw_video": out_mp4
        }, f)

if __name__ == "__main__":
    main()
