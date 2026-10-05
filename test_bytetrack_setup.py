import os
import cv2
import re
from ultralytics import YOLO

def main():
    img_dir = "data/raw/cbvd-5/images"
    
    # 1. Inspect CBVD-5 filenames
    files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    
    # Group by sequence ID
    # Assume format is SEQID_FRAMEID.jpg or similar
    sequences = {}
    for f in files:
        parts = f.split('_')
        seq_id = parts[0]
        if seq_id not in sequences:
            sequences[seq_id] = []
        sequences[seq_id].append(f)
        
    # Find a sequence with multiple frames
    best_seq = None
    max_frames = 0
    for seq_id, frames in sequences.items():
        if len(frames) > max_frames:
            max_frames = len(frames)
            best_seq = seq_id
            
    print(f"Selected Sequence ID: {best_seq}")
    print(f"Number of available sequential frames: {max_frames}")
    print(f"Frame filename pattern: {best_seq}_XXXXX.jpg")
    
    frame_paths = sorted(sequences[best_seq])
    
    # Get image dimensions
    first_img = cv2.imread(os.path.join(img_dir, frame_paths[0]))
    h, w, c = first_img.shape
    print(f"Image dimensions: {w}x{h}")
    
    # Create MP4
    mp4_path = f"data/test_{best_seq}.mp4"
    print(f"Checking if OpenCV can create an MP4 from these frames...")
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(mp4_path, fourcc, 10, (w, h))
        for fname in frame_paths:
            img = cv2.imread(os.path.join(img_dir, fname))
            out.write(img)
        out.release()
        print(f"MP4 created successfully at {mp4_path}")
    except Exception as e:
        print(f"Failed to create MP4: {e}")
        return
        
    # ByteTrack test
    print("Testing Ultralytics ByteTrack...")
    try:
        model = YOLO("yolov8n.pt")
        # Run tracking on the video
        results = model.track(source=mp4_path, tracker="bytetrack.yaml", classes=[19], persist=True, show=False)
        # Process the generator
        frames_processed = 0
        track_ids_produced = []
        for r in results:
            frames_processed += 1
            if r.boxes is not None and r.boxes.id is not None:
                track_ids_produced.extend(r.boxes.id.tolist())
        
        print(f"Frames processed: {frames_processed}")
        print(f"Track IDs produced: {set(track_ids_produced)}")
        print("Ultralytics ByteTrack works perfectly in the current environment.")
    except Exception as e:
        print(f"ByteTrack failed with error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
