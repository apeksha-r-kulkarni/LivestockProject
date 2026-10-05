import cv2
import numpy as np
from ultralytics import YOLO
import os

def create_dummy_video(filename, width=640, height=480, fps=10, frames=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for i in range(frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Add a moving white rectangle so a random object detector might find something,
        # but our model trained on black dummy images might not.
        x = 50 + i * 5
        cv2.rectangle(img, (x, 100), (x+100, 200), (255, 255, 255), -1)
        out.write(img)
    out.release()

def run_tracking():
    video_path = 'data/test_video.mp4'
    create_dummy_video(video_path)
    
    # Load trained model
    model = YOLO('models/vision/best.pt')
    
    # Run ByteTrack
    results = model.track(source=video_path, tracker='bytetrack.yaml', persist=True, save=True, project='outputs/reports', name='tracking')

if __name__ == '__main__':
    run_tracking()
