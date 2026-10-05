import os
import shutil
import json
from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    
    # Setup paths
    data_yaml = "data/detection_yolo/data.yaml"
    project_dir = "outputs/models"
    name = "yolo_cbvd5"
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=30,
        imgsz=640,
        batch=8,
        device="cpu",
        workers=2,
        patience=8,
        project=project_dir,
        name=name,
        exist_ok=True
    )
    
    print("Training finished.")
    
if __name__ == "__main__":
    main()
