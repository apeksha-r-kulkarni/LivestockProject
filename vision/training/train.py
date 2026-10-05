import os
import glob
import shutil
import time
from ultralytics import YOLO
import sys

def main(smoke_test=False):
    config_path = os.path.abspath('configs/vision.yaml')
    project_path = os.path.abspath('outputs/reports')
    name = 'smoke_test' if smoke_test else 'actual_train'
    epochs = 1 if smoke_test else 3
    
    print(f"--- Starting {'SMOKE TEST' if smoke_test else 'ACTUAL TRAINING'} ---")
    start_time = time.time()
    
    # Target lightweight model
    model = YOLO('yolov8n.yaml') # use untrained yaml for fast test, or yolov8n.pt
    model = YOLO('yolov8n.pt') 
    
    print(f"Ultralytics version: {__import__('ultralytics').__version__}")
    
    # Train
    results = model.train(
        data=config_path,
        epochs=epochs,
        imgsz=320, # reduced for CPU speed
        batch=4,   # small batch for CPU
        project=project_path,
        name=name,
        device='cpu',
        # Augmentation parameters
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, # color variation
        degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.5, # horizontal orientation
        workers=0
    )
    
    train_duration = time.time() - start_time
    print(f"Training completed in {train_duration:.2f} seconds.")
    
    if smoke_test:
        print("Smoke test complete. Proceeding to actual training...")
        return
        
    print("Evaluating on TEST set...")
    metrics = model.val(data=config_path, split='test')
    
    print("Validation / Test Evaluation Complete.")
    
    # Copy best/last models
    os.makedirs('models/vision', exist_ok=True)
    best_model_path = os.path.join(project_path, name, 'weights', 'best.pt')
    last_model_path = os.path.join(project_path, name, 'weights', 'last.pt')
    
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, 'models/vision/best.pt')
    if os.path.exists(last_model_path):
        shutil.copy(last_model_path, 'models/vision/last.pt')
        
    # Inference test
    print("Running inference test...")
    test_images = glob.glob('data/processed/vision/images/test/*.jpg')
    if test_images:
        test_img = test_images[0]
        preds = model(test_img)
        print("Inference Test Result on:", test_img)
        for box in preds[0].boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            print(f"Cow detection:\nBehavior: {name}\nConfidence: {conf:.2f}\n")
    else:
        print("No test images found for inference.")

if __name__ == '__main__':
    main(smoke_test=False)
