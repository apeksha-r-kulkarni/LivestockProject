import os
from ultralytics import YOLO

checkpoints = [
    "yolov8n.pt",
    "models/vision/best.pt",
    "models/vision/last.pt",
    "outputs/reports/actual_train/weights/best.pt"
]

img_dir = "data/raw/cbvd-5/images"
test_images = []
if os.path.exists(img_dir):
    test_images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")][:2]
print(f"Using test images: {test_images}\n")

for ckpt in checkpoints:
    print(f"--- Inspecting {ckpt} ---")
    if not os.path.exists(ckpt):
        print("Status: NOT FOUND\n")
        continue
        
    try:
        model = YOLO(ckpt)
        names = model.names
        num_classes = len(names)
        print(f"Number of classes: {num_classes}")
        
        if num_classes > 10:
            print(f"Classes (first 10): {[names[i] for i in range(10)]} ...")
        else:
            print(f"Classes: {names}")
            
        cow_class_id = None
        for idx, name in names.items():
            if "cow" in name.lower():
                cow_class_id = idx
                break
                
        print(f"Cow class present: {cow_class_id is not None} (ID: {cow_class_id})")
        
        is_coco = (num_classes == 80 and "person" in list(names.values()))
        if is_coco:
            print("Model Type: COCO Pretrained")
        else:
            print("Model Type: Project-Trained/Custom")
            
        imgsz = model.overrides.get("imgsz", "Default(640)")
        print(f"Input expectations: imgsz={imgsz}")
        
        if cow_class_id is not None and test_images:
            print("Detection Smoke Test:")
            results = model(test_images, verbose=False)
            for i, r in enumerate(results):
                boxes = r.boxes
                cow_boxes = [b for b in boxes if int(b.cls[0]) == cow_class_id]
                print(f"  Image {os.path.basename(test_images[i])}: Found {len(cow_boxes)} cows")
                for j, b in enumerate(cow_boxes[:3]):
                    conf = float(b.conf[0])
                    xyxy = [round(float(x), 2) for x in b.xyxy[0]]
                    print(f"    Cow {j}: Conf={conf:.4f}, BBox={xyxy}")
        else:
            print("Detection Smoke Test: SKIPPED (No cow class)")
            
    except Exception as e:
        print(f"Error: {e}")
        
    print("\n")
