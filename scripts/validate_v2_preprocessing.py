import os
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from scripts.train_behavior_classifier import SquarePadResize

def validate_preprocessing():
    out_dir = "outputs/reports/v2_preprocessing_check"
    report_path = "outputs/reports/v2_preprocessing_check.md"
    os.makedirs(out_dir, exist_ok=True)
    
    # Select some test images
    # We'll pick one from outside crops and one from cbvd-5 crops if available
    outside_crops_dir = "outputs/reports/outside_cattle_crops"
    cbvd_crops_dir = "data/crops/sample" # assuming we have some sample crops here from earlier
    
    test_images = []
    
    if os.path.exists(outside_crops_dir):
        crops = [f for f in os.listdir(outside_crops_dir) if f.startswith("crop_") and f.endswith(".jpg")]
        if crops:
            test_images.append(os.path.join(outside_crops_dir, crops[0]))
            
    if os.path.exists(cbvd_crops_dir):
        cbvd_crops = [f for f in os.listdir(cbvd_crops_dir) if f.endswith(".jpg")]
        if cbvd_crops:
            test_images.append(os.path.join(cbvd_crops_dir, cbvd_crops[0]))
            
    # Transforms
    v1_transform = transforms.Resize((224, 224))
    v2_transform = SquarePadResize(224)
    
    logs = []
    
    for img_path in test_images:
        name = os.path.basename(img_path)
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        
        # Apply V1
        v1_img = v1_transform(img)
        v1_path = os.path.join(out_dir, f"v1_{name}")
        v1_img.save(v1_path)
        
        # Apply V2
        v2_img = v2_transform(img)
        v2_path = os.path.join(out_dir, f"v2_{name}")
        v2_img.save(v2_path)
        
        logs.append({
            "name": name,
            "orig_size": f"{orig_w}x{orig_h}",
            "v1_size": f"{v1_img.size[0]}x{v1_img.size[1]}",
            "v2_size": f"{v2_img.size[0]}x{v2_img.size[1]}",
            "v1_path": v1_path,
            "v2_path": v2_path
        })
        
    with open(report_path, "w") as f:
        f.write("# V2 Preprocessing Validation Report\n\n")
        f.write("This report compares V1 (Direct Resize) vs V2 (Square Padding) preprocessing to verify aspect ratio preservation.\n\n")
        
        for log in logs:
            f.write(f"### Image: {log['name']}\n")
            f.write(f"- **Original Size**: {log['orig_size']}\n")
            f.write(f"- **V1 Size**: {log['v1_size']}\n")
            f.write(f"- **V2 Size**: {log['v2_size']}\n")
            f.write(f"\n*Visually, V2 should maintain the original aspect ratio by padding with black bars, preventing geometric distortion (especially for lying cattle).* \n\n")
            
    print(f"Validation complete. Report saved to {report_path}")

if __name__ == "__main__":
    validate_preprocessing()
