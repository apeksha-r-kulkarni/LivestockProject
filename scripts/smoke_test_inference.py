import os
import torch
from torchvision import transforms
from PIL import Image
from scripts.train_behavior_classifier import MultiTaskCowBehaviorModel, SquarePadResize
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default='outputs/models/best_behavior_classifier.pth')
    parser.add_argument("--sample_dir", type=str, default='data/crops/sample/')
    parser.add_argument("--report_path", type=str, default='outputs/reports/real_crop_inference_smoke_test.md')
    parser.add_argument("--v2_preprocessing", action="store_true")
    args = parser.parse_args()
    
    ckpt_path = args.ckpt_path
    sample_dir = args.sample_dir
    report_path = args.report_path
    
    device = torch.device('cpu')
    ckpt = torch.load(ckpt_path, map_location=device)
    
    model = MultiTaskCowBehaviorModel(backbone_name=ckpt.get('backbone', 'mobilenet_v3_large'))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model.to(device)
    
    if args.v2_preprocessing:
        base_resize = SquarePadResize(224)
    else:
        base_resize = transforms.Resize((224, 224))
        
    transform = transforms.Compose([
        base_resize,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    results = []
    posture_preds = []
    activity_preds = []
    posture_confs = []
    activity_confs = []
    
    correct_posture = 0
    correct_activity = 0
    total = 0
    
    for fname in sorted(os.listdir(sample_dir)):
        if not fname.endswith('.jpg'):
            continue
        
        # Parse ground truth from filename: sample_XX_posture_activity.jpg
        parts = fname.replace('.jpg', '').split('_')
        if len(parts) >= 4 and parts[0] == "sample":
            gt_posture = parts[2]
            gt_activity = parts[3]
        else:
            gt_posture = "unknown"
            gt_activity = "unknown"
            
        img_path = os.path.join(sample_dir, fname)
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img)
            out = model.predict_state(tensor, cow_id=fname, timestamp=0.0)
            
            p_pred = out['posture']
            p_conf = out['posture_confidence']
            a_pred = out['activity']
            a_conf = out['activity_confidence']
            
            posture_preds.append(p_pred)
            activity_preds.append(a_pred)
            posture_confs.append(p_conf)
            activity_confs.append(a_conf)
            
            if gt_posture == p_pred:
                correct_posture += 1
            if gt_activity == a_pred:
                correct_activity += 1
            total += 1
            
            results.append({
                'filename': fname,
                'p_gt': gt_posture,
                'a_gt': gt_activity,
                'p_pred': p_pred,
                'p_conf': p_conf,
                'a_pred': a_pred,
                'a_conf': a_conf
            })
        except Exception as e:
            print(f"Error on {fname}: {e}")
            
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write("# Real Crop Inference Smoke Test\n\n")
        f.write("## Summary Statistics\n")
        f.write(f"- **Total Samples Tested**: {total}\n")
        if total > 0:
            f.write(f"- **Average Posture Confidence**: {sum(posture_confs)/total:.4f}\n")
            f.write(f"- **Average Activity Confidence**: {sum(activity_confs)/total:.4f}\n")
            f.write(f"- **Posture Accuracy**: {correct_posture}/{total} ({correct_posture/total*100:.1f}%)\n")
            f.write(f"- **Activity Accuracy**: {correct_activity}/{total} ({correct_activity/total*100:.1f}%)\n\n")
        
        f.write("### Posture Prediction Distribution\n")
        for p in set(posture_preds):
            f.write(f"- {p}: {posture_preds.count(p)}\n")
            
        f.write("\n### Activity Prediction Distribution\n")
        for a in set(activity_preds):
            f.write(f"- {a}: {activity_preds.count(a)}\n")
            
        f.write("\n## Detailed Results\n")
        f.write("| Filename | GT Posture | Pred Posture | P-Conf | GT Activity | Pred Activity | A-Conf |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| `{r['filename']}` | {r['p_gt']} | {r['p_pred']} | {r['p_conf']:.4f} | {r['a_gt']} | {r['a_pred']} | {r['a_conf']:.4f} |\n")
            
    print("Report written to:", report_path)

if __name__ == '__main__':
    main()
