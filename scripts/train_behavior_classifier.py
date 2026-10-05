import os
import json
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

POSTURE_CLASSES = ["standing", "lying"]
ACTIVITY_CLASSES = ["none", "foraging", "drinking", "rumination"]

from PIL import ImageOps
class SquarePadResize:
    def __init__(self, size=224, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        max_dim = max(w, h)
        if max_dim == 0:
            return img.resize((self.size, self.size))
            
        ratio = self.size / max_dim
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        
        delta_w = self.size - new_w
        delta_h = self.size - new_h
        pad_left = delta_w // 2
        pad_top = delta_h // 2
        pad_right = delta_w - pad_left
        pad_bottom = delta_h - pad_top
        
        return ImageOps.expand(img, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)



class CowBehaviorDataset(Dataset):
    def __init__(self, split_json_path, images_dir, transform=None, padding_pct=0.1):
        with open(split_json_path, "r") as f:
            self.data = json.load(f)
        self.images_dir = images_dir
        self.transform = transform
        self.padding_pct = padding_pct

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.images_dir, item["filename"])
        
        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            _, bx, by, bw, bh = item["bbox"]
            
            pad_w = bw * self.padding_pct
            pad_h = bh * self.padding_pct
            
            left = max(0, bx - pad_w)
            top = max(0, by - pad_h)
            right = min(w, bx + bw + pad_w)
            bottom = min(h, by + bh + pad_h)
            
            crop = img.crop((left, top, right, bottom))
        except Exception as e:
            crop = Image.new("RGB", (224, 224))
            
        if self.transform:
            crop = self.transform(crop)
            
        posture_idx = POSTURE_CLASSES.index(item["posture"])
        activity_idx = ACTIVITY_CLASSES.index(item["activity"])
        
        return crop, posture_idx, activity_idx

class MultiTaskCowBehaviorModel(nn.Module):
    def __init__(self, backbone_name="mobilenet_v3_large", pretrained=True):
        super().__init__()
        
        if backbone_name == "mobilenet_v3_large":
            self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
            in_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone_name == "resnet18":
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone {backbone_name}")

        self.posture_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, len(POSTURE_CLASSES))
        )
        
        self.activity_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, len(ACTIVITY_CLASSES))
        )

    def forward(self, x):
        features = self.backbone(x)
        posture_logits = self.posture_head(features)
        activity_logits = self.activity_head(features)
        return posture_logits, activity_logits

    def predict_state(self, image_tensor, cow_id, timestamp):
        self.eval()
        with torch.no_grad():
            p_logits, a_logits = self(image_tensor.unsqueeze(0))
            
            p_probs = torch.softmax(p_logits, dim=1)[0]
            a_probs = torch.softmax(a_logits, dim=1)[0]
            
            p_conf, p_pred = torch.max(p_probs, dim=0)
            a_conf, a_pred = torch.max(a_probs, dim=0)
            
            return {
                "cow_id": cow_id,
                "timestamp": timestamp,
                "posture": POSTURE_CLASSES[p_pred.item()],
                "posture_confidence": p_conf.item(),
                "activity": ACTIVITY_CLASSES[a_pred.item()],
                "activity_confidence": a_conf.item()
            }

def evaluate(model, dataloader, criterion_posture, criterion_activity, device, posture_weight, activity_weight, split_name="VALIDATION"):
    model.eval()
    
    total_loss = 0.0
    posture_trues = []
    posture_preds = []
    activity_trues = []
    activity_preds = []
    
    with torch.no_grad():
        for crops, p_labels, a_labels in dataloader:
            crops = crops.to(device)
            p_labels = p_labels.to(device)
            a_labels = a_labels.to(device)
            
            p_logits, a_logits = model(crops)
            
            loss_p = criterion_posture(p_logits, p_labels)
            loss_a = criterion_activity(a_logits, a_labels)
            batch_loss = posture_weight * loss_p + activity_weight * loss_a
            total_loss += batch_loss.item()
            
            p_preds = torch.argmax(p_logits, dim=1)
            a_preds = torch.argmax(a_logits, dim=1)
            
            posture_trues.extend(p_labels.cpu().numpy())
            posture_preds.extend(p_preds.cpu().numpy())
            activity_trues.extend(a_labels.cpu().numpy())
            activity_preds.extend(a_preds.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    
    print(f"\n--- {split_name} EVALUATION ---")
    print(f"Avg Loss: {avg_loss:.4f}")
    
    # POSTURE METRICS
    print("\n[POSTURE]")
    print(f"Accuracy: {accuracy_score(posture_trues, posture_preds):.4f}")
    print(f"Macro F1: {f1_score(posture_trues, posture_preds, average='macro'):.4f}")
    print(f"Weighted F1: {f1_score(posture_trues, posture_preds, average='weighted'):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(posture_trues, posture_preds))
    print(classification_report(posture_trues, posture_preds, target_names=POSTURE_CLASSES, zero_division=0))
    
    # ACTIVITY METRICS
    print("\n[ACTIVITY]")
    print(f"Accuracy: {accuracy_score(activity_trues, activity_preds):.4f}")
    print(f"Macro F1: {f1_score(activity_trues, activity_preds, average='macro'):.4f}")
    print(f"Weighted F1: {f1_score(activity_trues, activity_preds, average='weighted'):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(activity_trues, activity_preds))
    print(classification_report(activity_trues, activity_preds, target_names=ACTIVITY_CLASSES, zero_division=0))
    
    # Explicit drinking metrics
    drinking_idx = ACTIVITY_CLASSES.index("drinking")
    dr_prec = precision_score(activity_trues, activity_preds, labels=[drinking_idx], average='macro', zero_division=0)
    dr_rec = recall_score(activity_trues, activity_preds, labels=[drinking_idx], average='macro', zero_division=0)
    dr_f1 = f1_score(activity_trues, activity_preds, labels=[drinking_idx], average='macro', zero_division=0)
    print(f"\nExplicit Drinking Class Metrics:")
    print(f"Precision: {dr_prec:.4f}, Recall: {dr_rec:.4f}, F1: {dr_f1:.4f}")
    
    act_macro_f1 = f1_score(activity_trues, activity_preds, average='macro')
    return avg_loss, act_macro_f1

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/splits")
    parser.add_argument("--images_dir", type=str, default="data/raw/cbvd-5/images")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_large")
    parser.add_argument("--posture_weight", type=float, default=1.0)
    parser.add_argument("--activity_weight", type=float, default=2.0)
    parser.add_argument("--activity_class_weights", type=float, nargs=4, default=[1.0, 2.5, 10.0, 2.0], help="Weights for [none, foraging, drinking, rumination]")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/models")
    parser.add_argument("--dry_run", action="store_true", help="Do not run training loop, just verify setup")
    parser.add_argument("--v2_preprocessing", action="store_true", help="Use V2 aspect-ratio-preserving square padding instead of squashing")
    parser.add_argument("--v3_perspective", action="store_true", help="Use V3 RandomPerspective augmentation (requires v2_preprocessing)")
    parser.add_argument("--pretrained_weights", type=str, default=None, help="Path to pretrained weights to load")
    parser.add_argument("--model_name", type=str, default="best_behavior_classifier.pth", help="Filename to save the best model")
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Augmentations (Safe for CCTV)
    if args.v2_preprocessing:
        print("Using V2 Preprocessing: SquarePadResize")
        base_resize = SquarePadResize(args.img_size)
    else:
        print("Using V1 Preprocessing: Direct Resize (Squash)")
        base_resize = transforms.Resize((args.img_size, args.img_size))

    train_transform_list = [
        base_resize,
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05))
    ]
    
    if args.v3_perspective:
        print("Using V3 Augmentation: RandomPerspective")
        train_transform_list.append(transforms.RandomPerspective(distortion_scale=0.4, p=0.5))
        
    train_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_transform = transforms.Compose(train_transform_list)
    
    val_test_transform = transforms.Compose([
        base_resize,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    train_ds = CowBehaviorDataset(os.path.join(args.data_dir, "train_split.json"), args.images_dir, train_transform)
    val_ds = CowBehaviorDataset(os.path.join(args.data_dir, "val_split.json"), args.images_dir, val_test_transform)
    test_ds = CowBehaviorDataset(os.path.join(args.data_dir, "test_split.json"), args.images_dir, val_test_transform)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    device = torch.device("cpu") # Enforce CPU
    model = MultiTaskCowBehaviorModel(backbone_name=args.backbone)
    
    if args.pretrained_weights:
        print(f"Loading pretrained weights from {args.pretrained_weights}...")
        ckpt = torch.load(args.pretrained_weights, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        
    model = model.to(device)

    activity_class_weights = torch.tensor(args.activity_class_weights, dtype=torch.float).to(device)
    
    criterion_posture = nn.CrossEntropyLoss()
    criterion_activity = nn.CrossEntropyLoss(weight=activity_class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("--- TRAINING INITIALIZATION ---")
    print(f"Model: {args.backbone}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Test samples: {len(test_ds)}")
    print(f"Loss weights -> Posture: {args.posture_weight}, Activity: {args.activity_weight}")
    print(f"Activity class weights: {args.activity_class_weights}")
    
    if args.dry_run:
        print("\n--- EXECUTING DRY RUN ---")
        print("1. All three splits loaded and initialized successfully.")
        print(f"2. Datasets initialized successfully.")
        print(f"3. DataLoaders initialized successfully.")
        
        print("4. Loading a representative batch...")
        crops, p_labels, a_labels = next(iter(train_loader))
        
        print("5 & 6. Batch verification:")
        print(f"   - Image tensor shape: {crops.shape}")
        print(f"   - Posture label shape: {p_labels.shape}")
        print(f"   - Activity label shape: {a_labels.shape}")
        print(f"   - Batch size: {crops.size(0)}")
        
        print("7. Label mapping verification:")
        print(f"   - Posture values min/max: {p_labels.min().item()} / {p_labels.max().item()} (Expected 0-1)")
        print(f"   - Activity values min/max: {a_labels.min().item()} / {a_labels.max().item()} (Expected 0-3)")
        
        print("8. Initializing backbone on CPU...")
        
        print("9. Running ONE forward-pass test...")
        crops = crops.to(device)
        model.eval()
        with torch.no_grad():
            p_logits, a_logits = model(crops)
            
        print(f"   - Posture logits shape: {p_logits.shape}")
        print(f"   - Activity logits shape: {a_logits.shape}")
        
        print("10. Activity class weights accepted:")
        print(f"   - Weights: {criterion_activity.weight.tolist()}")
        
        print("11. Loss functions instantiated correctly.")
        
        print("12. Verifying transforms...")
        print(f"   - Train transform includes augmentation: {'RandomHorizontalFlip' in str(train_ds.transform)}")
        print(f"   - Val/Test transform is deterministic: {'RandomHorizontalFlip' not in str(val_ds.transform)}")
        
        print("13. CPU execution verified.")
        print("14. No checkpoint saved.")
        
        print("\nREADY TO TRAIN (Aborting loop as requested via dry_run).")
        return

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_act_macro_f1 = -1.0
    best_checkpoint_path = os.path.join(args.checkpoint_dir, args.model_name)

    print("\n--- STARTING TRAINING ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        start_time = time.time()
        for crops, p_labels, a_labels in train_loader:
            crops = crops.to(device)
            p_labels = p_labels.to(device)
            a_labels = a_labels.to(device)
            
            optimizer.zero_grad()
            
            p_logits, a_logits = model(crops)
            
            loss_p = criterion_posture(p_logits, p_labels)
            loss_a = criterion_activity(a_logits, a_labels)
            
            total_loss = args.posture_weight * loss_p + args.activity_weight * loss_a
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
        avg_train_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {avg_train_loss:.4f} - Time: {epoch_time:.2f}s")
        
        # Validation
        val_loss, val_act_macro_f1 = evaluate(model, val_loader, criterion_posture, criterion_activity, device, args.posture_weight, args.activity_weight, split_name="VALIDATION")
        
        # Checkpointing
        if val_act_macro_f1 > best_val_act_macro_f1:
            best_val_act_macro_f1 = val_act_macro_f1
            print(f"*** New best validation activity macro F1: {best_val_act_macro_f1:.4f}. Saving checkpoint. ***")
            
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "backbone": args.backbone,
                "image_size": args.img_size,
                "posture_classes": POSTURE_CLASSES,
                "activity_classes": ACTIVITY_CLASSES,
                "loss_weights": {"posture": args.posture_weight, "activity": args.activity_weight},
                "activity_class_weights": args.activity_class_weights,
                "val_act_macro_f1": best_val_act_macro_f1
            }
            torch.save(checkpoint, best_checkpoint_path)

    print("\n--- TRAINING COMPLETE ---")
    
    # Test Evaluation
    print("Loading best checkpoint for TEST evaluation...")
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    print("\n--- FINAL TEST SET EVALUATION ---")
    evaluate(model, test_loader, criterion_posture, criterion_activity, device, args.posture_weight, args.activity_weight, split_name="TEST")

if __name__ == "__main__":
    train()
