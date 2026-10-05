import json
import os

def main():
    test_split = "data/splits/test_split.json"
    train_split = "data/splits/train_split.json"
    val_split = "data/splits/val_split.json"
    
    with open(test_split, 'r') as f: test_data = json.load(f)
    with open(train_split, 'r') as f: train_data = json.load(f)
    with open(val_split, 'r') as f: val_data = json.load(f)
    
    test_seqs = set([d["filename"].split('_')[0] for d in test_data])
    train_seqs = set([d["filename"].split('_')[0] for d in train_data])
    val_seqs = set([d["filename"].split('_')[0] for d in val_data])
    
    # Pure test sequences (not leaked into train/val)
    pure_test_seqs = test_seqs - train_seqs - val_seqs
    
    img_dir = "data/raw/cbvd-5/images/"
    
    best_seq = None
    best_len = 0
    
    for seq in pure_test_seqs:
        # Count frames in raw dir
        frames = [f for f in os.listdir(img_dir) if f.startswith(f"{seq}_")]
        if len(frames) > best_len:
            best_len = len(frames)
            best_seq = seq
            
    print(f"Best pure test sequence: {best_seq} with {best_len} frames")
    
if __name__ == "__main__":
    main()
