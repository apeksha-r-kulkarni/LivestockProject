import os
import zipfile
import wave

zip_path = 'data/raw/zenodo-bovine/CowRumIA-Dataset.zip'
extract_dir = 'data/raw/zenodo-bovine/extracted'

if not os.path.exists(extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

labels = {}
rates = set()
durations = []

for root, _, files in os.walk(extract_dir):
    for file in files:
        if file.endswith('.wav'):
            # infer label from filename or directory
            # typical format: label_idx.wav or directory is label
            path = os.path.join(root, file)
            parent = os.path.basename(root)
            
            # The structure might have labels in filenames e.g. "Calm_Beef_01.wav"
            # We'll just collect parent dirs and file prefixes
            label = parent if parent != 'extracted' else file.split('_')[0]
            
            labels[label] = labels.get(label, 0) + 1
            
            with wave.open(path, 'rb') as w:
                frames = w.getnframes()
                rate = w.getframerate()
                rates.add(rate)
                durations.append(frames / float(rate))

print(f"Total WAV files: {sum(labels.values())}")
print(f"Labels: {labels}")
print(f"Sample rates: {rates}")
if durations:
    print(f"Durations: Min={min(durations):.2f}s, Max={max(durations):.2f}s, Avg={sum(durations)/len(durations):.2f}s")
else:
    print("No durations found.")
