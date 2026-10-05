import os
import wave

extract_dir = 'data/raw/zenodo-bovine/CowRumIA-Dataset'

labels = {}
rates = set()
durations = []

for root, _, files in os.walk(extract_dir):
    for file in files:
        if file.endswith('.wav'):
            path = os.path.join(root, file)
            label = os.path.basename(root)
            
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
