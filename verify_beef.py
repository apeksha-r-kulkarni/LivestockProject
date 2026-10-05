import kaggle
import json
import collections

files = kaggle.api.dataset_list_files('lucyfirst/beef-cattle-behavior-data-set').files
directories = collections.defaultdict(int)
for f in files:
    name = f.name
    if name.startswith('Category Videos/cows/'):
        parts = name.split('/')
        if len(parts) >= 4:
            cat = parts[2]
            directories[cat] += 1
print("Beef Cattle Dataset:")
print(dict(directories))
