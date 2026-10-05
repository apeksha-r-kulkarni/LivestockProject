import kaggle
import json

api = kaggle.KaggleApi()
api.authenticate()
page = 1
directories = {}
total_files = 0

while True:
    try:
        files = api.dataset_list_files('lucyfirst/beef-cattle-behavior-data-set', pageToken=page if page > 1 else None)
        # However, Kaggle API python client doesn't support pageToken properly in some versions.
        # Alternatively, using the private kaggle API or just parsing dataset metadata.
        print(f"Got page {page}")
        break # We break because standard kaggle library dataset_list_files only returns top 20 or doesn't support pagination easily.
    except Exception as e:
        print(e)
        break

# Since Kaggle API pagination is complex, we might just look at the dataset metadata directly.
dataset_metadata = api.dataset_metadata('lucyfirst/beef-cattle-behavior-data-set', '.')
