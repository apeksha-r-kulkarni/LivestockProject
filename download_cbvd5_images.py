import os
import json
import csv
import urllib.request
import urllib.parse
from PIL import Image
import time
import shutil
import concurrent.futures

def get_unique_filenames(csv_path):
    filenames = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            file_list_str = row[1]
            try:
                f_list = json.loads(file_list_str)
                filenames.extend(f_list)
            except:
                pass
    return list(dict.fromkeys(filenames))

def is_valid_jpeg(filepath):
    if not os.path.exists(filepath):
        return False
    try:
        with Image.open(filepath) as img:
            img.verify()
        with Image.open(filepath) as img:
            if img.width <= 0 or img.height <= 0:
                return False
        return True
    except:
        return False

def download_with_retries(url, filepath, retries=3):
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, filepath)
            if is_valid_jpeg(filepath):
                return True
            else:
                if os.path.exists(filepath):
                    os.remove(filepath)
        except Exception as e:
            time.sleep(1)
    return False

def process_file(fname, images_dir, dataset):
    filepath = os.path.join(images_dir, fname)
    if is_valid_jpeg(filepath):
        return fname, True
    
    encoded_file = urllib.parse.quote(f"labelframes/labelframes/{fname}", safe="")
    url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset}/{encoded_file}"
    success = download_with_retries(url, filepath)
    return fname, success

def main():
    csv_path = "data/raw/cbvd-5/CBVD-5.csv"
    images_dir = "data/raw/cbvd-5/images"
    temp_dir = "data/raw/cbvd-5/temp_test"
    dataset = "fandaoerji/cbvd-5cow-behavior-video-dataset"
    
    os.makedirs(images_dir, exist_ok=True)
    
    with open(os.path.expanduser("~/.kaggle/kaggle.json"), "r") as f:
        creds = json.load(f)
    username, key = creds["username"], creds["key"]
    
    passman = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    passman.add_password(None, "https://www.kaggle.com", username, key)
    authhandler = urllib.request.HTTPBasicAuthHandler(passman)
    opener = urllib.request.build_opener(authhandler)
    urllib.request.install_opener(opener)
    
    print("Extracting filenames...")
    filenames = get_unique_filenames(csv_path)
    print(f"Found {len(filenames)} unique files.")
    
    failed_downloads = []
    
    print("Starting download and verification process (using 5 workers)...")
    processed_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_file, fname, images_dir, dataset): fname for fname in filenames}
        for future in concurrent.futures.as_completed(futures):
            fname, success = future.result()
            if not success:
                failed_downloads.append(fname)
            processed_count += 1
            if processed_count % 100 == 0 or processed_count == len(filenames):
                print(f"Processed: {processed_count}/{len(filenames)}")
            
    print("Validating all images...")
    valid_count = 0
    missing_or_corrupted = []
    
    for fname in filenames:
        if is_valid_jpeg(os.path.join(images_dir, fname)):
            valid_count += 1
        else:
            missing_or_corrupted.append(fname)
            
    # Calculate directory size
    total_size = 0
    for f in os.listdir(images_dir):
        fp = os.path.join(images_dir, f)
        if os.path.isfile(fp):
            total_size += os.path.getsize(fp)
            
    print("\n--- Final Report ---")
    print(f"Expected Images: {len(filenames)}")
    print(f"Successfully Validated: {valid_count}")
    print(f"Missing or Corrupted: {len(missing_or_corrupted)}")
    print(f"Total Directory Size: {total_size / (1024*1024):.2f} MB")
    
    if missing_or_corrupted:
        print("Missing/Corrupted Files:", missing_or_corrupted)
        
    if valid_count == len(filenames) or ("618_00002.jpg" not in missing_or_corrupted):
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print("Removed temporary test directory.")

if __name__ == "__main__":
    main()
