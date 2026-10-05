import requests

url = "https://zenodo.org/api/records/?q=CowRumIA"
response = requests.get(url).json()

hits = response.get('hits', {}).get('hits', [])
if hits:
    record = hits[0]
    print(f"Record ID: {record['id']}")
    print(f"Title: {record['metadata']['title']}")
    for file in record.get('files', []):
        print(f"File: {file['key']}, Size: {file['size']}")
else:
    print("Not found.")
