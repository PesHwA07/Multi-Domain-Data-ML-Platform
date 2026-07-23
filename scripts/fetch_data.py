import os
import shutil
# pyrefly: ignore [missing-import]
import kagglehub

def fetch_and_copy(dataset_id, target_dir):
    print(f"Downloading {dataset_id}...")
    # This downloads to the local kagglehub cache
    cached_path = kagglehub.dataset_download(dataset_id)
    print(f"Downloaded to cache: {cached_path}")
    
    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy all files from the cached path to our target directory
    for item in os.listdir(cached_path):
        source = os.path.join(cached_path, item)
        destination = os.path.join(target_dir, item)
        if os.path.isfile(source):
            shutil.copy2(source, destination)
            print(f"Copied {item} to {target_dir}")
        elif os.path.isdir(source):
