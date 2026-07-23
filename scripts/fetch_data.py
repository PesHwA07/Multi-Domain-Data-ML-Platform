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
            shutil.copytree(source, destination, dirs_exist_ok=True)
            print(f"Copied directory {item} to {target_dir}")

def main():
    # Set up our local data directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_raw_dir = os.path.join(base_dir, 'data', 'raw')
    
    print(f"Extracting all datasets into: {data_raw_dir}\n")
    
    # 1. Spotify Tracks
    spotify_dir = os.path.join(data_raw_dir, 'spotify')
    fetch_and_copy("maharshipandya/-spotify-tracks-dataset", spotify_dir)
    print("-" * 50)
    
    # 2. PJM Hourly Energy Consumption
    energy_dir = os.path.join(data_raw_dir, 'energy')
