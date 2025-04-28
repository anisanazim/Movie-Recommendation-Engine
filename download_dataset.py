#!/usr/bin/env python
"""
Download and setup the MovieLens dataset.
"""

import os
import sys
import shutil
import requests
import zipfile
from tqdm import tqdm

# Constants
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
DATA_DIR = "./data"
ZIP_PATH = os.path.join(DATA_DIR, "ml-25m.zip")
EXTRACT_DIR = DATA_DIR

def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("output", exist_ok=True)

def download_file(url, save_path):
    """
    Download a file with progress reporting.
    
    Args:
        url (str): URL to download
        save_path (str): Path to save the downloaded file
    """
    print(f"Downloading {url} to {save_path}...")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(save_path, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    
    progress_bar.close()
    
    if total_size != 0 and progress_bar.n != total_size:
        print("ERROR: Download incomplete")
        return False
    
    return True

def extract_zip(zip_path, extract_dir):
    """
    Extract a zip file.
    
    Args:
        zip_path (str): Path to the zip file
        extract_dir (str): Directory to extract to
    """
    print(f"Extracting {zip_path} to {extract_dir}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get total size for progress reporting
        total_size = sum(file.file_size for file in zip_ref.infolist())
        extracted_size = 0
        
        for file in tqdm(zip_ref.infolist(), desc="Extracting", unit="files"):
            zip_ref.extract(file, extract_dir)
            extracted_size += file.file_size
    
    print("Extraction complete!")

def verify_dataset(dataset_dir):
    """
    Verify that all necessary dataset files exist.
    
    Args:
        dataset_dir (str): Path to the dataset directory
    
    Returns:
        bool: True if all files exist, False otherwise
    """
    required_files = [
        "movies.csv",
        "ratings.csv",
        "tags.csv",
        "links.csv",
    ]
    
    print("Verifying dataset files...")
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(dataset_dir, "ml-25m", file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print(f"Missing files: {', '.join(missing_files)}")
        return False
    
    print("All dataset files present!")
    return True

def main():
    """Main function to download and setup the dataset."""
    try:
        # Create directories
        create_directories()
        
        # Check if dataset already exists
        dataset_dir = os.path.join(EXTRACT_DIR, "ml-25m")
        if os.path.exists(dataset_dir):
            print(f"Dataset directory {dataset_dir} already exists.")
            choice = input("Do you want to re-download the dataset? (y/n): ")
            if choice.lower() != 'y':
                if verify_dataset(EXTRACT_DIR):
                    print("Dataset is already set up and verified.")
                    return
                else:
                    print("Dataset verification failed. Re-downloading...")
            
            # Remove existing dataset
            shutil.rmtree(dataset_dir, ignore_errors=True)
        
        # Download dataset
        if not download_file(MOVIELENS_URL, ZIP_PATH):
            print("Failed to download the dataset.")
            return
        
        # Extract dataset
        extract_zip(ZIP_PATH, EXTRACT_DIR)
        
        # Verify dataset
        if verify_dataset(EXTRACT_DIR):
            print("Dataset download and setup complete!")
            
            # Ask if the zip file should be kept
            choice = input("Do you want to keep the zip file? (y/n): ")
            if choice.lower() != 'y':
                os.remove(ZIP_PATH)
                print(f"Deleted zip file: {ZIP_PATH}")
        else:
            print("Dataset verification failed. Please try again.")
    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()