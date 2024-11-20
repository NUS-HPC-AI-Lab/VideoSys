import os
import requests
import zipfile
from PIL import Image
from pytorch_fid import fid_score
import torch
import argparse


# Paths
COCO_DOWNLOAD_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_DIR = "coco_dataset"
REAL_IMAGES_DIR = os.path.join(COCO_DIR, "real_images")


def download_coco(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    coco_zip_path = os.path.join(output_dir, "val2017.zip")
    
    # Download MS-COCO validation set
    print("Downloading MS-COCO validation images...")
    with requests.get(COCO_DOWNLOAD_URL, stream=True) as r:
        r.raise_for_status()
        with open(coco_zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    print("Extracting images...")
    with zipfile.ZipFile(coco_zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    
    os.rename(os.path.join(output_dir, "val2017"), REAL_IMAGES_DIR)
    os.remove(coco_zip_path)
    print(f"MS-COCO images downloaded and extracted to {REAL_IMAGES_DIR}")

# Function to preprocess images (resize to 299x299)
def preprocess_images(input_dir, output_dir, target_size=(299, 299)):
    os.makedirs(output_dir, exist_ok=True)
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, img_name)
        with Image.open(img_path) as img:
            img = img.convert("RGB").resize(target_size)
            img.save(output_path)
    print(f"Preprocessed images saved to {output_dir}")

# Function to calculate FID
def calculate_fid(real_dir, generated_dir):
    print("Calculating FID score...")
    fid_value = fid_score.calculate_fid_given_paths(
        [real_dir, generated_dir],
        batch_size=50,  # Adjust based on your hardware
        device="cuda" if torch.cuda.is_available() else "cpu",
        dims=2048,  # Default feature dimensions for Inception-v3 pool3
    )
    print(f"FID Score: {fid_value}")
    return fid_value




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate FID for generated images.")
    parser.add_argument(
        "generated_images_dir",
        type=str,
        help="Path to the directory containing generated images."
    )
    args = parser.parse_args()

    GENERATED_IMAGES_DIR = args.generated_images_dir
    if not os.path.exists(REAL_IMAGES_DIR):
        download_coco(COCO_DIR)
    preprocessed_coco_dir = os.path.join(COCO_DIR, "preprocessed_real_images")
    preprocess_images(REAL_IMAGES_DIR, preprocessed_coco_dir)

    if not os.path.exists(GENERATED_IMAGES_DIR):
        raise ValueError("Generated images directory does not exist. Add your images to 'generated_images/'.")
    
    print(f"Calculating FID for {GENERATED_IMAGES_DIR}...")
    fid_value = calculate_fid(preprocessed_coco_dir, GENERATED_IMAGES_DIR)

    fid_score_file = os.path.join(GENERATED_IMAGES_DIR, "fid_score.txt")
    with open(fid_score_file, "w") as f:
        f.write(f"FID Score: {fid_value}\\n")
    print(f"FID score saved to {fid_score_file}")
    
    
    
# python calculate_FID.py --generated_images_dir outputs/flux-pab