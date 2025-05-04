import os
from pathlib import Path
from PIL import Image

# Define paths
RAW_DATASET_PATH = "socofing/socofing"  # Replace with the path to your SOCOFing dataset
ALTERED_PATH = os.path.join(RAW_DATASET_PATH, "Altered")
REAL_PATH = os.path.join(RAW_DATASET_PATH, "Real")
OUTPUT_PATH = "dataloader/altered"  # Output folder for the processed dataset
IMAGE_SIZE = (96, 96)  # Resize images to 96x96

def preprocess_socofing():
    """Preprocess the SOCOFing dataset."""
    print("Starting preprocessing of SOCOFing dataset...")

    # Create output directories
    train_dir = Path(OUTPUT_PATH) / "train"
    # val_dir = Path(OUTPUT_PATH) / "val"
    # for folder in [train_dir, val_dir]:
    #     folder.mkdir(parents=True, exist_ok=True)

    # # Process Altered images (train set)
    altered_subfolders = ["Altered-Easy", "Altered-Medium", "Altered-Hard"]
    for subfolder in altered_subfolders:
        subfolder_path = os.path.join(ALTERED_PATH, subfolder)
        for file in os.listdir(subfolder_path):
            if file.endswith(".BMP"):  # Process only BMP images
                # Extract the class label (e.g., "001" from "001_M_Left_little_finger.BMP")
                label = file.split("_")[0]

                # Create a subfolder for the class if it doesn't exist
                label_dir = train_dir / label
                label_dir.mkdir(parents=True, exist_ok=True)

                # Rename the file to avoid conflicts (add subfolder name)
                base_name, extension = os.path.splitext(file)
                new_filename = f"{base_name}_{subfolder.split('-')[-1]}.PNG"  # Add severity level and change to PNG

                # Load and preprocess the image
                img_path = os.path.join(subfolder_path, file)
                img = Image.open(img_path)  # Keep the image in grayscale
                img = img.resize(IMAGE_SIZE)  # Resize image to 96x96

                # Save the image to the corresponding class folder
                img.save(label_dir / new_filename)

    print("Altered images processed and saved to train folder.")

    # Process Real images (validation set)
    for file in os.listdir(REAL_PATH):
        if file.endswith(".BMP"):  # Process only BMP images
            # Extract the class label (e.g., "001" from "001_M_Left_little_finger.BMP")
            label = file.split("_")[0]

            # Create a subfolder for the class if it doesn't exist
            label_dir = train_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)

            # Keep the original filename but change the extension to PNG
            base_name, extension = os.path.splitext(file)
            new_filename = f"{base_name}.PNG"

            # Load and preprocess the image
            img_path = os.path.join(REAL_PATH, file)
            img = Image.open(img_path)  # Keep the image in grayscale
            img = img.resize(IMAGE_SIZE)  # Resize image to 96x96

            # Save the image to the corresponding class folder
            img.save(label_dir / new_filename)

    print("Real images processed and saved to val folder.")

    print(f"Preprocessing complete. Dataset saved to {OUTPUT_PATH}.")
    
def preprocess_realSocofing(): pass

if __name__ == "__main__":
    preprocess_socofing()