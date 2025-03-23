import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialize ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=20,       # Random rotation
    width_shift_range=0.2,   # Horizontal shift
    height_shift_range=0.2,  # Vertical shift
    shear_range=0.2,         # Shear transformation
    zoom_range=0.2,          # Random zoom
    horizontal_flip=True,    # Horizontal flip
    fill_mode='nearest'      # Fill missing pixels
)

# Define the output directory for augmented images
output_dir = "augmented_images"
os.makedirs(output_dir, exist_ok=True)

# Function to create label-specific directories
def create_label_directories(labels, base_dir):
    for label in labels:
        label_dir = os.path.join(base_dir, label)
        os.makedirs(label_dir, exist_ok=True)

# Function to save augmented images in respective label folders
def save_augmented_images(df, label, min_count, img_size, output_dir):
    # Filter images for the specific label
    subset = df[df['dx'] == label]
    current_count = subset.shape[0]
    additional_images_needed = min_count - current_count

    # Skip if the minimum count is already met
    if additional_images_needed <= 0:
        print(f"{label} already has {current_count} images. No augmentation needed.")
        return

    print(f"Augmenting {label}: {additional_images_needed} additional images needed.")
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    # Loop through each image in the subset
    for _, row in tqdm(subset.iterrows(), total=subset.shape[0], desc=f"Processing {label}"):
        image_array = row['image_array']
        if image_array is None:
            continue
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Generate augmented images
        count = 0
        for batch in datagen.flow(image_array, batch_size=1):
            augmented_image = (batch[0] * 255).astype(np.uint8)  # Rescale to 0-255
            augmented_image = cv2.resize(augmented_image, (img_size, img_size))

            # Save the augmented image to the respective folder
            image_name = f"{label}_{current_count + count + 1}.jpg"
            save_path = os.path.join(label_dir, image_name)
            cv2.imwrite(save_path, augmented_image)

            count += 1
            if count >= additional_images_needed:
                break

# Load the dataset and preprocess
df = pd.read_csv("datasets/HAM10000_metadata.csv")
img_directory = "datasets/HAM10000_images_part_1_org/"

def get_image_path(image_id):
    return os.path.join(img_directory, image_id + ".jpg")

df['image_path'] = df['image_id'].apply(get_image_path)

IMG_SIZE = 256  # Updated image size
channels = 3

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_array = img.astype(np.float32) / 255.0
    return img_array

tqdm.pandas()
df['image_array'] = df['image_path'].progress_apply(load_and_preprocess_image)

# Handle missing images
df = df[df['image_array'].notna()]

# Create label directories
unique_labels = df['dx'].unique()
create_label_directories(unique_labels, output_dir)

# Define minimum number of images per class
MIN_COUNT = 300

# Augment and save images for each label
for label in unique_labels:
    save_augmented_images(df, label, MIN_COUNT, IMG_SIZE, output_dir)

print("Image augmentation and saving completed.")
