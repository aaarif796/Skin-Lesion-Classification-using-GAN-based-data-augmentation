# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:15:08 2024

@author: aarif
"""



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
import os
import glob
import cv2
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Reshape, Conv2D, Conv2DTranspose, Dense, BatchNormalization, LeakyReLU, ReLU, Flatten, Embedding
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 16
BUFFER_SIZE = 500
Z_DIM = 100
IMG_SIZE = 128
channels = 3
num_classes = 4

def build_generator(z_dim, num_classes):
    noise_input = Input(shape=(z_dim,))
    label_input = Input(shape=(1,), dtype=tf.int32)

    label_embedding = Embedding(input_dim=num_classes, output_dim=z_dim)(label_input)
    label_embedding = Flatten()(label_embedding)

    model_input = tf.keras.layers.Concatenate()([noise_input, label_embedding])

    x = Dense(16 * 16 * 256, use_bias=False)(model_input)  # Adjust for 128x128 images
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Reshape((16, 16, 256))(x)
    
    # Add additional Conv2DTranspose layers to upscale to 128x128
    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)  # 32x32
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False)(x)  # 64x64
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2DTranspose(32, kernel_size=4, strides=2, padding='same', use_bias=False)(x)  # 128x128
    x = BatchNormalization()(x)
    x = ReLU()(x)

    img_output = Conv2DTranspose(3, kernel_size=4, strides=1, padding='same', activation='tanh', use_bias=False)(x)

    return tf.keras.Model([noise_input, label_input], img_output)

# Load the best generator model
generator = build_generator(Z_DIM, num_classes)  # Rebuild the generator structure
generator.load_weights('best_models_180/generator_epoch_800.h5')  # Load the best saved weights

# Function to generate a single image using the best model
def generate_single_image():
    noise = tf.random.normal([1, Z_DIM])  # Generate random noise for a single image
    label = tf.convert_to_tensor([np.random.randint(0, num_classes)])  # Generate a random label

    # Generate a fake image using the loaded generator model
    generated_image = generator([noise, label], training=False)[0].numpy()  # Get the single image
    generated_image = np.clip(generated_image, 0, 1)  # Ensure the pixel values are in the range [0, 1]

    # Convert the image to the range [0, 255] for saving
    generated_image = (generated_image * 255).astype(np.uint8)

    # Save the image
    folder = 'single_generated_images'
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_path = os.path.join(folder, 'single_generated_image_800ep.png')
    img = Image.fromarray(generated_image)
    img.save(file_path)
    
    print(f"Single image saved to {file_path}")

# Generate and save a single image
generate_single_image()


# Function to generate a single image for each label and save them
def generate_images_for_each_label():
    # Create folder if it doesn't exist
    folder = 'single_generated_images_per_label'
    if not os.path.exists(folder):
        os.makedirs(folder)

    for label in range(num_classes):
        noise = tf.random.normal([1, Z_DIM])  # Generate random noise for one image
        label_tensor = tf.convert_to_tensor([label])  # Convert label to tensor

        # Generate the image for the given label
        generated_image = generator([noise, label_tensor], training=False)[0].numpy()
        generated_image = np.clip(generated_image, 0, 1)  # Ensure values are in range [0, 1]
        
        # Convert the image to the range [0, 255] for saving
        generated_image = (generated_image * 255).astype(np.uint8)
        
        # Save the generated image for each label
        file_path = os.path.join(folder, f'generated_image_label__500_ep_{label}.png')
        img = Image.fromarray(generated_image)
        img.save(file_path)

        print(f"Image for label {label} saved to {file_path}")

# Generate and save a single image for each label
generate_images_for_each_label()



# Function to generate `n` images for class 0 and save them
def generate_images_for_class_0(n):
    # Create folder if it doesn't exist
    folder = 'gans_akiec'
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i in range(n):
        noise = tf.random.normal([1, Z_DIM])  # Generate random noise for one image
        label_tensor = tf.convert_to_tensor([0])  # Class 0

        # Generate the image for class 0
        generated_image = generator([noise, label_tensor], training=False)[0].numpy()
        generated_image = np.clip(generated_image, 0, 1)  # Ensure values are in range [0, 1]
        
        # Convert the image to the range [0, 255] for saving
        generated_image = (generated_image * 255).astype(np.uint8)
        
        # Save the generated image for class 0
        file_path = os.path.join(folder, f'akiec_{i+1}.jpg')
        img = Image.fromarray(generated_image)
        img.save(file_path)

        print(f"Image {i+1} for class 0 saved to {file_path}")
        
generate_images_for_class_0(673)


# Function to generate `n` images for class 0 and save them
def generate_images_for_class_1(n):
    # Create folder if it doesn't exist
    folder = 'gans_vasc'
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i in range(n):
        noise = tf.random.normal([1, Z_DIM])  # Generate random noise for one image
        label_tensor = tf.convert_to_tensor([1])  # Class 1

        # Generate the image for class 0
        generated_image = generator([noise, label_tensor], training=False)[0].numpy()
        generated_image = np.clip(generated_image, 0, 1)  # Ensure values are in range [0, 1]
        
        # Convert the image to the range [0, 255] for saving
        generated_image = (generated_image * 255).astype(np.uint8)
        
        # Save the generated image for class 0
        file_path = os.path.join(folder, f'vasc_{i+1}.jpg')
        img = Image.fromarray(generated_image)
        img.save(file_path)

        print(f"Image {i+1} for class 1 saved to {file_path}")
        
generate_images_for_class_1(858)



# Function to generate `n` images for class 0 and save them
def generate_images_for_class_2(n):
    # Create folder if it doesn't exist
    folder = 'gans_df'
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i in range(n):
        noise = tf.random.normal([1, Z_DIM])  # Generate random noise for one image
        label_tensor = tf.convert_to_tensor([2])  # Class 0

        # Generate the image for class 0
        generated_image = generator([noise, label_tensor], training=False)[0].numpy()
        generated_image = np.clip(generated_image, 0, 1)  # Ensure values are in range [0, 1]
        
        # Convert the image to the range [0, 255] for saving
        generated_image = (generated_image * 255).astype(np.uint8)
        
        # Save the generated image for class 0
        file_path = os.path.join(folder, f'df_{i+1}.jpg')
        img = Image.fromarray(generated_image)
        img.save(file_path)

        print(f"Image {i+1} for class 3 saved to {file_path}")

# Number of images to generate for class 0
n_images = 885  # Change this number to generate more or fewer images

# Generate and save `n` images for class 0
generate_images_for_class_2(n_images)







# Function to generate `n` images for class 0 and save them
def generate_images_for_class_3(n):
    # Create folder if it doesn't exist
    folder = 'gans_bcc'
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i in range(n):
        noise = tf.random.normal([1, Z_DIM])  # Generate random noise for one image
        label_tensor = tf.convert_to_tensor([3])  # Class 6

        # Generate the image for class 3
        generated_image = generator([noise, label_tensor], training=False)[0].numpy()
        generated_image = np.clip(generated_image, 0, 1)  # Ensure values are in range [0, 1]
        
        # Convert the image to the range [0, 255] for saving
        generated_image = (generated_image * 255).astype(np.uint8)
        
        # Save the generated image for class 0
        file_path = os.path.join(folder, f'bcc_{i+1}.jpg')
        img = Image.fromarray(generated_image)
        img.save(file_path)

        print(f"Image {i+1} for class 6 saved to {file_path}")

# Number of images to generate for class 0
n_images = 486  # Change this number to generate more or fewer images

# Generate and save `n` images for class 0
generate_images_for_class_3(n_images)











# Function to generate `n` images for class 0 and save them
def generate_images_for_class_6(n):
    # Create folder if it doesn't exist
    folder = 'generated_images_class_6'
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i in range(n):
        noise = tf.random.normal([1, Z_DIM])  # Generate random noise for one image
        label_tensor = tf.convert_to_tensor([6])  # Class 6

        # Generate the image for class 6
        generated_image = generator([noise, label_tensor], training=False)[0].numpy()
        generated_image = np.clip(generated_image, 0, 1)  # Ensure values are in range [0, 1]
        
        # Convert the image to the range [0, 255] for saving
        generated_image = (generated_image * 255).astype(np.uint8)
        
        # Save the generated image for class 0
        file_path = os.path.join(folder, f'generated_image_class_6_{i+1}.jpg')
        img = Image.fromarray(generated_image)
        img.save(file_path)

        print(f"Image {i+1} for class 6 saved to {file_path}")

# Number of images to generate for class 0
n_images = 185  # Change this number to generate more or fewer images

# Generate and save `n` images for class 0
generate_images_for_class_6(n_images)

