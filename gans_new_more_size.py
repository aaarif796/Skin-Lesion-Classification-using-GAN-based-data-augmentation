# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 02:51:48 2024

@author: aaari
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
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, BatchNormalization, ReLU, Reshape, Conv2DTranspose
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")

tf.keras.backend.clear_session()


df = pd.read_csv("datasets\HAM10000_metadata.csv")
df.head()



img_directory = "datasets/HAM10000_images_part_1_org/"
def get_image_path(image_id):
    return os.path.join(img_directory,image_id+".jpg")
df['image_path'] = df['image_id'].apply(get_image_path)
df.head()
df.shape

BATCH_SIZE = 8
BUFFER_SIZE = 500
Z_DIM = 100
IMG_SIZE = 128
channels = 3  
def load_and_preprocess_image(image_path):
    # Load the image from the given path
    img = cv2.imread(image_path)
    
    if img is None:
        print("No")
        # raise ValueError(f"Image at path '{image_path}' could not be loaded.")
        return None
    # Resize the image to the desired dimensions (256 * 256 in your case)
    img = cv2.resize(img, (128, 128))
    
    # Normalize the image to the range [0, 1]
    img_array = img.astype(np.float32) / 255.0
    
    return img_array

df['image_array'] = df['image_path'].apply(load_and_preprocess_image)



df.head()



df['dx'].value_counts()

# nv       6705
# mel      1113
# bkl      1099
# bcc       514
# akiec     327
# vasc      142
# df        115


df = df[['dx','image_path','image_array']]
df.head()

# Filter out records where 'dx' has values 'nv', 'mel', or 'bkl'
df_filtered = df[~df['dx'].isin(['nv', 'mel', 'bkl'])]

# Display the updated DataFrame
df_filtered.head()

df_filtered['dx'].value_counts()

label_mapping = {
    'akiec': 0,
    'vasc': 1,
    'df': 2,
    'bcc': 3
}

# Map the 'dx' column to numeric labels
df_filtered['label_encoded'] = df_filtered['dx'].map(label_mapping)

# Display the updated DataFrame
df_filtered.head()


image_data = np.stack(df_filtered['image_array'].values)

# Create a TensorFlow dataset with images and encoded labels
X = tf.data.Dataset.from_tensor_slices((image_data, df_filtered['label_encoded'].values)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


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


def build_discriminator(img_shape, num_classes):
    img_input = Input(shape=img_shape)
    label_input = Input(shape=(1,), dtype=tf.int32)

    # Embed the labels into a dense vector and reshape to match image dimensions
    label_embedding = Embedding(input_dim=num_classes, output_dim=np.prod(img_shape))(label_input)
    label_embedding = Reshape(img_shape)(label_embedding)

    # Combine image and label
    model_input = tf.keras.layers.Concatenate()([img_input, label_embedding])

    x = Conv2D(64, kernel_size=4, strides=2, padding='same', use_bias=False)(model_input)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    validity = Dense(1, activation='sigmoid', use_bias=False)(x)

    return tf.keras.Model([img_input, label_input], validity)



# Set parameters
generator = build_generator(Z_DIM, num_classes)

discriminator = build_discriminator((IMG_SIZE, IMG_SIZE, 3), num_classes)

# Optimizers for generator and discriminator
gen_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
disc_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)

criterion = BinaryCrossentropy(from_logits=True)


# Generator loss function
def generator_loss(fake_output):
    return criterion(tf.ones_like(fake_output), fake_output)

def gradient_penalty(real_images, fake_images, labels):
    # Generate random interpolation factor
    alpha = tf.random.uniform([tf.shape(real_images)[0], 1, 1, 1], 0., 1.)
    
    # Interpolate between real and fake images
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        # Ensure to pass the labels here
        pred = discriminator([interpolated, labels], training=True)

    gradients = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp


def discriminator_loss(real_output, fake_output, real_images, fake_images, labels):
    real_loss = criterion(tf.ones_like(real_output), real_output)  # Real label (1)
    fake_loss = criterion(tf.zeros_like(fake_output), fake_output)  # Fake label (0)

    # Add gradient penalty
    lambda_gp = 10  # Weight for the gradient penalty
    gp = gradient_penalty(real_images, fake_images, labels)  # Pass labels here

    # Total discriminator loss
    total_disc_loss = real_loss + fake_loss + lambda_gp * gp
    return total_disc_loss



@tf.function
def train_step(images, labels):
    current_batch_size = tf.shape(images)[0]  # Get the actual batch size
    
    noise = tf.random.normal([current_batch_size, Z_DIM])
    
    # Ensure labels have the correct shape
    labels = tf.reshape(labels, [current_batch_size, 1])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate fake images with labels
        fake_images = generator([noise, labels], training=True)

        # Discriminator outputs
        real_output = discriminator([images, labels], training=True)
        fake_output = discriminator([fake_images, labels], training=True)

        # Calculate losses
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output, images, fake_images, labels)  # Pass labels here

    # Get gradients
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Apply gradients
    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    return gen_loss, disc_loss





# Function to generate and save images without displaying them
def save_images(epoch):
    noise = tf.random.normal([BATCH_SIZE, Z_DIM])
    labels = tf.convert_to_tensor(np.random.randint(0, num_classes, BATCH_SIZE))
    
    preds = generator([noise, labels], training=False)
    
    # Create folder if it doesn't exist
    folder = 'cgans_image_180'
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    fig, axis = plt.subplots(1, BATCH_SIZE, figsize=(BATCH_SIZE * 3, 3))
    for i, ax in enumerate(axis.flat[:BATCH_SIZE]):
        img = preds[i].numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.axis('off')
    
    # Save the figure with the epoch number in the filename
    file_path = os.path.join(folder, f'gan_generated_images_epoch_{epoch}.png')
    plt.savefig(file_path)
    plt.close(fig)  # Close the figure to avoid display

    print(f"Images saved to {file_path}")

# Create a folder to store the best models
if not os.path.exists('best_models_180'):
    os.makedirs('best_models_180')
    
# Define model checkpoints for both generator and discriminator
gen_checkpoint = ModelCheckpoint(filepath='best_models_180/best_generator.h5', 
                                 monitor='gen_loss', 
                                 save_best_only=True, 
                                 save_weights_only=True, 
                                 mode='min', 
                                 verbose=1)

disc_checkpoint = ModelCheckpoint(filepath='best_models_180/best_discriminator.h5', 
                                  monitor='disc_loss', 
                                  save_best_only=True, 
                                  save_weights_only=True, 
                                  mode='min', 
                                  verbose=1)

# List to hold losses for each epoch
disc_losses = []
gen_losses = []



# Training loop with checkpoints
Epochs = 800
with tf.device("/GPU:0"):
    for epoch in tqdm(range(Epochs)):
        epoch_disc_loss = []
        epoch_gen_loss = []
        
        for images, labels in tqdm(X):
            gen_loss, disc_loss = train_step(images, labels)
            epoch_disc_loss.append(disc_loss)
            epoch_gen_loss.append(gen_loss)

        disc_losses.append(np.mean(epoch_disc_loss))
        gen_losses.append(np.mean(epoch_gen_loss))
        print(f"Epoch: {epoch + 1}, Generator Loss: {gen_losses[-1]}, Discriminator Loss: {disc_losses[-1]}")
        
        # Save checkpoints every 100 epochs with the epoch number in the filename
        if (epoch + 1) % 100 == 0:
            gen_checkpoint_path = f'best_models_180/generator_epoch_{epoch + 1}.h5'
            disc_checkpoint_path = f'best_models_180/discriminator_epoch_{epoch + 1}.h5'
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(gen_checkpoint_path), exist_ok=True)
            os.makedirs(os.path.dirname(disc_checkpoint_path), exist_ok=True)
            
            # Save generator and discriminator models
            generator.save_weights(gen_checkpoint_path)
            discriminator.save_weights(disc_checkpoint_path)
        
        # Save generated images every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_images(epoch + 1)
