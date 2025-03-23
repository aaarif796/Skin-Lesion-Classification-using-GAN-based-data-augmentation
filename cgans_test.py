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
import warnings
warnings.filterwarnings("ignore")



df = pd.read_csv("datasets/HAM10000_metadata.csv")
df.head()

lesion_type_mapping = {
    "nv": "Melanocytic Nevi (6705)",
    "mel": "Melanoma (1113)",
    "bkl": "Benign Keratosis-like Lesions (1099)",
    "bcc": "Basal Cell Carcinoma (514)",
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma (327)",
    "vasc": "Vascular Lesions (142)",
    "df": "Dermatofibroma (115)"
}
lesion_type_mapping

img_directory = "datasets/HAM10000_images_part_1_org/"
print(img_directory)
df.head()

def get_image_path(image_id):
    return os.path.join(img_directory, image_id + ".jpg")

df['image_path'] = df['image_id'].apply(get_image_path)
df.head()

df['dx'].value_counts()

BATCH_SIZE = 16
BUFFER_SIZE = 500
Z_DIM = 128
IMG_SIZE = 64
channels = 3

def load_and_preprocess_image(image_path):
    # Load the image from the given path
    img = cv2.imread(image_path)
    if img is None:
        # raise ValueError(f"Image at path '{image_path}' could not be loaded.")
        return None
    # Resize the image to the desired dimensions (128x128 in your case)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # Normalize the image to the range [0, 1]
    img_array = img.astype(np.float32) / 255.0
    return img_array

tqdm.pandas()
df['image_array'] = df['image_path'].progress_apply(load_and_preprocess_image)
print(df[['image_id', 'dx', 'image_path', 'image_array']].head())
df = df.dropna()
df.head()

# Label encode the labels instead of one-hot encoding
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['dx'])

image_data = np.stack(df['image_array'].values)
print(f"Shape of the image data: {image_data.shape}")

if any(isinstance(x, np.ndarray) for x in df['image_array']):
    image_data = np.stack(df['image_array'].values)
else:
    raise ValueError("Some images were not loaded correctly.")

# IMG_SIZE = 128


# Create a TensorFlow dataset with images and encoded labels
X = tf.data.Dataset.from_tensor_slices((image_data, df['label_encoded'].values)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

num_classes = len(label_encoder.classes_)


def build_generator(z_dim, num_classes):
    noise_input = Input(shape=(z_dim,))
    label_input = Input(shape=(1,), dtype=tf.int32)

    # Embed the labels into a dense vector and flatten
    label_embedding = Embedding(input_dim=num_classes, output_dim=z_dim)(label_input)
    label_embedding = Flatten()(label_embedding)

    # Combine noise and label embeddings
    model_input = tf.keras.layers.Concatenate()([noise_input, label_embedding])

    x = Dense(8 * 8 * 256, use_bias=False)(model_input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Reshape((8, 8, 256))(x)
    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    img_output = Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh', use_bias=False)(x)

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

# Discriminator loss function
def discriminator_loss(real_output, fake_output):
    real_loss = criterion(tf.ones_like(real_output), real_output)
    fake_loss = criterion(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Generator loss function
def generator_loss(fake_output):
    return criterion(tf.ones_like(fake_output), fake_output)


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
        disc_loss = discriminator_loss(real_output, fake_output)

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
    folder = 'cgans_image'
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    fig, axis = plt.subplots(3, 5, figsize=(15, 10))
    for i, ax in enumerate(axis.flat):
        img = preds[i].numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.axis('off')
    
    # Save the figure with the epoch number in the filename
    file_path = os.path.join(folder, f'gan_generated_images_epoch_{epoch}.png')
    plt.savefig(file_path)
    plt.close(fig)  # Close the figure to avoid display

    print(f"Images saved to {file_path}")

# Training the GAN
Epochs = 500
with tf.device("/GPU:0"):
    disc_losses = []
    gen_losses = []
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
        
        if (epoch + 1) % 10 == 0:
            save_images(epoch+1)