# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 10:46:49 2024

@author: aaari
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
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
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("datasets\HAM10000_metadata.csv")
df.head()



img_directory = "datasets/HAM10000_images_part_1_org/"
def get_image_path(image_id):
    return os.path.join(img_directory,image_id+".jpg")
df['image_path'] = df['image_id'].apply(get_image_path)
df.head()
df.shape

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

tqdm.pandas()
df['image_array'] = df['image_path'].progress_apply(load_and_preprocess_image)



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




gan_class_0_folder = "datasets/HAM10000_images_part_1_org/generated_images_class_0_akiec"
gan_class_1_folder = "datasets/HAM10000_images_part_1_org/generated_images_class_1_bcc"
gan_class_3_folder = "datasets/HAM10000_images_part_1_org/generated_images_class_3_df"
gan_class_6_folder = "datasets/HAM10000_images_part_1_org/generated_images_class_6_vasc"
def load_gan_images_and_add_to_df(folder_path, dx_label):
    data = []
    
    # Iterate over all image files in the folder
    for img_file in tqdm(os.listdir(folder_path)):
        # Get full path of the image
        img_path = os.path.join(folder_path, img_file)
        
        # Load and preprocess the image
        img_array = load_and_preprocess_image(img_path)
        
        # Append the image path, dx value, and image array to the list
        if img_array is not None:
            data.append({
                'dx': dx_label,
                'image_path': img_path,
                'image_array': img_array
            })
    
    return pd.DataFrame(data)

# Load GAN images from each class folder and create a DataFrame
df_gan_class_0 = load_gan_images_and_add_to_df(gan_class_0_folder, 'akiec')
df_gan_class_1 = load_gan_images_and_add_to_df(gan_class_1_folder, 'bcc')
df_gan_class_3 = load_gan_images_and_add_to_df(gan_class_3_folder, 'df')
df_gan_class_6 = load_gan_images_and_add_to_df(gan_class_6_folder, 'vasc')


df_gan_combined = pd.concat([df_gan_class_0,df_gan_class_1,df_gan_class_3, df_gan_class_6], ignore_index=True)

# Now append the GAN images DataFrame to the original DataFrame
df_combined = pd.concat([df, df_gan_combined], ignore_index=True)

df_combined.head()
df_combined['dx'].value_counts()

# df_gan_class_6.head()

# Function to balance each label to have exactly `target_count` instances
def balance_labels(df, target_count=1000):
    # List to store the balanced data for each label
    balanced_data = []
    
    # Iterate over each unique label (dx value)
    for label in df['dx'].unique():
        # Filter the data for the current label
        label_data = df[df['dx'] == label]
        current_count = len(label_data)
        
        if current_count > target_count:
            # Undersample: if the label has more than target_count, sample down to target_count
            balanced_label_data = label_data.sample(target_count, random_state=42)
        elif current_count < target_count:
            # Oversample: if the label has fewer than target_count, randomly duplicate until target_count
            oversample_data = label_data.sample(target_count - current_count, replace=True, random_state=42)
            balanced_label_data = pd.concat([label_data, oversample_data])
        else:
            # If already equal to target_count, no need to sample
            balanced_label_data = label_data
        
        # Append the balanced data for this label to the list
        balanced_data.append(balanced_label_data)
    
    # Concatenate the balanced data for all labels into a single DataFrame
    balanced_df = pd.concat(balanced_data, ignore_index=True)
    
    return balanced_df

# Balance the dataset to have 250 instances per label
df_balanced = balance_labels(df_combined, target_count=1000)


df_balanced['dx'].value_counts()

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam

# Assuming df_balanced is already defined and contains the image data and labels
X = np.array(df_balanced['image_array'].tolist())  # Shape will be (4997, 128, 128, 3)
y = df_balanced['dx'].values  # This will give you the diagnosis labels

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Load ResNet50 without the top layer
IMG_SIZE = 128  # Define image size
res = ResNet50(input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet', include_top=False)

# Freeze the layers in ResNet50
for layer in res.layers:
    layer.trainable = False

# Add custom layers on top
x = res.output  # Get the output of ResNet50
x = Flatten()(x)  # Flatten the output of the ResNet50 model
num_classes = len(np.unique(y))  # Number of unique classes in the labels
prediction = Dense(num_classes, activation='softmax')(x)  # Dense layer for classification

# Create the model
model = Model(inputs=res.input, outputs=prediction)

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Make predictions on the test data
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)  # Get class with the highest probability


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Create the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# View the structure of the model
model.summary()


tf.keras.backend.clear_session()

# custom resnet-18
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Assuming df_balanced is already defined and contains the image data and labels
X = np.array(df_balanced['image_array'].tolist())  # Shape will be (4997, 128, 128, 3)
y = df_balanced['dx'].values  # This will give you the diagnosis labels

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Load ResNet-50 (since ResNet-18 is not directly available in Keras, we can build it ourselves)
# ResNet-18 can be created using Keras Functional API, or use pre-trained weights from Keras
def resnet_block(input_tensor, filters, kernel_size=3, stride=1):
    x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def identity_block(input_tensor, filters):
    x = resnet_block(input_tensor, filters)
    x = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.add([x, input_tensor])  # Add the input tensor to the block output
    x = tf.keras.layers.Activation('relu')(x)
    return x

def conv_block(input_tensor, filters, stride=2):
    x = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same')(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def build_resnet_18(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Initial Conv Layer
    x = conv_block(inputs, 64, stride=1)
    
    # Stage 1
    x = identity_block(x, 64)
    x = identity_block(x, 64)
    
    # Stage 2
    x = conv_block(x, 128)
    x = identity_block(x, 128)
    x = identity_block(x, 128)
    
    # Stage 3
    x = conv_block(x, 256)
    x = identity_block(x, 256)
    x = identity_block(x, 256)
    
    # Stage 4
    x = conv_block(x, 512)
    x = identity_block(x, 512)
    x = identity_block(x, 512)

    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Output Layer
    num_classes = len(np.unique(y))  # Number of unique classes in the labels
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Build the ResNet-18 model
IMG_SIZE = 128  # Define image size
model = build_resnet_18((IMG_SIZE, IMG_SIZE, 3))

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


import gc
gc.collect()


# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2)

# Make predictions on the test data
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)  # Get class with the highest probability

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Create the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# View the structure of the model
model.summary()




# custom resnet-18
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator




# Assuming df_balanced is already defined and contains the image data and labels
X = np.array(df_balanced['image_array'].tolist())  # Shape will be (4997, 128, 128, 3)
y = df_balanced['dx'].values  # This will give you the diagnosis labels

from tensorflow.keras.utils import to_categorical
# Encode labels
label_encoder = LabelEncoder()
y_int = label_encoder.fit_transform(y)
y_encoded = to_categorical(y_int, num_classes=7)
num_classes = len(np.unique(y)) 

# Load InceptionV3 model with ImageNet weights, excluding the top layers
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=40)


for layer in base_model.layers:
    layer.trainable = False

# Add new layers for custom classification
x = base_model.output
x = GlobalAveragePooling2D()(x)           # Global average pooling to reduce dimensions
x = Dense(1024, activation='relu')(x)     # Fully connected layer
output = Dense(num_classes, activation='softmax')(x)  # Final output layer for classification

TF_ENABLE_MLIR=1

# Create the final model
model = Model(inputs=base_model.input, outputs=output)

model.summary()

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    batch_size=32
)


# Make predictions on the test data
# Predict class probabilities
y_pred_prob = model.predict(X_test)

# Convert probabilities to class labels
y_pred = np.argmax(y_pred_prob, axis=1)  # Predicted classes
y_true = np.argmax(y_test, axis=1)        # True classes (since y_test is one-hot encoded)

from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy Score:", accuracy)



# Create the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()