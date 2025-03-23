# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:46:37 2024

@author: aaari
"""

# akiec -> 0
# bcc -> 1
# bkl -> 2
# df -> 3
# mel -> 4
# nv -> 5
# vasc -> 6


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


IMG_SIZE = 128  
channels = 3    
def load_and_preprocess_image(image_path):
    # Load the image from the given path
    img = cv2.imread(image_path)
    
    if img is None:
        # raise ValueError(f"Image at path '{image_path}' could not be loaded.")
        return None
    # Resize the image to the desired dimensions (128x128 in your case)
    img = cv2.resize(img, (128, 128))
    
    # Normalize the image to the range [0, 1]
    img_array = img.astype(np.float32) / 255.0
    
    return img_array

tqdm.pandas()
df['image_array'] = df['image_path'].progress_apply(load_and_preprocess_image)

print(df[['image_id', 'dx', 'image_path', 'image_array']].head())



df = df.dropna()
df.head()




df['dx'].value_counts()


df = df[['dx','image_path','image_array']]
df.head()




gan_class_0_folder = "datasets/HAM10000_images_part_1_org/gans_class_0_ackie"
gan_class_3_folder = "datasets/HAM10000_images_part_1_org/gans_class_3_df"
gan_class_6_folder = "datasets/HAM10000_images_part_1_org/gans_class_6_vasc"
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
df_gan_class_3 = load_gan_images_and_add_to_df(gan_class_3_folder, 'df')
df_gan_class_6 = load_gan_images_and_add_to_df(gan_class_6_folder, 'vasc')

df_gan_combined = pd.concat([df_gan_class_0, df_gan_class_3, df_gan_class_6], ignore_index=True)

# Now append the GAN images DataFrame to the original DataFrame
df_combined = pd.concat([df, df_gan_combined], ignore_index=True)

df.head()

df_combined['dx'].value_counts()


# Function to balance each label to have exactly `target_count` instances
def balance_labels(df, target_count=250):
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
df_balanced = balance_labels(df_combined, target_count=250)


df_balanced['dx'].value_counts()





X = np.array(df_balanced['image_array'].tolist())  # Shape will be (4997, 128, 128, 3)
y = df_balanced['dx'].values  # This will give you the diagnosis labels

# Flatten the images for KNN
X_flattened = X.reshape(X.shape[0], -1)  # Shape will be (4997, 128*128*3)
print(f"Shape of flattened image data: {X_flattened.shape}")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_flattened, y_encoded, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
print(f"Validation data shape: {X_test.shape}, Validation labels shape: {y_test.shape}")



from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust n_neighbors

# Fit the model on the training data
knn.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification Report
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print("Classification Report:\n", report)




from sklearn.tree import DecisionTreeClassifier
# Create a decision tree classifier
clf = DecisionTreeClassifier(criterion='entropy',max_depth=20)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification Report
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print("Classification Report:\n", report)



from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 4. Train the classifier on the training data
clf.fit(X_train, y_train)

# 5. Make predictions on the test data
y_pred = clf.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification Report
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print("Classification Report:\n", report)




from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Initialize the AdaBoostClassifier
# We'll use a DecisionTreeClassifier with depth 1 as the base estimator (default behavior of AdaBoost)
clf = AdaBoostClassifier(n_estimators=100, random_state=42)

# 2. Train the AdaBoost classifier on the training data
clf.fit(X_train, y_train)

# 3. Make predictions on the test data
y_pred = clf.predict(X_test)

# 4. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# 5. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# 6. Classification Report
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print("Classification Report:\n", report)

