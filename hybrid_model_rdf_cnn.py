import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import glob
import cv2
from tensorflow.keras.utils import to_categorical
from tqdm.auto import tqdm


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50, ResNet152, ResNet101
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import warnings
warnings.filterwarnings("ignore")




import tensorflow as tf
tf.keras.backend.clear_session()




df = pd.read_csv("datasets\HAM10000_metadata.csv")
df.head()



img_directory = "datasets/HAM10000_images_part_1_org/"
def get_image_path(image_id):
    return os.path.join(img_directory,image_id+".jpg")
df['image_path'] = df['image_id'].apply(get_image_path)
df.head()
df.shape

IMG_SIZE = 96  
channels = 3    
def load_and_preprocess_image(image_path):
    # Load the image from the given path
    img = cv2.imread(image_path)
    
    if img is None:
        print("No")
        # raise ValueError(f"Image at path '{image_path}' could not be loaded.")
        return None
    # Resize the image to the desired dimensions (256 * 256 in your case)
    img = cv2.resize(img, (96, 96))
    
    # Normalize the image to the range [0, 1]
    img_array = img.astype(np.float32) / 255.0
    
    return img_array

tqdm.pandas()
df['image_array'] = df['image_path'].progress_apply(load_and_preprocess_image)



df.head()




df = df[['dx','image_path','image_array']]

gan_akiec_folder = "datasets\gans_images\gans_akiec"
gan_bcc_folder = "datasets\gans_images\gans_bcc"
gan_df_folder = "datasets\gans_images\gans_df"
gan_vasc_folder = "datasets\gans_images\gans_vasc"
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
df_gan_akiec = load_gan_images_and_add_to_df(gan_akiec_folder, 'akiec')
df_gan_bcc = load_gan_images_and_add_to_df(gan_bcc_folder, 'bcc')
df_gan_df = load_gan_images_and_add_to_df(gan_df_folder, 'df')
df_gan_vasc = load_gan_images_and_add_to_df(gan_vasc_folder, 'vasc')


df_gan_combined = pd.concat([df_gan_akiec,df_gan_bcc,df_gan_df, df_gan_vasc], ignore_index=True)

# Now append the GAN images DataFrame to the original DataFrame
df_combined = pd.concat([df, df_gan_combined], ignore_index=True)

df_combined.head()
df_combined['dx'].value_counts()


# Function to balance each label to have exactly `target_count` instances
def balance_labels(df, target_count=500):
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
df = balance_labels(df_combined, target_count=500)


df['dx'].value_counts()


# Assuming df_balanced is already defined and contains the image data and labels
X = np.array(df['image_array'].tolist())  
y = df['dx'].values  

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)
# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.25, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)



import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Build a simple CNN
def build_cnn(input_shape=(96, 96, 3)):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    model = Model(inputs, x)  # Feature extraction model
    return model

cnn_model = build_cnn()
cnn_model.summary()


# Extract features
def extract_features_in_batches(model, data, batch_size):
    features = []
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i + batch_size]
        features.append(model.predict(batch_data))
    return np.vstack(features)

batch_size = 8
X_train_features = extract_features_in_batches(cnn_model, X_train, batch_size)
X_val_features = extract_features_in_batches(cnn_model, X_val, batch_size)
X_test_features = extract_features_in_batches(cnn_model, X_test, batch_size)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Define the parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize Random Forest
rf = RandomForestClassifier(random_state=42)

# Use GridSearchCV to search for the best parameters
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train_features, y_train)

# Print the best parameters and their corresponding score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Evaluate on test data using the best model
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test_features)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy with Best Random Forest: {accuracy}")
print(best_rf)


from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Predict on the test set
y_pred = best_rf.predict(X_test_features)

# Generate the classification report
class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print("Classification Report:")
print(class_report)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix values (numerical output)
print("Confusion Matrix:")
print(cm)

# Optionally, visualize the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='viridis', xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.show()

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define the parameter grid for AdaBoost
param_grid = {
    'n_estimators': [50, 100, 200, 300],  # Number of weak learners
    'learning_rate': [0.01, 0.1, 1.0, 1.5],  # Learning rate
    'estimator__max_depth': [1, 5, 10, 20],  # Depth of the decision tree base estimator
}

# Initialize AdaBoost with DecisionTree as the base estimator
ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(), random_state=42)

# Use GridSearchCV to search for the best parameters
grid_search = GridSearchCV(estimator=ada, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train_features, y_train)

# Print the best parameters and their corresponding score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Evaluate the best model on the test data
best_ada = grid_search.best_estimator_
y_pred_ada = best_ada.predict(X_test_features)
accuracy_ada = accuracy_score(y_test, y_pred_ada)
print(f"Test Accuracy with Tuned AdaBoost: {accuracy_ada}")

# Generate classification report and confusion matrix
print("\nClassification Report:\n", classification_report(y_test, y_pred_ada))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_ada))
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred_ada), display_labels=label_encoder.classes_)
disp.plot(cmap='viridis', xticks_rotation='vertical')
plt.title("Confusion Matrix_adaboost")
plt.show()