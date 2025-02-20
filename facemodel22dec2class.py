# import os
# import json
# import cv2
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
# from tensorflow.keras.optimizers import SGD
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping

# # Dataset directory path
# IMAGE_SIZE = (128, 128)  # Resize dimensions
# LABELS = {"alert": 0, "yawning": 1}  # Updated class labels (excluding "microsleep")
# BATCH_SIZE = 4
# EPOCHS = 20

# # CNN Model Architecture (AlexNet-inspired)
# def create_model(input_shape=(128, 128, 3)):
#     model = Sequential([
#         # Block 1
#         Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
#         BatchNormalization(),
#         MaxPooling2D(pool_size=(2, 2)),

#         # Block 2
#         Conv2D(64, kernel_size=(3, 3), activation='relu'),
#         BatchNormalization(),
#         MaxPooling2D(pool_size=(2, 2)),

#         # Block 3
#         Conv2D(128, kernel_size=(3, 3), activation='relu'),
#         BatchNormalization(),
#         MaxPooling2D(pool_size=(2, 2)),

#         # Flatten and Dense layers
#         Flatten(),
#         Dense(256, activation='relu'),
#         Dropout(0.5),
#         Dense(1, activation='sigmoid')  # 2 classes (alert, yawning)
#     ])
#     model.compile(optimizer=SGD(learning_rate=0.005), loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# # Function to load data from a single folder
# def load_folder_data(folder_path):
#     images = []
#     labels = []

#     # JSON file with annotations
#     json_file = os.path.join(folder_path, "annotations_final.json")

#     if not os.path.exists(json_file):
#         print(f"No JSON file found in {folder_path}. Exiting.")
#         return np.array([]), np.array([])

#     with open(json_file, 'r') as f:
#         annotations = json.load(f)
    
#     # Load and preprocess images
#     for image_path, data in annotations.items():
#         if data['driver_state'] not in LABELS:  # Skip "microsleep" class
#             continue

#         full_image_path = os.path.join(folder_path, image_path)
#         if os.path.exists(full_image_path):
#             img = cv2.imread(full_image_path)
#             img = cv2.resize(img, IMAGE_SIZE)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = img / 255.0  # Normalize
#             images.append(img)
#             labels.append(LABELS[data['driver_state']])
    
#     return np.array(images), np.array(labels)

# # Function to balance classes in the dataset
# def balance_classes(images, labels):
#     unique_classes, counts = np.unique(labels, return_counts=True)
#     print("Class distribution before balancing:", dict(zip(unique_classes, counts)))

#     # Find the minimum number of samples for any class
#     min_samples = min(counts)
#     print(f"Balancing classes to {min_samples} samples each.")

#     # Collect indices for each class
#     balanced_indices = []
#     for class_label in unique_classes:
#         indices = np.where(labels == class_label)[0]
#         np.random.shuffle(indices)
#         balanced_indices.extend(indices[:min_samples])
    
#     # Shuffle and select balanced data
#     np.random.shuffle(balanced_indices)
#     balanced_images = images[balanced_indices]
#     balanced_labels = labels[balanced_indices]
#     print("Class distribution after balancing:", dict(zip(*np.unique(balanced_labels, return_counts=True))))
    
#     return balanced_images, balanced_labels

# # Training function for a single folder with data augmentation and validation split
# def train_on_single_folder(folder_path): 
#     model_path = "face_model4_2classes.keras"
    
#     # model = tf.keras.models.load_model(model_path)
#     # print("Model Loaded...")
    
#     model = create_model()
#     print("New Model Created...")

#     # Load data from the given folder
#     images, labels = load_folder_data(folder_path)
#     if len(images) == 0:
#         print("No data found in the folder. Exiting training.")
#         return

#     # Balance classes
#     images, labels = balance_classes(images, labels)

#     # Convert labels to one-hot encoding
#     labels = np.eye(len(LABELS))[labels]

#     # Split data into training and validation sets
#     X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

#     # Data Augmentation
#     train_datagen = ImageDataGenerator(rotation_range=15,
#                                        width_shift_range=0.1,
#                                        height_shift_range=0.1,
#                                        zoom_range=0.2,
#                                        horizontal_flip=True,
#                                        fill_mode="nearest")
    
#     val_datagen = ImageDataGenerator()

#     # Prepare generators
#     train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
#     val_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

#     # EarlyStopping callback
#     early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

#     # Train the model
#     print(f"Training on data from folder: {folder_path}")
#     history = model.fit(train_generator,
#                         epochs=EPOCHS,
#                         validation_data=val_generator,
#                         callbacks=[early_stopping],
#                         verbose=1)

#     # Save the model
#     model.save(model_path)
#     print(f"Model saved as {model_path}")

#     # Plot training history
#     plot_training_history(history)

# # Function to plot training and validation accuracy/loss
# def plot_training_history(history):
#     plt.figure(figsize=(12, 4))
    
#     # Plot accuracy
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['accuracy'], label='Train Accuracy')
#     plt.plot(history.history['val_accuracy'], label='Val Accuracy')
#     plt.title('Training and Validation Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()
    
#     # Plot loss
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['loss'], label='Train Loss')
#     plt.plot(history.history['val_loss'], label='Val Loss')
#     plt.title('Training and Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
    
#     plt.show()

# if __name__ == "__main__":
#     # Manually specify the folder path
#     folder_path = "archive (3)/classification_frames/P1043134_720"
#     if os.path.exists(folder_path):
#         train_on_single_folder(folder_path)
#     else:
#         print("Folder path does not exist. Please check the path and try again.")






# import os
# import json
# import cv2
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras import regularizers

# # Dataset directory path
# IMAGE_SIZE = (128, 128)  # Resize dimensions
# LABELS = {"alert": 0, "yawning": 1}  # Updated class labels (excluding "microsleep")
# BATCH_SIZE = 32
# EPOCHS = 50  # Increased epochs

# # CNN Model Architecture (AlexNet-inspired with improvements)
# def create_model(input_shape=(128, 128, 3)):
#     model = Sequential([
#         # Block 1
#         Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.01)),
#         BatchNormalization(),
#         MaxPooling2D(pool_size=(2, 2)),

#         # Block 2
#         Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
#         BatchNormalization(),
#         MaxPooling2D(pool_size=(2, 2)),

#         # Block 3
#         Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
#         BatchNormalization(),
#         MaxPooling2D(pool_size=(2, 2)),

#         # Block 4 (new)
#         Conv2D(512, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
#         BatchNormalization(),
#         MaxPooling2D(pool_size=(2, 2)),

#         # Flatten and Dense layers
#         Flatten(),
#         Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
#         Dropout(0.5),
#         Dense(1, activation='sigmoid')  # 2 classes (alert, yawning)
#     ])
#     model.compile(optimizer=Adam(learning_rate=0.05), loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# # Function to load data from a single folder
# def load_folder_data(folder_path):
#     images = []
#     labels = []

#     # JSON file with annotations
#     json_file = os.path.join(folder_path, "annotations_final.json")

#     if not os.path.exists(json_file):
#         print(f"No JSON file found in {folder_path}. Exiting.")
#         return np.array([]), np.array([])

#     with open(json_file, 'r') as f:
#         annotations = json.load(f)
    
#     # Load and preprocess images
#     for image_path, data in annotations.items():
#         if data['driver_state'] not in LABELS:  # Skip "microsleep" class
#             continue

#         full_image_path = os.path.join(folder_path, image_path)
#         if os.path.exists(full_image_path):
#             img = cv2.imread(full_image_path)
#             img = cv2.resize(img, IMAGE_SIZE)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = img / 255.0  # Normalize
#             images.append(img)
#             labels.append(LABELS[data['driver_state']])
    
#     return np.array(images), np.array(labels)

# # Function to balance classes in the dataset
# def balance_classes(images, labels):
#     unique_classes, counts = np.unique(labels, return_counts=True)
#     print("Class distribution before balancing:", dict(zip(unique_classes, counts)))

#     # Find the minimum number of samples for any class
#     min_samples = min(counts)
#     print(f"Balancing classes to {min_samples} samples each.")

#     # Collect indices for each class
#     balanced_indices = []
#     for class_label in unique_classes:
#         indices = np.where(labels == class_label)[0]
#         np.random.shuffle(indices)
#         balanced_indices.extend(indices[:min_samples])
    
#     # Shuffle and select balanced data
#     np.random.shuffle(balanced_indices)
#     balanced_images = images[balanced_indices]
#     balanced_labels = labels[balanced_indices]
#     print("Class distribution after balancing:", dict(zip(*np.unique(balanced_labels, return_counts=True))))
    
#     return balanced_images, balanced_labels

# # Training function for a single folder with data augmentation and validation split
# def train_on_single_folder(folder_path): 
#     model_path = "face_model4_2classes.keras"
    
#     # Create a new model
#     model = create_model()
#     print("New Model Created...")

#     # Load data from the given folder
#     images, labels = load_folder_data(folder_path)
#     if len(images) == 0:
#         print("No data found in the folder. Exiting training.")
#         return

#     # Balance classes
#     images, labels = balance_classes(images, labels)

#     # Convert labels to one-hot encoding
#     labels = np.eye(len(LABELS))[labels]

#     # Split data into training and validation sets
#     X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

#     # Data Augmentation
#     train_datagen = ImageDataGenerator(rotation_range=15,
#                                        width_shift_range=0.1,
#                                        height_shift_range=0.1,
#                                        zoom_range=0.2,
#                                        horizontal_flip=True,
#                                        fill_mode="nearest")
    
#     val_datagen = ImageDataGenerator()

#     # Prepare generators
#     train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
#     val_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

#     # EarlyStopping callback
#     early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#     # Train the model
#     print(f"Training on data from folder: {folder_path}")
#     history = model.fit(train_generator,
#                         epochs=EPOCHS,
#                         validation_data=val_generator,
#                         callbacks=[early_stopping],
#                         verbose=1)

#     # Save the model
#     model.save(model_path)
#     print(f"Model saved as {model_path}")

#     # Plot training history
#     plot_training_history(history)

# # Function to plot training and validation accuracy/loss
# def plot_training_history(history):
#     plt.figure(figsize=(12, 4))
    
#     # Plot accuracy
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['accuracy'], label='Train Accuracy')
#     plt.plot(history.history['val_accuracy'], label='Val Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()
    
#     # Plot loss
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['loss'], label='Train Loss')
#     plt.plot(history.history['val_loss'], label='Val Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
    
#     plt.show()

# # Start training on a folder (example usage)
# train_on_single_folder("archive (3)/classification_frames/P1043134_720")  # Replace with your actual path





import numpy as np
import os
import cv2
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt

# Constants
EPOCHS = 50
BATCH_SIZE = 32
MODEL_PATH = 'face_model_2classes.keras'
IMAGE_SIZE = (128, 128)  # Resize images to 128x128
LABELS = {'alert': 0, 'yawning': 1}  # Update with your driver states

# Function to load images and labels from a folder
def load_folder_data(folder_path):
    images = []
    labels = []

    # JSON file with annotations
    json_file = os.path.join(folder_path, "annotations_final.json")

    if not os.path.exists(json_file):
        print(f"No JSON file found in {folder_path}. Exiting.")
        return np.array([]), np.array([])

    with open(json_file, 'r') as f:
        annotations = json.load(f)
    
    # Load and preprocess images
    for image_path, data in annotations.items():
        full_image_path = os.path.join(folder_path, image_path)
        if os.path.exists(full_image_path):
            img = cv2.imread(full_image_path)
            img = cv2.resize(img, IMAGE_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img = img / 255.0  # Normalize the image
            images.append(img)
            labels.append(LABELS[data['driver_state']])  # Map the driver state to label
    
    return np.array(images), np.array(labels)

# Function to balance classes
def balance_classes(images, labels):
    unique_classes, counts = np.unique(labels, return_counts=True)
    print("Class distribution before balancing:", dict(zip(unique_classes, counts)))

    # Find the minimum number of samples for any class
    min_samples = min(counts)
    print(f"Balancing classes to {min_samples} samples each.")

    # Collect indices for each class
    balanced_indices = []
    for class_label in unique_classes:
        indices = np.where(labels == class_label)[0]
        np.random.shuffle(indices)
        balanced_indices.extend(indices[:min_samples])
    
    # Shuffle and select balanced data
    np.random.shuffle(balanced_indices)
    balanced_images = images[balanced_indices]
    balanced_labels = labels[balanced_indices]
    print("Class distribution after balancing:", dict(zip(*np.unique(balanced_labels, return_counts=True))))
    
    return balanced_images, balanced_labels

# Improved model architecture
def create_model(input_shape=(128, 128, 3)):  # RGB input shape
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # 2 classes (alert, yawning)
    ])
    
    # Use a learning rate scheduler
    optimizer = Adam(learning_rate=0.001)  # Even lower learning rate
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Adjusted EarlyStopping and Learning Rate Scheduler
def train_on_single_folder(folder_path): 
    model = create_model()

    # Load data from folder
    images, labels = load_folder_data(folder_path)
    if len(images) == 0:
        print("No data found in the folder. Exiting training.")
        return

    # Balance classes
    images, labels = balance_classes(images, labels)

    # Convert labels to one-hot encoding
    labels = np.eye(len(LABELS))[labels]

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Data Augmentation
    train_datagen = ImageDataGenerator(rotation_range=10,
                                       width_shift_range=0.05,
                                       height_shift_range=0.05,
                                       zoom_range=0.1,
                                       horizontal_flip=True,
                                       fill_mode="nearest")

    val_datagen = ImageDataGenerator()

    # Prepare generators
    train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

    # EarlyStopping callback with learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001)

    # Train the model
    print(f"Training on data from folder: {folder_path}")
    history = model.fit(train_generator,
                        epochs=EPOCHS,
                        validation_data=val_generator,
                        callbacks=[early_stopping, lr_scheduler],
                        verbose=1)

    # Save the model
    model.save(MODEL_PATH)
    print(f"Model saved as {MODEL_PATH}")

    # Plot training history
    plot_training_history(history)

# Function to plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

# Example usage
train_on_single_folder("archive (3)/classification_frames/P1043134_720")  # Replace with your actual path
