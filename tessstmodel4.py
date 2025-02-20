#vnev = py10

import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the pre-trained model
model = tf.keras.models.load_model('updated_eye_detection_19novalexnet_model4.keras')
# model = models.Sequential([
#     # First Convolutional Block
#     layers.Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=(256, 256, 3), kernel_regularizer=regularizers.l2(0.001)),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D((3, 3), strides=2),
    
#     # Second Convolutional Block
#     layers.Conv2D(256, (5, 5), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D((3, 3), strides=2),
    
#     # Third Convolutional Block
#     layers.Conv2D(384, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)),
#     layers.BatchNormalization(),
    
#     # Fourth Convolutional Block
#     layers.Conv2D(384, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)),
#     layers.BatchNormalization(),
    
#     # Fifth Convolutional Block
#     layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D((3, 3), strides=2),

#     # Fully Connected Layers
#     layers.Flatten(),
#     layers.Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
#     layers.Dropout(0.5),
#     layers.Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
#     layers.Dropout(0.5),
#     layers.Dense(1, activation='sigmoid')  # Binary classification
# ])



# Load the balanced dataset for further training
with open('balanced_data.pkl', 'rb') as f:
    balanced_data = pickle.load(f)

# Image preprocessing function
def preprocess_image(image, target_size=(256, 256)):
    img = image.resize(target_size)
    return img_to_array(img) / 255.0

# Define data augmentation for the training dataset
data_augmentation = ImageDataGenerator(
    rotation_range=15,          # Randomly rotate images by up to 15 degrees
    width_shift_range=0.1,      # Randomly shift images horizontally by 10%
    height_shift_range=0.1,     # Randomly shift images vertically by 10%
    zoom_range=0.2,             # Randomly zoom in on images
    horizontal_flip=True,       # Randomly flip images horizontally
    fill_mode='nearest'         # Fill empty pixels with the nearest value
)

# Generator function to load data in batches
def augmented_generator(data, batch_size, target_size=(256, 256), augment=False):
    i = 0
    while True:
        batch_images = []
        batch_labels = []
        for _ in range(batch_size):
            if i >= len(data):
                i = 0  # Reset to the beginning of the dataset
            item = data[i]
            image = item['Image_data']['file']  # Assuming this is already a PIL image
            label = 1 if item['Label'] == 'open_eyes' else 0
            image = preprocess_image(image, target_size)  # Preprocess image
            
            # Apply augmentation only to the training set
            if augment:
                image = data_augmentation.random_transform(image)

            batch_images.append(image)
            batch_labels.append(label)
            i += 1
        yield np.array(batch_images), np.array(batch_labels)

# Define batch size and target image size
batch_size = 8
target_size = (256, 256)

# Split the dataset into training and validation sets
train_data, val_data = train_test_split(balanced_data, test_size=0.2, random_state=42)

# Use augmented generator for training and standard generator for validation
train_gen = augmented_generator(train_data, batch_size, target_size, augment=True)
val_gen = augmented_generator(val_data, batch_size, target_size, augment=False)  # No augmentation for validation

# Update optimizer to SGD with a learning rate of 0.01
new_optimizer = SGD(learning_rate=0.01)

# Compile the model with the new optimizer
model.compile(optimizer=new_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model using the updated optimizer and augmented data
history = model.fit(train_gen, 
                    steps_per_epoch=len(train_data) // batch_size, 
                    validation_data=val_gen, 
                    validation_steps=len(val_data) // batch_size, 
                    epochs=20, 
                    callbacks=[early_stopping])

# Save the updated model
model.save('updated_eye_detection_19novalexnet_model_augmented.keras')

# Evaluate the model on validation set
val_loss, val_acc = model.evaluate(val_gen, steps=len(val_data) // batch_size)
print(f'Updated Validation Accuracy: {val_acc*100:.2f}%')

# Get true labels and predictions
true_labels = []
predicted_probs = []

for _ in range(len(val_data) // batch_size):
    images, labels = next(val_gen)
    true_labels.extend(labels)
    predicted_probs.extend(model.predict(images).flatten())  # Flatten to 1D array

# Convert to numpy arrays
true_labels = np.array(true_labels)
predicted_probs = np.array(predicted_probs)

# Threshold predictions to get binary classes
predicted_classes = (predicted_probs > 0.5).astype(int)

# Classification report
report = classification_report(true_labels, predicted_classes, target_names=['Closed Eye', 'Open Eye'])
print("Classification Report:\n", report)

# Confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_classes)
print("Confusion Matrix:\n", conf_matrix)

# Plot training history
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
