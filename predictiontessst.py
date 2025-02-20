# from tensorflow.keras.preprocessing.image import img_to_array
# from PIL import Image
# import numpy as np
# import tensorflow as tf

# # Load the trained model
# model = tf.keras.models.load_model('updated_eye_detection_19novalexnet_model4.keras')

# # Function to preprocess a single image
# def preprocess_single_image(image_path, target_size=(256, 256)):
#     image = Image.open(image_path)  # Open the image
#     image = image.resize(target_size)  # Resize to target size
#     image = img_to_array(image) / 255.0  # Convert to array and normalize
#     return np.expand_dims(image, axis=0)  # Add batch dimension

# # Path to the image
# image_path = 'vs test image/download (close4).jpeg'

# # Preprocess the image
# processed_image = preprocess_single_image(image_path)

# # Predict using the model
# prediction = model.predict(processed_image)

# # Interpret the result
# if prediction[0] > 0.5:  # Assuming sigmoid activation
#     print("Prediction: Open Eye")
# else:
#     print("Prediction: Closed Eye")


import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = tf.keras.models.load_model('updated_eye_detection_19novalexnet_model4.keras')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define target size for images (the same as used for training)
target_size = (256, 256)

# Preprocessing function for the webcam frames
def preprocess_image(image, target_size=(256, 256)):
    image_resized = cv2.resize(image, target_size)  # Resize image
    image_normalized = img_to_array(image_resized) / 255.0  # Normalize image
    return np.expand_dims(image_normalized, axis=0)  # Add batch dimension

# Loop to capture frames from the webcam
while True:
    # Capture a frame
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    processed_image = preprocess_image(frame, target_size)

    # Make predictions using the model
    prediction = model.predict(processed_image)
    
    # Determine the prediction (0: Closed Eye, 1: Open Eye)
    label = 'Open Eye' if prediction >= 0.6 else 'Closed Eye'
    color = (0, 255, 0) if label == 'Open Eye' else (0, 0, 255)  # Green for Open, Red for Closed
    
    # Display the label on the frame
    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # Display the webcam frame with the label
    cv2.imshow('Eye Detection', frame)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import img_to_array

# # Load the trained model
# model = tf.keras.models.load_model('updated_eye_detection_19novalexnet_model4.keras')

# # Define image preprocessing function
# def preprocess_image(image, target_size=(256, 256)):
#     image = cv2.resize(image, target_size)
#     image = img_to_array(image) / 255.0  # Normalize the image
#     return np.expand_dims(image, axis=0)  # Add batch dimension

# # Open the webcam
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# # Define font for text display
# font = cv2.FONT_HERSHEY_SIMPLEX

# print("Press 'q' to exit.")

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         print("Error: Could not read frame.")
#         break

#     # Convert frame to grayscale (optional, depending on preprocessing)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect face using Haarcascade (or any other face detection model)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

#     for (x, y, w, h) in faces:
#         # Extract the face region
#         face = frame[y:y+h, x:x+w]

#         # Preprocess the face image
#         processed_face = preprocess_image(face)

#         # Predict using the model
#         prediction = model.predict(processed_face).flatten()
#         label = "Open Eye" if prediction > 0.5 else "Closed Eye"

#         # Draw rectangle around the face and display the label
#         color = (0, 255, 0) if label == "Open Eye" else (0, 0, 255)
#         cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
#         cv2.putText(frame, label, (x, y-10), font, 0.7, color, 2)

#     # Display the frame
#     cv2.imshow("Eye State Detection", frame)

#     # Exit on pressing 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam and close the windows
# cap.release()
# cv2.destroyAllWindows()
