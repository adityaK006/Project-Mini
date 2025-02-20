# Project-Mini
Overview

Driver drowsiness is a major cause of road accidents, and detecting early signs of fatigue can prevent crashes and save lives. This project aims to develop a driver drowsiness detection system using deep learning models to analyze facial cues such as eye state (open/closed), yawning, and head pose.

Research Background

The project is based on research into facial-based drowsiness detection methods. The approach is inspired by existing studies that classify driver states based on facial expressions, eye movement, and head orientation. Our model integrates multiple detection components to improve accuracy and robustness.

Dataset

The model is trained using datasets containing labeled images of driver states, including:

FL3D Dataset (derived from NIT YMED): Contains 53,331 images labeled into three categories: Alert, Microsleep, and Yawning. Includes JSON annotations with facial landmarks.

Hugging Face Eye Dataset: Used for open/closed eye classification, balanced with 2000 images per class.

Custom Dataset: Augmented data for better generalization.

Model Architecture

The system consists of multiple CNN-based deep learning models:

Eye Detection Model

Custom CNN (inspired by AlexNet)

5 convolutional layers with batch normalization and max pooling

Fully connected layers with a sigmoid activation for binary classification (open/closed eyes)

Trained using SGD optimizer with a learning rate of 0.01

Face Detection & Drowsiness Classification Model

Another custom CNN trained on the FL3D dataset

Classifies facial states into Alert, Yawning, and Microsleep

Images resized to 256x256 before feeding into the model

Head Pose Estimation Model (Upcoming)

Planned to determine if the driver is looking forward or sideways as an additional safety measure

Implementation

Training Approach: Models trained incrementally, storing weights after each dataset folder for tracking performance.

Integration Strategy: The models work together with logic like:

Open Eyes + Alert Face = Awake

Closed Eyes + Yawning = Drowsy

Performance Metric: Achieved AUC = 0.97 for eye detection, with ongoing improvements for the face model.

Future Enhancements

Night Vision Adaptation: Improve detection in low-light conditions.

Real-Time Processing: Optimize models for real-time inference using OpenCV and TensorFlow Lite.

Drowsiness Scoring System: Combine eye, face, and head pose outputs into a weighted drowsiness score.
