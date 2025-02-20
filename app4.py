#updated the logic and threshold values.... the progress bar is working fine.....
#still i can add feature-> if face is detected then only use my models to calculate
#the score....
#Update-3/1/25  App working alert sound for drowsiness.

from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import threading
import time
import dlib

# Flask app initialization
app = Flask(__name__)

# Paths to the trained models
EYE_MODEL_PATH = "updated_eye_detection_19novalexnet_model4.keras"
FACE_MODEL_PATH = "face_model3_alexnet.keras"

# Load the trained models
eye_model = tf.keras.models.load_model(EYE_MODEL_PATH)
face_model = tf.keras.models.load_model(FACE_MODEL_PATH)

# Class Labels for the face model
FACE_LABELS = {0: "alert", 2: "yawning"}

# Image sizes
EYE_IMAGE_SIZE = (256, 256)
FACE_IMAGE_SIZE = (128, 128)

# Drowsiness thresholds
YAWNING_SCORE = 0.5
CLOSED_EYES_SCORE = 5
DROWSINESS_THRESHOLD = 100  # Threshold to declare drowsiness
EYE_CLOSURE_THRESHOLD = 2  # 2 seconds for eye closure to indicate drowsiness
YAWN_FREQUENCY_THRESHOLD = 3  # More than 2 yawns
YAWN_TIME_FRAME = 10  # Time frame in seconds to count yawns

# Global variables to store live predictions
live_predictions = {
    "eye_label": "N/A",
    "face_output": "N/A",
    "drowsiness_score": 0,
    "drowsiness_status": "N/A"
}
lock = threading.Lock()

# Global variable to track yawns within a time frame
yawn_timestamps = []

# Preprocessing functions
def preprocess_image(image, target_size=(256, 256)):
    image_resized = cv2.resize(image, target_size)
    image_normalized = img_to_array(image_resized) / 255.0
    return np.expand_dims(image_normalized, axis=0)

def preprocess_face_frame(frame):
    img = cv2.resize(frame, FACE_IMAGE_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def find_area(array):
    a = 0
    ox, oy = array[0]
    for x, y in array[1:]:
        a += abs(x * oy - y * ox)
        ox, oy = x, y
    return a / 2

# Persistent variables for eye closure tracking
static_eye_close_time = None
eye_closed_duration = 0

# Persistent variable to store the cumulative drowsiness score
cumulative_score = 0

def calculate_drowsiness(eye_label, face_output, current_time):
    global yawn_timestamps, static_eye_close_time, eye_closed_duration, cumulative_score

    # Initialize incremental score
    score_increment = 0

    # Track yawns in a rolling window
    if face_output == "yawning":
        with lock:  # Protect access to yawn_timestamps
            yawn_timestamps.append(current_time)

    # Remove yawns that are outside the time frame
    with lock:
        yawn_timestamps = [timestamp for timestamp in yawn_timestamps if current_time - timestamp <= YAWN_TIME_FRAME]

    # Yawn scoring: Add score only if yawn frequency threshold is met
    if len(yawn_timestamps) >= YAWN_FREQUENCY_THRESHOLD:
        score_increment += YAWNING_SCORE
        yawn_timestamps = []  # Reset yawn timestamps after scoring

    # Eye closure tracking
    if eye_label == "Closed Eye":
        if static_eye_close_time is None:  # Start tracking closed eye time
            static_eye_close_time = current_time
        else:
            eye_closed_duration = current_time - static_eye_close_time

        # Add score only if eye closure exceeds threshold
        if eye_closed_duration >= EYE_CLOSURE_THRESHOLD:
            score_increment += CLOSED_EYES_SCORE
            static_eye_close_time = None  # Reset tracking after scoring
    else:
        # Reset eye closure tracking if eyes are open
        static_eye_close_time = None
        eye_closed_duration = 0

    # No score increment for "Open Eye" and "Alert"
    if eye_label == "Open Eye" and face_output == "alert":
        score_increment = 0

    # Update cumulative score
    cumulative_score += score_increment

    # Debugging: Print current scores
    print(f"Eye Label: {eye_label}, Face Output: {face_output}")
    print(f"Score Increment: {score_increment}, Cumulative Score: {cumulative_score}")

    return cumulative_score


# Video stream generator for webcam
def generate_frames():
    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        return

    # Load dlib model for yawn detection
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks (1).dat")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Create a separate frame for face detection
        detection_frame = frame.copy()

        # Convert detection_frame to grayscale for face detection
        gray_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector(gray_frame)
        face_output = "alert"  # Default to alert unless yawn is detected

        for face in faces:
            # Detect landmarks for yawn detection
            landmarks = predictor(gray_frame, face)
            Landarkss = []
            for n in range(48, 55): 
                x = landmarks.part(n).x 
                y = landmarks.part(n).y 
                Landarkss.append((x, y))
                cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)
            Landarkss2 = []
            for n in range(54, 61): 
                x = landmarks.part(n).x 
                y = landmarks.part(n).y 
                Landarkss.append((x, y))
                cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)

            # Calculate the area of the mouth to detect yawning
            array = Landarkss + Landarkss2
            area = find_area(array)
            YAWN_THRESH = 26000
            if area > YAWN_THRESH:
                face_output = "yawning"  # Mark the face output as yawning

        # Preprocess inputs for eye model
        eye_roi = frame.copy()  # Whole frame for eye model
        eye_input = preprocess_image(eye_roi, target_size=EYE_IMAGE_SIZE)

        # Make predictions for eyes
        eye_pred = eye_model.predict(eye_input, verbose=0)[0][0]
        eye_label = "Open Eye" if eye_pred >= 0.6 else "Closed Eye"

        # Get current time to track yawns within a time window
        current_time = time.time()

        # Calculate drowsiness score
        drowsiness_score = calculate_drowsiness(eye_label, face_output, current_time)
        drowsiness_status = "Drowsy" if drowsiness_score >= DROWSINESS_THRESHOLD else "Alert"

        # Update live predictions
        with lock:
            live_predictions["eye_label"] = eye_label
            live_predictions["face_output"] = face_output
            live_predictions["drowsiness_score"] = drowsiness_score
            live_predictions["drowsiness_status"] = drowsiness_status

        # Overlay driver status on the original frame (not detection_frame)
        status_color = (0, 255, 0) if drowsiness_status == "Alert" else (0, 0, 255)
        cv2.putText(frame, f"Status: {drowsiness_status}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)

        # Encode the frame for display
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()



# Flask routes
@app.route('/')
def index():
    return '''
    <html>
        <head>
            <title>Drowsiness Detection System</title>
            <style>
                body {
                    font-family: 'Poppins', sans-serif;
                    background: linear-gradient(to bottom, #1e3c72, #2a5298);
                    margin: 0;
                    padding: 0;
                    color: white;
                }
                header {
                    text-align: center;
                    padding: 20px;
                    font-size: 2.5em;
                    font-weight: bold;
                    background: rgba(0, 0, 0, 0.5);
                }
                .container {
                    display: flex;
                    flex-direction: row;
                    justify-content: space-around;
                    padding: 20px;
                }
                .card {
                    background: rgba(255, 255, 255, 0.2);
                    backdrop-filter: blur(10px);
                    border-radius: 15px;
                    padding: 20px;
                    width: 45%;
                    text-align: center;
                    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
                }
                img {
                    border-radius: 10px;
                    border: 2px solid white;
                    width: 100%;
                }
                ul {
                    list-style: none;
                    padding: 0;
                    text-align: left;
                }
                ul li {
                    margin: 10px 0;
                    font-size: 1.2em;
                }
                button {
                    background-color: #4caf50;
                    border: none;
                    color: white;
                    padding: 15px 30px;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    font-size: 16px;
                    margin: 10px 5px;
                    cursor: pointer;
                    border-radius: 10px;
                    transition: all 0.3s ease-in-out;
                }
                button:hover {
                    background-color: #45a049;
                    transform: scale(1.1);
                }
                /* Progress Bar Container */
.progress-bar {
    height: 30px; /* Slightly taller for better visibility */
    border-radius: 15px; /* Rounded corners for a modern look */
    background: linear-gradient(to right, #f5f5f5, #e8e8e8); /* Subtle gradient background */
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Adds a shadow effect */
    border: 1px solid #dcdcdc; /* Light border */
}

/* Progress Bar Fill */
.progress-fill {
    height: 100%;
    width: 0%; /* Initialize as 0 */
    background: linear-gradient(90deg, #00b4ff, #0072ff); /* Gradient for the progress fill */
    border-radius: 15px; /* Match the container's rounded corners */
    text-align: center;
    line-height: 30px; /* Vertically center the text */
    color: white;
    font-weight: bold;
    font-size: 16px;
    font-family: 'Arial', sans-serif;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5); /* Add subtle text shadow */
    transition: width 0.5s ease, background-color 0.5s ease; /* Smooth animation for width and color */
    box-shadow: 0 0 8px rgba(0, 114, 255, 0.5); /* Glow effect for the progress fill */
}

/* Checkpoint Markers */
.checkpoint {
    position: absolute;
    top: -10px; /* Positioned slightly above the progress bar */
    width: 12px;
    height: 50px;
    background-color: #ffcc00; /* Yellow for visibility */
    border-radius: 6px; /* Rounded edges */
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2); /* Depth effect */
    transition: transform 0.3s ease, background-color 0.3s ease; /* Smooth hover animation */
    z-index: 1; /* Ensure checkpoints are above the fill */
}

/* Checkpoint Positions */
.checkpoint.level-1 { left: 25%; }
.checkpoint.level-2 { left: 50%; }
.checkpoint.level-3 { left: 75%; }

/* Hover Effect for Checkpoints */
.checkpoint:hover {
    background-color: #ffd700; /* Brighter yellow on hover */
    transform: scale(1.3); /* Slightly enlarge for emphasis */
}

/* Dynamic Progress Levels */
.progress-safe { background-color: #4caf50; } /* Green for safe */
.progress-warning { background-color: #ffcc00; } /* Yellow for warning */
.progress-danger { background-color: #ff8000; } /* Orange for danger */
.progress-critical { background-color: #ff0000; } /* Red for critical */

            </style>
        </head>
        <body>
    <header>Drowsiness Detection System</header>
    <div class="container">
        <div class="card">
            <h2>Live Webcam Feed</h2>
            <img id="video-feed" src="/video_feed" alt="Live Feed">
        </div>
        <div class="card">
            <h2>Detection Status</h2>
            <ul>
                <li><b>Eye Status:</b> <span id="eye-status">N/A</span></li>
                <li><b>Face Status:</b> <span id="face-status">N/A</span></li>
                <li><b>Drowsiness Score:</b> <span id="drowsiness-score">0</span></li>
                <li><b>Status:</b> <span id="alert-status">N/A</span></li>
            </ul>
            <div class="progress-container">
                <h3>Progress Bar (Drowsiness Levels)</h3>
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-fill">0%</div>
                    <div class="checkpoint level-1" style="left: 25%;"></div>
                    <div class="checkpoint level-2" style="left: 50%;"></div>
                    <div class="checkpoint level-3" style="left: 75%;"></div>
                </div>
            </div>
            <button onclick="startDetection()">Start Detection</button>
            <button onclick="exitApp()">Exit</button>
        </div>
    </div>
    <script>
        const updateStatus = () => {
            fetch('/live_status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('eye-status').innerText = data.eye_label || "N/A";
                    document.getElementById('face-status').innerText = data.face_output || "N/A";
                    document.getElementById('drowsiness-score').innerText = data.drowsiness_score || 0;
                    document.getElementById('alert-status').innerText = data.drowsiness_status || "N/A";
                    updateProgressBar(data.drowsiness_score || 0);
                })
                .catch(error => console.error('Error fetching live status:', error));
        };

        const updateProgressBar = (score) => {
    const progressBar = document.getElementById('progress-fill');

    // Determine progress level based on the score
    let progress = 0;
    if (score >= 75) {
        progress = 100; // Critical level
    } else if (score >= 50) {
        progress = 75; // Danger level
    } else if (score >= 25) {
        progress = 50; // Warning level
    } else {
        progress = 25; // Safe level
    }

    // Update progress bar width and text
    progressBar.style.width = `${progress}%`;
    progressBar.textContent = `${progress}%`;

    // Update color class based on progress level
    if (progress <= 25) {
        progressBar.className = 'progress-fill progress-safe'; // Green for safe
    } else if (progress <= 50) {
        progressBar.className = 'progress-fill progress-warning'; // Yellow for warning
    } else if (progress <= 75) {
        progressBar.className = 'progress-fill progress-danger'; // Orange for danger
    } else {
        progressBar.className = 'progress-fill progress-critical'; // Red for critical
    }

    // Play a sound when progress reaches 100%
    if (progress === 100) {
        if (!progressBar.dataset.soundPlayed) { // Ensure the sound is played only once
            playAlertSound(); // Call a helper function to play the sound
            progressBar.dataset.soundPlayed = 'true'; // Mark sound as played
        }
    } else {
        // Reset the soundPlayed attribute if the progress drops below 100%
        progressBar.dataset.soundPlayed = '';
    }
};

// Helper function to play alert sound
const playAlertSound = () => {
    try {
        const sound = new Audio('static/alertsound.mp3'); // Replace with your sound file's path
        sound.volume = 1.0; // Adjust volume if needed (0.0 to 1.0)
        sound.play().catch(error => {
            console.error('Error playing alert sound:', error);
        });
    } catch (error) {
        console.error('Audio playback failed:', error);
    }
};

        let interval;

        function startDetection() {
            // Start the periodic updates
            interval = setInterval(updateStatus, 1000);
            alert('Detection Started!');
        }

        function exitApp() {
            fetch('/shutdown', { method: 'POST' })
                .then(() => {
                    alert('Exiting application...');
                    window.location.href = 'about:blank';
                })
                .catch(console.error);
        }
    </script>
</body>

    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_status')
def live_status():
    with lock:
        print("Live Predictions:", live_predictions)  # Debugging line
        # Ensure drowsy_count is in the response
        return jsonify({
            "eye_label": live_predictions.get("eye_label", "N/A"),
            "face_output": live_predictions.get("face_output", "N/A"),
            "drowsiness_score": live_predictions.get("drowsiness_score", 0),
            "drowsiness_status": live_predictions.get("drowsiness_status", "N/A"),
            "drowsy_count": live_predictions.get("drowsy_count", 0)
        })

@app.route('/shutdown', methods=['POST'])
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return 'Server shutting down...'

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
