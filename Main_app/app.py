from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import logging
from werkzeug.utils import secure_filename
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for camera and model
face_cascade = None
emotion_model = None
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def initialize_models():
    """Initialize face detector and emotion model"""
    global face_cascade, emotion_model
    
    try:
        # Load face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        logger.info("Face cascade loaded successfully")
        
        # Try to load emotion model
        model_path = "../Task1/emotion_model.h5"
        if os.path.exists(model_path):
            try:
                from tensorflow.keras.models import load_model
                emotion_model = load_model(model_path, compile=False)
                logger.info("Emotion model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load emotion model: {e}")
                emotion_model = None
        else:
            logger.warning(f"Emotion model not found at {model_path}")
            emotion_model = None
            
    except Exception as e:
        logger.error(f"Error initializing models: {e}")

def preprocess_face(face_img):
    """Preprocess face image for emotion prediction"""
    try:
        # Resize to 48x48 grayscale and normalize (same as training)
        face_img = cv2.resize(face_img, (48, 48))
        face_img = face_img.astype('float32') / 255.0
        face_img = np.expand_dims(face_img, axis=-1)  # add channel dimension
        face_img = np.expand_dims(face_img, axis=0)   # add batch dimension
        return face_img
    except Exception as e:
        logger.error(f"Error preprocessing face: {e}")
        return None

def generate_frames():
    """Generate video frames with emotion detection"""
    cap = cv2.VideoCapture(0)
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        logger.error("Could not open camera")
        return
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                logger.warning("Failed to read frame from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            if face_cascade is not None:
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                for (x, y, w, h) in faces:
                    # Extract face region
                    face_img = gray[y:y+h, x:x+w]
                    
                    # Predict emotion if model is available
                    emotion_text = "No Model"
                    confidence = 0.0
                    
                    if emotion_model is not None:
                        try:
                            preprocessed = preprocess_face(face_img)
                            if preprocessed is not None:
                                preds = emotion_model.predict(preprocessed, verbose=0)
                                emotion_idx = np.argmax(preds)
                                confidence = float(preds[0][emotion_idx])
                                emotion_text = emotion_labels[emotion_idx]
                        except Exception as e:
                            logger.error(f"Error predicting emotion: {e}")
                            emotion_text = "Error"
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 3)
                    
                    # Draw emotion label with confidence
                    label = f"{emotion_text} ({confidence:.1%})"
                    
                    # Calculate text size for background
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                    )
                    
                    # Draw background rectangle for text
                    cv2.rectangle(frame, 
                                (x, y - text_height - 10), 
                                (x + text_width, y), 
                                (0, 255, 255), -1)
                    
                    # Draw text
                    cv2.putText(frame, label, (x, y - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # Add timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
    except Exception as e:
        logger.error(f"Error in frame generation: {e}")
    finally:
        cap.release()
        logger.info("Camera released")

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/camera_emotion')
def camera_emotion():
    """Camera emotion detection page"""
    return render_template('camera_emotion.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    """API endpoint to check system status"""
    status = {
        'face_detector': face_cascade is not None,
        'emotion_model': emotion_model is not None,
        'camera_available': True,  # We'll assume camera is available
        'supported_emotions': emotion_labels
    }
    return jsonify(status)

@app.route('/api/emotions')
def api_emotions():
    """API endpoint to get supported emotion labels"""
    return jsonify({
        'emotions': emotion_labels,
        'count': len(emotion_labels)
    })

# Placeholder routes for other features mentioned in the UI
@app.route('/activation_maps')
def activation_maps():
    """Placeholder for activation maps visualization"""
    return render_template('coming_soon.html', feature="Activation Maps Visualization")

@app.route('/train_emotion_model')
def train_emotion_model():
    """Placeholder for model training"""
    return render_template('coming_soon.html', feature="Emotion Model Training")

@app.route('/animal_detection')
def animal_detection_ui():
    """Animal detection UI"""
    return render_template('animal_detection.html')

@app.route('/animal_detect', methods=['POST'])
def animal_detect():
    """Run YOLO animal detection"""
    try:
        file = request.files['file']
        filename = secure_filename(file.filename)
        upload_path = os.path.join("static", "uploads", filename)
        os.makedirs(os.path.dirname(upload_path), exist_ok=True)
        file.save(upload_path)

        result_path = os.path.join("static", "results", "result_" + filename)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)

        # Run Task2/animal.py with input + output path
        subprocess.run(["python", "../Task2/animal.py", upload_path, result_path], check=True)

        return render_template("animal_detection.html", result_img=os.path.relpath(result_path, "static"))
    except Exception as e:
        logger.error(f"Animal detection error: {e}")
        return render_template("error.html", error_code=500, error_message="Animal detection failed")


@app.route('/detect_animals_video')
def detect_animals_video():
    """Run YOLO animal detection on video stream"""
    try:
        # This assumes Task2/animal.py has a mode for live video/webcam
        subprocess.Popen(["python", "../Task2/animal.py", "--video"])
        return render_template("animal_detection.html", video_mode=True)
    except Exception as e:
        logger.error(f"Animal video detection error: {e}")
        return render_template("error.html", error_code=500, error_message="Video detection failed")


@app.route('/train_animal_model')
def train_animal_model():
    """Placeholder for animal model training"""
    return render_template('coming_soon.html', feature="Animal Model Training")

@app.route('/audio_emotion')
def audio_emotion_ui():
    """Audio emotion recognition UI"""
    return render_template('audio_emotion.html')

@app.route('/audio_emotion_run', methods=['POST'])
def audio_emotion_run():
    """Run audio emotion recognition"""
    try:
        file = request.files['file']
        filename = secure_filename(file.filename)
        upload_path = os.path.join("static", "uploads", filename)
        os.makedirs(os.path.dirname(upload_path), exist_ok=True)
        file.save(upload_path)

        # Run Task3/female_emotion.py and capture result
        result = subprocess.check_output(["python", "../Task3/female_emotion.py", upload_path]).decode("utf-8").strip()

        return render_template("audio_emotion.html", emotion=result)
    except Exception as e:
        logger.error(f"Audio emotion recognition error: {e}")
        return render_template("error.html", error_code=500, error_message="Audio recognition failed")


@app.route('/record_audio')
def record_audio():
    """Record audio and run emotion recognition"""
    try:
        # This assumes Task3/female_emotion.py can record from mic if no arg is passed
        result = subprocess.check_output(["python", "../Task3/female_emotion.py"]).decode("utf-8").strip()
        return render_template("audio_emotion.html", emotion=result, recorded=True)
    except Exception as e:
        logger.error(f"Audio recording error: {e}")
        return render_template("error.html", error_code=500, error_message="Audio recording failed")


@app.route('/train_voice_model')
def train_voice_model():
    """Placeholder for voice model training"""
    return render_template('coming_soon.html', feature="Voice Emotion Model Training")

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('error.html', 
                         error_code=404, 
                         error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return render_template('error.html', 
                         error_code=500, 
                         error_message="Internal server error"), 500

if __name__ == "__main__":
    # Initialize models on startup
    initialize_models()
    
    # Run the app
    logger.info("Starting AI Vision Hub application...")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)