import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import sounddevice as sd
import wavio
import librosa
import numpy as np
import joblib
import os
import threading
from scipy.stats import skew, kurtosis
import warnings

# Suppress librosa warnings
warnings.filterwarnings('ignore')

class VoiceEmotionDetector:
    def __init__(self):
        # Audio parameters
        self.DURATION = 4
        self.FS = 22050
        self.MIN_VALID_PITCH = 80
        self.MAX_VALID_PITCH = 400
        self.FEMALE_PITCH_MIN = 165
        self.FEMALE_PITCH_MAX = 350
        
        # Emotion mapping
        self.emotion_labels = {
            0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
            4: "Neutral", 5: "Sad", 6: "Surprise"
        }
        
        # Load models
        self.load_models()
        
        # Setup GUI
        self.setup_gui()
    
    def load_models(self):
        """Load the emotion detection model and scaler"""
        try:
            self.model = joblib.load("emotion_model.pkl")
            self.scaler = joblib.load("scaler.pkl")
            print("Models loaded successfully")
        except FileNotFoundError as e:
            messagebox.showerror(
                "Model Error", 
                f"Model files not found: {e}\n\nPlease ensure 'emotion_model.pkl' and 'scaler.pkl' are in the same directory."
            )
            exit()
        except Exception as e:
            messagebox.showerror("Error", f"Could not load model/scaler: {e}")
            exit()
    
    def extract_advanced_features(self, file_path):
        """Extract comprehensive audio features for emotion detection"""
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=self.FS, duration=self.DURATION)
            
            # Ensure audio is not empty
            if len(audio) == 0:
                raise ValueError("Audio file is empty or corrupted")
            
            features = []
            
            # 1. MFCC features (40 coefficients)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mfcc_stats = [
                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1),
                np.max(mfcc, axis=1),
                np.min(mfcc, axis=1)
            ]
            features.extend(np.concatenate(mfcc_stats))
            
            # 2. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
            
            # Statistical measures for spectral features
            for feature in [spectral_centroids, spectral_rolloff, spectral_bandwidth, zero_crossing_rate]:
                features.extend([
                    np.mean(feature), np.std(feature), 
                    np.max(feature), np.min(feature),
                    skew(feature), kurtosis(feature)
                ])
            
            # 3. Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features.extend([
                np.mean(chroma, axis=1).mean(),
                np.std(chroma, axis=1).mean(),
                np.var(chroma, axis=1).mean()
            ])
            
            # 4. Tempo and rhythm
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            features.append(tempo)
            
            # 5. Energy and loudness
            rms_energy = librosa.feature.rms(y=audio)[0]
            features.extend([
                np.mean(rms_energy), np.std(rms_energy),
                np.max(rms_energy), np.min(rms_energy)
            ])
            
            return np.array(features)
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            raise
    
    def analyze_gender_advanced(self, file_path):
        """Advanced gender detection using multiple acoustic features"""
        try:
            audio, sr = librosa.load(file_path, sr=self.FS)
            
            # 1. Fundamental frequency analysis
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, threshold=0.1)
            pitch_values = pitches[magnitudes > np.percentile(magnitudes, 85)]
            pitch_values = pitch_values[(pitch_values > self.MIN_VALID_PITCH) & 
                                      (pitch_values < self.MAX_VALID_PITCH)]
            
            if len(pitch_values) < 10:
                return "unknown", "Insufficient pitch data for gender detection"
            
            # Statistical analysis of pitch
            mean_pitch = np.mean(pitch_values)
            median_pitch = np.median(pitch_values)
            pitch_std = np.std(pitch_values)
            
            # 2. Formant analysis (approximate)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            mean_spectral_centroid = np.mean(spectral_centroids)
            
            # 3. Spectral rolloff (voice brightness)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)[0]
            mean_rolloff = np.mean(spectral_rolloff)
            
            # Gender classification logic
            female_score = 0
            
            # Pitch-based scoring
            if self.FEMALE_PITCH_MIN <= median_pitch <= self.FEMALE_PITCH_MAX:
                female_score += 3
            elif median_pitch > self.FEMALE_PITCH_MAX * 0.8:
                female_score += 1
            
            # Spectral centroid scoring (females typically have higher)
            if mean_spectral_centroid > 2000:
                female_score += 2
            elif mean_spectral_centroid > 1500:
                female_score += 1
            
            # Spectral rolloff scoring
            if mean_rolloff > 4000:
                female_score += 1
            
            # Pitch variability (females often have more variable pitch)
            if pitch_std > 20:
                female_score += 1
            
            # Decision making
            confidence = min(female_score / 7.0, 1.0)
            
            if female_score >= 4:
                return "female", f"Confidence: {confidence:.2f}"
            else:
                return "male", f"Detected non-female voice (Score: {female_score}/7)"
                
        except Exception as e:
            print(f"Gender analysis error: {e}")
            return "unknown", f"Error in gender detection: {str(e)}"
    
    def predict_emotion(self, file_path):
        """Predict emotion from audio file"""
        try:
            # First check gender
            gender, gender_info = self.analyze_gender_advanced(file_path)
            
            if gender != "female":
                return None, gender_info
            
            # Extract features and predict emotion
            features = self.extract_advanced_features(file_path)
            
            # Handle feature scaling
            if hasattr(self.scaler, 'transform'):
                features_scaled = self.scaler.transform([features])
            else:
                features_scaled = [features]
            
            # Predict emotion
            emotion_pred = self.model.predict(features_scaled)[0]
            
            # Get emotion label
            if isinstance(emotion_pred, (int, np.integer)):
                emotion_name = self.emotion_labels.get(emotion_pred, "Unknown")
            else:
                emotion_name = str(emotion_pred).capitalize()
            
            # Get prediction confidence if available
            try:
                probabilities = self.model.predict_proba(features_scaled)[0]
                confidence = np.max(probabilities)
                return emotion_name, f"Detected with {confidence:.1%} confidence"
            except:
                return emotion_name, "Emotion detected"
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, f"Error in emotion prediction: {str(e)}"
    
    def process_audio_file(self, file_path):
        """Process audio file and update GUI"""
        try:
            self.status_label.config(text="Analyzing audio...", fg="blue")
            self.progress_bar.start()
            self.window.update()
            
            emotion, message = self.predict_emotion(file_path)
            
            self.progress_bar.stop()
            
            if emotion:
                self.status_label.config(
                    text=f"🎭 Emotion: {emotion}\n{message}", 
                    fg="green"
                )
                self.result_frame.pack(pady=10)
                self.emotion_display.config(text=emotion)
            else:
                self.status_label.config(
                    text=f"❌ {message}\n\nThis system only works with female voices.", 
                    fg="red"
                )
                self.result_frame.pack_forget()
                
        except Exception as e:
            self.progress_bar.stop()
            self.status_label.config(
                text="❌ Error processing audio file.", 
                fg="red"
            )
            print(f"Processing error: {e}")
    
    def record_voice_thread(self):
        """Record voice in separate thread to prevent GUI freezing"""
        filename = "temp_record.wav"
        try:
            self.status_label.config(text="🎤 Recording... Speak clearly for 4 seconds", fg="orange")
            self.record_button.config(state="disabled")
            self.window.update()
            
            # Record audio
            audio = sd.rec(int(self.DURATION * self.FS), samplerate=self.FS, channels=1, dtype='float64')
            sd.wait()
            
            # Save to file
            wavio.write(filename, audio, self.FS, sampwidth=2)
            
            # Process the recording
            self.process_audio_file(filename)
            
        except Exception as e:
            self.status_label.config(text="❌ Recording failed. Check your microphone.", fg="red")
            messagebox.showerror("Recording Error", f"Could not record audio: {str(e)}")
        finally:
            self.record_button.config(state="normal")
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                except:
                    pass
    
    def record_voice(self):
        """Start recording in a separate thread"""
        thread = threading.Thread(target=self.record_voice_thread, daemon=True)
        thread.start()
    
    def upload_file(self):
        """Handle file upload"""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.m4a"),
                ("WAV files", "*.wav"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            # Process in separate thread to prevent GUI freezing
            thread = threading.Thread(
                target=self.process_audio_file, 
                args=(file_path,), 
                daemon=True
            )
            thread.start()
    
    def setup_gui(self):
        """Setup the GUI interface"""
        self.window = tk.Tk()
        self.window.title("🎭 Female Voice Emotion Detector")
        self.window.geometry("500x400")
        self.window.resizable(False, False)
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main frame
        main_frame = tk.Frame(self.window, bg='#f0f0f0', padx=30, pady=20)
        main_frame.pack(expand=True, fill='both')
        
        # Title
        title_label = tk.Label(
            main_frame, 
            text="🎭 Emotion Detection System",
            font=("Arial", 16, "bold"),
            bg='#f0f0f0',
            fg='#333'
        )
        title_label.pack(pady=(0, 5))
        
        subtitle_label = tk.Label(
            main_frame,
            text="Female Voices Only",
            font=("Arial", 10, "italic"),
            bg='#f0f0f0',
            fg='#666'
        )
        subtitle_label.pack(pady=(0, 20))
        
        # Buttons frame
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(pady=10)
        
        # Upload button
        upload_btn = tk.Button(
            button_frame,
            text="📁 Upload Audio File",
            font=("Arial", 12),
            width=20,
            height=2,
            command=self.upload_file,
            bg='#4CAF50',
            fg='white',
            relief='flat',
            cursor='hand2'
        )
        upload_btn.pack(pady=5)
        
        # Record button
        self.record_button = tk.Button(
            button_frame,
            text="🎤 Record Voice",
            font=("Arial", 12),
            width=20,
            height=2,
            command=self.record_voice,
            bg='#2196F3',
            fg='white',
            relief='flat',
            cursor='hand2'
        )
        self.record_button.pack(pady=5)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(
            main_frame, 
            mode='indeterminate',
            length=300
        )
        self.progress_bar.pack(pady=10)
        
        # Status label
        self.status_label = tk.Label(
            main_frame,
            text="Select an option above to begin emotion detection",
            font=("Arial", 11),
            wraplength=400,
            justify="center",
            bg='#f0f0f0'
        )
        self.status_label.pack(pady=15)
        
        # Result frame (initially hidden)
        self.result_frame = tk.Frame(main_frame, bg='#e8f5e8', relief='ridge', bd=2)
        
        result_title = tk.Label(
            self.result_frame,
            text="Detected Emotion:",
            font=("Arial", 12, "bold"),
            bg='#e8f5e8'
        )
        result_title.pack(pady=5)
        
        self.emotion_display = tk.Label(
            self.result_frame,
            text="",
            font=("Arial", 18, "bold"),
            bg='#e8f5e8',
            fg='#2E7D32'
        )
        self.emotion_display.pack(pady=5)
        
        # Instructions
        instructions = tk.Label(
            main_frame,
            text="💡 Tips: Speak clearly, avoid background noise, ensure good audio quality",
            font=("Arial", 9),
            wraplength=400,
            justify="center",
            bg='#f0f0f0',
            fg='#888'
        )
        instructions.pack(side='bottom', pady=10)
    
    def run(self):
        """Start the application"""
        self.window.mainloop()

# Run the application
if __name__ == "__main__":
    try:
        app = VoiceEmotionDetector()
        app.run()
    except KeyboardInterrupt:
        print("\nApplication closed by user")
    except Exception as e:
        print(f"Application error: {e}")
        messagebox.showerror("Fatal Error", f"Application failed to start: {e}")