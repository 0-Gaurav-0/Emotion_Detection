import warnings
warnings.filterwarnings('ignore')

import scipy
import scipy.signal
try:
    _ = scipy.signal.hann
except AttributeError:
    from scipy.signal.windows import hann
    scipy.signal.hann = hann

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import sounddevice as sd
import wavio
import librosa
import numpy as np
import os
import threading
import pickle

# --- Emotion Mapping ---
EMOTION_MAP = {
    0: 'neutral',
    1: 'happy',
    2: 'sad',
    3: 'angry',
    4: 'fear',
    5: 'disgust',
    6: 'surprise'
}

class VoiceEmotionDetector:
    def __init__(self):
        self.DURATION = 4
        self.FS = 22050
        self.MIN_VALID_PITCH = 80
        self.MAX_VALID_PITCH = 400
        self.FEMALE_PITCH_MIN = 165
        self.FEMALE_PITCH_MAX = 350

        self.load_models()
        self.setup_gui()

    def load_models(self):
        try:
            with open("emotion_model.pkl", "rb") as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.scaler = data["scaler"]
                self.selector = data["selector"]
                self.label_encoder = data["label_encoder"]
                self.gender = data.get("gender", "unknown")
            print("✅ Model loaded successfully")
        except Exception as e:
            messagebox.showerror("Model Error", f"Could not load model: {e}")
            exit()

    def extract_features(self, file_path):
        audio, sr = librosa.load(file_path, sr=self.FS, duration=self.DURATION)
        if len(audio) == 0:
            raise ValueError("Empty audio")

        features = []

        # MFCC (40)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_stats = [np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
                      np.max(mfcc, axis=1), np.min(mfcc, axis=1)]
        features.extend(np.concatenate(mfcc_stats))

        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]

        from scipy.stats import skew, kurtosis
        for f in [spectral_centroids, spectral_rolloff, spectral_bandwidth, zero_crossing_rate]:
            features.extend([np.mean(f), np.std(f), np.max(f), np.min(f),
                             skew(f), kurtosis(f)])

        # Chroma
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features.extend([np.mean(chroma), np.std(chroma), np.var(chroma)])

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        features.append(tempo)

        # Energy
        rms_energy = librosa.feature.rms(y=audio)[0]
        features.extend([np.mean(rms_energy), np.std(rms_energy),
                         np.max(rms_energy), np.min(rms_energy)])

        features = np.array(features)

        # --- pad/trim to expected size (192, same as train.py) ---
        EXPECTED_DIM = 192
        if features.shape[0] != EXPECTED_DIM:
            if features.shape[0] < EXPECTED_DIM:
                features = np.pad(features, (0, EXPECTED_DIM - features.shape[0]))
            else:
                features = features[:EXPECTED_DIM]

        return features


    def analyze_gender_strict(self, file_path):
        audio, sr = librosa.load(file_path, sr=self.FS)
        pitches, mags = librosa.piptrack(y=audio, sr=sr, threshold=0.1)
        pitch_values = pitches[mags > np.percentile(mags, 90)]
        pitch_values = pitch_values[(pitch_values > self.MIN_VALID_PITCH) & (pitch_values < self.MAX_VALID_PITCH)]

        if len(pitch_values) < 20:
            return "unknown", 0.0

        median_pitch = np.median(pitch_values)
        prob_female = np.clip((median_pitch - self.FEMALE_PITCH_MIN) /
                              (self.FEMALE_PITCH_MAX - self.FEMALE_PITCH_MIN), 0, 1)

        if prob_female > 0.6:
            return "female", prob_female
        else:
            return "male", prob_female

    def predict_emotion(self, file_path):
        gender, prob = self.analyze_gender_strict(file_path)
        if gender != "female":
            return None, f"Non-female detected (prob_female={prob:.2f})"

        features = self.extract_features(file_path)
        features_scaled = self.scaler.transform([features])
        features_selected = self.selector.transform(features_scaled)

        emotion_pred = self.model.predict(features_selected)[0]
        emotion_name = EMOTION_MAP.get(emotion_pred, "Unknown")

        try:
            probs = self.model.predict_proba(features_selected)[0]
            confidence = np.max(probs)
            return emotion_name, confidence
        except:
            return emotion_name, None

    # --- GUI ---
    def process_audio_file(self, file_path):
        self.status_label.config(text="Analyzing...", fg="blue")
        self.progress_bar.start()
        self.window.update()

        emotion, conf = self.predict_emotion(file_path)
        self.progress_bar.stop()

        if emotion:
            self.result_frame.pack(pady=10)
            self.emotion_display.config(text=emotion)
            if isinstance(conf, float):
                self.confidence_bar['value'] = conf * 100
                self.status_label.config(text=f"🎭 {emotion} ({conf*100:.1f}%)", fg="green")
        else:
            self.status_label.config(text=f"❌ {conf}", fg="red")
            self.result_frame.pack_forget()

    def record_voice_thread(self):
        filename = "temp_record.wav"
        try:
            self.status_label.config(text="🎤 Recording 4s...", fg="orange")
            self.record_button.config(state="disabled")
            self.window.update()

            audio = sd.rec(int(self.DURATION * self.FS), samplerate=self.FS,
                           channels=1, dtype='float64')
            sd.wait()
            wavio.write(filename, audio, self.FS, sampwidth=2)

            self.process_audio_file(filename)
        finally:
            self.record_button.config(state="normal")
            if os.path.exists(filename):
                os.remove(filename)

    def record_voice(self):
        threading.Thread(target=self.record_voice_thread, daemon=True).start()

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3 *.flac *.m4a")])
        if file_path:
            threading.Thread(target=self.process_audio_file, args=(file_path,), daemon=True).start()

    def setup_gui(self):
        self.window = tk.Tk()
        self.window.title("🎭 Female Emotion Detector")
        self.window.geometry("500x450")
        self.window.resizable(False, False)

        style = ttk.Style()
        style.theme_use('clam')

        main_frame = tk.Frame(self.window, bg='#f0f0f0', padx=30, pady=20)
        main_frame.pack(expand=True, fill='both')

        tk.Label(main_frame, text="🎭 Emotion Detection", font=("Arial", 16, "bold"),
                 bg='#f0f0f0').pack(pady=(0,5))
        tk.Label(main_frame, text="Female Voices Only", font=("Arial", 10, "italic"),
                 bg='#f0f0f0', fg='#666').pack(pady=(0,20))

        tk.Button(main_frame, text="📁 Upload Audio File", command=self.upload_file,
                  font=("Arial", 12), width=20, height=2, bg='#4CAF50', fg='white').pack(pady=5)

        self.record_button = tk.Button(main_frame, text="🎤 Record Voice", command=self.record_voice,
                                       font=("Arial", 12), width=20, height=2, bg='#2196F3', fg='white')
        self.record_button.pack(pady=5)

        self.progress_bar = ttk.Progressbar(main_frame, mode='indeterminate', length=300)
        self.progress_bar.pack(pady=10)

        self.status_label = tk.Label(main_frame, text="Ready", font=("Arial", 11), bg='#f0f0f0')
        self.status_label.pack(pady=15)

        self.result_frame = tk.Frame(main_frame, bg='#e8f5e8', relief='ridge', bd=2)
        tk.Label(self.result_frame, text="Detected Emotion:", font=("Arial", 12, "bold"),
                 bg='#e8f5e8').pack(pady=5)
        self.emotion_display = tk.Label(self.result_frame, text="", font=("Arial", 18, "bold"),
                                        bg='#e8f5e8', fg='#2E7D32')
        self.emotion_display.pack(pady=5)

        tk.Label(main_frame, text="Confidence:", font=("Arial", 10), bg='#f0f0f0').pack()
        self.confidence_bar = ttk.Progressbar(main_frame, mode='determinate', length=300)
        self.confidence_bar.pack(pady=5)

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = VoiceEmotionDetector()
    app.run()
