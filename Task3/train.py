import pandas as pd
import numpy as np
import pickle
import librosa
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import scipy.signal
import warnings
import os
import re

warnings.filterwarnings('ignore')

# --- PATCH scipy.signal.hann if missing ---
if not hasattr(scipy.signal, 'hann'):
    from scipy.signal import get_window
    scipy.signal.hann = lambda M, sym=True: get_window('hann', M, fftbins=sym)

# === SETTINGS ===
DATASET_PATH = "female_emotion_dataset.csv"  # CSV with audio file paths + labels
LABEL_COLUMN = "label"
FILE_COLUMN = "filepath"
MODEL_SAVE_PATH = "emotion_model.pkl"

EXPECTED_DIM = 192  # fixed input size


# === FEATURE EXTRACTION ===
def extract_features(file_path, DURATION=4, FS=22050):
    audio, sr = librosa.load(file_path, sr=FS, duration=DURATION)
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

    # --- pad/trim to expected size ---
    if features.shape[0] != EXPECTED_DIM:
        if features.shape[0] < EXPECTED_DIM:
            features = np.pad(features, (0, EXPECTED_DIM - features.shape[0]))
        else:
            features = features[:EXPECTED_DIM]

    return features


# === LOAD DATASET ===
df = pd.read_csv(DATASET_PATH)
if LABEL_COLUMN not in df.columns or FILE_COLUMN not in df.columns:
    raise ValueError("CSV must have columns: filepath,label")

# --- Filter female voices (even Actor IDs) ---
actor_ids = df[FILE_COLUMN].str.extract(r"Actor_(\d+)")
if actor_ids.isnull().all().values:
    print("⚠️ No Actor IDs detected in file paths. Skipping female filter.")
else:
    df = df[actor_ids[0].astype(float) % 2 == 0]
print(f"Filtered dataset: {len(df)} female samples")

X_paths = df[FILE_COLUMN].values
y = df[LABEL_COLUMN].values

# === EXTRACT FEATURES ===
print("Extracting features...")
X, valid_y = [], []
for fp, label in zip(X_paths, y):
    if not os.path.exists(fp):
        print(f"⚠️ File missing: {fp}")
        continue
    try:
        X.append(extract_features(fp))
        valid_y.append(label)
    except Exception as e:
        print(f"Skipping {fp}: {e}")

X = np.array(X)
y = np.array(valid_y)
print(f"Final dataset: {X.shape}, Labels: {len(y)}")

if len(X) == 0:
    raise ValueError("❌ No samples found. Check dataset paths/CSV formatting.")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection (keep all)
selector = SelectKBest(score_func=f_classif, k='all')
X_selected = selector.fit_transform(X_scaled, y_encoded)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# --- Define model ---
model = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500,
                      learning_rate_init=1e-3, random_state=42)

print("Training MLP...")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc*100:.2f}%")

# --- Save model ---
with open(MODEL_SAVE_PATH, 'wb') as f:
    pickle.dump({
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "selector": selector,
        "gender": "female-only"
    }, f)

print(f"💾 Model saved as '{MODEL_SAVE_PATH}'")
