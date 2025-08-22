import os
import librosa
import numpy as np
import pandas as pd

# Fix for scipy hann deprecation in older code
import scipy.signal as signal
try:
    from scipy.signal.windows import hann
    signal.hann = hann
except ImportError:
    pass

# Path to dataset
DATA_PATH = "RAVDESS_Audio"  # Change to your dataset path

# Map emotion codes from RAVDESS dataset
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Function to extract MFCC features
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=22050)  # Load audio
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)  # Extract 40 MFCCs
    return np.mean(mfccs.T, axis=0)  # Take mean over time

# Function to check if the voice is female (RAVDESS: even-numbered actors = female)
def is_female(file_name):
    actor_id = int(file_name.split("-")[-1].split(".")[0])
    return actor_id % 2 == 0

# Storage for features
features = []
files_processed = 0

print(f"🎙 Extracting features from female voices in '{DATA_PATH}'...")

# Loop over all files in dataset
for root, _, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav") and is_female(file):
            try:
                emotion_code = file.split("-")[2]  # Extract emotion code
                emotion_label = EMOTION_MAP.get(emotion_code)
                if emotion_label is None:
                    continue  # Skip if emotion code not found

                file_path = os.path.join(root, file)
                mfcc = extract_features(file_path)

                # Append features + label
                features.append([*mfcc, emotion_label])
                files_processed += 1

                if files_processed % 50 == 0: 
                    print(f"✅ Processed {files_processed} files...")
            except Exception as e:
                print(f"❌ Error processing {file}: {e}")

# Create dataframe and save
columns = [f'mfcc_{i}' for i in range(40)] + ['label']
df = pd.DataFrame(features, columns=columns)
df.to_csv("female_emotion_dataset.csv", index=False)

print(f"\n🎯 Done! Extracted features from {files_processed} female audio files.")
print("📁 Saved dataset as 'female_emotion_dataset.csv'")
