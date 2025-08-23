import os
import csv

# Folder path where WAV files are stored
DATASET_DIR = "path_to_your_dataset_folder"  # replace with actual path

# Emotion mapping
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Output CSV
output_csv = "female_emotion_dataset.csv"

with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "emotion", "gender", "path"])

    for root, dirs, files in os.walk(DATASET_DIR):
        for filename in files:
            if filename.endswith(".wav"):
                parts = filename.split("-")
                emotion_code = parts[2]
                actor_id = int(parts[-1].split(".")[0])

                gender = "female" if actor_id % 2 == 0 else "male"

                if gender == "female":
                    emotion = emotion_map.get(emotion_code, "unknown")
                    full_path = os.path.join(root, filename)
                    writer.writerow([filename, emotion, gender, full_path])

print(f"Dataset CSV saved to: {output_csv}")
