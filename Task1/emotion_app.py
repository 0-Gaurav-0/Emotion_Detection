# emotion_app.py

import os
import sys
import cv2
import numpy as np
from train_fer_model import train_model
from predict_emotion_realtime import run_realtime_emotion_detection
from activation_maps import visualize_activation_maps

def predict_from_image(model_path="emotion_model.h5"):
    from keras.models import load_model
    from keras.preprocessing.image import img_to_array
    import matplotlib.pyplot as plt

    model = load_model(model_path)

    # Load image
    img_path = input("Enter path to image (48x48 grayscale face): ")
    if not os.path.exists(img_path):
        print("File not found.")
        return

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Invalid image.")
        return
    img = cv2.resize(img, (48, 48))
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 48, 48, 1)

    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    preds = model.predict(img)
    emotion = emotions[np.argmax(preds)]
    print(f"Predicted Emotion: {emotion}")

    plt.imshow(cv2.imread(img_path), cmap="gray")
    plt.title(f"Predicted: {emotion}")
    plt.show()

def main_menu():
    while True:
        print("\n==== EMOTION DETECTION APP ====")
        print("1. Train Emotion Detection Model (Task 2)")
        print("2. Predict Emotion from Image")
        print("3. Predict Emotion in Real-time (Webcam)")
        print("4. Visualize CNN Activation Maps (Task 3)")
        print("5. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            train_model()
        elif choice == '2':
            predict_from_image()
        elif choice == '3':
            run_realtime_emotion_detection()
        elif choice == '4':
            visualize_activation_maps()
        elif choice == '5':
            print("Goodbye!")
            sys.exit()
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()
