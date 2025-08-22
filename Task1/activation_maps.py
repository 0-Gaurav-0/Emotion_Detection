import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow_addons as tfa
from tensorflow.keras.models import load_model
from tensorflow.keras import Model


# Load your pretrained CNN model
model = load_model(
    "emotion_model.h5",
    custom_objects={"AdamW": tfa.optimizers.AdamW}
)
print("Model loaded successfully!")



# ------------------------
# Helper: Get all Conv2D layers
# ------------------------
conv_layers = [layer.name for layer in model.layers if 'conv' in layer.name]
print(f"Convolutional layers: {conv_layers}")

# ------------------------
# Load and preprocess a test image
# ------------------------
# Use any grayscale face image of size 48x48
img_path = "Hello.png"  # Replace with your image file
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"Image not found at path: {img_path}")

# Resize to 48x48 (input size expected by model)
img = cv2.resize(img, (48, 48))
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)      # (1, 48, 48)
img = np.expand_dims(img, axis=-1)     # (1, 48, 48, 1)

# ------------------------
# Visualize activation maps for each conv layer
# ------------------------
for layer_name in conv_layers:
    intermediate_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    activations = intermediate_model.predict(img)

    num_filters = activations.shape[-1]
    size = activations.shape[1]

    print(f"\nVisualizing activations from layer: {layer_name}")
    print(f"Shape: {activations.shape} | Number of filters: {num_filters}")

    cols = 8
    rows = num_filters // cols + int(num_filters % cols != 0)

    plt.figure(figsize=(15, 2 * rows))
    for i in range(num_filters):
        ax = plt.subplot(rows, cols, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(activations[0, :, :, i], cmap='viridis')
        ax.set_title(f'F{i}', fontsize=8)
    plt.suptitle(f'Activation Maps - Layer: {layer_name}', fontsize=16)
    plt.tight_layout()
    plt.show()
