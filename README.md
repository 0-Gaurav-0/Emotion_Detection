# Emotion Detection

A full-stack project that combines **YOLOv8 object detection** and **deep learning audio emotion recognition**, featuring both a **Flask web app** and **desktop GUIs (PyQt5 + Tkinter)**.  

---

## 🚀 Features  

- ✅ Real-time **animal detection** using YOLOv8 + OpenCV  
- ✅ **Voice emotion recognition** with TensorFlow & Librosa  
- ✅ Flask web app with modern UIs (`animal_detection.html`, `audio_emotion.html`)  
- ✅ Standalone GUI apps:  
  - **PyQt5 GUI** for YOLO detection  
  - **Tkinter GUI** for voice emotion recognition  
- ✅ Audio recording + feature extraction (`sounddevice`, `wavio`)  
- ✅ Model training pipelines with `scikit-learn`, `XGBoost`, and `TensorFlow`  

---

## 🛠️ Installation  

### 🔹 Option 1: Conda (Recommended for GPU/CUDA)  
```bash
conda env create -f environment.yml
conda activate animal-audio-env
```

### 🔹 Option 2: Pip (Lightweight / CPU only)  
```bash
python -m venv venv
# Activate
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

---

## ⚡ Usage  

### 🌐 Run Flask Web App  
```bash
python app.py
```
Open in browser: **http://127.0.0.1:5000**

### 🖥️ Run YOLO Animal Detection GUI  
```bash
python animal_detection.py
```

### 🎤 Run Voice Emotion Recognition GUI  
```bash
python audio_emotion.py
```

---

## 📂 Project Structure  

```
project-root/
│── app.py                  # Flask web app
│── animal_detection.py     # YOLOv8 + PyQt5 GUI
│── audio_emotion.py        # Tkinter voice emotion GUI
│── requirements.txt        # pip dependencies
│── environment.yml         # Conda env with CUDA + GPU support
│── setup.txt               # File/folder structure reference
│── static/                 # CSS, JS, images for web app
│── templates/              # HTML templates
│── models/                 # Pretrained & trained ML/DL models
│── data/                   # Sample datasets
│── README.md               # Project documentation
│── .gitignore
```

---

## ⚙️ System Requirements  

- **Python:** 3.10  
- **TensorFlow:** 2.10.0 (last with official GPU support on Windows)  
- **PyTorch:** 2.0.1 + torchvision 0.15.2 + torchaudio 2.0.1  
- **YOLOv8 (Ultralytics):** 8.0.227  
- **CUDA/cuDNN:** versions matching your GPU + TensorFlow/PyTorch builds  
- **Windows GPU users:** ensure `zlibwapi.dll` and NVIDIA drivers are installed  

---

## 📊 Tech Stack  

- **Deep Learning:** TensorFlow, PyTorch, YOLOv8  
- **ML Tools:** scikit-learn, XGBoost, pandas, numpy  
- **Audio:** librosa, sounddevice, wavio, scipy  
- **Vision:** OpenCV, matplotlib  
- **Frontend (Web):** Flask, HTML, CSS, JS  
- **GUI:** PyQt5, Tkinter  

