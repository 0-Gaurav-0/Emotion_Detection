import sys
import cv2
import torch
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent
from PyQt5.QtCore import Qt, QTimer

# Define carnivorous animals
CARNIVORES = {"cat", "dog"}  # Expand if needed

# Load your trained model
model = YOLO("runs/detect/train/weights/best.pt")

class AnimalDetectorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Animal Detector")
        self.setGeometry(100, 100, 800, 600)
        self.setAcceptDrops(True)

        self.image_label = QLabel("Upload or drag an image/video here.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed gray;")
        self.image_label.setMinimumSize(400, 300)

        self.image_button = QPushButton("Upload Image")
        self.video_button = QPushButton("Upload Video")

        self.image_button.clicked.connect(self.load_image)
        self.video_button.clicked.connect(self.load_video)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.image_button)
        button_layout.addWidget(self.video_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.detect_and_display(path, is_video=False)
            elif path.lower().endswith(('.mp4', '.avi', '.mov')):
                self.detect_and_display(path, is_video=True)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.detect_and_display(path, is_video=False)

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Videos (*.mp4 *.avi *.mov)")
        if path:
            self.detect_and_display(path, is_video=True)

    def compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area != 0 else 0

    def detect_and_display(self, path, is_video=False):
        if is_video:
            cap = cv2.VideoCapture(path)
            seen_carnivores = []
            iou_threshold = 0.5

            while cap.isOpened():
                if not self.isVisible():
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(source=frame, save=False, imgsz=640, conf=0.3)[0]
                boxes = results.boxes
                names = model.names
                frame_height, frame_width = frame.shape[:2]

                for box in boxes:
                    cls_id = int(box.cls[0])
                    label = names[cls_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bbox = (x1, y1, x2, y2)

                    if label in CARNIVORES:
                        matched = False
                        for prev_box in seen_carnivores:
                            if self.compute_iou(bbox, prev_box) > iou_threshold:
                                matched = True
                                break
                        if not matched:
                            seen_carnivores.append(bbox)

                    # Draw bounding box and label
                    box_height = y2 - y1
                    font_scale = max(0.5, min(2.0, box_height / 100))
                    thickness = max(1, int(box_height / 100))
                    color = (0, 0, 255) if label in CARNIVORES else (0, 255, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                    )

                    if y1 - text_height - baseline - 4 > 0:
                        text_y = y1 - baseline - 4
                    elif y2 + text_height + 4 < frame_height:
                        text_y = y2 + text_height + 4
                    else:
                        text_y = y1 + text_height + 4

                    text_x = x1 + 2
                    bg_top_left = (x1, text_y - text_height - 4)
                    bg_bottom_right = (x1 + text_width + 4, text_y)

                    cv2.rectangle(frame, bg_top_left, bg_bottom_right, color, -1)
                    cv2.putText(frame, label, (text_x, text_y - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

                # Convert to QPixmap and show in QLabel
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                bytes_per_line = ch * w
                qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg).scaled(
                    self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.image_label.setPixmap(pixmap)

                QApplication.processEvents()
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            cap.release()

            if seen_carnivores:
                QMessageBox.information(self, "Carnivores Detected", f"{len(seen_carnivores)} unique carnivorous animals detected.")

        else:
            image = cv2.imread(path)
            image, carnivore_count = self.process_frame(image)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(pixmap)

            if carnivore_count > 0:
                QMessageBox.information(self, "Carnivores Detected", f"{carnivore_count} carnivorous animals detected.")

    def process_frame(self, frame):
        results = model.predict(source=frame, save=False, imgsz=640, conf=0.3)[0]
        boxes = results.boxes
        names = model.names
        carnivore_count = 0

        frame_height, frame_width = frame.shape[:2]

        for box in boxes:
            cls_id = int(box.cls[0])
            label = names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_height = y2 - y1

            # Adjust font size and thickness based on box height
            font_scale = max(0.5, min(2.0, box_height / 100))
            thickness = max(1, int(box_height / 100))

            color = (0, 255, 0)
            if label in CARNIVORES:
                carnivore_count += 1
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Calculate size of text
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # Choose label position: above the box if there's space, else below or inside
            if y1 - text_height - baseline - 4 > 0:
                text_y = y1 - baseline - 4
            elif y2 + text_height + 4 < frame_height:
                text_y = y2 + text_height + 4
            else:
                # fallback: put it inside the box
                text_y = y1 + text_height + 4

            text_x = x1 + 2
            bg_top_left = (x1, text_y - text_height - 4)
            bg_bottom_right = (x1 + text_width + 4, text_y)

            # Draw background for text
            cv2.rectangle(frame, bg_top_left, bg_bottom_right, color, -1)
            # Put the label
            cv2.putText(frame, label, (text_x, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        return frame, carnivore_count


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AnimalDetectorApp()
    window.show()
    sys.exit(app.exec_())
