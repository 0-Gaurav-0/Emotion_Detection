import subprocess

command = [
    "yolo",
    "task=detect",
    "mode=train",
    "model=yolov8n.pt",
    "data=dataset/data.yaml",
    "epochs=10",
    "imgsz=640"
]

result = subprocess.run(command, capture_output=True, text=True)

print("STDOUT:\n", result.stdout)
print("STDERR:\n", result.stderr)
