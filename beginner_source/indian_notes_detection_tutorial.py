"""
Title: Indian Currency Note Detection using YOLOv5
Author: Your Name
Description:
    This tutorial demonstrates how to detect Indian currency notes using YOLOv5 and Roboflow.
    It covers dataset download, training, inference, and retraining for improved accuracy.

    Requirements:
        - Roboflow account with access to the dataset.
        - Google Colab or local environment with GPU.
        - Replace 'YOUR_API_KEY_HERE' with your Roboflow API key.
"""

import os

# Step 1: Clone YOLOv5 repository and install requirements
print("Cloning YOLOv5 and installing dependencies...")
os.system("git clone https://github.com/ultralytics/yolov5")
os.chdir("yolov5")
os.system("pip install -r requirements.txt")
os.system("pip install roboflow")

# Step 2: Download Dataset from Roboflow
from roboflow import Roboflow

print("Downloading dataset from Roboflow...")
rf = Roboflow(api_key="YOUR_API_KEY_HERE")  # Replace with your Roboflow API key
project = rf.workspace("omkar-patkar-fes59").project("indian-currency-notes")
version = project.version(4)
dataset = version.download("yolov5")

# Step 3: Explore dataset structure
print("\n📂 Dataset location:", dataset.location)
for root, dirs, files in os.walk(dataset.location):
    print(f"📁 {root}")
    for file in files[:5]:
        print("   📄", file)
    break

# Step 4: Train the model (Initial training)
print("\n🧠 Training YOLOv5 model...")
os.system("""
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 30 \
  --data indian-currency/notes-4/data.yaml \
  --weights yolov5s.pt \
  --project currency-project \
  --name yolo_currency \
  --cache
""")

# Step 5: Run inference on test images
print("\n🔍 Running detection on test images...")
os.system("""
python detect.py \
  --weights currency-project/yolo_currency/weights/best.pt \
  --img 640 \
  --conf 0.25 \
  --source indian-currency/notes-4/test/images
""")

# Step 6: Retrain with more epochs (optional)
print("\n📈 Retraining with 50 epochs for improved accuracy...")
os.system("""
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 50 \
  --data indian-currency/notes-4/data.yaml \
  --weights yolov5s.pt \
  --project currency-project/yolo_currency_v2 \
  --name improved_run
""")
