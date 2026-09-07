#!/usr/bin/env python3
"""
train_yolo.py
One-Click YOLOv11n Object Detection (Bounding Box) Training Script.

Trains Ultralytics YOLOv11n on the generated dataset with auto-detected
hardware acceleration (Apple Silicon MPS, NVIDIA CUDA, or CPU).
"""

import sys
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

def detect_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "0"
    return "cpu"

def train(data_yaml="dataset/data.yaml", epochs=50, batch=16, imgsz=640, device=None):
    yaml_path = Path(data_yaml).resolve()
    if not yaml_path.exists():
        print(f"Error: Dataset config '{yaml_path}' not found!")
        print("Please run 'python generate_dataset.py' first.")
        sys.exit(1)
        
    chosen_device = device if device is not None else detect_device()
    print("=" * 60)
    print("🚀 YOLOv11n Bounding Box Detection Training")
    print(f"  • Dataset:  {yaml_path}")
    print(f"  • Model:    yolo11n.pt (Pure Detection, NO segmentation)")
    print(f"  • Device:   {chosen_device}")
    print(f"  • Epochs:   {epochs}")
    print(f"  • Batch:    {batch}")
    print(f"  • Img Size: {imgsz}")
    print("=" * 60)
    
    # Load YOLOv11n pretrained detection model
    model = YOLO("yolo11n.pt")
    
    # Start training
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=chosen_device,
        project="runs/detect",
        name="train_yolov11n",
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print(" Training Complete!")
    best_weights = Path("runs/detect/train_yolov11n/weights/best.pt")
    if best_weights.exists():
        print(f"  • Best Model Saved: {best_weights.resolve()}")
    print("=" * 60)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv11n Bounding Box Detector")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (default: 640)")
    parser.add_argument("--device", type=str, default=None, help="Device to train on ('mps', '0', 'cpu')")
    args = parser.parse_args()
    
    train(
        data_yaml=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device
    )
