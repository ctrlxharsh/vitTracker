#!/usr/bin/env python3
"""
test_detector.py
Real-Time Inference / Verification for YOLOv11n Trained Weights.

Runs detection on an image, directory of images, video, or live webcam.
"""

import sys
import argparse
from pathlib import Path
import cv2
from camera import open_video_capture
from ultralytics import YOLO

def run_inference(weights="runs/detect/train_yolov11n/weights/best.pt", source="dataset/previews", conf=0.4):
    weights_path = Path(weights)
    if not weights_path.exists():
        print(f"Weights file '{weights_path}' not found. Defaulting to 'yolo11n.pt'...")
        weights_path = "yolo11n.pt"
        
    print(f"Loading detector weights from {weights_path}...")
    model = YOLO(str(weights_path))
    
    # Check if webcam
    if str(source).isdigit() or str(source).lower() in ("picam", "rpicam", "csi"):
        cap = open_video_capture(source)
        print("Starting camera live detection (Press 'q' to quit)...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame, conf=conf, verbose=False)
            annotated_frame = results[0].plot()
            cv2.imshow("YOLOv11n Blue Object Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        results = model.predict(source=source, conf=conf, save=True, project="runs/detect", name="test_results")
        print(f"Results saved to runs/detect/test_results")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test YOLOv11n Blue Object Detector")
    parser.add_argument("--weights", type=str, default="runs/detect/train_yolov11n/weights/best.pt", help="Path to best.pt")
    parser.add_argument("--source", type=str, default="dataset/previews", help="Image, directory, or webcam index (0)")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    args = parser.parse_args()
    
    run_inference(weights=args.weights, source=args.source, conf=args.conf)
