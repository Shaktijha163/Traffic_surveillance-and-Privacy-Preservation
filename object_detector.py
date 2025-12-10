"""
YOLOv8 Object Detector & Tracker
Fixes: 
1. Uses Built-in ByteTrack (No external tracker needed)
2. Increases resolution (imgsz) to detect far away cars
3. Lowers confidence threshold to catch missed cars
"""

import cv2
import numpy as np
from typing import List, Dict
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics not installed! Run: pip install ultralytics")
    exit(1)

class CarPartsConfig:
    MODEL_PATH = "models/best.pt"
    # CRITICAL FIX 1: Lower confidence to catch cars further away/blurry
    CONF_THRESHOLD = 0.25 
    IOU_THRESHOLD = 0.45
    # CRITICAL FIX 2: Increase inference size. 
    # Default is 640. 1280 allows detecting small objects (cars far away).
    IMG_SIZE = 1280 
    
    # Your class names
    class_names = {
        0: 'Car boot', 1: 'Car hood', 2: 'Driver-s door', 
        3: 'Fender-FL', 4: 'Fender-FR', 5: 'Fender-RL', 6: 'Fender-RR',
        7: 'Front bumper', 8: 'Headlight-L', 9: 'Headlight-R',
        10: 'Passenger-door-FL', 11: 'Passenger-door-RL', 12: 'Passenger-door-RR',
        13: 'Rear bumper', 14: 'Rear light-L', 15: 'Rear light-R',
        16: 'Side bumper-L', 17: 'Side bumper-R', 18: 'Side mirror-L', 
        19: 'Side mirror-R', 20: 'sticker'
    }

config = CarPartsConfig()

class CarPartsDetector:
    def __init__(self, model_path: str = None, device: str = "cuda", conf_threshold: float = None):
        self.device = device
        self.model_path = model_path or config.MODEL_PATH
        self.conf_threshold = conf_threshold or config.CONF_THRESHOLD
        
        print(f"🚗 Loading YOLOv8: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.class_names = self.model.names
        print(f"✓ Model loaded. Conf: {self.conf_threshold}, ImgSz: {config.IMG_SIZE}")

    def detect_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Runs tracking directly in YOLO.
        Returns list of detections WITH track_ids.
        """
        # CRITICAL FIX 3: Enable 'persist=True' for tracking
        # CRITICAL FIX 4: tracker="bytetrack.yaml" is SOTA for cars
        results = self.model.track(
            frame,
            persist=True,  # Keeps track ID between frames
            tracker="bytetrack.yaml", # Uses internal ByteTrack
            conf=self.conf_threshold,
            iou=config.IOU_THRESHOLD,
            imgsz=config.IMG_SIZE, # High res processing
            verbose=False,
            device=self.device
        )
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = self.class_names.get(class_id, f'class_{class_id}')
                
                # Check if YOLO assigned a Track ID (it might be None for unstable detections)
                track_id = int(boxes.id[i].cpu().numpy()) if boxes.id is not None else -1
                
                detection = {
                    'bbox': bbox,
                    'confidence': conf,
                    'class_id': class_id,
                    'class_name': class_name,
                    'track_id': track_id # Now coming directly from model
                }
                detections.append(detection)
        
        return detections