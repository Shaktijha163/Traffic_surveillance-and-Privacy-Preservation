"""
YOLOv8 Object Detector for Car and Car Parts Detection
Uses custom trained weights (best.pt)
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ ultralytics not installed!")
    print("Run: pip install ultralytics")
    exit(1)


class CarPartsConfig:
    """Configuration for car parts detection"""
    MODEL_PATH = "models/best.pt"  # Path to your trained model
    CONF_THRESHOLD = 0.5  # Increased from 0.25 for better quality
    IOU_THRESHOLD = 0.45
    
    # Your model's class names (update based on your training)
    # These should match what you trained on
    class_names = {
        0: 'Car boot',
        1: 'Car hood',
        2: 'Driver-s door - -F-R-',
        3: 'Fender - -F-L-',
        4: 'Fender - -F-R-',
        5: 'Fender - -R-L-',
        6: 'Fender - -R-R-',
        7: 'Front bumper',
        8: 'Headlight - -L-',
        9: 'Headlight - -R-',
        10: 'Passenger-s door - -F-L-',
        11: 'Passenger-s door - -R-L-',
        12: 'Passenger-s door - -R-R-',
        13: 'Rear bumper',
        14: 'Rear light - -L-',
        15: 'Rear light - -R-',
        16: 'Side bumper - -L-',
        17: 'Side bumper - -R-',
        18: 'Side mirror - -L-',
        19: 'Side mirror - -R-',
        20: 'sticker'
    }



config = CarPartsConfig()


class CarPartsDetector:
    """YOLOv8 detector for cars and car parts using custom trained model"""
    
    def __init__(self, model_path: str = None, device: str = "cuda", conf_threshold: float = None):
        """
        Initialize YOLOv8 detector with custom trained weights
        
        Args:
            model_path: Path to best.pt model
            device: 'cuda' or 'cpu'
            conf_threshold: Confidence threshold (default: 0.5)
        """
        self.device = device
        self.model_path = model_path or config.MODEL_PATH
        self.conf_threshold = conf_threshold or config.CONF_THRESHOLD
        self.iou_threshold = config.IOU_THRESHOLD
        
        print(f"🚗 Loading Custom YOLOv8 Car Parts Detector: {self.model_path}")
        self.model = self._load_model()
        
        # Get class names from model
        self.class_names = self.model.names
        print(f"✓ Custom YOLOv8 loaded on {device}")
        print(f"  Detected classes: {list(self.class_names.values())}")
        print(f"  Confidence threshold: {self.conf_threshold}\n")
    
    def _load_model(self) -> YOLO:
        """Load custom trained YOLOv8 model"""
        try:
            model_path = Path(self.model_path)
            
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found: {model_path}\n"
                    f"Please download best.pt from Google Drive and place it in {model_path.parent}/"
                )
            
            # Load custom model
            model = YOLO(str(model_path))
            
            # Move to device
            model.to(self.device)
            
            return model
            
        except Exception as e:
            print(f"❌ Failed to load YOLOv8: {e}")
            raise
    
    def detect_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect car parts in a single frame
        
        Args:
            frame: BGR image (numpy array)
        
        Returns:
            List of detections:
            [
                {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float,
                    'class_id': int,
                    'class_name': str
                },
                ...
            ]
        """
        # Run inference
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device
        )
        
        # Parse results
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                # Get box coordinates (xyxy format)
                bbox = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                
                # Get confidence
                conf = float(boxes.conf[i].cpu().numpy())
                
                # Get class
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = self.class_names.get(class_id, f'class_{class_id}')
                
                detection = {
                    'bbox': bbox,
                    'confidence': conf,
                    'class_id': class_id,
                    'class_name': class_name
                }
                
                detections.append(detection)
        
        return detections
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """
        Detect car parts in multiple frames (batch processing)
        
        Args:
            frames: List of BGR images
        
        Returns:
            List of detection lists (one per frame)
        """
        # Run batch inference
        results = self.model.predict(
            frames,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device
        )
        
        # Parse results for each frame
        all_detections = []
        
        for result in results:
            frame_detections = []
            
            if result.boxes is not None:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                    conf = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = self.class_names.get(class_id, f'class_{class_id}')
                    
                    detection = {
                        'bbox': bbox,
                        'confidence': conf,
                        'class_id': class_id,
                        'class_name': class_name
                    }
                    
                    frame_detections.append(detection)
            
            all_detections.append(frame_detections)
        
        return all_detections
    
    def visualize_detections(self, 
                           frame: np.ndarray, 
                           detections: List[Dict],
                           show_conf: bool = True) -> np.ndarray:
        """
        Draw detection boxes on frame
        
        Args:
            frame: BGR image
            detections: List of detections from detect_frame()
            show_conf: Show confidence scores
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Color map for different classes
        np.random.seed(42)
        num_classes = len(self.class_names)
        colors = {}
        for class_id in range(num_classes):
            colors[class_id] = tuple(np.random.randint(0, 255, 3).tolist())
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_id = det['class_id']
            class_name = det['class_name']
            
            # Get color
            color = colors.get(class_id, (255, 255, 255))
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}"
            if show_conf:
                label += f" {conf:.2f}"
            
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - text_h - 10),
                (x1 + text_w, y1),
                color,
                -1
            )
            
            # Text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        
        return annotated
    
    def get_detection_stats(self, detections: List[Dict]) -> Dict:
        """Get statistics about detections"""
        if not detections:
            return {
                'total': 0,
                'by_class': {},
                'avg_confidence': 0.0
            }
        
        # Count by class
        by_class = {}
        for det in detections:
            class_name = det['class_name']
            by_class[class_name] = by_class.get(class_name, 0) + 1
        
        # Average confidence
        avg_conf = sum(d['confidence'] for d in detections) / len(detections)
        
        return {
            'total': len(detections),
            'by_class': by_class,
            'avg_confidence': avg_conf
        }


# ============================================================================
# TESTING
# ============================================================================

def test_detector():
    """Test detector on sample image/video"""
    print("\n" + "="*60)
    print("Testing Custom YOLOv8 Car Parts Detector")
    print("="*60 + "\n")
    
    # Initialize detector
    detector = CarPartsDetector(device="cuda", conf_threshold=0.5)
    
    # Test on sample image
    print("Testing on sample image...")
    
    test_image_path = "test_images"
    test_dir = Path(test_image_path)
    
    if test_dir.exists():
        images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")) + list(test_dir.glob("*.webp"))
        
        if images:
            img_path = str(images[0])
            print(f"Loading: {img_path}")
            
            frame = cv2.imread(img_path)
            
            if frame is not None:
                # Detect
                detections = detector.detect_frame(frame)
                
                # Stats
                stats = detector.get_detection_stats(detections)
                
                print(f"\n✓ Detection Results:")
                print(f"  Total detections: {stats['total']}")
                print(f"  By class: {stats['by_class']}")
                print(f"  Avg confidence: {stats['avg_confidence']:.3f}")
                
                # Visualize
                annotated = detector.visualize_detections(frame, detections)
                
                # Save
                output_path = "results/test_detection.jpg"
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(output_path, annotated)
                print(f"\n✓ Saved annotated image: {output_path}")
                
                return True
    
    print("⚠ No test images found. Create 'test_images/' folder with images.")
    return False


if __name__ == "__main__":
    test_detector()