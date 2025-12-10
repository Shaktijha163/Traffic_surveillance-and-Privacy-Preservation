"""
Simplified Object Tracker
Visualizes tracks provided by YOLOv8 ByteTrack
"""

import cv2
import numpy as np
from typing import List, Dict
from collections import defaultdict, deque

class CarPartsTracker:
    def __init__(self, device: str = "cuda"):
        # We store history for visualization (trails)
        # deque(maxlen=30) automatically removes old points
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        print("✓ Simplified Tracker Ready (Visualization Only)")
    
    def update(self, frame: np.ndarray, detections: List[Dict], frame_idx: int) -> List[Dict]:
        """
        Since YOLO now does the tracking, this function primarily 
        updates the history buffer for visualization.
        """
        active_tracks = []
        
        for det in detections:
            track_id = det.get('track_id', -1)
            
            # If YOLO didn't assign an ID yet, skip history update
            if track_id == -1:
                continue
                
            # Update history for trails
            bbox = det['bbox']
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            self.track_history[track_id].append((center_x, center_y))
            
            # Format output to match what the pipeline expects
            track_info = {
                'track_id': track_id,
                'bbox': bbox,
                'confidence': det['confidence'],
                'class_name': det['class_name'],
                'class_id': det['class_id'],
                'age': 0 # Not needed for ByteTrack
            }
            active_tracks.append(track_info)
            
        return active_tracks

    def visualize_tracks(self, frame: np.ndarray, tracks: List[Dict], trail_length: int = 30) -> np.ndarray:
        annotated = frame.copy()
        
        # Consistent colors based on ID
        def get_color(id):
            np.random.seed(id)
            return tuple(np.random.randint(0, 255, 3).tolist())

        for track in tracks:
            track_id = track['track_id']
            x1, y1, x2, y2 = track['bbox']
            
            color = get_color(track_id)
            
            # Draw Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw Label
            label = f"ID:{track_id} {track['class_name']}"
            (w, h), _ = cv2.getTextSize(label, 0, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 0, 0.6, (255,255,255), 2)
            
            # Draw Trail
            if track_id in self.track_history:
                points = list(self.track_history[track_id])
                for i in range(1, len(points)):
                    if points[i - 1] is None or points[i] is None: continue
                    thickness = int(np.sqrt(float(i + 1)) * 1.5)
                    cv2.line(annotated, points[i - 1], points[i], color, thickness)
                    
        return annotated

    def clear_cache(self):
        self.track_history.clear()