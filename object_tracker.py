"""
DeepSORT Multi-Object Tracker for Car Parts Tracking
Tracks detected car parts across video frames
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from collections import defaultdict

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    print("❌ deep-sort-realtime not installed!")
    print("Run: pip install deep-sort-realtime")
    exit(1)


class DeepSortConfig:
    """Configuration for DeepSORT tracker"""
    MAX_AGE = 30  # Keep tracks for 30 frames without detection
    MIN_HITS = 3  # Require 3 hits before confirming track
    IOU_THRESHOLD = 0.3
    N_INIT = 3
    MAX_IOU_DISTANCE = 0.7
    MAX_DIST = 0.2
    NN_BUDGET = 100


config = DeepSortConfig()


class CarPartsTracker:
    """DeepSORT tracker for multi-object tracking of car parts"""
    
    def __init__(self, max_age: int = None, device: str = "cuda"):
        """
        Initialize DeepSORT tracker with pretrained feature extractor
        
        Args:
            max_age: Maximum frames to keep lost tracks
            device: 'cuda' or 'cpu' (note: deep_sort_realtime uses CPU for features)
        """
        self.max_age = max_age or config.MAX_AGE
        self.min_hits = config.MIN_HITS
        self.iou_threshold = config.IOU_THRESHOLD
        
        print(f"🔍 Initializing DeepSORT tracker for car parts...")
        self.tracker = self._initialize_tracker()
        
        # Track history storage
        self.track_history = defaultdict(list)  # track_id -> list of {frame, bbox, ...}
        self.track_metadata = {}  # track_id -> {first_frame, last_frame, class, ...}
        
        print(f"✓ DeepSORT tracker ready")
        print(f"  Max age: {self.max_age} frames")
        print(f"  Min hits: {self.min_hits} frames\n")
    
    def _initialize_tracker(self) -> DeepSort:
        """Initialize DeepSORT tracker with pretrained weights"""
        try:
            tracker = DeepSort(
                max_age=self.max_age,
                n_init=config.N_INIT,
                max_iou_distance=config.MAX_IOU_DISTANCE,
                max_cosine_distance=config.MAX_DIST,
                nn_budget=config.NN_BUDGET,
                embedder="mobilenet",  # Pretrained feature extractor
                embedder_gpu=False,    # CPU is fine for this
            )
            return tracker
            
        except Exception as e:
            print(f"❌ Failed to initialize DeepSORT: {e}")
            raise
    
    def update(self, 
               frame: np.ndarray,
               detections: List[Dict],
               frame_idx: int) -> List[Dict]:
        """
        Update tracker with new detections
        
        Args:
            frame: Current frame (BGR image)
            detections: List of detections from YOLOv8 detector
            frame_idx: Current frame number
        
        Returns:
            List of tracks:
            [
                {
                    'track_id': int,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float,
                    'class_id': int,
                    'class_name': str,
                    'age': int  # Frames since last detection
                },
                ...
            ]
        """
        # Convert detections to DeepSORT format
        # Format: ([x1, y1, w, h], confidence, class_name)
        ds_detections = []
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            w, h = x2 - x1, y2 - y1
            
            ds_det = (
                [x1, y1, w, h],
                det['confidence'],
                det['class_name']
            )
            ds_detections.append(ds_det)
        
        # Update tracker
        tracks = self.tracker.update_tracks(ds_detections, frame=frame)
        
        # Convert tracks to our format
        active_tracks = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            bbox = track.to_ltrb()  # [x1, y1, x2, y2]
            bbox = [int(x) for x in bbox]
            
            # Find matching detection for class_id
            class_id = -1
            for det in detections:
                det_bbox = det['bbox']
                # Check if bboxes overlap (simple IoU check)
                if self._bbox_iou(bbox, det_bbox) > 0.5:
                    class_id = det['class_id']
                    break
            
            # Get track info
            track_info = {
                'track_id': track_id,
                'bbox': bbox,
                'confidence': track.get_det_conf() if track.get_det_conf() else 0.0,
                'class_name': track.get_det_class(),
                'class_id': class_id,
                'age': track.time_since_update
            }
            
            # Store in history
            self._update_track_history(track_id, frame_idx, track_info)
            
            active_tracks.append(track_info)
        
        return active_tracks
    
    def _update_track_history(self, 
                             track_id: int, 
                             frame_idx: int, 
                             track_info: Dict):
        """Store track information in history"""
        # Add to history
        history_entry = {
            'frame': frame_idx,
            'bbox': track_info['bbox'],
            'confidence': track_info['confidence'],
            'class_name': track_info['class_name']
        }
        self.track_history[track_id].append(history_entry)
        
        # Update metadata
        if track_id not in self.track_metadata:
            self.track_metadata[track_id] = {
                'first_frame': frame_idx,
                'last_frame': frame_idx,
                'class_name': track_info['class_name'],
                'total_detections': 1
            }
        else:
            self.track_metadata[track_id]['last_frame'] = frame_idx
            self.track_metadata[track_id]['total_detections'] += 1
    
    def _bbox_iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def get_track_history(self, track_id: int) -> List[Dict]:
        """Get full history of a track"""
        return self.track_history.get(track_id, [])
    
    def get_track_metadata(self, track_id: int) -> Optional[Dict]:
        """Get metadata about a track"""
        return self.track_metadata.get(track_id)
    
    def get_all_tracks(self) -> Dict[int, List[Dict]]:
        """Get all track histories"""
        return dict(self.track_history)
    
    def get_active_track_ids(self) -> List[int]:
        """Get list of all track IDs"""
        return list(self.track_metadata.keys())
    
    def get_track_duration(self, track_id: int) -> int:
        """Get duration (in frames) of a track"""
        meta = self.track_metadata.get(track_id)
        if meta:
            return meta['last_frame'] - meta['first_frame'] + 1
        return 0
    
    def visualize_tracks(self,
                        frame: np.ndarray,
                        tracks: List[Dict],
                        show_id: bool = True,
                        show_class: bool = True,
                        show_conf: bool = True,
                        trail_length: int = 30) -> np.ndarray:
        """
        Draw tracked objects on frame with trails
        
        Args:
            frame: BGR image
            tracks: List of tracks from update()
            show_id: Show track ID
            show_class: Show object class
            show_conf: Show confidence score
            trail_length: Length of motion trail (0 to disable)
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Color map (consistent colors per track_id)
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 165, 255),  # Orange
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (128, 0, 128),  # Purple
            (0, 255, 128),  # Spring green
            (255, 128, 0),  # Dark orange
            (255, 200, 0),  # Cyan
            (200, 100, 255),# Pink
        ]
        
        for track in tracks:
            track_id = track['track_id']
            x1, y1, x2, y2 = track['bbox']
            class_name = track['class_name']
            confidence = track['confidence']
            
            # Get consistent color for this track
            color = colors[int(track_id) % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Build label
            label_parts = []
            if show_id:
                label_parts.append(f"ID:{track_id}")
            if show_class:
                label_parts.append(class_name)
            if show_conf:
                label_parts.append(f"{confidence:.2f}")
            
            label = " ".join(label_parts)
            
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - text_h - 10),
                (x1 + text_w + 5, y1),
                color,
                -1
            )
            
            # Text
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
            
            # Draw motion trail
            if trail_length > 0:
                history = self.track_history[track_id]
                if len(history) > 1:
                    # Get recent positions
                    recent = history[-trail_length:]
                    centers = [
                        ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                        for entry in recent
                        for bbox in [entry['bbox']]
                    ]
                    
                    # Draw trail
                    for i in range(1, len(centers)):
                        # Fade trail
                        alpha = i / len(centers)
                        thickness = max(1, int(3 * alpha))
                        
                        cv2.line(
                            annotated,
                            centers[i-1],
                            centers[i],
                            color,
                            thickness
                        )
        
        return annotated
    
    def get_statistics(self) -> Dict:
        """Get tracker statistics"""
        total_tracks = len(self.track_metadata)
        
        if total_tracks == 0:
            return {
                'total_tracks': 0,
                'avg_duration': 0,
                'by_class': {}
            }
        
        # Count by class
        by_class = defaultdict(int)
        durations = []
        
        for track_id, meta in self.track_metadata.items():
            by_class[meta['class_name']] += 1
            duration = meta['last_frame'] - meta['first_frame'] + 1
            durations.append(duration)
        
        return {
            'total_tracks': total_tracks,
            'avg_duration': np.mean(durations),
            'max_duration': max(durations),
            'min_duration': min(durations),
            'by_class': dict(by_class)
        }
    
    def reset(self):
        """Reset tracker (clear all tracks)"""
        self.tracker = self._initialize_tracker()
        self.track_history.clear()
        self.track_metadata.clear()
        print("✓ Tracker reset")


# ============================================================================
# TESTING
# ============================================================================

def test_tracker():
    """Test tracker with detector"""
    print("\n" + "="*60)
    print("Testing DeepSORT Car Parts Tracker")
    print("="*60 + "\n")
    
    # Initialize
    from object_detector import CarPartsDetector
    
    detector = CarPartsDetector(device="cuda", conf_threshold=0.5)
    tracker = CarPartsTracker()
    
    print("✓ Detector and Tracker initialized\n")
    
    # Test on sample images
    test_dir = Path("test_images")
    
    if test_dir.exists():
        images = sorted(list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")) + list(test_dir.glob("*.webp")))
        
        if len(images) > 0:
            print(f"Testing on {len(images)} images (simulating video)...\n")
            
            for frame_idx, img_path in enumerate(images[:20]):  # First 20 frames
                frame = cv2.imread(str(img_path))
                
                if frame is None:
                    continue
                
                # Detect
                detections = detector.detect_frame(frame)
                
                # Track
                tracks = tracker.update(frame, detections, frame_idx)
                
                print(f"Frame {frame_idx}: {len(detections)} detections, {len(tracks)} tracks")
                
                # Visualize last frame
                if frame_idx == len(images) - 1 or frame_idx == 19:
                    annotated = tracker.visualize_tracks(frame, tracks, trail_length=10)
                    
                    output_path = f"results/test_tracking_frame{frame_idx}.jpg"
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(output_path, annotated)
                    print(f"  Saved: {output_path}")
            
            # Print statistics
            stats = tracker.get_statistics()
            print(f"\n✓ Tracking Statistics:")
            print(f"  Total tracks: {stats['total_tracks']}")
            print(f"  Avg duration: {stats['avg_duration']:.1f} frames")
            print(f"  By class: {stats['by_class']}")
            
            return True
    
    print("⚠ No test images found. Create 'test_images/' folder with images.")
    return False

def test_video_tracker(video_path="video/traffic.mp4", output_path="results/tracking_output.mp4"):
    """Test DeepSORT car parts tracker on a video"""
    print("\n" + "="*60)
    print(f"🎥 Testing tracker on video: {video_path}")
    print("="*60 + "\n")

    from object_detector import CarPartsDetector

    # Initialize detector + tracker
    detector = CarPartsDetector(device="cuda", conf_threshold=0.5)
    tracker = CarPartsTracker()

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return

    # Video writer setup
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    print(f"Total frames: {total_frames}, FPS: {fps}\n")

    frame_idx = 0
    while frame_idx < 400:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection + tracking
        detections = detector.detect_frame(frame)
        tracks = tracker.update(frame, detections, frame_idx)

        # Visualize
        annotated = tracker.visualize_tracks(frame, tracks, trail_length=20)

        # Write frame
        writer.write(annotated)

        if frame_idx % 20 == 0:
            print(f"Frame {frame_idx}/{total_frames} | Detections: {len(detections)} | Tracks: {len(tracks)}")

        frame_idx += 1

    cap.release()
    writer.release()

    print(f"\n✅ Done! Output saved to: {output_path}")
    print(f"Total tracks: {tracker.get_statistics()['total_tracks']}")

if __name__ == "__main__":
    # test_tracker()
    test_video_tracker("video/traffic1.mp4", "results/tracking_output.mp4")

