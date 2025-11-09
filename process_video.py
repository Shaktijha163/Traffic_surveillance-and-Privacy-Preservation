"""
Video Processing Script for Car Parts Detection and Tracking
Uses custom YOLOv8 model (best.pt) + DeepSORT tracking
"""

import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
import time

from object_detector import CarPartsDetector
from object_tracker import CarPartsTracker


def process_video(
    video_path: str,
    output_path: str,
    model_path: str = "models/best.pt",
    conf_threshold: float = 0.5,
    device: str = "cuda",
    show_trails: bool = True,
    trail_length: int = 30,
    skip_frames: int = 0,
    max_frames: int = None
):
    """
    Process video with car parts detection and tracking
    
    Args:
        video_path: Input video path
        output_path: Output video path
        model_path: Path to best.pt model
        conf_threshold: Detection confidence threshold
        device: 'cuda' or 'cpu'
        show_trails: Show motion trails
        trail_length: Length of trails (frames)
        skip_frames: Process every Nth frame (0 = process all)
        max_frames: Maximum number of frames to process (None = all frames)
    """
    
    print("\n" + "="*70)
    print("Video Processing: Car Parts Detection + Tracking")
    print("="*70 + "\n")
    
    # Initialize detector and tracker
    print("Initializing models...")
    detector = CarPartsDetector(
        model_path=model_path,
        device=device,
        conf_threshold=conf_threshold
    )
    tracker = CarPartsTracker(device=device)
    
    # Open video
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo Info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f}s\n")
    
    # Setup output video
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )
    
    # Process video
    print("Processing video...\n")
    
    frame_idx = 0
    processed_frames = 0
    start_time = time.time()
    
    # Determine frames to process
    frames_to_process = min(total_frames, max_frames) if max_frames else total_frames
    
    pbar = tqdm(total=frames_to_process, desc="Processing")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Stop if max_frames reached
        if max_frames and frame_idx >= max_frames:
            break
        
        # Skip frames if needed
        if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
            out.write(frame)  # Write original frame
            frame_idx += 1
            pbar.update(1)
            continue
        
        # Detect objects
        detections = detector.detect_frame(frame)
        
        # Track objects
        tracks = tracker.update(frame, detections, frame_idx)
        
        # Visualize
        annotated = tracker.visualize_tracks(
            frame,
            tracks,
            show_id=True,
            show_class=True,
            show_conf=True,
            trail_length=trail_length if show_trails else 0
        )
        
        # Add frame info
        info_text = f"Frame: {frame_idx} | Detections: {len(detections)} | Tracks: {len(tracks)}"
        cv2.putText(
            annotated,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Write frame
        out.write(annotated)
        
        frame_idx += 1
        processed_frames += 1
        pbar.update(1)
    
    pbar.close()
    
    # Cleanup
    cap.release()
    out.release()
    
    # Print statistics
    elapsed_time = time.time() - start_time
    avg_fps = processed_frames / elapsed_time
    
    print(f"\n{'='*70}")
    print("Processing Complete!")
    print(f"{'='*70}")
    print(f"  Processed Frames: {processed_frames}")
    print(f"  Time Elapsed: {elapsed_time:.2f}s")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Output saved to: {output_path}")
    
    # Tracking statistics
    stats = tracker.get_statistics()
    print(f"\nTracking Statistics:")
    print(f"  Total Tracks: {stats['total_tracks']}")
    print(f"  Avg Track Duration: {stats['avg_duration']:.1f} frames")
    print(f"  Detections by Class:")
    for class_name, count in stats['by_class'].items():
        print(f"    {class_name}: {count}")
    
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Process video with car parts detection and tracking"
    )
    
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Input video path"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results/output.mp4",
        help="Output video path (default: results/output.mp4)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="models/best.pt",
        help="Path to trained model (default: models/best.pt)"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)"
    )
    
    parser.add_argument(
        "--no-trails",
        action="store_true",
        help="Disable motion trails"
    )
    
    parser.add_argument(
        "--trail-length",
        type=int,
        default=30,
        help="Motion trail length in frames (default: 30)"
    )
    
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="Process every Nth frame (0=all frames, 1=every other frame, etc.)"
    )
    
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: all frames)"
    )
    
    args = parser.parse_args()
    
    try:
        process_video(
            video_path=args.video,
            output_path=args.output,
            model_path=args.model,
            conf_threshold=args.conf,
            device=args.device,
            show_trails=not args.no_trails,
            trail_length=args.trail_length,
            skip_frames=args.skip_frames,
            max_frames=args.max_frames
        )
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        raise


if __name__ == "__main__":
    main()