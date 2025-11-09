"""
Video Processing with Memorability-Based Perturbation
Processes videos to reduce memorability of detected car parts
"""

import cv2
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import json

from amnet_integration import IntegratedMemorabilityPipeline


def process_video_with_memorability(
    video_path: str,
    output_path: str,
    yolo_model: str = "models/best.pt",
    amnet_model: str = "models/amnet_weights.pkl",
    conf_threshold: float = 0.5,
    memorability_threshold: float = 0.6,
    device: str = "cuda",
    apply_perturbation: bool = True,
    show_attention: bool = True,
    max_frames: int = None,
    save_stats: bool = True
):
    """
    Process video with memorability analysis and perturbation
    
    Args:
        video_path: Input video path
        output_path: Output video path
        yolo_model: Path to YOLO weights
        amnet_model: Path to AMNet weights
        conf_threshold: Detection confidence threshold
        memorability_threshold: Threshold for high memorability (0-1)
        device: 'cuda' or 'cpu'
        apply_perturbation: Apply perturbations to reduce memorability
        show_attention: Show attention maps on high-memorability regions
        max_frames: Maximum frames to process (None = all)
        save_stats: Save statistics to JSON
    """
    
    print("\n" + "="*70)
    print("Video Processing: Memorability Analysis & Reduction")
    print("="*70 + "\n")
    
    # Initialize pipeline
    pipeline = IntegratedMemorabilityPipeline(
        yolo_model_path=yolo_model,
        amnet_model_path=amnet_model,
        device=device,
        conf_threshold=conf_threshold,
        memorability_threshold=memorability_threshold,
        attention_maps=show_attention
    )
    
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
    
    print(f"Video Info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f}s")
    print(f"\nMemorability Settings:")
    print(f"  Threshold: {memorability_threshold}")
    print(f"  Apply perturbation: {apply_perturbation}")
    print(f"  Show attention: {show_attention}\n")
    
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
    start_time = time.time()
    
    # Statistics tracking
    all_stats = []
    total_high_mem = 0
    total_detections = 0
    mem_scores_all = []
    
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
        
        # Process frame
        result_frame, stats = pipeline.process_frame(
            frame,
            frame_idx=frame_idx,
            apply_perturbation=apply_perturbation,
            visualize=True
        )
        
        # Update statistics
        total_high_mem += stats['high_memorability_count']
        total_detections += stats['total_detections']
        if stats['avg_memorability'] > 0:
            mem_scores_all.append(stats['avg_memorability'])
        
        stats['frame_idx'] = frame_idx
        all_stats.append(stats)
        
        # Add info text
        info_text = (f"Frame: {frame_idx} | Detections: {stats['total_detections']} | "
                    f"High Mem: {stats['high_memorability_count']} | "
                    f"Avg Mem: {stats['avg_memorability']:.3f}")
        
        cv2.putText(
            result_frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        # Add perturbation indicator
        if apply_perturbation:
            status_text = "PERTURBATION: ON"
            color = (0, 255, 0)
        else:
            status_text = "PERTURBATION: OFF"
            color = (0, 0, 255)
        
        cv2.putText(
            result_frame,
            status_text,
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )
        
        # Write frame
        out.write(result_frame)
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    
    # Cleanup
    cap.release()
    out.release()
    
    # Compute final statistics
    elapsed_time = time.time() - start_time
    avg_fps = frame_idx / elapsed_time
    
    print(f"\n{'='*70}")
    print("Processing Complete!")
    print(f"{'='*70}")
    print(f"  Processed Frames: {frame_idx}")
    print(f"  Time Elapsed: {elapsed_time:.2f}s")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Output saved to: {output_path}")
    
    print(f"\nMemorability Statistics:")
    print(f"  Total Detections: {total_detections}")
    print(f"  High Memorability Detections: {total_high_mem}")
    print(f"  High Mem Percentage: {100*total_high_mem/max(total_detections,1):.1f}%")
    if mem_scores_all:
        print(f"  Overall Avg Memorability: {np.mean(mem_scores_all):.3f}")
        print(f"  Memorability Std Dev: {np.std(mem_scores_all):.3f}")
    
    # Tracking statistics
    track_stats = pipeline.tracker.get_statistics()
    print(f"\nTracking Statistics:")
    print(f"  Total Tracks: {track_stats['total_tracks']}")
    if track_stats['total_tracks'] > 0:
        print(f"  Avg Track Duration: {track_stats['avg_duration']:.1f} frames")
        print(f"  Detections by Class:")
        for class_name, count in track_stats['by_class'].items():
            print(f"    {class_name}: {count}")
    
    print(f"\n{'='*70}\n")
    
    # Save statistics to JSON
    if save_stats:
        stats_path = output_path.parent / (output_path.stem + "_stats.json")
        
        summary = {
            'video_info': {
                'input': str(video_path),
                'output': str(output_path),
                'resolution': f"{width}x{height}",
                'fps': fps,
                'frames_processed': frame_idx,
                'duration_seconds': frame_idx / fps
            },
            'processing': {
                'elapsed_time': elapsed_time,
                'avg_fps': avg_fps,
                'memorability_threshold': memorability_threshold,
                'perturbation_enabled': apply_perturbation
            },
            'memorability': {
                'total_detections': total_detections,
                'high_memorability_count': total_high_mem,
                'high_mem_percentage': 100*total_high_mem/max(total_detections,1),
                'avg_memorability': float(np.mean(mem_scores_all)) if mem_scores_all else 0,
                'std_memorability': float(np.std(mem_scores_all)) if mem_scores_all else 0
            },
            'tracking': track_stats,
            'frame_stats': all_stats
        }
        
        with open(stats_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Statistics saved to: {stats_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Process video with memorability analysis and reduction"
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
        default="results/memorability_reduced.mp4",
        help="Output video path (default: results/memorability_reduced.mp4)"
    )
    
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="models/best.pt",
        help="Path to YOLO model (default: models/best.pt)"
    )
    
    parser.add_argument(
        "--amnet-model",
        type=str,
        default="models/amnet_weights.pkl",
        help="Path to AMNet model (default: models/amnet_weights.pkl)"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)"
    )
    
    parser.add_argument(
        "--mem-threshold",
        type=float,
        default=0.6,
        help="Memorability threshold (0-1, default: 0.6)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)"
    )
    
    parser.add_argument(
        "--no-perturbation",
        action="store_true",
        help="Disable perturbation (analysis only)"
    )
    
    parser.add_argument(
        "--no-attention",
        action="store_true",
        help="Disable attention map visualization"
    )
    
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: all)"
    )
    
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Don't save statistics JSON"
    )
    
    args = parser.parse_args()
    
    try:
        process_video_with_memorability(
            video_path=args.video,
            output_path=args.output,
            yolo_model=args.yolo_model,
            amnet_model=args.amnet_model,
            conf_threshold=args.conf,
            memorability_threshold=args.mem_threshold,
            device=args.device,
            apply_perturbation=not args.no_perturbation,
            show_attention=not args.no_attention,
            max_frames=args.max_frames,
            save_stats=not args.no_stats
        )
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        raise


if __name__ == "__main__":
    main()