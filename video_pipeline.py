"""
video_pipeline.py
Process video: Output both original (annotated) and perturbed versions
Plus optional side-by-side comparison
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import time
from amnet_integration import IntegratedMemorabilityPipeline


class VideoPipeline:
    """
    Video processing pipeline that outputs:
    1. Original video with annotations
    2. Perturbed video with annotations
    3. Side-by-side comparison (optional)
    """
    
    def __init__(self,
                 yolo_model_path: str = "models/best.pt",
                 amnet_model_path: str = "models/amnet_weights.pth",
                 device: str = "cuda",
                 memorability_threshold: float = 0.6):
        
        print(f"\n{'='*80}")
        print("Video Processing Pipeline - Initialization")
        print(f"{'='*80}\n")
        
        # Initialize pipeline (loads models once)
        self.pipeline = IntegratedMemorabilityPipeline(
            yolo_model_path=yolo_model_path,
            amnet_model_path=amnet_model_path,
            device=device,
            memorability_threshold=memorability_threshold,
            attention_maps=True
        )
        
        print("✓ Video pipeline ready!\n")
    
    
    def process_video(self,
                     video_path: str,
                     output_dir: str = "results/video_output",
                     max_frames: Optional[int] = None,
                     create_comparison: bool = True,
                     save_original: bool = True,
                     save_perturbed: bool = True):
        """
        Process video and save outputs
        
        Args:
            video_path: Path to input video
            output_dir: Output directory
            max_frames: Process only first N frames (None = all frames)
            create_comparison: Create side-by-side comparison video
            save_original: Save original video with annotations
            save_perturbed: Save perturbed video
        
        Returns:
            Dictionary with statistics
        """
        
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not video_path.exists():
            print(f"❌ Error: Video not found: {video_path}")
            return None
        
        print(f"\n{'='*80}")
        print(f"Processing Video: {video_path.name}")
        print(f"{'='*80}\n")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Error: Could not open video")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Limit frames if specified
        if max_frames is not None:
            frames_to_process = min(max_frames, total_frames)
        else:
            frames_to_process = total_frames
        
        print(f"📊 Video Information:")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Total frames: {total_frames}")
        print(f"   Frames to process: {frames_to_process}")
        print(f"   Duration: {total_frames/fps:.1f}s (processing {frames_to_process/fps:.1f}s)\n")
        
        # Create video writers
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        writers = {}
        
        if save_original:
            original_path = output_dir / f"{video_path.stem}_original_annotated.mp4"
            writers['original'] = cv2.VideoWriter(
                str(original_path), fourcc, fps, (width, height)
            )
            print(f"✓ Will save original (annotated): {original_path}")
        
        if save_perturbed:
            perturbed_path = output_dir / f"{video_path.stem}_perturbed.mp4"
            writers['perturbed'] = cv2.VideoWriter(
                str(perturbed_path), fourcc, fps, (width, height)
            )
            print(f"✓ Will save perturbed: {perturbed_path}")
        
        if create_comparison:
            comparison_path = output_dir / f"{video_path.stem}_comparison.mp4"
            writers['comparison'] = cv2.VideoWriter(
                str(comparison_path), fourcc, fps, (width * 2, height)
            )
            print(f"✓ Will save comparison: {comparison_path}\n")
        
        # Processing statistics
        stats = {
            'total_frames': 0,
            'total_detections': 0,
            'total_high_mem': 0,
            'total_edits': 0,
            'processing_time': 0
        }
        
        frame_idx = 0
        start_time = time.time()
        
        print(f"{'='*80}")
        print("Processing frames...")
        print(f"{'='*80}\n")
        
        # Process frames
        while frame_idx < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.time()
            
            # Process WITHOUT perturbation (for original video)
            original_annotated, stats_orig = self.pipeline.process_frame(
                frame.copy(),
                frame_idx=frame_idx,
                apply_perturbation=False,
                visualize=True
            )
            
            # Process WITH perturbation
            perturbed_annotated, stats_pert = self.pipeline.process_frame(
                frame.copy(),
                frame_idx=frame_idx,
                apply_perturbation=True,
                visualize=False
            )
            
            frame_time = time.time() - frame_start
            
            # Write to output videos
            if 'original' in writers:
                writers['original'].write(original_annotated)
            
            if 'perturbed' in writers:
                writers['perturbed'].write(perturbed_annotated)
            
            if 'comparison' in writers:
                # Create side-by-side comparison
                comparison_frame = np.hstack([original_annotated, perturbed_annotated])
                
                # Add labels
                cv2.putText(comparison_frame, "ORIGINAL (Annotated)", 
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv2.putText(comparison_frame, "PERTURBED (Diffusion Applied)", 
                           (width + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                writers['comparison'].write(comparison_frame)
            
            # Update statistics
            stats['total_frames'] += 1
            stats['total_detections'] += stats_pert['total_detections']
            stats['total_high_mem'] += stats_pert['high_memorability_count']
            stats['total_edits'] += stats_pert['edits_applied']
            stats['processing_time'] += frame_time
            
            # Progress update every 30 frames
            if frame_idx % 30 == 0 or frame_idx == frames_to_process - 1:
                elapsed = time.time() - start_time
                fps_proc = (frame_idx + 1) / elapsed if elapsed > 0 else 0
                eta = (frames_to_process - frame_idx - 1) / fps_proc if fps_proc > 0 else 0
                
                progress = ((frame_idx + 1) / frames_to_process) * 100
                
                print(f"  Frame {frame_idx + 1}/{frames_to_process} ({progress:.1f}%) | "
                      f"Detections: {stats_pert['total_detections']} | "
                      f"High-mem: {stats_pert['high_memorability_count']} | "
                      f"Edits: {stats_pert['edits_applied']} | "
                      f"Speed: {fps_proc:.1f} fps | "
                      f"ETA: {eta:.0f}s")
            
            frame_idx += 1
        
        # Cleanup
        cap.release()
        for writer in writers.values():
            writer.release()
        
        total_time = time.time() - start_time
        avg_fps = stats['total_frames'] / total_time if total_time > 0 else 0
        
        # Print final statistics
        print(f"\n{'='*80}")
        print("Processing Complete!")
        print(f"{'='*80}\n")
        
        print(f"📊 Final Statistics:")
        print(f"   Frames processed: {stats['total_frames']}")
        print(f"   Total detections: {stats['total_detections']}")
        print(f"   High memorability detections: {stats['total_high_mem']}")
        print(f"   Total edits applied: {stats['total_edits']}")
        print(f"   Processing time: {total_time:.1f}s")
        print(f"   Average speed: {avg_fps:.2f} fps")
        print(f"   Time per frame: {stats['processing_time']/stats['total_frames']:.3f}s\n")
        
        print(f"✅ Output files saved to: {output_dir}/")
        if save_original:
            print(f"   - {video_path.stem}_original_annotated.mp4")
        if save_perturbed:
            print(f"   - {video_path.stem}_perturbed.mp4")
        if create_comparison:
            print(f"   - {video_path.stem}_comparison.mp4  ⭐ RECOMMENDED FOR REVIEW\n")
        
        return stats


def main():
    """Main execution with examples"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Video Processing Pipeline with Memorability Reduction'
    )
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video')
    parser.add_argument('--output', type=str, default='results/video_output',
                       help='Output directory')
    parser.add_argument('--frames', type=int, default=None,
                       help='Process only first N frames (default: all)')
    parser.add_argument('--no-comparison', action='store_true',
                       help='Skip creating comparison video')
    parser.add_argument('--no-original', action='store_true',
                       help='Skip saving original video')
    parser.add_argument('--no-perturbed', action='store_true',
                       help='Skip saving perturbed video')
    parser.add_argument('--threshold', type=float, default=0.6,
                       help='Memorability threshold (default: 0.6)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = VideoPipeline(
        yolo_model_path="models/best.pt",
        amnet_model_path="models/amnet_weights.pkl",
        device=args.device,
        memorability_threshold=args.threshold
    )
    
    # Process video
    stats = pipeline.process_video(
        video_path=args.video,
        output_dir=args.output,
        max_frames=args.frames,
        create_comparison=not args.no_comparison,
        save_original=not args.no_original,
        save_perturbed=not args.no_perturbed
    )
    
    if stats:
        print("\n✨ Video processing complete!\n")
    else:
        print("\n❌ Video processing failed.\n")


if __name__ == "__main__":
    main()