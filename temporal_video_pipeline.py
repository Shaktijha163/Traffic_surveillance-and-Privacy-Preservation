"""
temporal_video_pipeline.py
Two-pass video processing with temporally consistent edits

Pass 1: Collect track data (fast)
Pass 2: Apply synchronized diffusion edits (slow)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import time
import json
from amnet_integration import TemporalMemorabilityPipeline


class TemporalVideoPipeline:
    """
    Two-pass video pipeline for temporally consistent memorability reduction
    
    WORKFLOW:
    1. Pass 1: Detect -> Track -> Score Memorability (save all data)
    2. Pass 2: Apply synchronized edits per track (temporally consistent)
    3. Render final video with consistent edits
    """
    
    def __init__(self,
                 yolo_model_path: str = "models/best.pt",
                 amnet_model_path: str = "models/amnet_weights.pkl",
                 device: str = "cuda",
                 memorability_threshold: float = 0.6):
        
        print(f"\n{'='*80}")
        print("Temporal Video Processing Pipeline - Initialization")
        print(f"{'='*80}\n")
        
        # Initialize temporal pipeline
        self.pipeline = TemporalMemorabilityPipeline(
            yolo_model_path=yolo_model_path,
            amnet_model_path=amnet_model_path,
            device=device,
            memorability_threshold=memorability_threshold,
            attention_maps=True,
            guidance_scale=7.5,
            num_inference_steps=25
        )
        
        print("✓ Temporal video pipeline ready!\n")
    
    
    def process_video_two_pass(self,
                              video_path: str,
                              output_dir: str = "results/temporal_video",
                              max_frames: Optional[int] = None,
                              save_metadata: bool = True,
                              create_comparison: bool = True):
        """
        Process video with two-pass approach for temporal consistency
        
        Args:
            video_path: Path to input video
            output_dir: Output directory
            max_frames: Process only first N frames (None = all)
            save_metadata: Save track metadata JSON
            create_comparison: Create side-by-side comparison
        
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
        
        # Get video properties
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Error: Could not open video")
            return None
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        frames_to_process = min(max_frames, total_frames) if max_frames else total_frames
        
        print(f"📊 Video Information:")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Total frames: {total_frames}")
        print(f"   Frames to process: {frames_to_process}")
        print(f"   Duration: {frames_to_process/fps:.1f}s\n")
        
        # ========================================================================
        # PASS 1: COLLECT TRACK DATA
        # ========================================================================
        
        print(f"\n{'='*80}")
        print("PASS 1: Collecting Track Data")
        print(f"{'='*80}\n")
        
        pass1_start = time.time()
        
        # Create temporary video if max_frames specified
        temp_video = None
        if max_frames:
            temp_video = output_dir / "temp_input.mp4"
            self._create_trimmed_video(video_path, temp_video, max_frames)
            video_to_process = temp_video
        else:
            video_to_process = video_path
        
        # Collect track data
        metadata_path = output_dir / f"{video_path.stem}_metadata.json" if save_metadata else None
        
        completed_tracks = self.pipeline.collect_track_data(
            str(video_to_process),
            output_metadata=str(metadata_path) if metadata_path else None
        )
        
        pass1_time = time.time() - pass1_start
        
        print(f"\n✓ Pass 1 completed in {pass1_time:.1f}s")
        print(f"  Collected {len(completed_tracks)} tracks")
        print(f"  Total frames stored: {len(self.pipeline.all_frames)}")
        
        # ========================================================================
        # PASS 2: APPLY TEMPORAL EDITS
        # ========================================================================
        
        print(f"\n{'='*80}")
        print("PASS 2: Applying Temporally Consistent Edits")
        print(f"{'='*80}\n")
        
        pass2_start = time.time()
        
        # Output paths
        output_video = output_dir / f"{video_path.stem}_temporal_edited.mp4"
        
        self.pipeline.apply_temporal_edits(str(output_video))
        
        pass2_time = time.time() - pass2_start
        
        print(f"\n✓ Pass 2 completed in {pass2_time:.1f}s")
        
        # ========================================================================
        # OPTIONAL: CREATE COMPARISON VIDEO
        # ========================================================================
        
        if create_comparison:
            print(f"\n{'='*80}")
            print("Creating Comparison Video")
            print(f"{'='*80}\n")
            
            comparison_path = output_dir / f"{video_path.stem}_comparison.mp4"
            self._create_comparison_video(
                str(video_to_process),
                str(output_video),
                str(comparison_path),
                max_frames=frames_to_process
            )
        
        # Cleanup temp video
        if temp_video and temp_video.exists():
            temp_video.unlink()
        
        # ========================================================================
        # STATISTICS
        # ========================================================================
        
        total_time = pass1_time + pass2_time
        
        # Calculate statistics
        total_detections = sum(len(frames) for frames in completed_tracks.values())
        high_mem_tracks = sum(
            1 for frames in completed_tracks.values()
            if any(f['mem_score'] > self.pipeline.memorability_threshold for f in frames)
        )
        
        stats = {
            'total_frames': len(self.pipeline.all_frames),
            'total_tracks': len(completed_tracks),
            'high_memorability_tracks': high_mem_tracks,
            'total_detections': total_detections,
            'pass1_time': pass1_time,
            'pass2_time': pass2_time,
            'total_time': total_time,
            'fps_pass1': frames_to_process / pass1_time if pass1_time > 0 else 0,
            'fps_pass2': frames_to_process / pass2_time if pass2_time > 0 else 0
        }
        
        # Print final statistics
        print(f"\n{'='*80}")
        print("Processing Complete!")
        print(f"{'='*80}\n")
        
        print(f"📊 Final Statistics:")
        print(f"   Frames processed: {stats['total_frames']}")
        print(f"   Total tracks: {stats['total_tracks']}")
        print(f"   High memorability tracks: {stats['high_memorability_tracks']}")
        print(f"   Total detections: {stats['total_detections']}")
        print(f"")
        print(f"   Pass 1 time: {pass1_time:.1f}s ({stats['fps_pass1']:.1f} fps)")
        print(f"   Pass 2 time: {pass2_time:.1f}s ({stats['fps_pass2']:.1f} fps)")
        print(f"   Total time: {total_time:.1f}s")
        print(f"")
        
        print(f"✅ Output files saved to: {output_dir}/")
        print(f"   - {output_video.name}  ⭐ EDITED VIDEO")
        if save_metadata:
            print(f"   - {metadata_path.name}  📊 METADATA")
        if create_comparison:
            print(f"   - {comparison_path.name}  🔍 COMPARISON\n")
        
        return stats
    
    
    def _create_trimmed_video(self, input_path: Path, output_path: Path, max_frames: int):
        """Create trimmed version of input video"""
        cap = cv2.VideoCapture(str(input_path))
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        for i in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        cap.release()
        out.release()
        
        print(f"✓ Created trimmed video: {output_path} ({max_frames} frames)")
    
    
    def _create_comparison_video(self, 
                                 original_path: str, 
                                 edited_path: str, 
                                 output_path: str,
                                 max_frames: Optional[int] = None):
        """Create side-by-side comparison video"""
        
        cap_orig = cv2.VideoCapture(original_path)
        cap_edit = cv2.VideoCapture(edited_path)
        
        if not cap_orig.isOpened() or not cap_edit.isOpened():
            print("⚠ Cannot create comparison: videos not found")
            return
        
        fps = int(cap_orig.get(cv2.CAP_PROP_FPS))
        width = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
        
        frame_idx = 0
        while True:
            if max_frames and frame_idx >= max_frames:
                break
            
            ret_orig, frame_orig = cap_orig.read()
            ret_edit, frame_edit = cap_edit.read()
            
            if not ret_orig or not ret_edit:
                break
            
            # Stack horizontally
            comparison = np.hstack([frame_orig, frame_edit])
            
            # Add labels
            cv2.putText(comparison, "ORIGINAL", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.putText(comparison, "TEMPORAL EDITED", 
                       (width + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            out.write(comparison)
            frame_idx += 1
        
        cap_orig.release()
        cap_edit.release()
        out.release()
        
        print(f"✓ Created comparison video: {output_path}")


def main():
    """Main execution"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Temporal Video Processing Pipeline (Two-Pass)'
    )
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video')
    parser.add_argument('--output', type=str, default='results/temporal_video',
                       help='Output directory')
    parser.add_argument('--frames', type=int, default=None,
                       help='Process only first N frames (default: all)')
    parser.add_argument('--no-comparison', action='store_true',
                       help='Skip creating comparison video')
    parser.add_argument('--no-metadata', action='store_true',
                       help='Skip saving metadata JSON')
    parser.add_argument('--threshold', type=float, default=0.6,
                       help='Memorability threshold (default: 0.6)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = TemporalVideoPipeline(
        yolo_model_path="models/best.pt",
        amnet_model_path="models/amnet_weights.pkl",
        device=args.device,
        memorability_threshold=args.threshold
    )
    
    # Process video
    stats = pipeline.process_video_two_pass(
        video_path=args.video,
        output_dir=args.output,
        max_frames=args.frames,
        save_metadata=not args.no_metadata,
        create_comparison=not args.no_comparison
    )
    
    if stats:
        print("\n✨ Temporal video processing complete!\n")
    else:
        print("\n❌ Video processing failed.\n")


if __name__ == "__main__":
    main()