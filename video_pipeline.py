"""
video_pipeline.py
Video processing loop optimized for Batched Video Diffusion
Outputs:
1. Clean Diffusion Video (Just the edits)
2. Visualization Video (Edits + Bounding Boxes + Heatmaps)
3. Comparison Video (Split screen: Original vs Clean Diffusion)
"""
import cv2
import numpy as np
from pathlib import Path
import time
import argparse
from amnet_integration import AdvancedMemorabilityPipeline

class AdvancedVideoPipeline:
    def __init__(self, yolo_model_path="models/best.pt", amnet_model_path="models/amnet_weights.pkl", 
                 device="cuda", memorability_threshold=0.6, use_lcm=True):
        
        self.pipeline = AdvancedMemorabilityPipeline(
            yolo_model_path=yolo_model_path, 
            amnet_model_path=amnet_model_path, 
            device=device,
            memorability_threshold=memorability_threshold, 
            attention_maps=True,
            use_lcm=use_lcm
        )
        # AnimateDiff works best with 16 frames context. 
        # We use a multiple of 16 (e.g., 32) for efficiency.
        self.batch_size = 32 
        
    def process_video(self, video_path, output_dir="results/output", max_frames=None):
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not video_path.exists():
            print(f" Error: Video not found: {video_path}")
            return None
            
        # --- Setup Inputs ---
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames_to_process = min(max_frames, total_frames) if max_frames else total_frames
        
        # --- Setup Writers ---
        # 1. Clean output: Shows ONLY the diffusion edits
        out_path_clean = output_dir / f"{video_path.stem}_clean.mp4"
        
        # 2. Visualized output: Shows boxes, heatmaps, and stats overlay
        out_path_viz = output_dir / f"{video_path.stem}_visualized.mp4"
        
        # 3. Comparison output: Left (Original) | Right (Clean Edit)
        out_path_comp = output_dir / f"{video_path.stem}_comparison.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        writer_clean = cv2.VideoWriter(str(out_path_clean), fourcc, fps, (width, height))
        writer_viz = cv2.VideoWriter(str(out_path_viz), fourcc, fps, (width, height))
        writer_comp = cv2.VideoWriter(str(out_path_comp), fourcc, fps, (width * 2, height))
        
        print(f" Processing {frames_to_process} frames in batches of {self.batch_size}...")
        print(f"   Outputs will be saved to: {output_dir}")
        
        processed_count = 0
        start_time = time.time()
        
        while processed_count < frames_to_process:
            # 1. Read a batch of frames
            current_batch = []
            while len(current_batch) < self.batch_size and processed_count + len(current_batch) < frames_to_process:
                ret, frame = cap.read()
                if not ret: break
                current_batch.append(frame)
            
            if not current_batch:
                break
                
            batch_start_idx = processed_count
            print(f"\n Batch Loaded: Frames {batch_start_idx} to {batch_start_idx + len(current_batch)}")

            # 2. Process the batch
            try:
                # We work on copies to preserve originals logic if needed
                batch_to_process = [f.copy() for f in current_batch]
                
                # Unpack the THREE return values from amnet_integration
                clean_batch, viz_batch, batch_stats = self.pipeline.process_transfer_batch(
                    batch_to_process, 
                    start_frame_idx=batch_start_idx
                )
                
                # 3. Write results
                for i in range(len(clean_batch)):
                    original_frame = current_batch[i]
                    clean_frame = clean_batch[i]
                    viz_frame = viz_batch[i]
                    
                    # 1. Save CLEAN edit
                    writer_clean.write(clean_frame)
                    
                    # 2. Save VISUALIZED edit
                    writer_viz.write(viz_frame)
                    
                    # 3. Save COMPARISON (Left: Original, Right: Clean Edit)
                    # Add simple labels
                    cv2.putText(original_frame, "ORIGINAL", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(clean_frame, "DIFFUSION EDIT", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    comparison_frame = np.hstack([original_frame, clean_frame])
                    writer_comp.write(comparison_frame)
                
                processed_count += len(current_batch)
                
            except Exception as e:
                print(f" Batch Failed: {e}")
                import traceback
                traceback.print_exc()
                # Fallback: Write original frames if diffusion fails to keep video length correct
                for frame in current_batch:
                    writer_clean.write(frame)
                    writer_viz.write(frame)
                    dbl_frame = np.hstack([frame, frame])
                    writer_comp.write(dbl_frame)
                processed_count += len(current_batch)

            # Clear VRAM after every batch to prevent leaks
            self.pipeline.clear_cache()
            
        cap.release()
        writer_clean.release()
        writer_viz.release()
        writer_comp.release()
        
        elapsed = time.time() - start_time
        print(f"\n Processing Complete!")
        print(f"    Clean Video:  {out_path_clean}")
        print(f"    Debug Video:  {out_path_viz}")
        print(f"    Comparison:   {out_path_comp}")
        print(f"    Time taken: {elapsed:.2f}s ({processed_count/elapsed:.2f} fps)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--frames', type=int, default=None)
    parser.add_argument('--use-lcm', action='store_true', default=True, help="Use LCM for faster inference")
    args = parser.parse_args()
    
    pipeline = AdvancedVideoPipeline(
        yolo_model_path="models/best.pt",
        amnet_model_path="models/amnet_weights.pkl",
        use_lcm=args.use_lcm
    )
    
    pipeline.process_video(args.video, max_frames=args.frames)

if __name__ == "__main__":
    main()