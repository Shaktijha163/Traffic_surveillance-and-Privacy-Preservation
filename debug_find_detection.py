"""
Scan video to find frames with detections
"""

import cv2
from pathlib import Path
from amnet_integration import IntegratedMemorabilityPipeline


def scan_video_for_detections(video_path: str, max_frames: int = 200, sample_every: int = 10):
    """
    Scan video and find frames with high-memorability detections
    """
    
    print(f"\n{'='*80}")
    print(f"Scanning Video for Detections")
    print(f"{'='*80}\n")
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = IntegratedMemorabilityPipeline(
        yolo_model_path="models/best.pt",
        amnet_model_path="models/amnet_weights.pkl",
        device="cuda",
        memorability_threshold=0.5,  # Lower threshold
        attention_maps=False  # Faster without attention
    )
    print("✓ Pipeline initialized\n")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Video info:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {total_frames/fps:.1f}s\n")
    
    print(f"Scanning first {max_frames} frames (every {sample_every} frames)...\n")
    
    # Results
    frames_with_detections = []
    frames_with_high_mem = []
    
    frame_idx = 0
    scanned = 0
    
    while frame_idx < min(max_frames, total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample every N frames
        if frame_idx % sample_every == 0:
            # Quick detection (no perturbation, no visualization)
            result, stats = pipeline.process_frame(
                frame,
                frame_idx=frame_idx,
                apply_perturbation=False,
                visualize=False
            )
            
            scanned += 1
            
            if stats['total_detections'] > 0:
                frames_with_detections.append({
                    'frame': frame_idx,
                    'time': frame_idx / fps,
                    'detections': stats['total_detections'],
                    'high_mem': stats['high_memorability_count'],
                    'avg_mem': stats['avg_memorability']
                })
                
                print(f"Frame {frame_idx:4d} ({frame_idx/fps:6.2f}s): "
                      f"Detections={stats['total_detections']:2d}, "
                      f"High-mem={stats['high_memorability_count']:2d}, "
                      f"Avg={stats['avg_memorability']:.3f}")
                
                if stats['high_memorability_count'] > 0:
                    frames_with_high_mem.append(frame_idx)
        
        frame_idx += 1
    
    cap.release()
    
    # Summary
    print(f"\n{'='*80}")
    print("SCAN RESULTS")
    print(f"{'='*80}\n")
    
    print(f"Frames scanned: {scanned} (sampled every {sample_every} frames)")
    print(f"Frames with detections: {len(frames_with_detections)}")
    print(f"Frames with high memorability: {len(frames_with_high_mem)}\n")
    
    if len(frames_with_high_mem) > 0:
        print(f"✅ GOOD! Found frames with high memorability!")
        print(f"\nBest frames to test (with high-memorability detections):")
        
        # Show top 5
        sorted_frames = sorted(frames_with_detections, 
                              key=lambda x: x['high_mem'], 
                              reverse=True)
        
        for i, f in enumerate(sorted_frames[:5], 1):
            print(f"  {i}. Frame {f['frame']:4d} ({f['time']:6.2f}s): "
                  f"{f['detections']} detections, {f['high_mem']} high-mem")
        
        print(f"\n💡 Run debug on best frame:")
        best_frame = sorted_frames[0]['frame']
        print(f"   python debug_diffusion.py /home/survprivacy/shakti/AMNet/video/traffic.mp4 {best_frame}")
        
        return sorted_frames[0]['frame']
    
    elif len(frames_with_detections) > 0:
        print(f"⚠️ Found detections but none with high memorability (> 0.5)")
        print(f"\nFrames with detections (try lower threshold):")
        
        for i, f in enumerate(frames_with_detections[:10], 1):
            print(f"  {i}. Frame {f['frame']:4d} ({f['time']:6.2f}s): "
                  f"{f['detections']} detections, avg={f['avg_mem']:.3f}")
        
        print(f"\n💡 Try with lower threshold:")
        best_frame = frames_with_detections[0]['frame']
        print(f"   Modify pipeline to use memorability_threshold=0.3")
        print(f"   Then run: python debug_diffusion.py /home/survprivacy/shakti/AMNet/video/traffic.mp4 {best_frame}")
        
        return best_frame
    
    else:
        print(f"❌ NO DETECTIONS FOUND in any scanned frames!")
        print(f"\nPossible issues:")
        print(f"  1. YOLO model not detecting car parts in this video")
        print(f"  2. Model trained on different type of images")
        print(f"  3. Confidence threshold too high")
        
        print(f"\n💡 Solutions:")
        print(f"  1. Check YOLO model classes: python -c 'from ultralytics import YOLO; m=YOLO(\"models/best.pt\"); print(m.names)'")
        print(f"  2. Lower confidence threshold in pipeline (current: 0.5)")
        print(f"  3. Test on your training images first")
        
        return None


if __name__ == "__main__":
    import sys
    
    video_path = "/home/survprivacy/shakti/AMNet/video/traffic.mp4"
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    
    best_frame = scan_video_for_detections(
        video_path, 
        max_frames=200,
        sample_every=10  # Check every 10th frame for speed
    )
    
    if best_frame is not None:
        print(f"\n✅ Best frame found: {best_frame}")
        print(f"\nNext step:")
        print(f"python debug_diffusion.py {video_path} {best_frame}")
    else:
        print(f"\n❌ No suitable frames found. Check YOLO model.")