"""
Example script to run the memorability reduction pipeline
"""

import cv2
from pathlib import Path
from amnet_integration import IntegratedMemorabilityPipeline


def process_single_image(image_path, output_path):
    """Process a single image"""
    
    print(f"\n{'='*70}")
    print(f"Processing: {image_path}")
    print(f"{'='*70}\n")
    
    # Initialize pipeline
    pipeline = IntegratedMemorabilityPipeline(
        yolo_model_path="models/best.pt",
        amnet_model_path="models/amnet_weights.pkl",
        device="cuda",  # or "cpu" if no GPU
        conf_threshold=0.5,  # YOLO confidence threshold
        memorability_threshold=0.6,  # High memorability threshold
        attention_maps=True  # Enable attention visualization
    )
    
    # Load image
    frame = cv2.imread(str(image_path))
    
    if frame is None:
        print(f"❌ Error: Could not load image {image_path}")
        return
    
    # Process frame
    result_frame, stats = pipeline.process_frame(
        frame,
        frame_idx=0,
        apply_perturbation=True,  # Apply privacy protection
        visualize=True  # Show bounding boxes and scores
    )
    
    # Print statistics
    print(f"\n📊 Statistics:")
    print(f"  Total detections: {stats['total_detections']}")
    print(f"  Tracked objects: {stats['total_tracks']}")
    print(f"  High memorability parts: {stats['high_memorability_count']}")
    print(f"  Average memorability: {stats['avg_memorability']:.3f}")
    
    # Save result
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result_frame)
    print(f"\n✅ Saved result to: {output_path}\n")
    
    return result_frame, stats


def process_video(video_path, output_path):
    """Process an entire video"""
    
    print(f"\n{'='*70}")
    print(f"Processing Video: {video_path}")
    print(f"{'='*70}\n")
    
    # Initialize pipeline
    pipeline = IntegratedMemorabilityPipeline(
        yolo_model_path="models/best.pt",
        amnet_model_path="models/amnet_weights.pkl",
        device="cuda",
        conf_threshold=0.5,
        memorability_threshold=0.6,
        attention_maps=True
    )
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_idx = 0
    total_high_mem = 0
    
    print(f"\nProcessing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        result_frame, stats = pipeline.process_frame(
            frame,
            frame_idx=frame_idx,
            apply_perturbation=True,
            visualize=True
        )
        
        # Write to output
        out.write(result_frame)
        
        total_high_mem += stats['high_memorability_count']
        
        # Print progress every 30 frames
        if frame_idx % 30 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"  Frame {frame_idx}/{total_frames} ({progress:.1f}%) - "
                  f"High mem parts: {stats['high_memorability_count']}")
        
        frame_idx += 1
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"\n✅ Video processing complete!")
    print(f"  Total frames processed: {frame_idx}")
    print(f"  Total high memorability parts detected: {total_high_mem}")
    print(f"  Output saved to: {output_path}\n")


def process_folder(input_folder, output_folder):
    """Process all images in a folder"""
    
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(ext))
    
    if len(image_files) == 0:
        print(f"❌ No images found in {input_folder}")
        return
    
    print(f"\nFound {len(image_files)} images to process")
    
    # Initialize pipeline once for all images
    pipeline = IntegratedMemorabilityPipeline(
        yolo_model_path="models/best.pt",
        amnet_model_path="models/amnet_weights.pkl",
        device="cuda",
        conf_threshold=0.5,
        memorability_threshold=0.6,
        attention_maps=True
    )
    
    # Process each image
    for idx, img_path in enumerate(image_files):
        print(f"\n[{idx+1}/{len(image_files)}] Processing: {img_path.name}")
        
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  ⚠ Skipping (could not load)")
            continue
        
        # Process
        result_frame, stats = pipeline.process_frame(
            frame,
            frame_idx=idx,
            apply_perturbation=True,
            visualize=True
        )
        
        # Save
        output_file = output_path / f"processed_{img_path.name}"
        cv2.imwrite(str(output_file), result_frame)
        
        print(f"  Detections: {stats['total_detections']}, "
              f"High mem: {stats['high_memorability_count']}, "
              f"Avg: {stats['avg_memorability']:.3f}")
        print(f"  ✓ Saved: {output_file}")
    
    print(f"\n✅ Batch processing complete! Results in: {output_folder}\n")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print("Memorability Reduction Pipeline - Demo")
    print("="*70)
    
    # Example 1: Process single image
    print("\n📸 Example 1: Single Image")
    print("-" * 70)
    
    if Path("test_images/car1.jpg").exists():
        process_single_image(
            image_path="test_images/car1.jpg",
            output_path="results/output_car1.jpg"
        )
    else:
        print("⚠ test_images/car1.jpg not found. Skipping...")
    
    
    # Example 2: Process folder
    print("\n📁 Example 2: Process Entire Folder")
    print("-" * 70)
    
    if Path("test_images").exists():
        process_folder(
            input_folder="test_images",
            output_folder="results/batch_output"
        )
    else:
        print("⚠ test_images/ folder not found. Skipping...")
    
    
    # Example 3: Process video (if you have one)
    print("\n🎬 Example 3: Process Video")
    print("-" * 70)
    
    if Path("test_video.mp4").exists():
        process_video(
            video_path="test_video.mp4",
            output_path="results/output_video.mp4"
        )
    else:
        print("⚠ test_video.mp4 not found. Skipping...")
    
    
    print("\n" + "="*70)
    print("✨ All examples complete!")
    print("="*70 + "\n")
    
    # Show usage instructions
    print("💡 Usage Tips:")
    print("  1. Single image:  python run_pipeline.py --image path/to/image.jpg")
    print("  2. Folder:        python run_pipeline.py --folder path/to/folder/")
    print("  3. Video:         python run_pipeline.py --video path/to/video.mp4")
    print("  4. Run examples:  python run_pipeline.py")
    print()