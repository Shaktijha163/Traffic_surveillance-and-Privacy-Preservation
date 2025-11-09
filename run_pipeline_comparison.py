"""
Improved pipeline that saves comparison images to verify diffusion edits
"""

import cv2
import numpy as np
from pathlib import Path
from amnet_integration import IntegratedMemorabilityPipeline


def process_single_image_with_comparison(image_path, output_dir):
    """
    Process image and save 3 versions for comparison:
    1. Original with annotations
    2. Edited (diffusion applied, no annotations)
    3. Edited with annotations
    """
    
    print(f"\n{'='*70}")
    print(f"Processing: {image_path}")
    print(f"{'='*70}\n")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline
    pipeline = IntegratedMemorabilityPipeline(
        yolo_model_path="models/best.pt",
        amnet_model_path="models/amnet_weights.pkl",
        device="cuda",
        conf_threshold=0.5,
        memorability_threshold=0.6,
        attention_maps=True
    )
    
    # Load image
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"❌ Error: Could not load image {image_path}")
        return
    
    img_name = Path(image_path).stem
    
    # ========================================================================
    # Step 1: Process WITHOUT perturbation (just detection + visualization)
    # ========================================================================
    print("📊 Step 1: Analyzing memorability (no edits)...")
    original_annotated, stats_original = pipeline.process_frame(
        frame.copy(),
        frame_idx=0,
        apply_perturbation=False,  # No diffusion
        visualize=True              # Show annotations
    )
    
    output_path_1 = output_dir / f"{img_name}_1_original_annotated.jpg"
    cv2.imwrite(str(output_path_1), original_annotated)
    print(f"✓ Saved: {output_path_1}")
    
    # ========================================================================
    # Step 2: Process WITH perturbation, NO visualization
    # ========================================================================
    print("\n🎨 Step 2: Applying diffusion edits...")
    edited_clean, stats_edited = pipeline.process_frame(
        frame.copy(),
        frame_idx=0,
        apply_perturbation=True,   # Apply diffusion
        visualize=False             # No annotations - see pure edit
    )
    
    output_path_2 = output_dir / f"{img_name}_2_edited_clean.jpg"
    cv2.imwrite(str(output_path_2), edited_clean)
    print(f"✓ Saved: {output_path_2}")
    
    # ========================================================================
    # Step 3: Process WITH perturbation AND visualization
    # ========================================================================
    print("\n📊 Step 3: Applying diffusion + annotations...")
    edited_annotated, stats_final = pipeline.process_frame(
        frame.copy(),
        frame_idx=0,
        apply_perturbation=True,   # Apply diffusion
        visualize=True              # Show annotations
    )
    
    output_path_3 = output_dir / f"{img_name}_3_edited_annotated.jpg"
    cv2.imwrite(str(output_path_3), edited_annotated)
    print(f"✓ Saved: {output_path_3}")
    
    # ========================================================================
    # Step 4: Create side-by-side comparison
    # ========================================================================
    print("\n🖼️  Step 4: Creating comparison image...")
    comparison = create_comparison_image(
        frame, 
        original_annotated, 
        edited_clean, 
        edited_annotated,
        stats_final
    )
    
    output_path_comparison = output_dir / f"{img_name}_COMPARISON.jpg"
    cv2.imwrite(str(output_path_comparison), comparison)
    print(f"✓ Saved: {output_path_comparison}")
    
    # ========================================================================
    # Print Statistics
    # ========================================================================
    print(f"\n📊 Statistics:")
    print(f"  Total detections: {stats_final['total_detections']}")
    print(f"  High memorability parts: {stats_final['high_memorability_count']}")
    print(f"  Average memorability: {stats_final['avg_memorability']:.3f}")
    print(f"  Edits applied: {stats_final['edits_applied']}")
    
    if stats_final['edits_applied'] > 0:
        print(f"\n  Edit Details:")
        for edit in stats_final['edit_details']:
            print(f"    - {edit['class_name']} (M:{edit['mem_score']:.2f}): {edit['method']}")
    
    print(f"\n✅ Results saved to: {output_dir}/")
    print(f"\n💡 Check these files:")
    print(f"   1. {img_name}_1_original_annotated.jpg  <- Original with boxes")
    print(f"   2. {img_name}_2_edited_clean.jpg        <- PURE DIFFUSION RESULT ⭐")
    print(f"   3. {img_name}_3_edited_annotated.jpg    <- Edited with boxes")
    print(f"   4. {img_name}_COMPARISON.jpg            <- Side-by-side view ⭐⭐⭐")
    
    return edited_clean, stats_final


def create_comparison_image(original, original_annotated, edited_clean, edited_annotated, stats):
    """Create a 2x2 grid comparison image"""
    
    # Resize all images to same size for comparison
    h, w = original.shape[:2]
    target_h, target_w = 600, 800
    
    def resize_img(img):
        return cv2.resize(img, (target_w, target_h))
    
    img1 = resize_img(original)
    img2 = resize_img(original_annotated)
    img3 = resize_img(edited_clean)
    img4 = resize_img(edited_annotated)
    
    # Add text labels
    def add_label(img, text, color=(255, 255, 255)):
        img_copy = img.copy()
        cv2.rectangle(img_copy, (0, 0), (target_w, 50), (0, 0, 0), -1)
        cv2.putText(img_copy, text, (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        return img_copy
    
    img1 = add_label(img1, "1. ORIGINAL", (255, 255, 255))
    img2 = add_label(img2, "2. ORIGINAL + DETECTIONS", (0, 255, 255))
    img3 = add_label(img3, "3. DIFFUSION APPLIED", (0, 255, 0))
    img4 = add_label(img4, "4. DIFFUSION + DETECTIONS", (0, 255, 255))
    
    # Create 2x2 grid
    top_row = np.hstack([img1, img2])
    bottom_row = np.hstack([img3, img4])
    grid = np.vstack([top_row, bottom_row])
    
    # Add statistics panel at the bottom
    stats_panel_h = 120
    stats_panel = np.zeros((stats_panel_h, grid.shape[1], 3), dtype=np.uint8)
    
    y_offset = 30
    cv2.putText(stats_panel, f"Detections: {stats['total_detections']} | High Memorability: {stats['high_memorability_count']} | Edits: {stats['edits_applied']}", 
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    y_offset += 35
    cv2.putText(stats_panel, f"Average Memorability: {stats['avg_memorability']:.3f}", 
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Add instruction
    y_offset += 35
    cv2.putText(stats_panel, "COMPARE: Image 1 vs Image 3 to see diffusion changes!", 
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Combine
    final = np.vstack([grid, stats_panel])
    
    return final


def process_folder_with_comparison(input_folder, output_folder):
    """Process all images in folder with comparison outputs"""
    
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
    
    print(f"\n{'='*70}")
    print(f"Found {len(image_files)} images to process")
    print(f"{'='*70}\n")
    
    for idx, img_path in enumerate(image_files):
        print(f"\n{'='*70}")
        print(f"[{idx+1}/{len(image_files)}] Processing: {img_path.name}")
        print(f"{'='*70}")
        
        # Create subfolder for this image
        img_output_dir = output_path / img_path.stem
        
        process_single_image_with_comparison(
            image_path=str(img_path),
            output_dir=str(img_output_dir)
        )
    
    print(f"\n{'='*70}")
    print(f"✅ Batch processing complete!")
    print(f"   Results in: {output_folder}/")
    print(f"{'='*70}\n")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("Memorability Reduction Pipeline - Comparison Mode")
    print("="*70)
    
    # Example 1: Process single image with comparison
    print("\n📸 Example 1: Single Image with Comparison")
    print("-" * 70)
    
    if Path("test_images/car1.jpg").exists():
        process_single_image_with_comparison(
            image_path="test_images/car1.jpg",
            output_dir="results/comparison_output"
        )
    else:
        print("⚠ test_images/car1.jpg not found.")
        print("  Place your test image and update the path.")
    
    # Example 2: Process entire folder
    print("\n\n📁 Example 2: Process Entire Folder with Comparisons")
    print("-" * 70)
    
    if Path("test_images").exists():
        process_folder_with_comparison(
            input_folder="test_images",
            output_folder="results/batch_comparison"
        )
    else:
        print("⚠ test_images/ folder not found.")
    
    print("\n" + "="*70)
    print("✨ All examples complete!")
    print("="*70 + "\n")
    
    print("💡 Check the results:")
    print("   results/comparison_output/        <- Single image results")
    print("   results/batch_comparison/         <- Batch results")
    print("\n   Look for *_COMPARISON.jpg files for best visualization!")
    print()