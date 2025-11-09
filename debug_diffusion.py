"""
Debug script to verify diffusion is working on a single frame
"""

import cv2
import numpy as np
from pathlib import Path
from amnet_integration import IntegratedMemorabilityPipeline


def debug_single_frame(video_path: str, frame_number: int = 0):
    """
    Extract and process a single frame to debug diffusion
    """
    
    print(f"\n{'='*80}")
    print(f"DEBUG: Testing Diffusion on Single Frame")
    print(f"{'='*80}\n")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    # Skip to target frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"❌ Cannot read frame {frame_number}")
        return
    
    print(f"✓ Loaded frame {frame_number} from video")
    print(f"  Frame shape: {frame.shape}\n")
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = IntegratedMemorabilityPipeline(
        yolo_model_path="models/best.pt",
        amnet_model_path="models/amnet_weights.pkl",
        device="cuda",
        memorability_threshold=0.6,
        attention_maps=True
    )
    print("✓ Pipeline initialized\n")
    
    # Save original
    Path("debug_output").mkdir(exist_ok=True)
    cv2.imwrite("debug_output/0_original.jpg", frame)
    print("✓ Saved: debug_output/0_original.jpg")
    
    # Step 1: Process WITHOUT perturbation
    print("\n" + "="*80)
    print("STEP 1: Processing WITHOUT perturbation (detection only)")
    print("="*80 + "\n")
    
    result_no_perturb, stats_no_perturb = pipeline.process_frame(
        frame.copy(),
        frame_idx=0,
        apply_perturbation=False,
        visualize=True
    )
    
    cv2.imwrite("debug_output/1_no_perturbation_annotated.jpg", result_no_perturb)
    print(f"✓ Saved: debug_output/1_no_perturbation_annotated.jpg")
    
    print(f"\n📊 Detection Stats:")
    print(f"  Total detections: {stats_no_perturb['total_detections']}")
    print(f"  High memorability: {stats_no_perturb['high_memorability_count']}")
    print(f"  Avg memorability: {stats_no_perturb['avg_memorability']:.3f}")
    
    # Step 2: Process WITH perturbation (no visualization)
    print("\n" + "="*80)
    print("STEP 2: Processing WITH perturbation (diffusion, no annotations)")
    print("="*80 + "\n")
    
    result_perturb_clean, stats_perturb = pipeline.process_frame(
        frame.copy(),
        frame_idx=0,
        apply_perturbation=True,
        visualize=False  # No annotations to see pure diffusion
    )
    
    cv2.imwrite("debug_output/2_perturbed_clean.jpg", result_perturb_clean)
    print(f"✓ Saved: debug_output/2_perturbed_clean.jpg")
    
    print(f"\n📊 Perturbation Stats:")
    print(f"  Edits applied: {stats_perturb['edits_applied']}")
    
    if stats_perturb['edits_applied'] > 0:
        print(f"\n  Edit Details:")
        for edit in stats_perturb['edit_details']:
            print(f"    - Class: {edit['class_name']}")
            print(f"      Memorability: {edit['mem_score']:.3f}")
            print(f"      Method: {edit['method']}")  # This should say "diffusion" not "fallback"
            print(f"      BBox: {edit['bbox']}")
    else:
        print(f"  ⚠️ WARNING: No edits were applied!")
        print(f"     This means either:")
        print(f"     1. No detections had memorability > 0.6")
        print(f"     2. Diffusion is not running properly")
    
    # Step 3: Process WITH perturbation (with visualization)
    print("\n" + "="*80)
    print("STEP 3: Processing WITH perturbation (diffusion + annotations)")
    print("="*80 + "\n")
    
    result_perturb_annotated, _ = pipeline.process_frame(
        frame.copy(),
        frame_idx=0,
        apply_perturbation=True,
        visualize=True
    )
    
    cv2.imwrite("debug_output/3_perturbed_annotated.jpg", result_perturb_annotated)
    print(f"✓ Saved: debug_output/3_perturbed_annotated.jpg")
    
    # Step 4: Create comparison
    print("\n" + "="*80)
    print("STEP 4: Creating comparison images")
    print("="*80 + "\n")
    
    # Side-by-side: Original vs Perturbed (clean)
    comparison_clean = np.hstack([frame, result_perturb_clean])
    cv2.putText(comparison_clean, "ORIGINAL", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
    cv2.putText(comparison_clean, "PERTURBED (Diffusion)", 
                (frame.shape[1] + 10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    
    cv2.imwrite("debug_output/4_comparison_clean.jpg", comparison_clean)
    print(f"✓ Saved: debug_output/4_comparison_clean.jpg")
    
    # Side-by-side: Original annotated vs Perturbed annotated
    comparison_annotated = np.hstack([result_no_perturb, result_perturb_annotated])
    cv2.putText(comparison_annotated, "ORIGINAL (Annotated)", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
    cv2.putText(comparison_annotated, "PERTURBED (Annotated)", 
                (frame.shape[1] + 10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    
    cv2.imwrite("debug_output/5_comparison_annotated.jpg", comparison_annotated)
    print(f"✓ Saved: debug_output/5_comparison_annotated.jpg")
    
    # Calculate pixel difference
    diff = cv2.absdiff(frame, result_perturb_clean)
    diff_sum = np.sum(diff)
    
    print(f"\n📊 Pixel Difference Analysis:")
    print(f"  Total pixel difference: {diff_sum:,}")
    
    if diff_sum < 1000:
        print(f"  ⚠️ WARNING: Images are nearly identical!")
        print(f"     Diffusion may not be working properly.")
    else:
        print(f"  ✓ Images are different - diffusion appears to be working!")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    print("📁 Output files in debug_output/:")
    print("  0_original.jpg                  - Original frame")
    print("  1_no_perturbation_annotated.jpg - Detections only")
    print("  2_perturbed_clean.jpg           - Pure diffusion (no boxes)")
    print("  3_perturbed_annotated.jpg       - Diffusion + boxes")
    print("  4_comparison_clean.jpg          - Side-by-side (clean) ⭐⭐⭐")
    print("  5_comparison_annotated.jpg      - Side-by-side (annotated)")
    
    print("\n💡 What to check:")
    print("  1. Compare 0_original.jpg vs 2_perturbed_clean.jpg")
    print("  2. Check if edit method shows 'diffusion' (not 'fallback')")
    print("  3. Look at 4_comparison_clean.jpg for visual differences")
    
    if stats_perturb['edits_applied'] == 0:
        print("\n⚠️ ISSUE: No edits were applied!")
        print("   Try lowering memorability_threshold to 0.5 or 0.4")
    elif diff_sum < 1000:
        print("\n⚠️ ISSUE: Edits applied but no visual change!")
        print("   Check if diffusion model is loading correctly")
    else:
        print("\n✅ Diffusion appears to be working!")
    
    print()


if __name__ == "__main__":
    import sys
    
    # Default video path
    video_path = "/home/survprivacy/shakti/AMNet/video/traffic.mp4"
    frame_num = 50  # Frame with the red truck
    
    # Allow command line override
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    if len(sys.argv) > 2:
        frame_num = int(sys.argv[2])
    
    print(f"\n🔍 Debugging frame {frame_num} from: {video_path}")
    debug_single_frame(video_path, frame_num)