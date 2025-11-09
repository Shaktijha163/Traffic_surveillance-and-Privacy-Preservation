"""
amnet_integration.py (COMPLETE FIXED VERSION - Full Temporal Consistency)
AMNet Integration with YOLO Detection Pipeline
Predicts memorability of detected car parts and applies diffusion-based perturbation

FIXED:
- Stores diffused crops per (track_id, class_name)
- Reapplies cached edits every frame (paste stored crop back)
- Pass track_id to diffusion editor
- Accumulate edits on working frame buffer
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import torch.nn.functional as F

from object_detector import CarPartsDetector
from object_tracker import CarPartsTracker
from amnet import AMNet
from config import get_config, get_memorability_category, get_perturbation_method, MEMORABILITY_THRESHOLD_HIGH

from diffusion_editor import DiffusionEditor


# ============================
# Memorability Analyzer
# ============================
class MemorabilityAnalyzer:
    """
    Analyzes memorability of detected car parts using AMNet
    """
    
    def __init__(self, 
                 amnet_model_path: str,
                 device: str = "cuda",
                 attention_maps: bool = True):
        """
        Initialize AMNet memorability analyzer
        
        Args:
            amnet_model_path: Path to AMNet weights (.pkl file)
            device: 'cuda' or 'cpu'
            attention_maps: Enable attention map visualization
        """
        print(f"🧠 Initializing AMNet Memorability Analyzer...")
        
        self.device = device
        self.attention_maps_enabled = attention_maps
        
        # Initialize AMNet
        self.amnet = AMNet()
        
        # Get AMNet hyperparameters
        hps = get_config()
        hps.use_cuda = (device == "cuda")
        hps.cuda_device = 0
        hps.model_weights = amnet_model_path
        hps.use_attention = attention_maps  # Enable/disable attention
        
        self.amnet.init(hps)
        self.amnet.model.eval()
        
        print(f"✓ AMNet loaded from: {amnet_model_path}")
        print(f"  Attention maps: {'ON' if attention_maps else 'OFF'}")
        print(f"  Device: {device}\n")
    
    
    def predict_memorability_crops(self, 
                                   frame: np.ndarray,
                                   detections: List[Dict]) -> Dict[int, Dict]:
        """
        Predict memorability for each detected object (crops)
        
        Args:
            frame: Original BGR frame
            detections: List of detections from YOLO
            
        Returns:
            Dictionary mapping detection index to memorability info:
            {
                0: {
                    'memorability_score': float (0-1),
                    'attention_map': np.ndarray if enabled,
                    'bbox': [x1, y1, x2, y2],
                    'class_name': str
                },
                ...
            }
        """
        if len(detections) == 0:
            return {}
        
        # Extract crops for each detection
        crops = []
        crop_info = []
        
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            
            # Add padding to crops (10%)
            h, w = y2 - y1, x2 - x1
            pad_h, pad_w = int(h * 0.1), int(w * 0.1)
            
            x1_pad = max(0, x1 - pad_w)
            y1_pad = max(0, y1 - pad_h)
            x2_pad = min(frame.shape[1], x2 + pad_w)
            y2_pad = min(frame.shape[0], y2 + pad_h)
            
            # Extract crop
            crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if crop.size == 0:
                continue
            
            # Convert BGR to RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(Image.fromarray(crop_rgb))
            
            crop_info.append({
                'det_idx': idx,
                'bbox': det['bbox'],
                'class_name': det['class_name']
            })
        
        if len(crops) == 0:
            return {}
        
        # Predict memorability for all crops
        pr = self.amnet.predict_memorability_image_batch(crops)
        
        # Build result dictionary
        results = {}
        
        for i, info in enumerate(crop_info):
            det_idx = info['det_idx']
            
            result = {
                'memorability_score': float(pr.predictions[i]),
                'bbox': info['bbox'],
                'class_name': info['class_name']
            }
            
            # Add attention map if enabled
            if self.attention_maps_enabled and pr.attention_masks is not None:
                attention_map = pr.attention_masks[i]
                result['attention_map'] = attention_map
            
            results[det_idx] = result
        
        return results
    
    
    def predict_memorability_full_frame(self, frame: np.ndarray) -> Dict:
        """
        Predict memorability for entire frame
        
        Args:
            frame: BGR image
            
        Returns:
            {
                'memorability_score': float,
                'attention_map': np.ndarray (if enabled)
            }
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        
        # Predict
        pr = self.amnet.predict_memorability_image_batch([img_pil])
        
        result = {
            'memorability_score': float(pr.predictions[0])
        }
        
        # Add attention map if enabled
        if self.attention_maps_enabled and pr.attention_masks is not None:
            result['attention_map'] = pr.attention_masks[0]
        
        return result
    
    
    def visualize_attention_on_detection(self,
                                        frame: np.ndarray,
                                        detection: Dict,
                                        attention_map: np.ndarray,
                                        alpha: float = 0.5) -> np.ndarray:
        """
        Overlay attention map on detected region
        
        Args:
            frame: Original frame
            detection: Detection dict with bbox
            attention_map: Attention map from AMNet [seq_len, spatial_locations]
            alpha: Blend factor
            
        Returns:
            Frame with attention overlay
        """
        x1, y1, x2, y2 = detection['bbox']
        w, h = x2 - x1, y2 - y1
        
        # Take last step attention (most refined)
        att = attention_map[-1]  # [spatial_locations]
        
        # Reshape to spatial grid
        ares = int(np.sqrt(att.shape[0]))
        att = att.reshape((ares, ares))
        
        # Normalize
        att = (att - att.min()) / (att.max() - att.min() + 1e-8)
        att = (att * 255).astype(np.uint8)
        
        # Resize to bbox size
        att_resized = cv2.resize(att, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(att_resized, cv2.COLORMAP_JET)
        
        # Blend with original region
        roi = frame[y1:y2, x1:x2]
        blended = cv2.addWeighted(roi, alpha, heatmap, 1-alpha, 0)
        
        # Copy back
        result = frame.copy()
        result[y1:y2, x1:x2] = blended
        
        return result


# ============================
# Integrated Pipeline (FULLY FIXED)
# ============================
class IntegratedMemorabilityPipeline:
    """
    Complete pipeline: Detection -> Tracking -> Memorability -> Perturbation
    
    FIXED: Proper temporal consistency with crop caching and reapplication
    """
    
    def __init__(self,
                 yolo_model_path: str = "models/best.pt",
                 amnet_model_path: str = "models/amnet_weights.pkl",
                 diffusion_model_id: str = "stabilityai/stable-diffusion-2-inpainting",
                 device: str = "cuda",
                 conf_threshold: float = 0.5,
                 memorability_threshold: float = 0.6,
                 attention_maps: bool = True,
                 guidance_scale: float = 7.5,
                 num_inference_steps: int = 25):
        """
        Initialize integrated pipeline
        
        Args:
            yolo_model_path: Path to YOLO weights
            amnet_model_path: Path to AMNet weights
            diffusion_model_id: Hugging Face model ID for diffusion inpainting
            device: 'cuda' or 'cpu'
            conf_threshold: YOLO confidence threshold
            memorability_threshold: Threshold for high memorability
            attention_maps: Enable attention visualization
            guidance_scale: CFG scale for diffusion
            num_inference_steps: Number of diffusion steps
        """
        print("\n" + "="*70)
        print("Integrated Memorability Reduction Pipeline (FIXED VERSION)")
        print("="*70 + "\n")
        
        # Initialize detector
        self.detector = CarPartsDetector(
            model_path=yolo_model_path,
            device=device,
            conf_threshold=conf_threshold
        )
        
        # Initialize tracker
        self.tracker = CarPartsTracker(device=device)
        
        # Initialize memorability analyzer
        self.mem_analyzer = MemorabilityAnalyzer(
            amnet_model_path=amnet_model_path,
            device=device,
            attention_maps=attention_maps
        )
        
        # Initialize diffusion editor (from diffusion_editor.py)
        self.diffusion_editor = DiffusionEditor(
            model_id=diffusion_model_id,
            device=device,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            resize_to=None,  # Optional: resize to 512 for faster processing
            enable_caching=True  # ENABLE CACHING
        )
        
        self.memorability_threshold = memorability_threshold
        self.attention_maps = attention_maps
        
        # ============================================================
        # FIXED: Store actual diffused crops, not just flags
        # ============================================================
        self.edit_cache = {}  # {(track_id, class_name): {
                               #     'crop': np.ndarray,
                               #     'bbox_size': (w, h),
                               #     'method': str,
                               #     'mem_score': float
                               # }}
        
        print("✓ Pipeline initialized successfully!")
        print("✓ Temporal consistency with crop caching ENABLED\n")
    
    
    def process_frame(self,
                     frame: np.ndarray,
                     frame_idx: int,
                     apply_perturbation: bool = True,
                     visualize: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Process single frame: detect -> track -> analyze -> perturb
        
        FIXED: Properly maintains temporal consistency by:
        1. Reapplying all cached edits first
        2. Only running diffusion on NEW high-memorability detections
        3. Storing diffused crops for future reuse
        
        Args:
            frame: BGR image
            frame_idx: Frame number
            apply_perturbation: Apply perturbations to memorable regions
            visualize: Add visualizations
            
        Returns:
            (processed_frame, statistics)
        """
        # 1. Detect objects
        detections = self.detector.detect_frame(frame)
        
        # 2. Track objects
        tracks = self.tracker.update(frame, detections, frame_idx)
        
        # 3. Match detections with tracks (to get track_id for each detection)
        detection_to_track = self._match_detections_to_tracks(detections, tracks)
        
        # 4. Start with original frame
        result_frame = frame.copy()
        
        # ============================================================
        # FIX STEP 1: Reapply ALL cached edits first
        # This ensures previously diffused regions stay diffused
        # ============================================================
        cached_count = 0
        for track in tracks:
            track_id = track['track_id']
            class_name = track['class_name']
            cache_key = (track_id, class_name)
            
            if cache_key in self.edit_cache:
                # Get cached diffused crop
                cached = self.edit_cache[cache_key]
                cached_crop = cached['crop']
                
                # Get current bbox
                x1, y1, x2, y2 = track['bbox']
                w, h = x2 - x1, y2 - y1
                
                # Ensure bbox is valid
                if w <= 0 or h <= 0:
                    continue
                
                # Resize cached crop to current bbox size (handles slight bbox changes)
                if cached_crop.shape[:2] != (h, w):
                    cached_crop_resized = cv2.resize(
                        cached_crop, 
                        (w, h), 
                        interpolation=cv2.INTER_LINEAR
                    )
                else:
                    cached_crop_resized = cached_crop
                
                # Paste cached diffused crop back into frame
                try:
                    result_frame[y1:y2, x1:x2] = cached_crop_resized
                    cached_count += 1
                except Exception as e:
                    print(f"  Warning: Could not paste cached crop for T{track_id}: {e}")
                    continue
        
        if cached_count > 0:
            print(f"  Reapplied {cached_count} cached edits")
        
        # 5. Analyze memorability (for all detections)
        mem_results = self.mem_analyzer.predict_memorability_crops(frame, detections)
        
        # ============================================================
        # FIX STEP 2: Find NEW high-memorability detections (not yet cached)
        # ============================================================
        new_high_mem = []
        for det_idx, mem_info in mem_results.items():
            # Check if memorability is high
            if mem_info['memorability_score'] <= self.memorability_threshold:
                continue
            
            # Get track_id for this detection
            track_id = detection_to_track.get(det_idx)
            if track_id is None:
                continue  # Can't cache without track_id
            
            det = detections[det_idx]
            class_name = det['class_name']
            cache_key = (track_id, class_name)
            
            # Skip if already cached
            if cache_key in self.edit_cache:
                continue  # Already processed - don't re-edit
            
            # This is a NEW high-memorability detection
            new_high_mem.append({
                'det_idx': det_idx,
                'track_id': track_id,
                'class_name': class_name,
                'mem_info': mem_info,
                'bbox': det['bbox']
            })
        
        # ============================================================
        # FIX STEP 3: Run diffusion on NEW high-mem detections + cache result
        # ============================================================
        edits = []
        
        if apply_perturbation and len(new_high_mem) > 0:
            print(f"  Found {len(new_high_mem)} new high-memorability detections")
            
            for item in new_high_mem:
                det_idx = item['det_idx']
                track_id = item['track_id']
                class_name = item['class_name']
                bbox = item['bbox']
                x1, y1, x2, y2 = bbox
                
                # Validate bbox
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Create single-detection list for diffusion editor
                single_det = [detections[det_idx]]
                single_mem = {0: item['mem_info']}
                
                # Run diffusion on result_frame (which may already have other edits)
                try:
                    edited_frame, edit_info = self.diffusion_editor.edit_frame(
                        frame_bgr=result_frame,
                        detections=single_det,
                        mem_results=single_mem,
                        mem_threshold=self.memorability_threshold,
                        process_entire_bbox=True,
                        pad=2
                    )
                    
                    if len(edit_info) > 0:
                        # Update working frame
                        result_frame = edited_frame
                        
                        # Extract the diffused crop from the edited frame
                        diffused_crop = result_frame[y1:y2, x1:x2].copy()
                        
                        # Cache the diffused crop for future frames
                        cache_key = (track_id, class_name)
                        self.edit_cache[cache_key] = {
                            'crop': diffused_crop,
                            'bbox_size': (x2 - x1, y2 - y1),
                            'method': edit_info[0]['method'],
                            'mem_score': item['mem_info']['memorability_score']
                        }
                        
                        # Record edit
                        edit = edit_info[0]
                        edit['track_id'] = track_id
                        edits.append(edit)
                        
                        print(f"    [NEW] T{track_id} {class_name} "
                              f"(mem={item['mem_info']['memorability_score']:.3f}) "
                              f"- {edit_info[0]['method']}")
                
                except Exception as e:
                    print(f"  Error processing T{track_id} {class_name}: {e}")
                    continue
        
        # 6. Visualize (if enabled)
        if visualize:
            result_frame = self._visualize_results(
                result_frame,
                detections,
                tracks,
                mem_results,
                new_high_mem,
                edits
            )
        
        # 7. Statistics
        stats = {
            'total_detections': len(detections),
            'total_tracks': len(tracks),
            'high_memorability_count': len(new_high_mem),
            'avg_memorability': np.mean([m['memorability_score'] 
                                        for m in mem_results.values()]) if mem_results else 0,
            'cached_edits_applied': cached_count,
            'new_edits': len(edits),
            'edits_applied': len(edits),  # For backward compatibility with video_pipeline.py
            'cache_size': len(self.edit_cache),
            'edit_details': edits
        }
        
        return result_frame, stats
    
    
    def _match_detections_to_tracks(self, 
                                   detections: List[Dict], 
                                   tracks: List[Dict]) -> Dict[int, int]:
        """
        Match each detection to its track_id using IoU
        
        Args:
            detections: List of detections
            tracks: List of tracks
            
        Returns:
            Dictionary: {det_idx: track_id}
        """
        detection_to_track = {}
        
        for det_idx, det in enumerate(detections):
            det_bbox = det['bbox']
            best_iou = 0
            best_track_id = None
            
            for track in tracks:
                track_bbox = track['bbox']
                iou = self._compute_iou(det_bbox, track_bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track['track_id']
            
            if best_iou > 0.5:  # Threshold for matching
                detection_to_track[det_idx] = best_track_id
        
        return detection_to_track
    
    
    def _compute_iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    
    def _visualize_results(self,
                          frame: np.ndarray,
                          detections: List[Dict],
                          tracks: List[Dict],
                          mem_results: Dict,
                          high_mem_detections: List,
                          edits: List[Dict]) -> np.ndarray:
        """Add visualization overlays"""
        
        result = frame.copy()
        
        # Draw detections with memorability scores
        for det_idx, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            
            # Get memorability
            mem_score = mem_results.get(det_idx, {}).get('memorability_score', 0)
            
            # Color based on memorability (green=low, red=high)
            color_val = int(mem_score * 255)
            color = (0, 255 - color_val, color_val)  # BGR
            
            # Draw bbox
            thickness = 3 if mem_score > self.memorability_threshold else 2
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            
            # Label
            label = f"{det['class_name']} M:{mem_score:.2f}"
            
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                result,
                (x1, y1 - text_h - 10),
                (x1 + text_w, y1),
                color,
                -1
            )
            cv2.putText(
                result,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
            
            # Draw attention map if available and high memorability
            if self.attention_maps and mem_score > self.memorability_threshold:
                att_map = mem_results[det_idx].get('attention_map')
                if att_map is not None:
                    result = self.mem_analyzer.visualize_attention_on_detection(
                        result, det, att_map, alpha=0.6
                    )
        
        # Draw track IDs
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            track_id = track['track_id']
            
            # Draw track ID in top-right corner of bbox
            track_label = f"T{track_id}"
            cv2.putText(
                result,
                track_label,
                (x2 - 40, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )
        
        return result
    
    
    def clear_cache(self):
        """Clear edit cache"""
        self.edit_cache.clear()
        print("✓ Cleared edit cache")
    
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        if len(self.edit_cache) == 0:
            return {
                'cache_size': 0,
                'cached_tracks': 0,
                'cached_part_types': 0
            }
        
        cached_tracks = set(k[0] for k in self.edit_cache.keys())
        cached_parts = set(k[1] for k in self.edit_cache.keys())
        
        return {
            'cache_size': len(self.edit_cache),
            'cached_tracks': len(cached_tracks),
            'cached_part_types': len(cached_parts)
        }


# ============================================================================
# Testing & Demo
# ============================================================================

def test_memorability_integration():
    """Test the integrated pipeline"""
    
    print("\n" + "="*70)
    print("Testing Memorability Integration (FIXED VERSION)")
    print("="*70 + "\n")
    
    # Initialize pipeline
    pipeline = IntegratedMemorabilityPipeline(
        yolo_model_path="models/best.pt",
        amnet_model_path="models/amnet_weights.pkl",
        diffusion_model_id="stabilityai/stable-diffusion-2-inpainting",
        device="cuda",
        memorability_threshold=0.6,
        attention_maps=True,
        guidance_scale=7.5,
        num_inference_steps=25
    )
    
    # Test on images
    test_dir = Path("test_images")
    
    if test_dir.exists():
        images = sorted(list(test_dir.glob("*.jpg")) + 
                       list(test_dir.glob("*.png")) + 
                       list(test_dir.glob("*.webp")))
        
        if len(images) > 0:
            for idx, img_path in enumerate(images[:5]):  # First 5
                print(f"\n{'='*60}")
                print(f"Processing: {img_path.name}")
                print('='*60)
                
                frame = cv2.imread(str(img_path))
                
                if frame is None:
                    continue
                
                # Process with perturbation
                result, stats = pipeline.process_frame(
                    frame,
                    frame_idx=idx,
                    apply_perturbation=True,
                    visualize=True
                )
                
                print(f"\n📊 Frame Statistics:")
                print(f"  Detections: {stats['total_detections']}")
                print(f"  Tracks: {stats['total_tracks']}")
                print(f"  High memorability: {stats['high_memorability_count']}")
                print(f"  Avg memorability: {stats['avg_memorability']:.3f}")
                print(f"  Cached edits applied: {stats['cached_edits_applied']}")
                print(f"  New edits: {stats['new_edits']}")
                print(f"  Cache size: {stats['cache_size']}")
                
                # Save
                output_path = f"results/memorability_test_{idx}.jpg"
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(output_path, result)
                print(f"\n✓ Saved: {output_path}")
            
            # Print final cache stats
            cache_stats = pipeline.get_cache_stats()
            print(f"\n{'='*60}")
            print("📦 Final Cache Statistics:")
            print('='*60)
            print(f"  Total cached entries: {cache_stats['cache_size']}")
            print(f"  Unique tracks cached: {cache_stats['cached_tracks']}")
            print(f"  Part types cached: {cache_stats['cached_part_types']}")
            
            print("\n✓ Test complete!")
            return True
    
    print("⚠ No test images found in 'test_images/' directory")
    return False


if __name__ == "__main__":
    test_memorability_integration()