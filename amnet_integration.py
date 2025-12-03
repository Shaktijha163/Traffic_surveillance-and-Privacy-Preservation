"""
amnet_integration.py (FIXED VERSION - Scale-Aware Temporal Consistency)
AMNet Integration with YOLO Detection Pipeline
Predicts memorability of detected car parts and applies diffusion-based perturbation

FIXES APPLIED:
- Maximum bbox size limit for diffusion processing
- Scale-aware caching (different edits for different distances)
- Size similarity check before reusing cached crops
- Adaptive diffusion strength based on bbox size
- Proportional context padding
- Fallback to lighter methods for very large bboxes
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
# Configuration Constants
# ============================
MAX_BBOX_DIMENSION = 450  # Maximum width or height for diffusion processing
MAX_BBOX_AREA = 50000  # Maximum area for full diffusion (450*450 ~ 200k, being conservative)
CACHE_SIZE_TOLERANCE = 0.25  # 25% size difference tolerance for cache reuse
SCALE_BINS = {
    'small': (0, 10000),      # Area < 10k pixels
    'medium': (10000, 30000), # Area 10k-30k pixels
    'large': (30000, 50000),  # Area 30k-50k pixels
    'xlarge': (50000, float('inf'))  # Area > 50k pixels
}


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
        print(f" Initializing AMNet Memorability Analyzer...")
        
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
# Helper Functions
# ============================
def get_scale_bin(bbox_area: float) -> str:
    """Determine scale bin for bbox area"""
    for bin_name, (min_area, max_area) in SCALE_BINS.items():
        if min_area <= bbox_area < max_area:
            return bin_name
    return 'xlarge'


def compute_size_similarity(size1: Tuple[int, int], size2: Tuple[int, int]) -> float:
    """
    Compute size similarity between two (w, h) tuples
    Returns relative difference (0 = identical, 1 = 100% different)
    """
    w1, h1 = size1
    w2, h2 = size2
    
    if w1 == 0 or h1 == 0 or w2 == 0 or h2 == 0:
        return 1.0
    
    w_diff = abs(w1 - w2) / max(w1, w2)
    h_diff = abs(h1 - h2) / max(h1, h2)
    
    return max(w_diff, h_diff)


# ============================
# Integrated Pipeline (FIXED)
# ============================
class IntegratedMemorabilityPipeline:
    """
    Complete pipeline: Detection -> Tracking -> Memorability -> Perturbation
    
    FIXED: Scale-aware temporal consistency with intelligent caching
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
        print("Integrated Memorability Reduction Pipeline ")
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
            resize_to=512,
            enable_caching=True
        )
        
        self.memorability_threshold = memorability_threshold
        self.attention_maps = attention_maps
        
        # Scale-aware edit cache: {(track_id, class_name, scale_bin): {...}}
        self.edit_cache = {}
        
        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'skipped_large_bbox': 0,
            'fallback_edits': 0
        }
        
        print("✓ Pipeline initialized successfully!")
        print("✓ Scale-aware temporal consistency ENABLED")
        print(f"✓ Max bbox dimension: {MAX_BBOX_DIMENSION}px")
        print(f"✓ Max bbox area: {MAX_BBOX_AREA}px²")
        print(f"✓ Cache tolerance: {CACHE_SIZE_TOLERANCE*100}%\n")
    
    
    def process_frame(self,
                    frame: np.ndarray,
                    frame_idx: int,
                    apply_perturbation: bool = True,
                    visualize: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Process single frame: detect -> track -> analyze -> perturb

        FIXED: Scale-aware caching with size similarity checks

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

        # 3. Match detections with tracks
        detection_to_track = self._match_detections_to_tracks(detections, tracks)

        # 4. Start with original frame
        result_frame = frame.copy()

        # 5. Reapply cached edits (with size similarity check)
        cached_count = 0
        H, W = result_frame.shape[:2]

        for track in tracks:
            track_id = track['track_id']
            class_name = track['class_name']
            x1, y1, x2, y2 = map(int, track['bbox'])
            
            # Clamp bbox
            x1 = max(0, min(W - 1, x1))
            x2 = max(0, min(W, x2))
            y1 = max(0, min(H - 1, y1))
            y2 = max(0, min(H, y2))
            
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            
            # Compute scale bin
            bbox_area = w * h
            scale_bin = get_scale_bin(bbox_area)
            
            # Check cache with scale bin
            cache_key = (track_id, class_name, scale_bin)
            
            if cache_key not in self.edit_cache:
                continue
            
            cached = self.edit_cache[cache_key]
            cached_crop = cached.get('crop')
            cached_size = cached.get('bbox_size')
            
            if cached_crop is None or cached_size is None:
                continue
            
            # Check size similarity
            current_size = (w, h)
            size_diff = compute_size_similarity(current_size, cached_size)
            
            if size_diff > CACHE_SIZE_TOLERANCE:
                # Size too different - skip reuse
                continue
            
            # Paste cached crop
            try:
                src_h, src_w = cached_crop.shape[:2]
                
                # Resize to current bbox size
                if (src_h, src_w) != (h, w):
                    cached_resized = cv2.resize(cached_crop, (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    cached_resized = cached_crop
                
                # Safe paste
                dst_y1, dst_y2 = y1, y2
                dst_x1, dst_x2 = x1, x2
                
                src_y1 = 0
                src_x1 = 0
                src_y2 = src_y1 + (dst_y2 - dst_y1)
                src_x2 = src_x1 + (dst_x2 - dst_x1)
                
                if dst_y1 < 0:
                    src_y1 += -dst_y1
                    dst_y1 = 0
                if dst_x1 < 0:
                    src_x1 += -dst_x1
                    dst_x1 = 0
                if dst_y2 > H:
                    src_y2 -= (dst_y2 - H)
                    dst_y2 = H
                if dst_x2 > W:
                    src_x2 -= (dst_x2 - W)
                    dst_x2 = W
                
                final_h = dst_y2 - dst_y1
                final_w = dst_x2 - dst_x1
                if final_h <= 0 or final_w <= 0:
                    continue
                
                src_region = cached_resized[src_y1:src_y1 + final_h, src_x1:src_x1 + final_w]
                
                if src_region.shape[0] != final_h or src_region.shape[1] != final_w:
                    src_region = cv2.resize(src_region, (final_w, final_h), interpolation=cv2.INTER_LINEAR)
                
                result_frame[dst_y1:dst_y2, dst_x1:dst_x2] = src_region
                cached_count += 1
                self.stats['cache_hits'] += 1
                
            except Exception as e:
                print(f"  Warning: Could not paste cached crop for T{track_id}: {e}")
                continue

        if cached_count > 0:
            print(f"  Reapplied {cached_count} cached edits")

        # 6. Analyze memorability
        mem_results = self.mem_analyzer.predict_memorability_crops(frame, detections)

        # 7. Find NEW high-memorability detections
        new_high_mem = []
        for det_idx, mem_info in mem_results.items():
            if mem_info['memorability_score'] <= self.memorability_threshold:
                continue
            
            track_id = detection_to_track.get(det_idx)
            if track_id is None:
                continue
            
            det = detections[det_idx]
            class_name = det['class_name']
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            
            w = x2 - x1
            h = y2 - y1
            bbox_area = w * h
            scale_bin = get_scale_bin(bbox_area)
            
            cache_key = (track_id, class_name, scale_bin)
            
            # Skip if already cached for this scale
            if cache_key in self.edit_cache:
                continue
            
            new_high_mem.append({
                'det_idx': det_idx,
                'track_id': track_id,
                'class_name': class_name,
                'mem_info': mem_info,
                'bbox': bbox,
                'bbox_area': bbox_area,
                'scale_bin': scale_bin
            })

        # 8. Process NEW high-memorability detections
        edits = []
        skipped = 0
        fallback = 0

        if apply_perturbation and len(new_high_mem) > 0:
            print(f"  Found {len(new_high_mem)} new high-memorability detections")

            for item in new_high_mem:
                det_idx = item['det_idx']
                track_id = item['track_id']
                class_name = item['class_name']
                bbox = item['bbox']
                bbox_area = item['bbox_area']
                scale_bin = item['scale_bin']
                x1, y1, x2, y2 = bbox
                
                w = x2 - x1
                h = y2 - y1
                
                # Check if bbox is too large
                if w > MAX_BBOX_DIMENSION or h > MAX_BBOX_DIMENSION or bbox_area > MAX_BBOX_AREA:
                    # Use fallback: light blur
                    print(f"    [SKIP] T{track_id} {class_name} too large ({w}x{h}), using light blur")
                    
                    try:
                        # Apply light Gaussian blur
                        x1c = max(0, min(W - 1, int(x1)))
                        x2c = max(0, min(W, int(x2)))
                        y1c = max(0, min(H - 1, int(y1)))
                        y2c = max(0, min(H, int(y2)))
                        
                        if x2c > x1c and y2c > y1c:
                            roi = result_frame[y1c:y2c, x1c:x2c]
                            kernel_size = min(15, max(3, min(roi.shape[0], roi.shape[1]) // 10))
                            if kernel_size % 2 == 0:
                                kernel_size += 1
                            blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
                            result_frame[y1c:y2c, x1c:x2c] = blurred
                            
                            # Cache the blurred version
                            cache_key = (track_id, class_name, scale_bin)
                            self.edit_cache[cache_key] = {
                                'crop': blurred.copy(),
                                'bbox_size': (x2c - x1c, y2c - y1c),
                                'method': 'fallback_blur',
                                'mem_score': item['mem_info']['memorability_score']
                            }
                            
                            edits.append({
                                'track_id': track_id,
                                'bbox': bbox,
                                'class_name': class_name,
                                'method': 'fallback_blur',
                                'mem_score': item['mem_info']['memorability_score']
                            })
                            
                            fallback += 1
                            self.stats['fallback_edits'] += 1
                    except Exception as e:
                        print(f"    Error applying blur fallback: {e}")
                    
                    skipped += 1
                    self.stats['skipped_large_bbox'] += 1
                    continue
                
                # Validate bbox
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Create single-detection list for diffusion editor
                single_det = [detections[det_idx]]
                single_mem = {0: item['mem_info']}
                
                # Compute adaptive strength based on bbox size
                strength = self._compute_adaptive_strength(bbox_area)
                
                # Run diffusion
                try:
                    edited_frame, edit_info = self.diffusion_editor.edit_frame(
                        frame_bgr=result_frame,
                        detections=single_det,
                        mem_results=single_mem,
                        mem_threshold=self.memorability_threshold,
                        process_entire_bbox=True,
                        pad=2,
                        strength=strength  # Pass adaptive strength
                    )
                    
                    if len(edit_info) > 0:
                        # Update working frame
                        result_frame = edited_frame
                        
                        # Extract and cache the edited crop
                        Hf, Wf = result_frame.shape[:2]
                        x1c = max(0, min(Wf - 1, int(x1)))
                        x2c = max(0, min(Wf, int(x2)))
                        y1c = max(0, min(Hf - 1, int(y1)))
                        y2c = max(0, min(Hf, int(y2)))
                        
                        if x2c <= x1c or y2c <= y1c:
                            continue
                        
                        diffused_crop = result_frame[y1c:y2c, x1c:x2c].copy()
                        
                        # Cache with scale bin
                        cache_key = (track_id, class_name, scale_bin)
                        self.edit_cache[cache_key] = {
                            'crop': diffused_crop,
                            'bbox_size': (x2c - x1c, y2c - y1c),
                            'method': edit_info[0]['method'],
                            'mem_score': item['mem_info']['memorability_score']
                        }
                        
                        # Record edit
                        edit = edit_info[0]
                        edit['track_id'] = track_id
                        edit['scale_bin'] = scale_bin
                        edits.append(edit)
                        
                        self.stats['cache_misses'] += 1
                        
                        print(f"    [NEW] T{track_id} {class_name} ({scale_bin}) "
                              f"(mem={item['mem_info']['memorability_score']:.3f}) "
                              f"- {edit_info[0]['method']}")
                
                except Exception as e:
                    print(f"  Error processing T{track_id} {class_name}: {e}")
                    continue

        # 9. Visualize (if enabled)
        if visualize:
            result_frame = self._visualize_results(
                result_frame,
                detections,
                tracks,
                mem_results,
                new_high_mem,
                edits
            )

        # 10. Statistics
        stats = {
            'total_detections': len(detections),
            'total_tracks': len(tracks),
            'high_memorability_count': len(new_high_mem),
            'avg_memorability': np.mean([m['memorability_score']
                                        for m in mem_results.values()]) if mem_results else 0,
            'cached_edits_applied': cached_count,
            'new_edits': len(edits),
            'edits_applied': len(edits),
            'cache_size': len(self.edit_cache),
            'skipped_large_bbox': skipped,
            'fallback_edits': fallback,
            'edit_details': edits
        }

        return result_frame, stats
    
    
    def _compute_adaptive_strength(self, bbox_area: float) -> float:
        """
        Compute adaptive diffusion strength based on bbox size
        Larger bboxes get lower strength for more subtle edits
        """
        if bbox_area > 40000:
            return 0.5  # Very subtle for large regions
        elif bbox_area > 25000:
            return 0.6
        elif bbox_area > 15000:
            return 0.7
        else:
            return 0.8  # Default strength for small regions
    
    
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
                'cached_part_types': 0,
                'cache_hits': self.stats['cache_hits'],
                'cache_misses': self.stats['cache_misses'],
                'skipped_large_bbox': self.stats['skipped_large_bbox'],
                'fallback_edits': self.stats['fallback_edits']
            }
        
        cached_tracks = set(k[0] for k in self.edit_cache.keys())
        cached_parts = set(k[1] for k in self.edit_cache.keys())
        scale_distribution = {}
        for k in self.edit_cache.keys():
            scale_bin = k[2]
            scale_distribution[scale_bin] = scale_distribution.get(scale_bin, 0) + 1
        
        return {
            'cache_size': len(self.edit_cache),
            'cached_tracks': len(cached_tracks),
            'cached_part_types': len(cached_parts),
            'scale_distribution': scale_distribution,
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'skipped_large_bbox': self.stats['skipped_large_bbox'],
            'fallback_edits': self.stats['fallback_edits']
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
                print(f"  Skipped (large bbox): {stats['skipped_large_bbox']}")
                print(f"  Fallback edits: {stats['fallback_edits']}")
                
                # Save
                output_path = f"results/memorability_test_{idx}.jpg"
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(output_path, result)
                print(f"\n✓ Saved: {output_path}")
            
            # Print final cache stats
            cache_stats = pipeline.get_cache_stats()
            print(f"\n{'='*60}")
            print(" Final Cache Statistics:")
            print('='*60)
            print(f"  Total cached entries: {cache_stats['cache_size']}")
            print(f"  Unique tracks cached: {cache_stats['cached_tracks']}")
            print(f"  Part types cached: {cache_stats['cached_part_types']}")
            print(f"  Scale distribution: {cache_stats['scale_distribution']}")
            print(f"  Cache hits: {cache_stats['cache_hits']}")
            print(f"  Cache misses: {cache_stats['cache_misses']}")
            print(f"  Skipped large bbox: {cache_stats['skipped_large_bbox']}")
            print(f"  Fallback edits: {cache_stats['fallback_edits']}")
            
            print("\n✓ Test complete!")
            return True
    
    print("⚠ No test images found in 'test_images/' directory")
    return False


if __name__ == "__main__":
    test_memorability_integration()