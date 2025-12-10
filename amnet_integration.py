"""
amnet_integration.py
Complete pipeline adapted for Batch Video Diffusion (AnimateDiff)
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image

# Import your custom modules
from object_detector import CarPartsDetector
from object_tracker import CarPartsTracker
from amnet import AMNet
from config import get_config
# Import the NEW editor we created
from diffusion_editor import AdvancedTemporalVideoEditor

class MemorabilityAnalyzer:
    """Analyzes memorability using AMNet"""
    
    def __init__(self, amnet_model_path: str, device: str = "cuda", attention_maps: bool = True):
        print(f" Initializing AMNet Memorability Analyzer...")
        self.device = device
        self.attention_maps_enabled = attention_maps
        self.amnet = AMNet()
        hps = get_config()
        hps.use_cuda = (device == "cuda")
        hps.cuda_device = 0
        hps.model_weights = amnet_model_path
        hps.use_attention = attention_maps
        self.amnet.init(hps)
        self.amnet.model.eval()
        print(f"✓ AMNet loaded\n")
    
    def predict_memorability_crops(self, frame, detections):
        if len(detections) == 0: return {}
        crops, crop_info = [], []
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            # Safety checks for image boundaries
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1: continue

            # Add padding context for better AMNet analysis
            box_h, box_w = y2 - y1, x2 - x1
            pad_h, pad_w = int(box_h * 0.1), int(box_w * 0.1)
            
            x1_pad, y1_pad = max(0, x1 - pad_w), max(0, y1 - pad_h)
            x2_pad, y2_pad = min(w, x2 + pad_w), min(h, y2 + pad_h)
            
            crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            if crop.size == 0: continue
            
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(Image.fromarray(crop_rgb))
            crop_info.append({'det_idx': idx, 'bbox': det['bbox'], 'class_name': det['class_name']})
            
        if len(crops) == 0: return {}
        
        try:
            pr = self.amnet.predict_memorability_image_batch(crops)
            results = {}
            for i, info in enumerate(crop_info):
                result = {
                    'memorability_score': float(pr.predictions[i]), 
                    'bbox': info['bbox'], 
                    'class_name': info['class_name']
                }
                if self.attention_maps_enabled and pr.attention_masks is not None: 
                    result['attention_map'] = pr.attention_masks[i]
                results[info['det_idx']] = result
            return results
        except Exception as e:
            print(f"AMNet Batch Error: {e}")
            return {}
    
    def visualize_attention_on_detection(self, frame, detection, attention_map, alpha=0.5):
        x1, y1, x2, y2 = detection['bbox']
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0: return frame

        att = attention_map[-1]
        ares = int(np.sqrt(att.shape[0]))
        att = att.reshape((ares, ares))
        att = (att - att.min()) / (att.max() - att.min() + 1e-8)
        att = (att * 255).astype(np.uint8)
        att_resized = cv2.resize(att, (w, h), interpolation=cv2.INTER_CUBIC)
        heatmap = cv2.applyColorMap(att_resized, cv2.COLORMAP_JET)
        
        roi = frame[y1:y2, x1:x2]
        blended = cv2.addWeighted(roi, alpha, heatmap, 1-alpha, 0)
        result = frame.copy()
        result[y1:y2, x1:x2] = blended
        return result


class AdvancedMemorabilityPipeline:
    def __init__(self, yolo_model_path="models/best.pt", amnet_model_path="models/amnet_weights.pkl", 
                 device="cuda", conf_threshold=0.5, memorability_threshold=0.6, 
                 attention_maps=True, use_lcm=True):
        
        print("\n" + "="*70)
        print("🎬 Advanced Video Diffusion Pipeline (Batch Mode)")
        print("="*70 + "\n")
        
        self.detector = CarPartsDetector(model_path=yolo_model_path, device=device, conf_threshold=conf_threshold)
        self.tracker = CarPartsTracker(device=device)
        self.mem_analyzer = MemorabilityAnalyzer(amnet_model_path=amnet_model_path, device=device, attention_maps=attention_maps)
        
        # Initialize the NEW Video Editor (AnimateDiff)
        self.video_editor = AdvancedTemporalVideoEditor(device=device, use_lcm=use_lcm)
        
        self.memorability_threshold = memorability_threshold
        self.attention_maps = attention_maps
        
        print(f"✓ Pipeline ready! Mode: {'LCM (Fast)' if use_lcm else 'Standard'}\n")
    
    def process_transfer_batch(self, frame_batch: List[np.ndarray], start_frame_idx: int) -> Tuple[List[np.ndarray], List[np.ndarray], Dict]:
        batch_detections = []
        batch_tracks = []
        batch_mem_results = []
        
        # 1. Detection, Tracking & Memorability (Frame by Frame)
        for i, frame in enumerate(frame_batch):
            current_idx = start_frame_idx + i
            detections = self.detector.detect_frame(frame)
            tracks = self.tracker.update(frame, detections, current_idx)
            mem_results = self.mem_analyzer.predict_memorability_crops(frame, detections)
            
            batch_detections.append(detections)
            batch_tracks.append(tracks)
            batch_mem_results.append(mem_results)

        # 2. Run Diffusion
        print(f"  ⚡ Running Video Diffusion on {len(frame_batch)} frames...")
        
        # edited_frames now contains the CLEAN video with diffusion edits
        edited_frames, edit_stats = self.video_editor.edit_frame_batch(
            frames=frame_batch,
            detections_batch=batch_detections,
            tracks_batch=batch_tracks,
            mem_results_batch=batch_mem_results,
            mem_threshold=self.memorability_threshold
        )
        
        # 3. Create Visualization Copy (WITH Boxes/Heatmaps)
        visualized_frames = []
        
        total_high_mem = 0
        total_mem_score = 0
        
        for i, frame in enumerate(edited_frames):
            # We make a COPY for visualization so we don't draw on the clean edit
            viz_frame = self._visualize_results(
                frame.copy(), 
                batch_detections[i], 
                batch_tracks[i], 
                batch_mem_results[i]
            )
            visualized_frames.append(viz_frame)
            
            # Stats
            high_mem = sum(1 for m in batch_mem_results[i].values() if m['memorability_score'] > self.memorability_threshold)
            avg_mem = np.mean([m['memorability_score'] for m in batch_mem_results[i].values()]) if batch_mem_results[i] else 0
            total_high_mem += high_mem
            total_mem_score += avg_mem

        batch_stats = {
            'processed_frames': len(frame_batch),
            'total_edits': edit_stats.get('edits', 0),
            'tracks_edited': edit_stats.get('tracks_edited', 0),
            'avg_high_mem_per_frame': total_high_mem / len(frame_batch) if frame_batch else 0,
            'avg_memorability': total_mem_score / len(frame_batch) if frame_batch else 0
        }
        
        # RETURN BOTH: edited_frames (Clean) AND visualized_frames (With Boxes)
        return edited_frames, visualized_frames, batch_stats

    def _visualize_results(self, frame, detections, tracks, mem_results):
        result = frame.copy()
        
        # Draw Tracks
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            # Track ID text
            cv2.putText(result, f"ID:{track['track_id']}", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Draw Detections & Memorability
        for det_idx, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            mem_score = mem_results.get(det_idx, {}).get('memorability_score', 0)
            
            # Color based on memorability (Green = High, Red = Low)
            is_high = mem_score > self.memorability_threshold
            color = (0, 255, 0) if is_high else (0, 0, 255)
            thickness = 2 if is_high else 1
            
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            
            label = f"{det['class_name']} {mem_score:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(result, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
            cv2.putText(result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Draw Attention Heatmap if High Memorability
            if self.attention_maps and is_high:
                att_map = mem_results[det_idx].get('attention_map')
                if att_map is not None:
                    result = self.mem_analyzer.visualize_attention_on_detection(result, det, att_map, alpha=0.6)
                    
        return result

    def clear_cache(self):
        self.video_editor.clear_cache()