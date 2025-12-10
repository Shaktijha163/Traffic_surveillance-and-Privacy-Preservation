import torch
import torch.nn.functional as F
import numpy as np
import cv2
import sys
import os
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import warnings

# --- DIFFUSERS & ANIMATEDIFF IMPORTS ---
try:
    from diffusers import MotionAdapter, AnimateDiffVideoToVideoPipeline, LCMScheduler
    from diffusers.utils import export_to_gif
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False
    warnings.warn("diffusers/peft not installed. Install: pip install diffusers transformers accelerate peft")

class AdvancedTemporalVideoEditor:
    CLASS_PROMPTS = {
        # DEBUG MODE: Use distinct textures to verify it works
        'sticker': "carbon fiber texture, high contrast, pattern, 4k",
        'bumper': "shiny chrome bumper, metallic reflection, silver, 4k",
        'door': "rusty metal door, old paint, weathered, texture, 4k",
        'hood': "carbon fiber car hood, black woven pattern, racing style, 4k",
        'fender': "matte black fender, smooth, 4k",
        'default': "shiny chrome car part, metallic, 4k"
    }

    def __init__(self, device="cuda", use_lcm=True):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.has_diffusion = False
        
        # Configuration
        self.context_length = 16  # AnimateDiff works best with 16 frames
        self.use_lcm = use_lcm    

        if HAS_DIFFUSERS:
            try:
                print(f" Loading Video Diffusion (AnimateDiff Video-to-Video)...")
                
                # 1. Load Motion Adapter
                adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)

                # 2. Load Pipeline
                model_id = "runwayml/stable-diffusion-v1-5" 
                
                self.pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(
                    model_id,
                    motion_adapter=adapter,
                    torch_dtype=torch.float16
                )

                # 3. Optimize for Efficiency (LCM-LoRA)
                if self.use_lcm:
                    print("  ⚡ Applying LCM-LoRA for efficiency (4-8 steps)...")
                    self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
                    self.pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm")
                    self.pipe.set_adapters(["lcm"], adapter_weights=[1.0])

                # 4. Enable Optimizations
                self.pipe.enable_vae_slicing()
                
                # --- FIX: CPU Offload handles device placement automatically ---
                # Do NOT call self.pipe.to("cuda") if using cpu_offload
                self.pipe.enable_model_cpu_offload() 
                
                self.has_diffusion = True
                print("✓ AnimateDiff Video Editor ready!\n")
            except Exception as e:
                warnings.warn(f"Could not load editor: {e}")
                import traceback
                traceback.print_exc()

    def _get_prompt(self, class_name):
        for key in self.CLASS_PROMPTS:
            if key in class_name.lower(): return self.CLASS_PROMPTS[key]
        return self.CLASS_PROMPTS['default']

    def edit_frame_batch(self, frames: List[np.ndarray], detections_batch, tracks_batch, mem_results_batch, mem_threshold):
        if not self.has_diffusion: 
            return frames, {'method': 'no_diffusion', 'edits': 0}

        # Identify tracks
        track_edits = self._collect_track_edits(frames, detections_batch, tracks_batch, mem_results_batch, mem_threshold)
        
        if len(track_edits) == 0: 
            return frames, {'method': 'no_edits', 'edits': 0}

        # We must copy frames to avoid overwriting originals during processing
        edited_frames = [f.copy() for f in frames]
        total_edits = 0

        for track_id, edit_info in track_edits.items():
            print(f"   Batch Processing Track {track_id} ({edit_info['class_name']})...")
            try:
                edited_frames = self._process_video_track(edited_frames, edit_info)
                total_edits += len(edit_info['frame_indices'])
            except Exception as e:
                print(f"  ⚠ Failed to process track {track_id}: {e}")
                continue

        return edited_frames, {
            'method': 'animatediff_video',
            'edits': total_edits,
            'tracks_edited': len(track_edits)
        }

    def _process_video_track(self, full_frames, edit_info):
        frame_indices = sorted(edit_info['frame_indices'])
        bboxes = edit_info['bboxes']
        prompt = self._get_prompt(edit_info['class_name'])
        neg_prompt = "text, watermark, distortion, blurry, low quality, warping, flickering, humans, hands"

        # Chunking strategy
        chunk_size = self.context_length
        
        for i in range(0, len(frame_indices), chunk_size):
            chunk_indices = frame_indices[i : i + chunk_size]
            chunk_bboxes = bboxes[i : i + chunk_size]
            
            if len(chunk_indices) < 2: 
                continue

            batch_crops = []
            batch_masks = []
            crop_coords_list = []
            valid_chunk_indices = []
            
            # Extract
            for idx, global_frame_idx in enumerate(chunk_indices):
                bbox = chunk_bboxes[idx]
                frame = full_frames[global_frame_idx]
                crop, mask, coords = self._extract_crop(frame, bbox)
                
                if crop is not None:
                    batch_crops.append(crop)
                    batch_masks.append(mask)
                    crop_coords_list.append(coords)
                    valid_chunk_indices.append(global_frame_idx)

            if not batch_crops: continue

            # Run Diffusion
            generated_frames = self._run_video_generation(
                batch_crops, 
                prompt, 
                neg_prompt
            )

            # Paste Back
            for idx, gen_img in enumerate(generated_frames):
                if idx >= len(valid_chunk_indices): break 
                
                global_idx = valid_chunk_indices[idx]
                coords = crop_coords_list[idx]
                mask = batch_masks[idx]
                
                gen_cv2 = cv2.cvtColor(np.array(gen_img), cv2.COLOR_RGB2BGR)
                
                full_frames[global_idx] = self._paste_crop(
                    full_frames[global_idx], 
                    gen_cv2, 
                    coords, 
                    mask
                )

        return full_frames

    def _run_video_generation(self, images: List[Image.Image], prompt, negative_prompt):
        # Resize to 512x512
        w, h = 512, 512
        resized_images = [img.resize((w, h), Image.LANCZOS) for img in images]

        steps = 6 if self.use_lcm else 20
        guidance = 1.5 if self.use_lcm else 7.5
        strength = 0.8

        output = self.pipe(
            video=resized_images, 
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            strength=strength,
            generator=torch.Generator("cuda").manual_seed(42)
        ).frames[0]

        # Resize back
        final_output = []
        for i, img in enumerate(output):
            orig_w, orig_h = images[i].size
            final_output.append(img.resize((orig_w, orig_h), Image.LANCZOS))
            
        return final_output

    def _extract_crop(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        pad = 20 
        h, w = frame.shape[:2]
        x1_c = max(0, x1 - pad)
        y1_c = max(0, y1 - pad)
        x2_c = min(w, x2 + pad)
        y2_c = min(h, y2 + pad)
        
        crop = frame[y1_c:y2_c, x1_c:x2_c]
        if crop.size == 0: return None, None, None

        mask = np.zeros((crop.shape[0], crop.shape[1]), dtype=np.uint8)
        bx1 = max(0, x1 - x1_c)
        by1 = max(0, y1 - y1_c)
        bx2 = min(crop.shape[1], x2 - x1_c)
        by2 = min(crop.shape[0], y2 - y1_c)
        mask[by1:by2, bx1:bx2] = 255
        
        return (
            Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
            Image.fromarray(mask),
            (x1_c, y1_c, x2_c, y2_c)
        )

    def _paste_crop(self, frame, crop, crop_coords, mask):
        x1, y1, x2, y2 = crop_coords
        target_h, target_w = y2 - y1, x2 - x1
        
        if crop.shape[:2] != (target_h, target_w):
            crop = cv2.resize(crop, (target_w, target_h))
            
        mask_np = np.array(mask.resize((target_w, target_h))) / 255.0
        mask_blur = cv2.GaussianBlur(mask_np, (21, 21), 10)
        mask_3ch = np.stack([mask_blur] * 3, axis=-1)
        
        roi = frame[y1:y2, x1:x2]
        blended = (crop.astype(np.float32) * mask_3ch + roi.astype(np.float32) * (1 - mask_3ch)).astype(np.uint8)
        
        result = frame.copy()
        result[y1:y2, x1:x2] = blended
        return result
        
    def _collect_track_edits(self, frames, detections_batch, tracks_batch, mem_results_batch, mem_threshold):
        track_edits = {}
        for frame_idx in range(len(frames)):
            detections = detections_batch[frame_idx]
            tracks = tracks_batch[frame_idx]
            mem_results = mem_results_batch[frame_idx]
            det_to_track = self._match_detections_to_tracks(detections, tracks)
            for det_idx, track_id in det_to_track.items():
                mem_info = mem_results.get(det_idx)
                if mem_info and mem_info['memorability_score'] > mem_threshold:
                    if track_id not in track_edits:
                        track_edits[track_id] = {
                            'class_name': detections[det_idx]['class_name'], 
                            'frame_indices': [], 
                            'bboxes': []
                        }
                    track_edits[track_id]['frame_indices'].append(frame_idx)
                    track_edits[track_id]['bboxes'].append(tuple(detections[det_idx]['bbox']))
        return track_edits
    
    def _match_detections_to_tracks(self, detections, tracks):
        det_to_track = {}
        for det_idx, det in enumerate(detections):
            best_iou, best_id = 0, None
            for track in tracks:
                iou = self._compute_iou(det['bbox'], track['bbox'])
                if iou > best_iou: best_iou, best_id = iou, track['track_id']
            if best_iou > 0.5: det_to_track[det_idx] = best_id
        return det_to_track

    def _compute_iou(self, bbox1, bbox2):
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        inter_x_min, inter_y_min = max(x1_min, x2_min), max(y1_min, y2_min)
        inter_x_max, inter_y_max = min(x1_max, x2_max), min(y1_max, y2_max)
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min: return 0.0
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        union_area = (x1_max - x1_min) * (y1_max - y1_min) + (x2_max - x2_min) * (y2_max - y2_min) - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def clear_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()