"""
diffusion_editor.py (FIXED VERSION - Adaptive Strength Support)
Diffusion-based editor for reducing memorability of detected regions.

FIXES APPLIED:
- Support for adaptive strength parameter
- Proportional context padding based on bbox size
- Better error handling and fallbacks
- Optimized processing for different bbox sizes
"""

from typing import List, Dict, Tuple, Optional
from pathlib import Path
import math
import warnings

import numpy as np
from PIL import Image, ImageDraw
import cv2
import torch

# Attempt to import diffusers / StableDiffusionInpaintPipeline
try:
    from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
    _HAS_DIFFUSERS = True
except Exception:
    _HAS_DIFFUSERS = False


def _ensure_bbox_in_bounds(bbox: Tuple[int, int, int, int], img_w: int, img_h: int):
    """Ensure bbox coordinates are within image bounds"""
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(img_w - 1, int(x1)))
    y1 = max(0, min(img_h - 1, int(y1)))
    x2 = max(0, min(img_w, int(x2)))
    y2 = max(0, min(img_h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _create_bbox_mask(image_size: Tuple[int, int], bbox: Tuple[int, int, int, int], pad: int = 8):
    """
    Create a binary mask PIL Image (white inside bbox, black outside).
    pad: expand bbox by some pixels to avoid hard edges artifacts.
    """
    w, h = image_size
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x1, y1, x2, y2], fill=255)
    return mask


def _pil_from_bgr(bgr: np.ndarray) -> Image.Image:
    """Convert BGR numpy array to PIL Image"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _bgr_from_pil(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL Image to BGR numpy array"""
    arr = np.array(pil_img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr


# ============================
# DiffusionEditor Class
# ============================

class DiffusionEditor:
    """
    Enhanced diffusion editor with adaptive strength and intelligent processing
    
    FEATURES:
    - Adaptive diffusion strength based on bbox size
    - Proportional context padding
    - Optimized processing for different scales
    - Robust error handling with fallbacks
    """

    CLASS_PROMPT_MAP = {
        'sticker': "plain car surface, no text, no stickers, photorealistic",
        'bumper': "generic car bumper, plain, no logos, photorealistic",
        'door': "generic car door, plain surface, no logos, photorealistic",
        'hood': "generic car hood, smooth surface, no logos, photorealistic",
        'car_hood': "generic car hood, smooth surface, no logos, photorealistic",
        'windshield': "generic car windshield, clear glass, photorealistic",
        'lights': "generic car headlight, standard design, photorealistic",
        'fender': "generic car fender, smooth surface, no logos, photorealistic",
        'mirror': "generic car mirror, plain design, photorealistic",
        'default': "generic car part, plain appearance, no logos, photorealistic"
    }

    def __init__(self,
                 device: str = "cuda",
                 model_id: str = "stabilityai/stable-diffusion-2-inpainting",
                 guidance_scale: float = 7.5,
                 num_inference_steps: int = 25,
                 resize_to: Optional[int] = 512,
                 enable_caching: bool = True):
        """
        Initialize diffusion editor
        
        Args:
            device: 'cuda' or 'cpu'
            model_id: Hugging Face model ID for inpainting
            guidance_scale: CFG scale
            num_inference_steps: Number of diffusion steps
            resize_to: Resize images to this size (None = keep original)
            enable_caching: Cache edits for reuse
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model_id = model_id
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.resize_to = resize_to
        self.enable_caching = enable_caching

        self.pipe = None
        self.has_diffusion = False

        if _HAS_DIFFUSERS:
            try:
                self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,
                    variant="fp16" if self.device == "cuda" else None
                )
                try:
                    self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
                except Exception:
                    pass

                if self.device == "cuda":
                    self.pipe = self.pipe.to("cuda")

                self.has_diffusion = True
                print(f"✓ Diffusion inpainting model loaded: {model_id} (device={self.device})")
                print(f"  Caching: {'ENABLED' if enable_caching else 'DISABLED'}")
            except Exception as e:
                warnings.warn(f"Could not load diffusion pipeline '{model_id}': {e}. Falling back to blur.")
                self.pipe = None
                self.has_diffusion = False
        else:
            warnings.warn("diffusers library not found. Install 'diffusers transformers accelerate'.")
            self.has_diffusion = False


    def _get_prompt_for_class(self, class_name: str) -> str:
        """Get diffusion prompt for class"""
        # Try exact match first
        if class_name.lower() in self.CLASS_PROMPT_MAP:
            return self.CLASS_PROMPT_MAP[class_name.lower()]
        
        # Try partial match
        for key in self.CLASS_PROMPT_MAP:
            if key in class_name.lower():
                return self.CLASS_PROMPT_MAP[key]
        
        return self.CLASS_PROMPT_MAP['default']


    def _compute_context_padding(self, bbox_width: int, bbox_height: int) -> int:
        """
        Compute proportional context padding based on bbox size
        Returns padding in pixels
        """
        # Use 15% of smaller dimension, capped at 50 pixels
        smaller_dim = min(bbox_width, bbox_height)
        pad = int(0.15 * smaller_dim)
        return min(50, max(10, pad))


    def edit_frame(self,
                frame_bgr: np.ndarray,
                detections: List[Dict],
                mem_results: Dict[int, Dict],
                mem_threshold: float = 0.6,
                process_entire_bbox: bool = True,
                pad: int = 0,
                strength: float = 0.8) -> Tuple[np.ndarray, List[Dict]]:
        """
        Edit frame: run inpainting on high-memorability detections

        Args:
            frame_bgr: BGR image
            detections: List of detections
            mem_results: Memorability results {det_idx: {...}}
            mem_threshold: Threshold for editing
            process_entire_bbox: Process entire bbox vs attention regions
            pad: Additional padding around bbox (added to adaptive padding)
            strength: Diffusion strength (0-1, higher = more change)

        Returns:
            (edited_frame, list_of_edits)
        """
        img_h, img_w = frame_bgr.shape[:2]

        # Determine which detections to edit
        to_edit = []
        for idx, det in enumerate(detections):
            mem_info = mem_results.get(idx)
            if mem_info is None:
                continue
            score = float(mem_info.get('memorability_score', 0.0))
            if score > mem_threshold:
                bbox = det.get('bbox')
                safe_bbox = _ensure_bbox_in_bounds(tuple(bbox), img_w, img_h)
                if safe_bbox:
                    to_edit.append({
                        'det_idx': idx,
                        'bbox': safe_bbox,
                        'class_name': det.get('class_name', 'default'),
                        'mem_score': score
                    })

        if len(to_edit) == 0:
            return frame_bgr, []

        # Process largest boxes first
        to_edit.sort(key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]), reverse=True)

        working_img = frame_bgr.copy()
        edits = []

        for item in to_edit:
            bbox = item['bbox']
            class_name = item['class_name']
            x1, y1, x2, y2 = bbox

            w = x2 - x1
            h = y2 - y1

            # Compute adaptive context padding
            adaptive_pad = self._compute_context_padding(w, h)
            total_pad = adaptive_pad + int(pad)

            # Extract crop with context padding
            x1_ctx = max(0, x1 - total_pad)
            y1_ctx = max(0, y1 - total_pad)
            x2_ctx = min(img_w, x2 + total_pad)
            y2_ctx = min(img_h, y2 + total_pad)

            # Defensive check
            if x2_ctx <= x1_ctx or y2_ctx <= y1_ctx:
                continue

            crop = working_img[y1_ctx:y2_ctx, x1_ctx:x2_ctx].copy()
            crop_h, crop_w = crop.shape[:2]

            # Create mask for detection region within crop
            mask_crop = np.zeros((crop_h, crop_w), dtype=np.uint8)
            mask_x1 = max(0, x1 - x1_ctx)
            mask_y1 = max(0, y1 - y1_ctx)
            mask_x2 = min(crop_w, x2 - x1_ctx)
            mask_y2 = min(crop_h, y2 - y1_ctx)

            # Validate mask coords
            if mask_x2 <= mask_x1 or mask_y2 <= mask_y1:
                continue

            mask_crop[mask_y1:mask_y2, mask_x1:mask_x2] = 255

            # Generate edit with diffusion
            prompt = self._get_prompt_for_class(class_name)
            negative_prompt = "wheel, tire, rim, circular objects, round shapes, stripes, lines, text, logos, patterns, distortion, warping"

            if self.has_diffusion and self.pipe is not None:
                try:
                    # Convert crop to PIL
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop_pil = Image.fromarray(crop_rgb)
                    mask_pil = Image.fromarray(mask_crop)

                    # Resize for processing (maintain aspect ratio)
                    target_size = self.resize_to if self.resize_to is not None else 512
                    try:
                        target_size = int(max(64, target_size))
                    except Exception:
                        target_size = 512

                    aspect = float(crop_w) / float(max(1, crop_h))
                    if aspect > 1.0:
                        new_w = target_size
                        new_h = max(64, int(round(target_size / aspect)))
                    else:
                        new_h = target_size
                        new_w = max(64, int(round(target_size * aspect)))

                    # Ensure dimensions are multiples of 8 (required by stable diffusion)
                    new_w = (new_w // 8) * 8
                    new_h = (new_h // 8) * 8
                    new_w = max(64, new_w)
                    new_h = max(64, new_h)

                    crop_resized = crop_pil.resize((new_w, new_h), resample=Image.LANCZOS)
                    mask_resized = mask_pil.resize((new_w, new_h), resample=Image.NEAREST)

                    # Run inpainting with adaptive strength
                    out = self.pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=crop_resized,
                        mask_image=mask_resized,
                        num_inference_steps=self.num_inference_steps,
                        guidance_scale=self.guidance_scale,
                        strength=strength  # Use adaptive strength
                    )

                    # Validate output
                    if not hasattr(out, 'images') or len(out.images) == 0:
                        raise RuntimeError("Diffusion pipeline returned no images")

                    out_img = out.images[0]

                    # Resize back to original crop size
                    out_img = out_img.resize((crop_w, crop_h), resample=Image.LANCZOS)
                    out_bgr = _bgr_from_pil(out_img)

                    # Blend masked region into crop
                    mask_bool = mask_crop > 127
                    crop_edited = crop.copy()
                    for c in range(3):
                        crop_edited[:, :, c][mask_bool] = out_bgr[:, :, c][mask_bool]

                    # Paste edited crop back into working image
                    working_img[y1_ctx:y2_ctx, x1_ctx:x2_ctx] = crop_edited

                    edits.append({
                        'det_idx': item['det_idx'],
                        'bbox': bbox,
                        'class_name': class_name,
                        'prompt': prompt,
                        'method': 'diffusion_crop',
                        'mem_score': item['mem_score'],
                        'strength': strength
                    })

                except Exception as e:
                    # Robust fallback: blur the context region
                    warnings.warn(f"Crop inpainting failed for bbox {bbox}: {e}. Using blur fallback.")
                    roi = working_img[y1_ctx:y2_ctx, x1_ctx:x2_ctx]
                    # Ensure kernel size is odd and reasonable
                    k = max(3, min(51, min(roi.shape[0], roi.shape[1]) // 8))
                    if k % 2 == 0:
                        k += 1
                    blurred = cv2.GaussianBlur(roi, (k, k), 0)
                    working_img[y1_ctx:y2_ctx, x1_ctx:x2_ctx] = blurred
                    edits.append({
                        'det_idx': item['det_idx'],
                        'bbox': bbox,
                        'class_name': class_name,
                        'prompt': prompt,
                        'method': 'fallback_blur',
                        'mem_score': item['mem_score']
                    })
            else:
                # No diffusion available: blur fallback
                roi = working_img[y1_ctx:y2_ctx, x1_ctx:x2_ctx]
                k = max(3, min(51, min(roi.shape[0], roi.shape[1]) // 8))
                if k % 2 == 0:
                    k += 1
                blurred = cv2.GaussianBlur(roi, (k, k), 0)
                working_img[y1_ctx:y2_ctx, x1_ctx:x2_ctx] = blurred
                edits.append({
                    'det_idx': item['det_idx'],
                    'bbox': bbox,
                    'class_name': class_name,
                    'prompt': prompt,
                    'method': 'fallback_blur',
                    'mem_score': item['mem_score']
                })

        return working_img, edits