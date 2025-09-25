#!/usr/bin/env python3
"""
diffusion_perturbation.py

Process a folder of images using AMNet to get attention maps and memorability scores,
create masks for high-attention regions, run Stable Diffusion inpainting to perturb
those regions (with prompts depending on the memorability score), and re-score the
perturbed images.

Outputs:
 - perturbed images saved to output_root/perturbed/<imagename>
 - visualization images saved to output_root/vis/<imagename>_vis.png
 - CSV log saved to output_root/logs.csv
"""

import os
import sys
import time
import tempfile
import csv
import traceback
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch

# diffusion
from diffusers import StableDiffusionInpaintPipeline

# AMNet from this repo
from amnet import AMNet
from config import get_amnet_config
import argparse


class MemorabilityPerturbator:
    def __init__(self, memorability_model_path, device="cuda"):
        """
        memorability_model_path: path to the AMNet checkpoint (.pkl)
        device: 'cuda' or 'cpu' - used for diffusion pipeline
        """
        self.device = device
        print("Loading AMNet memorability model...")
        self.mem_model = self.load_memorability_model(memorability_model_path)
        print("AMNet loaded.")

        # load diffusion inpainting pipeline
        print("Loading Stable Diffusion Inpaint pipeline (this may take some time)...")
        # Using float16 for VRAM efficiency; make sure GPU supports it
        self.diffusion_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        )
        if self.device == "cuda" and torch.cuda.is_available():
            self.diffusion_pipe = self.diffusion_pipe.to("cuda")
        else:
            self.diffusion_pipe = self.diffusion_pipe.to("cpu")
        print("Diffusion pipeline loaded and moved to", self.device)

    def load_memorability_model(self, model_path):
        """
        Initialize AMNet and load checkpoint specified by model_path.
        """
        args = argparse.Namespace(
            dataset="memcat_vehicle",    # fine-tuned dataset
            experiment="",
            cnn="ResNet50FC",
            model_weights=model_path,
            dataset_root="datasets/memcat_vehicle",  # adjust path if needed
            images_dir="",
            splits_dir="",
            eval_images="",
            test_split="",
            val_split="",
            train_split="train_5",
            epoch_max=30,
            epoch_start=0,
            train_batch_size=128,
            test_batch_size=128,
            gpu=0,
            lstm_steps=3,
            last_step_prediction=False,
            att_off=False
        )

        hps = get_amnet_config(args)

        model = AMNet()
        model.init(hps)
        return model


    def _generate_prompt_from_score(self, score: float) -> str:
        """Map memorability score to inpainting prompt."""
        if score > 0.8:
            return "make the vehicle less distinctive, reduce unique features, make more generic"
        elif score > 0.6:
            return "slightly reduce vehicle memorability, soften distinctive features"
        else:
            return "minimal changes, preserve vehicle appearance"

    def _create_mask_from_attention(self, attention_map: np.ndarray, threshold: float = 0.7) -> np.ndarray:
        """
        Convert an attention heatmap (values in [0,1]) to a smooth mask for inpainting.
        Returns uint8 mask (0..255) as single-channel image.
        """
        if attention_map.max() <= 1.0 and attention_map.min() >= 0.0:
            am = attention_map.copy()
        else:
            # normalize if not normalized
            am = (attention_map - attention_map.min()) / (1e-8 + attention_map.max() - attention_map.min())

        # binary mask
        mask = (am > threshold).astype(np.uint8) * 255

        # morphological smoothing
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)

        return mask.astype(np.uint8)

    def _visualize_and_save(self, original_img_pil, attention_map, mask_img_pil, perturbed_img_pil,
                            orig_score, new_score, outpath_vis):
        """
        Create 2x3 visualization like in your earlier code and save as PNG.
        """
        # Ensure numpy arrays for diff
        orig_np = np.array(original_img_pil.resize((512, 512)))
        pert_np = np.array(perturbed_img_pil.resize((512, 512)))

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0, 0].imshow(original_img_pil)
        axes[0, 0].set_title(f"Original\nScore: {orig_score:.3f}")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(attention_map, cmap="hot")
        axes[0, 1].set_title("Attention Heatmap")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(mask_img_pil, cmap="gray")
        axes[0, 2].set_title("Perturbation Mask")
        axes[0, 2].axis("off")

        axes[1, 0].imshow(perturbed_img_pil)
        axes[1, 0].set_title(f"Perturbed\nScore: {new_score:.3f}")
        axes[1, 0].axis("off")

        diff = pert_np.astype(float) - orig_np.astype(float)
        axes[1, 1].imshow(np.abs(diff).astype(np.uint8))
        axes[1, 1].set_title("Absolute Difference")
        axes[1, 1].axis("off")

        axes[1, 2].bar(['Original', 'Perturbed'], [orig_score, new_score])
        axes[1, 2].set_ylabel('Memorability Score')
        axes[1, 2].set_title(f'Reduction: {orig_score-new_score:.3f}')

        plt.tight_layout()
        os.makedirs(os.path.dirname(outpath_vis), exist_ok=True)
        plt.savefig(outpath_vis, dpi=300, bbox_inches='tight')
        plt.close(fig)

    
    def process_folder(self, input_folder: str, output_root: str,
                       mask_threshold: float = 0.7,
                       sd_strength: float = 0.6,
                       guidance_scale: float = 7.5,
                       num_inference_steps: int = 30,
                       overwrite: bool = False):
        """
        Main entry: process all images in input_folder, save results under output_root.
        """
        input_folder = os.path.expanduser(input_folder)
        output_root = os.path.expanduser(output_root)
        os.makedirs(output_root, exist_ok=True)
        perturbed_dir = os.path.join(output_root, "perturbed")
        vis_dir = os.path.join(output_root, "vis")
        os.makedirs(perturbed_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

        # 1) Run AMNet on the whole folder once to get original scores & attention masks
        print("Running AMNet on input folder to obtain original scores & attention masks...")
        pr = self.mem_model.predict_memorability(input_folder)  # PredictionResult
        if not hasattr(pr, "predictions") or not hasattr(pr, "attention_masks"):
            raise RuntimeError("AMNet prediction result missing expected fields (predictions, attention_masks).")

        image_paths = pr.image_names
        original_scores = pr.predictions
        attention_masks_all = pr.attention_masks  # expected shape (N, T, L) numpy

        # Basic logging CSV
        csv_path = os.path.join(output_root, "logs.csv")
        write_header = not os.path.exists(csv_path)
        csv_file = open(csv_path, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if write_header:
            csv_writer.writerow([
                "image", "orig_score", "new_score",
                "reduction", "reduction_pct",
                "perturbed_path", "vis_path"
            ])

        n_images = len(image_paths)
        print(f"Found {n_images} images in AMNet result. Beginning per-image processing...")

        for i, img_path in enumerate(image_paths):
            try:
                base_name = os.path.basename(img_path)
                print(f"[{i+1}/{n_images}] Processing {base_name} ...")

                orig_score = float(original_scores[i])
                # Get attention map for this image (last timestep)
                att_raw = attention_masks_all[i]  # shape (T, L)
                att_last = att_raw[-1] if att_raw.ndim == 2 else att_raw[-1, :]
                L = att_last.shape[0]
                side = int(np.round(np.sqrt(L)))
                if side * side != L:
                    side = int(np.ceil(np.sqrt(L)))
                att_map = att_last.reshape(side, side)
                att_map = (att_map - att_map.min()) / (1e-8 + att_map.max() - att_map.min())

                # create mask
                mask = self._create_mask_from_attention(att_map, threshold=mask_threshold)

                # Load original image
                orig_pil = Image.open(img_path).convert("RGB")

                # Resize for SD
                image_sd = orig_pil.resize((512, 512))
                mask_sd = Image.fromarray(mask).convert("L").resize((512, 512))

                # Prompt from memorability
                prompt = self._generate_prompt_from_score(orig_score)

                # Run diffusion inpainting
                print(f"  Running SD inpainting (prompt: {prompt}) ...")
                generator = None
                if self.device == "cuda":
                    generator = torch.Generator(device="cuda").manual_seed(
                        int(time.time() % (2**31-1))
                    )

                out = self.diffusion_pipe(
                    prompt=prompt,
                    image=image_sd,
                    mask_image=mask_sd,
                    strength=sd_strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                )
                perturbed_img = out.images[0]

                # Save perturbed image
                perturbed_path = os.path.join(perturbed_dir, base_name)
                if not overwrite and os.path.exists(perturbed_path):
                    pass
                perturbed_img.save(perturbed_path)
                
                with tempfile.TemporaryDirectory() as td:
                    tmp_path = os.path.join(td, base_name)
                    perturbed_img.save(tmp_path)
                    pr_pert = self.mem_model.predict_memorability(td)
                    new_score = float(pr_pert.predictions[0])

                reduction = orig_score - new_score
                reduction_pct = (reduction / orig_score * 100.0) if orig_score != 0 else 0.0

                # Visualization
                mask_vis = Image.fromarray(mask).convert("L").resize(orig_pil.size)
                vis_path = os.path.join(vis_dir, f"{Path(base_name).stem}_vis.png")
                self._visualize_and_save(
                    orig_pil,
                    cv2.resize((att_map*255).astype(np.uint8), (224, 224)),
                    mask_vis,
                    perturbed_img,
                    orig_score,
                    new_score,
                    vis_path
                )

                # Write CSV row
                csv_writer.writerow([
                    img_path, orig_score, new_score,
                    reduction, reduction_pct,
                    perturbed_path, vis_path
                ])
                csv_file.flush()
                print(
                    f"  Done. Orig: {orig_score:.3f}, "
                    f"New: {new_score:.3f}, "
                    f"Reduction: {reduction:.3f} ({reduction_pct:.1f}%)"
                )

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                traceback.print_exc()
                csv_writer.writerow([img_path, "ERROR", str(e), "", "", "", ""])
                csv_file.flush()
                continue

        csv_file.close()
        print("Processing finished. Results saved to", output_root)
        return

# ---------------------------
# CLI
# ---------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch inpaint vehicle images to reduce memorability using AMNet attention.")
    parser.add_argument("--model", required=True, help="Path to AMNet checkpoint (weights_*.pkl)")
    parser.add_argument("--input_folder", required=True, help="Folder containing input images (all images will be scored & processed)")
    parser.add_argument("--output_root", default="outputs", help="Root folder for perturbed images, visualizations & logs")
    parser.add_argument("--mask_threshold", type=float, default=0.7, help="Attention threshold for mask creation (0-1)")
    parser.add_argument("--sd_strength", type=float, default=0.6, help="Stable Diffusion strength parameter (0-1) for inpainting")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale (CFG) for diffusion")
    parser.add_argument("--num_steps", type=int, default=30, help="Number of diffusion inference steps")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to place diffusion pipeline on")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing perturbed images")
    args = parser.parse_args()

    perturbator = MemorabilityPerturbator(args.model, device=args.device)
    perturbator.process_folder(args.input_folder, args.output_root,
                               mask_threshold=args.mask_threshold,
                               sd_strength=args.sd_strength,
                               guidance_scale=args.guidance_scale,
                               num_inference_steps=args.num_steps,
                               overwrite=args.overwrite)

if __name__ == "__main__":
    main()