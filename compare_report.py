#!/usr/bin/env python3
"""
compare_report.py

Generate a PDF report comparing original vs perturbed images
side by side with their predicted memorability scores.
"""

import os
import argparse
import tempfile
import shutil
import matplotlib.pyplot as plt
from PIL import Image

import torch
from amnet import AMNet
from config import HParameters
# AMNet from this repo
from amnet import AMNet
from config import get_amnet_config
import argparse


def load_memorability_model(model_path):
    """
    Initialize AMNet and load checkpoint specified by model_path.
    """
    args = argparse.Namespace(
        dataset="memcat_vehicle",    # fine-tuned dataset
        experiment="report_inference",  # give it a name
        cnn="ResNet50FC",
        model_weights=model_path,
        dataset_root="datasets/memcat_vehicle",  
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




def score_image(mem_model, image_path):
    """Run AMNet on a single image by wrapping it in a temp dir."""
    import numpy as np
    with tempfile.TemporaryDirectory() as td:
        fname = os.path.basename(image_path)
        tmp_img = os.path.join(td, fname)
        shutil.copy(image_path, tmp_img)
        pr = mem_model.predict_memorability(td)
        return float(pr.predictions[0])


def make_report(mem_model, original_dir, perturbed_dir, output_pdf):
    """Generate a PDF with side-by-side comparisons."""
    originals = sorted(os.listdir(original_dir))
    perturbed = sorted(os.listdir(perturbed_dir))
    common = sorted(set(originals) & set(perturbed))
    print(f"Found {len(common)} images for report.")

    from matplotlib.backends.backend_pdf import PdfPages
    pdf = PdfPages(output_pdf)

    for i, fname in enumerate(common, 1):
        print(f"[{i}/{len(common)}] Processing {fname} ...")
        orig_path = os.path.join(original_dir, fname)
        pert_path = os.path.join(perturbed_dir, fname)

        # Score both images
        try:
            pred_orig = score_image(mem_model, orig_path)
            pred_pert = score_image(mem_model, pert_path)
        except Exception as e:
            print(f"  Skipping {fname}, error while scoring: {e}")
            continue

        # Load images
        img_orig = Image.open(orig_path).convert("RGB")
        img_pert = Image.open(pert_path).convert("RGB")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img_orig)
        axes[0].set_title(f"Original\nScore: {pred_orig:.3f}")
        axes[0].axis("off")

        axes[1].imshow(img_pert)
        axes[1].set_title(f"Perturbed\nScore: {pred_pert:.3f}")
        axes[1].axis("off")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    pdf.close()
    print("PDF saved to", output_pdf)


def main():
    parser = argparse.ArgumentParser(description="Generate side-by-side PDF comparison report")
    parser.add_argument("--model", required=True, help="Path to AMNet checkpoint (weights_*.pkl)")
    parser.add_argument("--original_dir", required=True, help="Folder with original images")
    parser.add_argument("--perturbed_dir", required=True, help="Folder with perturbed images")
    parser.add_argument("--output_pdf", required=True, help="Output PDF path")
    args = parser.parse_args()

    mem_model = load_memorability_model(args.model)
    make_report(mem_model, args.original_dir, args.perturbed_dir, args.output_pdf)


if __name__ == "__main__":
    main()
