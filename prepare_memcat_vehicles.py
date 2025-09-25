import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Paths (adjust if needed)
memcat_root = os.path.expanduser("~/Downloads/MemCat_data/MemCat/vehicle")
csv_path = os.path.expanduser("~/Downloads/MemCat_data/memcat_image_data.csv")
output_root = os.path.expanduser("~/proj/AMNet/datasets/memcat_vehicle")

# Output dirs
images_out = os.path.join(output_root, "images")
splits_out = os.path.join(output_root, "splits")
os.makedirs(images_out, exist_ok=True)
os.makedirs(splits_out, exist_ok=True)

# Load CSV with scores
df = pd.read_csv(csv_path)
# Clean column names
df.columns = [c.strip().lower() for c in df.columns]

# Use the correct column names from your CSV
if "image_file" not in df.columns:
    raise ValueError(f"CSV columns found: {df.columns}, expected 'image_file'")

# Choose which memorability score to use (with or without FA correction)
memorability_col = "memorability_w_fa_correction"  # or "memorability_wo_fa_correction"
if memorability_col not in df.columns:
    raise ValueError(f"CSV columns found: {df.columns}, expected '{memorability_col}'")

# Filter for vehicle category only (if needed)
if "category" in df.columns:
    df = df[df["category"].str.lower() == "vehicle"]
    print(f" Filtered to {len(df)} vehicle images")

# Collect all vehicle images and scores
records = []
for root, _, files in os.walk(memcat_root):
    for fname in files:
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            # Match using the correct column name
            row = df[df["image_file"] == fname]
            if row.empty:
                print(f"No score found for {fname}, skipping")
                continue
            score = float(row[memorability_col].values[0])

            # Copy into output/images/
            dst_name = fname
            src_path = os.path.join(root, fname)
            dst_path = os.path.join(images_out, dst_name)
            if not os.path.exists(dst_path):
                shutil.copy(src_path, dst_path)

            records.append((dst_name, score))

print(f" Collected {len(records)} vehicle images with scores")

# Save annotations.txt
ann_path = os.path.join(output_root, "annotations.txt")
with open(ann_path, "w") as f:
    for fname, score in records:
        f.write(f"{fname} {score:.6f}\n")
print(f"Saved annotations.txt with {len(records)} entries")

# Train/Val/Test split (70/15/15)
filenames = [r[0] for r in records]
train_files, test_files = train_test_split(filenames, test_size=0.30, random_state=42)
val_files, test_files = train_test_split(test_files, test_size=0.50, random_state=42)

def save_split(name, items):
    path = os.path.join(splits_out, f"{name}.txt")
    with open(path, "w") as f:
        for fname in items:
            f.write(f"{fname}\n")
    print(f" Saved {name}.txt with {len(items)} samples")

save_split("train", train_files)
save_split("val", val_files)
save_split("test", test_files)

print(" Done! Dataset prepared at:", output_root)
