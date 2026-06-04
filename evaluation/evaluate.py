import os
import sys
import csv

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.unet3d import UNet3D
from datasets.brats_dataset import BraTSDataset
from training.losses import dice_score, iou_score, dice_score_regions

DATASET_PATH = r"D:\ENZO\PFC1\datasets\BraTS2020"
MODEL_PATH = r"../training/checkpoints/best_model.pth"
OUTPUT_DIR = "outputs/predictions"

BATCH_SIZE = 1
NUM_VISUALIZATIONS = 5
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CMAP_SEG = mcolors.ListedColormap(["black", "#1f77b4", "#2ca02c", "#d62728"])
NORM_SEG = mcolors.BoundaryNorm([0, 1, 2, 3, 4], CMAP_SEG.N)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\nDevice: {DEVICE}")
print("\nLoading dataset...\n")

dataset = BraTSDataset(DATASET_PATH, augment=False)
dataset_size = len(dataset)
test_size = int(dataset_size*0.15)
val_size = int(dataset_size*0.15)
train_size = dataset_size - val_size - test_size

generator = torch.Generator().manual_seed(SEED)
_, _, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Test samples: {len(test_dataset)}")

print("\nLoading model...\n")

model = UNet3D().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model loaded successfully!")

print("\nStarting evaluation...\n")

totals = {"dice": 0.0, "iou": 0.0, "WT": 0.0, "TC": 0.0, "ET": 0.0}
saved_images = 0
csv_rows = []

with torch.no_grad():
    for idx, (images, masks) in enumerate(tqdm(test_loader, desc="Evaluating")):

        images  = images.to(DEVICE)
        masks = masks.to(DEVICE)
        outputs = model(images)

        d = dice_score(outputs, masks)
        iou = iou_score(outputs, masks)
        regions = dice_score_regions(outputs, masks)

        totals["dice"] += d
        totals["iou"] += iou
        for k in ["WT", "TC", "ET"]:
            totals[k] += regions[k]

        csv_rows.append({
            "sample": idx,
            "dice": round(d, 4),
            "iou": round(iou, 4),
            "WT": round(regions["WT"], 4),
            "TC": round(regions["TC"], 4),
            "ET": round(regions["ET"], 4),
        })
        
        if saved_images < NUM_VISUALIZATIONS:

            pred_labels = torch.argmax(outputs, dim=1)
            image_np = images[0, 3].cpu().numpy() 
            mask_np = masks[0].cpu().numpy()
            pred_np = pred_labels[0].cpu().numpy()

            tumor_per_slice = (mask_np > 0).sum(axis=(1, 2))
            middle = int(tumor_per_slice.argmax())

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f"Sample {idx}", fontsize=13)

            axes[0].imshow(image_np[middle], cmap="gray")
            axes[0].set_title("FLAIR MRI")
            axes[0].axis("off")

            axes[1].imshow(mask_np[middle], cmap=CMAP_SEG, norm=NORM_SEG, interpolation="nearest")
            axes[1].set_title(f"Ground Truth  (Dice={d:.3f})")
            axes[1].axis("off")

            axes[2].imshow(pred_np[middle], cmap=CMAP_SEG, norm=NORM_SEG, interpolation="nearest")
            axes[2].set_title("Prediction")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"prediction_{idx:03d}.png"), dpi=150, bbox_inches="tight")
            plt.close()
            saved_images += 1

n = len(test_loader)

print("\n" + "="*50)
print("FINAL TEST RESULTS")
print("─"*50)
print(f"Average Dice (mean WT/TC/ET) : {totals['dice']/n:.4f}")
print(f"Average IoU  (mean WT/TC/ET) : {totals['iou']/n:.4f}")
print("─"*50)
print(f"Dice WT (Whole Tumor)        : {totals['WT']/n:.4f}")
print(f"Dice TC (Tumor Core)         : {totals['TC']/n:.4f}")
print(f"Dice ET (Enhancing Tumor)    : {totals['ET']/n:.4f}")
print("="*50 + "\n")

csv_path = os.path.join(OUTPUT_DIR, "results_per_sample.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["sample", "dice", "iou", "WT", "TC", "ET"])
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"Results saved to {csv_path}")
