import os
import sys
import time
import csv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.unet3d import UNet3D
from datasets.brats_dataset import BraTSDataset
from gradcam_3d import GradCAM3D
from lime_3d import LIME3D

DATASET_PATH = r"D:\ENZO\PFC1\datasets\BraTS2020"
MODEL_PATH = r"../training/checkpoints/best_model.pth"
OUTPUT_DIR = "outputs/xai_comparison"

N_SAMPLES_EVAL = 5
SAMPLE_START = 0
TUMOR_CLASS = 3

LIME_BLOCK_SIZE = 16
LIME_N_SAMPLES = 200

N_STEPS_DELETION = 20

HEATMAP_THRESH_GRADCAM = 0.20
HEATMAP_THRESH_LIME = 0.40

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CMAP_SEG = mcolors.ListedColormap(["black", "#1f77b4", "#2ca02c", "#d62728"])
NORM_SEG = mcolors.BoundaryNorm([0, 1, 2, 3, 4], CMAP_SEG.N)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\nDevice: {DEVICE}")
print("\nLoading model...")

model = UNet3D().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("\nLoading dataset...")
dataset = BraTSDataset(DATASET_PATH, augment=False)

def get_tumor_score(model, tensor, tumor_class=TUMOR_CLASS):
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)
    return probs[:, tumor_class].mean().item()


def compute_fidelity(model, input_tensor, importance_map, k_fractions=None, tumor_class=TUMOR_CLASS):

    if k_fractions is None:
        k_fractions = [0.05, 0.10, 0.20, 0.30, 0.50]

    base_score = get_tumor_score(model, input_tensor, tumor_class)
    flat_imp = importance_map.flatten()
    drops = []
    imp_means = []

    for k in k_fractions:
        n_mask = max(1, int(len(flat_imp) * k))
        thresh = np.sort(flat_imp)[::-1][n_mask - 1]

        mask_3d = (importance_map >= thresh).astype(np.float32)
        mask_t = torch.tensor(mask_3d, device=DEVICE).unsqueeze(0).unsqueeze(0)
        masked = input_tensor * (1 - mask_t)

        masked_score = get_tumor_score(model, masked, tumor_class)
        drops.append(base_score - masked_score)
        imp_means.append(importance_map[importance_map >= thresh].mean())

    corr = float(np.corrcoef(imp_means, drops)[0, 1]) if len(drops) > 1 else 0.0
    return corr


def compute_infidelity(model, input_tensor, importance_map, n_pert=50, noise_std=0.1, tumor_class=TUMOR_CLASS):
    imp_flat = importance_map.flatten().astype(np.float32)
    base_score = get_tumor_score(model, input_tensor, tumor_class)
    sq_errors = []

    rng = np.random.default_rng(0)
    for _ in range(n_pert):
        noise = rng.normal(0, noise_std, input_tensor.shape).astype(np.float32)
        noise_t = torch.tensor(noise, device=DEVICE)
        perturbed = input_tensor - noise_t

        pert_score = get_tumor_score(model, perturbed, tumor_class)
        delta_f = base_score-pert_score
        noise_flat = noise.mean(axis=1).flatten()
        i_dot_d = float(np.dot(imp_flat, noise_flat))

        sq_errors.append((i_dot_d-delta_f)**2)

    return float(np.mean(sq_errors))


def compute_deletion_insertion(model, input_tensor, importance_map, n_steps=N_STEPS_DELETION, tumor_class=TUMOR_CLASS):

    flat_imp = importance_map.flatten()
    sorted_idx = np.argsort(flat_imp)[::-1]
    n_voxels = len(flat_imp)

    D, H, W = importance_map.shape
    base_score = get_tumor_score(model, input_tensor, tumor_class)
    blank = torch.zeros_like(input_tensor)

    steps = np.linspace(0, 1, n_steps + 1)
    del_scores = np.zeros(n_steps + 1)
    ins_scores = np.zeros(n_steps + 1)

    for i, frac in enumerate(steps):
        n_mod = int(frac * n_voxels)
        mask_flat = np.ones(n_voxels, dtype=np.float32)
        if n_mod > 0:
            mask_flat[sorted_idx[:n_mod]] = 0.0
        mask_3d = torch.tensor(mask_flat.reshape(D, H, W), device=DEVICE)
        mask_bc = mask_3d.unsqueeze(0).unsqueeze(0)
        del_input = input_tensor * mask_bc
        del_scores[i] = get_tumor_score(model, del_input, tumor_class)

        reveal_flat = np.zeros(n_voxels, dtype=np.float32)
        if n_mod > 0:
            reveal_flat[sorted_idx[:n_mod]] = 1.0
        reveal_3d = torch.tensor(reveal_flat.reshape(D, H, W), device=DEVICE)
        reveal_bc = reveal_3d.unsqueeze(0).unsqueeze(0)
        ins_input = blank + input_tensor * reveal_bc
        ins_scores[i] = get_tumor_score(model, ins_input, tumor_class)

    del_auc = float(np.trapezoid(del_scores, steps))
    ins_auc = float(np.trapezoid(ins_scores, steps))
    return steps, del_scores, ins_scores, del_auc, ins_auc


def plot_qualitative(image, mask_gt, pred_labels, gradcam_map, lime_map, slice_idx, sample_idx, output_dir):
    mri = image[3, :, :, slice_idx].numpy()
    gt_s = mask_gt[:, :, slice_idx].numpy()
    pr_s = pred_labels[:, :, slice_idx]
    gc_s = gradcam_map[:, :, slice_idx]
    li_s = lime_map[:, :, slice_idx]

    gc_masked = np.where(gc_s > HEATMAP_THRESH_GRADCAM, gc_s, np.nan)
    li_masked = np.where(li_s > HEATMAP_THRESH_LIME,    li_s, np.nan)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f"Sample {sample_idx} - axial z={slice_idx}", fontsize=13)

    axes[0].imshow(mri, cmap="gray")
    axes[0].set_title("FLAIR MRI", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(gt_s, cmap=CMAP_SEG, norm=NORM_SEG, interpolation="nearest")
    axes[1].set_title("Ground Truth", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(pr_s, cmap=CMAP_SEG, norm=NORM_SEG, interpolation="nearest")
    axes[2].set_title("U-Net Prediction", fontsize=11)
    axes[2].axis("off")

    axes[3].imshow(mri, cmap="gray")
    im_gc = axes[3].imshow(gc_masked, cmap="jet", alpha=0.55, vmin=0, vmax=1)
    axes[3].contour(gt_s > 0, colors="white", linewidths=0.8)
    axes[3].set_title("Grad-CAM", fontsize=11)
    axes[3].axis("off")
    plt.colorbar(im_gc, ax=axes[3], fraction=0.046, pad=0.04)

    axes[4].imshow(mri, cmap="gray")
    im_li = axes[4].imshow(li_masked, cmap="jet", alpha=0.55, vmin=0, vmax=1)
    axes[4].contour(gt_s > 0, colors="white", linewidths=0.8)
    axes[4].set_title("LIME 3D", fontsize=11)
    axes[4].axis("off")
    plt.colorbar(im_li, ax=axes[4], fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = os.path.join(output_dir, f"qualitative_sample{sample_idx:02d}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def plot_multistrip(image, heatmap_np, mask_gt, slice_idx, label, sample_idx, output_dir, thresh=0.20):
    n_strips = 5
    half = n_strips // 2
    D = image.shape[1]
    indices = [max(0, min(D - 1, slice_idx + off)) for off in range(-half, half + 1)]

    fig, axes = plt.subplots(2, n_strips, figsize=(4 * n_strips, 8))
    fig.suptitle(f"{label} — Multi-slice strip  (sample {sample_idx})", fontsize=13, y=1.01)

    for col, si in enumerate(indices):
        mri_s = image[3, :, :, si].numpy()
        heat_s = heatmap_np[:, :, si]
        gt_s = mask_gt[:, :, si].numpy()
        heat_m = np.where(heat_s > thresh, heat_s, np.nan)

        axes[0,col].imshow(mri_s, cmap="gray")
        axes[0,col].set_title(f"FLAIR  z={si}", fontsize=9)
        axes[0,col].axis("off")

        axes[1,col].imshow(mri_s, cmap="gray")
        axes[1,col].imshow(heat_m, cmap="jet", alpha=0.55, vmin=0, vmax=1)
        axes[1,col].contour(gt_s > 0, colors="white", linewidths=0.8)
        axes[1,col].set_title(f"{label}  z={si}", fontsize=9)
        axes[1,col].axis("off")

    plt.tight_layout()
    fname = f"multistrip_{label.lower().replace('-','').replace(' ','_')}_sample{sample_idx:02d}.png"
    path = os.path.join(output_dir, fname)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def plot_deletion_insertion_curves(all_gc_del, all_gc_ins, all_li_del, all_li_ins, steps, output_dir):
    gc_del_mean = np.mean(all_gc_del, axis=0)
    gc_ins_mean = np.mean(all_gc_ins, axis=0)
    li_del_mean = np.mean(all_li_del, axis=0)
    li_ins_mean = np.mean(all_li_ins, axis=0)

    gc_del_std = np.std(all_gc_del, axis=0)
    gc_ins_std = np.std(all_gc_ins, axis=0)
    li_del_std = np.std(all_li_del, axis=0)
    li_ins_std = np.std(all_li_ins, axis=0)

    fig,axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Deletion & Insertion Curves — Grad-CAM vs LIME 3D", fontsize=13)

    ax = axes[0]
    ax.plot(steps, gc_del_mean, "b-o", markersize=4, label="Grad-CAM")
    ax.fill_between(steps, gc_del_mean - gc_del_std, gc_del_mean + gc_del_std, alpha=0.2, color="blue")
    ax.plot(steps, li_del_mean, "r-s", markersize=4, label="LIME 3D")
    ax.fill_between(steps, li_del_mean - li_del_std, li_del_mean + li_del_std, alpha=0.2, color="red")
    ax.set_xlabel("Fraction of voxels deleted")
    ax.set_ylabel("Model score (ET class)")
    ax.set_title("Deletion ( - AUC = better)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(steps, gc_ins_mean, "b-o", markersize=4, label="Grad-CAM")
    ax.fill_between(steps, gc_ins_mean - gc_ins_std, gc_ins_mean + gc_ins_std, alpha=0.2, color="blue")
    ax.plot(steps, li_ins_mean, "r-s", markersize=4, label="LIME 3D")
    ax.fill_between(steps, li_ins_mean - li_ins_std, li_ins_mean + li_ins_std, alpha=0.2, color="red")
    ax.set_xlabel("Fraction of voxels inserted")
    ax.set_ylabel("Model score (ET class)")
    ax.set_title("Insertion ( + AUC = better)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "deletion_insertion_curves.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path

print(f"\nAnalizando {N_SAMPLES_EVAL} muestras...\n")

results = []

all_gc_del, all_gc_ins = [],[]
all_li_del, all_li_ins = [],[]
steps_ref = None

for sample_idx in range(SAMPLE_START, SAMPLE_START + N_SAMPLES_EVAL):
    print(f"{'='*60}")
    print(f"Sample {sample_idx}")
    print(f"{'='*60}")

    image, mask_gt = dataset[sample_idx]
    input_tensor = image.unsqueeze(0).to(DEVICE)

    tumor_per_axial = (mask_gt > 0).sum(dim=(0, 1)).numpy()
    slice_idx = int(tumor_per_axial.argmax())
    print(f"Corte axial con más tumor: z={slice_idx}")

    print("\n[Grad-CAM] Computando...")
    gc_explainer = GradCAM3D(model, target_layer=model.up4.conv)

    t0 = time.perf_counter()
    gc_map, gc_pred = gc_explainer.compute(input_tensor, target_class=TUMOR_CLASS)
    gc_time = time.perf_counter()-t0
    gc_explainer.remove_hooks()

    print(f"[Grad-CAM] Tiempo: {gc_time:.2f}s")

    print(f"\n[LIME] Computando ({LIME_N_SAMPLES} perturbaciones)...")
    lime_explainer = LIME3D(model, block_size = LIME_BLOCK_SIZE, n_samples = LIME_N_SAMPLES, tumor_class = TUMOR_CLASS, seed = 42 + sample_idx,)
    t0 = time.perf_counter()
    li_map, li_pred = lime_explainer.compute(input_tensor, verbose=True)
    lime_time = time.perf_counter()-t0
    print(f"[LIME] Tiempo: {lime_time:.2f}s")
    print("\nCalculando métricas...")

    gc_fidelity = compute_fidelity(model, input_tensor, gc_map)
    gc_infidelity = compute_infidelity(model, input_tensor, gc_map)
    li_fidelity = compute_fidelity(model, input_tensor, li_map)
    li_infidelity = compute_infidelity(model, input_tensor, li_map)

    steps_ref, gc_del, gc_ins, gc_del_auc, gc_ins_auc = compute_deletion_insertion(model, input_tensor, gc_map)
    _, li_del, li_ins, li_del_auc, li_ins_auc = compute_deletion_insertion(model, input_tensor, li_map)

    all_gc_del.append(gc_del)
    all_gc_ins.append(gc_ins)
    all_li_del.append(li_del)
    all_li_ins.append(li_ins)

    print(f"\n  Grad-CAM  |  Fidelity: {gc_fidelity:.4f}  Infidelity: {gc_infidelity:.6f} "f"Del AUC: {gc_del_auc:.4f}  Ins AUC: {gc_ins_auc:.4f}  Time: {gc_time:.2f}s")
    print(f"  LIME 3D   |  Fidelity: {li_fidelity:.4f}  Infidelity: {li_infidelity:.6f} "f"Del AUC: {li_del_auc:.4f}  Ins AUC: {li_ins_auc:.4f}  Time: {lime_time:.2f}s")

    results.append({
        "sample": sample_idx,
        "gc_fidelity": gc_fidelity,
        "gc_infidelity": gc_infidelity,
        "gc_deletion_auc": gc_del_auc,
        "gc_insertion_auc": gc_ins_auc,
        "gc_time_s": round(gc_time, 3),
        "li_fidelity": li_fidelity,
        "li_infidelity": li_infidelity,
        "li_deletion_auc": li_del_auc,
        "li_insertion_auc": li_ins_auc,
        "li_time_s": round(lime_time, 3),
    })

    print("\nGenerando figuras...")
    plot_qualitative(image, mask_gt, gc_pred, gc_map, li_map, slice_idx, sample_idx, OUTPUT_DIR)

    plot_multistrip(image, gc_map, mask_gt, slice_idx, "Grad-CAM", sample_idx, OUTPUT_DIR, thresh=HEATMAP_THRESH_GRADCAM)
    plot_multistrip(image, li_map, mask_gt, slice_idx, "LIME 3D",  sample_idx, OUTPUT_DIR, thresh=HEATMAP_THRESH_LIME)

    print(f"Sample {sample_idx} completado.\n")

csv_path = os.path.join(OUTPUT_DIR, "comparison_table.csv")
fieldnames = list(results[0].keys())

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

avg = {k: round(float(np.mean([r[k] for r in results])), 5)
       for k in fieldnames if k != "sample"}
avg["sample"] = "MEAN"

with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writerow(avg)

print(f"\nTabla guardada: {csv_path}")

print("\n" + "="*70)
print("RESUMEN COMPARATIVO  (promedios sobre {} muestras)".format(N_SAMPLES_EVAL))
print("─"*70)
print(f"{'Método':<12} {'Fidelity↑':>12} {'Infidelity↓':>13} {'Del AUC↓':>10} {'Ins AUC↑':>10} {'Time(s)↓':>10}")
print("─"*70)
print(f"{'Grad-CAM':<12} " f"{avg['gc_fidelity']:>12.4f} " f"{avg['gc_infidelity']:>13.6f}" f"{avg['gc_deletion_auc']:>10.4f}" f"{avg['gc_insertion_auc']:>10.4f}" f"{avg['gc_time_s']:>10.3f}")
print(f"{'LIME 3D':<12} " f"{avg['li_fidelity']:>12.4f} " f"{avg['li_infidelity']:>13.6f} " f"{avg['li_deletion_auc']:>10.4f} " f"{avg['li_insertion_auc']:>10.4f} " f"{avg['li_time_s']:>10.3f}")
print("="*70)
print("\nInterpretación:")
print("Fidelity   +  más alto = mejor alineación con el modelo")
print("Infidelity -  más bajo = mejor consistencia local")
print("Del AUC    -  más bajo = regiones importantes más precisas")
print("Ins AUC    +  más alto = regiones importantes más precisas")
print("Time(s)    -  más bajo = más eficiente")

if steps_ref is not None:
    plot_deletion_insertion_curves(all_gc_del, all_gc_ins, all_li_del, all_li_ins, steps_ref, OUTPUT_DIR)
print("\n¡Análisis XAI completado!")
print(f"Resultados en: {OUTPUT_DIR}")
