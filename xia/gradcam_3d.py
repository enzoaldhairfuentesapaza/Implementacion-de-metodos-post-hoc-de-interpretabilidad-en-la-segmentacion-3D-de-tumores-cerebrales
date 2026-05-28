import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from train_unet3d import (
    BraTSDataset,
    UNet3D
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


DATASET_PATH = r"D:\ENZO\PFC1\datasets\BraTS_subset"

MODEL_PATH   = r"D:\ENZO\PFC1\models\unet3d_brats.pth"

OUTPUT_DIR   = r"D:\ENZO\PFC1\outputs\gradcam"

os.makedirs(OUTPUT_DIR, exist_ok=True)

dataset = BraTSDataset(DATASET_PATH)

image, mask = dataset[0]


model = UNet3D().to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()


input_tensor = image.unsqueeze(0).to(device)   # [1,4,128,128,128]


activations = {}
gradients   = {}

def forward_hook(module, input, output):
    activations["value"] = output

def backward_hook(module, grad_input, grad_output):
    gradients["value"] = grad_output[0]


target_layer = model.up4.conv      # DoubleConv inside the last Up block
                                   # shape: [1, 32, 128, 128, 128]

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)



output = model(input_tensor)      # [1, 4, 128, 128, 128]


pred = torch.argmax(output, dim=1).squeeze(0)   # [128,128,128]


tumor_class = 3    # enhancing tumor (BraTS label 4 → remapped to 3)



probs = torch.softmax(output, dim=1)           # [1,4,128,128,128]

score = (output[0, tumor_class] * probs[0, tumor_class]).sum()

model.zero_grad()

score.backward()


act  = activations["value"]     # [1, C, D, H, W]
grad = gradients["value"]       # [1, C, D, H, W]

pooled_gradients = grad.mean(dim=[0, 2, 3, 4])

weighted = act.clone()

for i in range(weighted.shape[1]):
    weighted[:, i] *= pooled_gradients[i]

heatmap = weighted.sum(dim=1).squeeze()    # [D, H, W]
heatmap = torch.relu(heatmap)


heatmap = F.interpolate(
    heatmap.unsqueeze(0).unsqueeze(0),
    size=(128, 128, 128),
    mode="trilinear",
    align_corners=False
).squeeze()



p1  = torch.quantile(heatmap, 0.01)
p99 = torch.quantile(heatmap, 0.99)

heatmap = torch.clamp(heatmap, p1, p99)
heatmap = (heatmap - p1) / (p99 - p1 + 1e-8)

heatmap_np = heatmap.detach().cpu().numpy()   # [0, 1]



slice_idx = int(
    torch.argmax(
        (mask > 0).float().sum(dim=(0, 1))
    ).item()
)

print(f"Best slice (axial): {slice_idx}")

modalities = ["T1", "T1ce", "T2", "FLAIR"]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for i in range(4):
    axes[i].imshow(image[i, :, :, slice_idx], cmap="gray")
    axes[i].set_title(modalities[i])
    axes[i].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "modalities.png"), dpi=300, bbox_inches="tight")
plt.show()

mri_slice     = image[3, :, :, slice_idx].numpy()          # FLAIR
gt_slice      = mask[:, :, slice_idx].numpy()
pred_slice    = pred[:, :, slice_idx].detach().cpu().numpy()
heatmap_slice = heatmap_np[:, :, slice_idx]



heatmap_masked = np.where(
    heatmap_slice > 0.2,
    heatmap_slice,
    np.nan
)


fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(mri_slice, cmap="gray")
axes[0].set_title("FLAIR MRI", fontsize=13)
axes[0].axis("off")

axes[1].imshow(gt_slice, cmap="jet", vmin=0, vmax=3)
axes[1].set_title("Ground Truth", fontsize=13)
axes[1].axis("off")

axes[2].imshow(pred_slice, cmap="jet", vmin=0, vmax=3)
axes[2].set_title("U-Net Prediction", fontsize=13)
axes[2].axis("off")

axes[3].imshow(mri_slice, cmap="gray")

im = axes[3].imshow(
    heatmap_masked,
    cmap="jet",
    alpha=0.55,
    vmin=0,
    vmax=1
)

axes[3].contour(gt_slice, colors="white", linewidths=1)

axes[3].set_title("Grad-CAM (up4.conv · class 3)", fontsize=13)
axes[3].axis("off")

plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gradcam_result.png"), dpi=300, bbox_inches="tight")
plt.show()

print("Grad-CAM visualization saved.")



n_strips = 5
half     = n_strips // 2

strip_indices = [
    max(0, min(127, slice_idx + offset))
    for offset in range(-half, half + 1)
]

fig, axes = plt.subplots(2, n_strips, figsize=(4 * n_strips, 8))

for col, si in enumerate(strip_indices):

    mri_s  = image[3, :, :, si].numpy()
    heat_s = heatmap_np[:, :, si]
    gt_s   = mask[:, :, si].numpy()

    heat_m = np.where(heat_s > 0.2, heat_s, np.nan)

    axes[0, col].imshow(mri_s, cmap="gray")
    axes[0, col].set_title(f"FLAIR  z={si}", fontsize=9)
    axes[0, col].axis("off")

    axes[1, col].imshow(mri_s, cmap="gray")
    axes[1, col].imshow(heat_m, cmap="jet", alpha=0.55, vmin=0, vmax=1)
    axes[1, col].contour(gt_s, colors="white", linewidths=0.8)
    axes[1, col].set_title(f"Grad-CAM  z={si}", fontsize=9)
    axes[1, col].axis("off")

plt.suptitle("Grad-CAM — Multi-slice Strip", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gradcam_multistrip.png"), dpi=300, bbox_inches="tight")
plt.show()

print("Multi-slice strip saved.")
