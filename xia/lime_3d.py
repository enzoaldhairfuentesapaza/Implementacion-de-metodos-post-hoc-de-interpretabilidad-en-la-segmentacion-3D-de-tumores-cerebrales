import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from sklearn.linear_model import Ridge

from train_unet3d import (
    BraTSDataset,
    UNet3D
)

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# ============================================================
# PATHS
# ============================================================

DATASET_PATH = r"D:\ENZO\PFC1\datasets\BraTS_subset"

MODEL_PATH = r"D:\ENZO\PFC1\models\unet3d_brats.pth"

OUTPUT_DIR = r"D:\ENZO\PFC1\outputs\lime"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# LOAD DATASET
# ============================================================

dataset = BraTSDataset(DATASET_PATH)

image, mask = dataset[0]

# image:
# [4,128,128,128]

# mask:
# [128,128,128]

# ============================================================
# LOAD MODEL
# ============================================================

model = UNet3D().to(device)

model.load_state_dict(
    torch.load(MODEL_PATH)
)

model.eval()

# ============================================================
# INPUT
# ============================================================

input_tensor = image.unsqueeze(0).to(device)

# ============================================================
# FORWARD
# ============================================================

with torch.no_grad():

    output = model(input_tensor)

# ============================================================
# PREDICTION
# ============================================================

pred = torch.argmax(
    output,
    dim=1
).squeeze(0)

# ============================================================
# BEST SLICE
# ============================================================

slice_idx = torch.argmax(
    (mask > 0).sum(dim=(0,1))
).item()

print("Best slice:", slice_idx)

# ============================================================
# CREATE BLOCKS
# ============================================================

def create_blocks(
    volume_shape,
    block_size=16
):

    blocks = []

    D,H,W = volume_shape

    for z in range(0, D, block_size):

        for y in range(0, H, block_size):

            for x in range(0, W, block_size):

                z2 = min(z + block_size, D)
                y2 = min(y + block_size, H)
                x2 = min(x + block_size, W)

                blocks.append(
                    (
                        z,z2,
                        y,y2,
                        x,x2
                    )
                )

    return blocks

# ============================================================
# PERTURBATION FUNCTION
# ============================================================

def perturb_image(
    image,
    blocks,
    mask_vector
):

    perturbed = image.clone()

    for i, active in enumerate(mask_vector):

        if active == 0:

            z1,z2,y1,y2,x1,x2 = blocks[i]

            perturbed[
                :,
                :,
                z1:z2,
                y1:y2,
                x1:x2
            ] = 0

    return perturbed

# ============================================================
# CREATE BLOCKS
# ============================================================

blocks = create_blocks(
    (128,128,128),
    block_size=16
)

print("Number of blocks:", len(blocks))

# ============================================================
# GENERATE LIME DATASET
# ============================================================

num_samples = 50

tumor_class = 3

masks = np.random.randint(
    0,
    2,
    size=(num_samples, len(blocks))
)

scores = []

# ============================================================
# RUN PERTURBATIONS
# ============================================================

for i, mask_vector in enumerate(masks):

    print(
        f"Sample {i+1}/{num_samples}"
    )

    perturbed = perturb_image(
        input_tensor,
        blocks,
        mask_vector
    )

    with torch.no_grad():

        output = model(perturbed)

        probs = torch.softmax(
            output,
            dim=1
        )

        score = probs[
            :,
            tumor_class
        ].mean().item()

    scores.append(score)

# ============================================================
# LIME REGRESSION
# ============================================================

explainer = Ridge(alpha=1.0)

explainer.fit(
    masks,
    scores
)

importance = explainer.coef_

# ============================================================
# CREATE LIME MAP
# ============================================================

lime_map = np.zeros(
    (128,128,128)
)

for i, block in enumerate(blocks):

    z1,z2,y1,y2,x1,x2 = block

    lime_map[
        z1:z2,
        y1:y2,
        x1:x2
    ] = importance[i]

# ============================================================
# NORMALIZE
# ============================================================

lime_map -= lime_map.min()

lime_map /= (
    lime_map.max() + 1e-8
)

# ============================================================
# VISUALIZATION
# ============================================================

mri_slice = image[
    3,
    :,
    :,
    slice_idx
].numpy()

gt_slice = mask[
    :,
    :,
    slice_idx
].numpy()

pred_slice = pred[
    :,
    :,
    slice_idx
].detach().cpu().numpy()

lime_slice = lime_map[
    :,
    :,
    slice_idx
]

# ============================================================
# CLEAN HEATMAP
# ============================================================

lime_slice = np.where(
    lime_slice > 0.4,
    lime_slice,
    np.nan
)

# ============================================================
# MULTIMODAL MRI
# ============================================================

modalities = [
    "T1",
    "T1ce",
    "T2",
    "FLAIR"
]

fig, axes = plt.subplots(
    1,
    4,
    figsize=(16,4)
)

for i in range(4):

    axes[i].imshow(
        image[
            i,
            :,
            :,
            slice_idx
        ],
        cmap="gray"
    )

    axes[i].set_title(
        modalities[i]
    )

    axes[i].axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "modalities.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# MAIN FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(20,5)
)

# ------------------------------------------------------------
# MRI
# ------------------------------------------------------------

axes[0].imshow(
    mri_slice,
    cmap="gray"
)

axes[0].set_title(
    "FLAIR MRI"
)

axes[0].axis("off")

# ------------------------------------------------------------
# GROUND TRUTH
# ------------------------------------------------------------

axes[1].imshow(
    gt_slice,
    cmap="jet"
)

axes[1].set_title(
    "Ground Truth"
)

axes[1].axis("off")

# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

axes[2].imshow(
    pred_slice,
    cmap="jet"
)

axes[2].set_title(
    "U-Net Prediction"
)

axes[2].axis("off")

# ------------------------------------------------------------
# LIME
# ------------------------------------------------------------

axes[3].imshow(
    mri_slice,
    cmap="gray"
)

axes[3].imshow(
    lime_slice,
    cmap="jet",
    alpha=0.5
)

axes[3].contour(
    gt_slice,
    colors="white",
    linewidths=1
)

axes[3].set_title(
    "LIME 3D"
)

axes[3].axis("off")

# ============================================================
# SAVE
# ============================================================

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "lime_result.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("LIME visualization saved.")