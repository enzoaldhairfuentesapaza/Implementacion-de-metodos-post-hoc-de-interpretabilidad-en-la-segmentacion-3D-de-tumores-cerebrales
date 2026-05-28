import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import (
    DataLoader,
    random_split
)

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

from models.unet3d import UNet3D

from datasets.brats_dataset import (BraTSDataset)

from training.losses import (dice_score,iou_score)

# Dataset path
DATASET_PATH = (r"D:\ENZO\PFC1\datasets\BraTS2020_TrainingData")

# Path del modelo entrenado
MODEL_PATH = (r"checkpoints/best_model.pth")

# Batch size
BATCH_SIZE = 1

# Número de imágenes a guardar
NUM_VISUALIZATIONS = 5

# Output folder
OUTPUT_DIR = ("outputs/predictions")

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nLoading dataset...\n")

dataset = BraTSDataset(DATASET_PATH,augment=False)


dataset_size = len(dataset)
test_size = int(dataset_size * 0.15)
val_size = int(dataset_size * 0.15)
train_size = (dataset_size - val_size - test_size)

_, _, test_dataset = random_split(dataset, [train_size, val_size, test_size])

test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)

print("\nLoading model...\n")

model = UNet3D().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("Model loaded successfully!")


total_dice = 0.0
total_iou = 0.0
saved_images = 0

print("\nStarting evaluation...\n")

with torch.no_grad():

    for idx, (images, masks) in enumerate(tqdm(test_loader)):

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)
        dice = dice_score(outputs, masks)

        iou = iou_score(outputs, masks)

        total_dice += dice

        total_iou += iou

        if saved_images < NUM_VISUALIZATIONS:

            pred = torch.argmax(
                outputs,
                dim=1
            )

            image_np = (
                images[0, 0]
                .cpu()
                .numpy()
            )

            mask_np = (
                masks[0]
                .cpu()
                .numpy()
            )

            pred_np = (
                pred[0]
                .cpu()
                .numpy()
            )

            middle = image_np.shape[0] // 2

            fig, axes = plt.subplots(1,3,figsize=(15, 5))

            axes[0].imshow(image_np[middle], cmap="gray")
            axes[0].set_title("MRI")
            axes[0].axis("off")

            axes[1].imshow(mask_np[middle])
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            axes[2].imshow(pred_np[middle])
            axes[2].set_title("Prediction")
            axes[2].axis("off")

            plt.tight_layout()

            save_path = os.path.join(
                OUTPUT_DIR,
                f"prediction_{idx}.png"
            )

            plt.savefig(save_path)

            plt.close()

            saved_images += 1


avg_dice = (
    total_dice /
    len(test_loader)
)

avg_iou = (
    total_iou /
    len(test_loader)
)

print("\n========================================")
print("FINAL TEST RESULTS")
print("----------------------------------------")
print(f"Average Dice: {avg_dice:.4f}")
print(f"Average IoU : {avg_iou:.4f}")
print("========================================\n")