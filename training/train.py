import os
import sys

import torch
import torch.optim as optim

from torch.utils.data import (
    DataLoader,
    random_split
)

from tqdm import tqdm

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)


from models.unet3d import UNet3D

from datasets.brats_dataset import (
    BraTSDataset
)

from training.losses import (
    SegmentationLoss,
    dice_score,
    iou_score
)


# ===========
# PARAMETERS
# ===========

# Dataset path
DATASET_PATH = (r"D:\ENZO\PFC1\datasets\BraTS2020_TrainingData")

# Batch size
BATCH_SIZE = 1

# Número de epochs
EPOCHS = 50

# Learning rate
LEARNING_RATE = 1e-4

# Validation split
VALIDATION_SPLIT = 0.15

# Test split
TEST_SPLIT = 0.15

# Número de workers
NUM_WORKERS = 0

# Checkpoint path
CHECKPOINT_DIR = "checkpoints"

# Device
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

print("\nLoading dataset...\n")

dataset = BraTSDataset(
    DATASET_PATH,
    augment=True
)

dataset_size = len(dataset)

test_size = int(
    dataset_size * TEST_SPLIT
)

val_size = int(
    dataset_size * VALIDATION_SPLIT
)

train_size = (
    dataset_size -
    val_size -
    test_size
)

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size]
)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

print("\nLoading model...\n")

model = UNet3D().to(DEVICE)

print(model)

criterion = SegmentationLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


best_val_dice = 0.0


print("\nStarting training...\n")

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0
    train_dice = 0.0
    train_iou = 0.0

    train_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}"
    )

    for images, masks in train_bar:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        outputs = model(images)

        loss = criterion(
            outputs,
            masks
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        train_dice += dice_score(
            outputs.detach(),
            masks
        )

        train_iou += iou_score(
            outputs.detach(),
            masks
        )

        train_bar.set_postfix({
            "Loss": loss.item()
        })

    train_loss /= len(train_loader)
    train_dice /= len(train_loader)
    train_iou /= len(train_loader)

    model.eval()

    val_loss = 0.0

    val_dice = 0.0

    val_iou = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)

            masks = masks.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                masks
            )

            val_loss += loss.item()

            val_dice += dice_score(
                outputs,
                masks
            )

            val_iou += iou_score(
                outputs,
                masks
            )

    val_loss /= len(val_loader)
    val_dice /= len(val_loader)
    val_iou /= len(val_loader)

    print("\n========================================")
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print("----------------------------------------")
    print(f"Train Loss : {train_loss:.4f}")
    print(f"Train Dice : {train_dice:.4f}")
    print(f"Train IoU  : {train_iou:.4f}")
    print("----------------------------------------")
    print(f"Val Loss   : {val_loss:.4f}")
    print(f"Val Dice   : {val_dice:.4f}")
    print(f"Val IoU    : {val_iou:.4f}")
    print("========================================\n")


    if val_dice > best_val_dice:
        best_val_dice = val_dice
        save_path = os.path.join(
            CHECKPOINT_DIR,
            "best_model.pth"
        )
        torch.save(
            model.state_dict(),
            save_path
        )
        print(
            f"Best model saved! "
            f"Dice: {best_val_dice:.4f}"
        )

print("\nTraining completed!")

print(f"Best Validation Dice: "f"{best_val_dice:.4f}")