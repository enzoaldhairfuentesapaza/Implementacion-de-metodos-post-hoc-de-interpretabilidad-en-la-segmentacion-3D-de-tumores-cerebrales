import os
import sys
import json

import torch
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.unet3d import UNet3D
from datasets.brats_dataset import BraTSDataset
from training.losses import SegmentationLoss, dice_score, iou_score, dice_score_regions


DATASET_PATH = r"D:\ENZO\PFC1\datasets\BraTS2020"
CHECKPOINT_DIR = "../checkpoints"
HISTORY_PATH = "checkpoints/history.json"

BATCH_SIZE = 1
EPOCHS = 50
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
NUM_WORKERS = 0
SEED = 42
EARLY_STOPPING_PATIENCE = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"\nDevice: {DEVICE}")
print("\nLoading dataset . . .\n")

dataset = BraTSDataset(DATASET_PATH, augment=True)
dataset_size = len(dataset)
test_size = int(dataset_size * TEST_SPLIT)
val_size = int(dataset_size * VALIDATION_SPLIT)
train_size = dataset_size - val_size - test_size

generator = torch.Generator().manual_seed(SEED)
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

print(f"Train samples      : {len(train_dataset)}")
print(f"Validation samples : {len(val_dataset)}")
print(f"Test samples       : {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print("\nLoading model . . .\n")

model = UNet3D().to(DEVICE)
criterion = SegmentationLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

history = {
    "train_loss": [], 
    "train_dice": [], 
    "train_iou": [],
    "val_loss": [], 
    "val_dice": [], 
    "val_iou": [],
    "val_dice_wt": [], 
    "val_dice_tc": [], 
    "val_dice_et": [],
    "lr": []
}

best_val_dice = 0.0
epochs_without_improv = 0

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    model.train()
    train_loss = train_dice = train_iou = 0.0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    for images, masks in train_bar:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_dice += dice_score(outputs.detach(), masks)
        train_iou += iou_score(outputs.detach(), masks)

        train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    n_train = len(train_loader)
    train_loss /= n_train
    train_dice /= n_train
    train_iou /= n_train

    model.eval()
    val_loss = val_dice = val_iou = 0.0
    val_regions = {"WT": 0.0, "TC": 0.0, "ET": 0.0}

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, masks)

            val_loss += loss.item()
            val_dice += dice_score(outputs, masks)
            val_iou += iou_score(outputs, masks)

            r = dice_score_regions(outputs, masks)
            for k in val_regions:
                val_regions[k] += r[k]

    n_val = len(val_loader)
    val_loss /= n_val
    val_dice /= n_val
    val_iou /= n_val
    for k in val_regions:
        val_regions[k] /= n_val

    current_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()

    print(f"\n{'='*50}")
    print(f"Epoch {epoch+1}/{EPOCHS} LR: {current_lr:.2e}")
    print(f"{'─'*50}")
    print(f"Train Loss : {train_loss:.4f}  |  Dice: {train_dice:.4f}  |  IoU: {train_iou:.4f}")
    print(f"Val Loss : {val_loss:.4f}  |  Dice: {val_dice:.4f}  |  IoU: {val_iou:.4f}")
    print(f"Val Regions — WT: {val_regions['WT']:.4f}  TC: {val_regions['TC']:.4f}  ET: {val_regions['ET']:.4f}")
    print(f"{'='*50}\n")

    history["train_loss"].append(train_loss)
    history["train_dice"].append(train_dice)
    history["train_iou"].append(train_iou)
    history["val_loss"].append(val_loss)
    history["val_dice"].append(val_dice)
    history["val_iou"].append(val_iou)
    history["val_dice_wt"].append(val_regions["WT"])
    history["val_dice_tc"].append(val_regions["TC"])
    history["val_dice_et"].append(val_regions["ET"])
    history["lr"].append(current_lr)

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    if val_dice > best_val_dice:
        best_val_dice         = val_dice
        epochs_without_improv = 0
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
        print(f"Best model saved - Dice: {best_val_dice:.4f}\n")
    else:
        epochs_without_improv += 1
        print(f"No improvement ({epochs_without_improv}/{EARLY_STOPPING_PATIENCE})\n")
        if epochs_without_improv >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

print("\n" + "="*50)
print("EVALUACIÓN EN TEST SET")
print("="*50)

model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pth")))
model.eval()

test_dice = test_iou = 0.0
test_regions = {"WT": 0.0, "TC": 0.0, "ET": 0.0}

with torch.no_grad():
    for images, masks in tqdm(test_loader, desc="Test"):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        outputs = model(images)

        test_dice += dice_score(outputs, masks)
        test_iou += iou_score(outputs, masks)

        r = dice_score_regions(outputs, masks)
        for k in test_regions:
            test_regions[k] += r[k]

n_test = len(test_loader)
test_dice /= n_test
test_iou /= n_test
for k in test_regions:
    test_regions[k] /= n_test

print(f"\nTest Dice (mean)  : {test_dice:.4f}")
print(f"Test IoU  (mean)  : {test_iou:.4f}")
print(f"Test Dice WT      : {test_regions['WT']:.4f}")
print(f"Test Dice TC      : {test_regions['TC']:.4f}")
print(f"Test Dice ET      : {test_regions['ET']:.4f}")
print(f"\nBest Validation Dice: {best_val_dice:.4f}")
print("\nTraining completed")
