import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# ============================================================
# DATASET PATH
# ============================================================

DATASET_PATH = r"D:\ENZO\PFC1\datasets\BraTS_subset"

# ============================================================
# DATASET
# ============================================================

class BraTSDataset(Dataset):

    def __init__(self, root_dir):

        self.root_dir = root_dir

        self.patients = sorted(os.listdir(root_dir))

    def __len__(self):

        return len(self.patients)

    def load_nifti(self, path):

        return nib.load(path).get_fdata()

    def __getitem__(self, idx):

        patient = self.patients[idx]

        patient_path = os.path.join(
            self.root_dir,
            patient
        )

        # ----------------------------------------------------
        # LOAD FILES
        # ----------------------------------------------------

        t1_path = glob.glob(
            os.path.join(patient_path, "*_t1*")
        )[0]

        t1ce_path = glob.glob(
            os.path.join(patient_path, "*_t1ce*")
        )[0]

        t2_path = glob.glob(
            os.path.join(patient_path, "*_t2*")
        )[0]

        flair_path = glob.glob(
            os.path.join(patient_path, "*_flair*")
        )[0]

        seg_path = glob.glob(
            os.path.join(patient_path, "*_seg*")
        )[0]

        # ----------------------------------------------------
        # LOAD MRI
        # ----------------------------------------------------

        t1 = self.load_nifti(t1_path)
        t1ce = self.load_nifti(t1ce_path)
        t2 = self.load_nifti(t2_path)
        flair = self.load_nifti(flair_path)

        seg = self.load_nifti(seg_path)

        # ----------------------------------------------------
        # FIX LABELS
        # BraTS uses 0,1,2,4
        # Convert to 0,1,2,3
        # ----------------------------------------------------

        seg[seg == 4] = 3

        # ----------------------------------------------------
        # STACK CHANNELS
        # Shape:
        # [4, D, H, W]
        # ----------------------------------------------------

        image = np.stack(
            [t1, t1ce, t2, flair],
            axis=0
        )

        # ----------------------------------------------------
        # CROP FOR MEMORY
        # 128x128x128
        # ----------------------------------------------------

        image = image[:, 56:184, 56:184, 13:141]

        seg = seg[56:184, 56:184, 13:141]

        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        image = (image - image.mean()) / (
            image.std() + 1e-8
        )

        # ----------------------------------------------------
        # TO TENSOR
        # ----------------------------------------------------

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        seg = torch.tensor(
            seg,
            dtype=torch.long
        )

        return image, seg


# ============================================================
# DOUBLE CONV
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm3d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm3d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.conv(x)
    

# ============================================================
# DOWN BLOCK
# ============================================================

class Down(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.MaxPool3d(2),

            DoubleConv(
                in_channels,
                out_channels
            )
        )

    def forward(self, x):

        return self.block(x)
    
# ============================================================
# UP BLOCK
# ============================================================

class Up(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.up = nn.ConvTranspose3d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(
            in_channels,
            out_channels
        )

    def forward(self, x1, x2):

        x1 = self.up(x1)

        # ----------------------------------------------------
        # FIX SHAPE DIFFERENCES
        # ----------------------------------------------------

        diffD = x2.size(2) - x1.size(2)
        diffH = x2.size(3) - x1.size(3)
        diffW = x2.size(4) - x1.size(4)

        x1 = nn.functional.pad(
            x1,
            [
                diffW // 2,
                diffW - diffW // 2,

                diffH // 2,
                diffH - diffH // 2,

                diffD // 2,
                diffD - diffD // 2
            ]
        )

        # ----------------------------------------------------
        # CONCAT
        # ----------------------------------------------------

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)
    

# ============================================================
# U-NET 3D
# ============================================================

class UNet3D(nn.Module):

    def __init__(self):

        super().__init__()

        self.inc = DoubleConv(4, 32)

        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        self.bottleneck = Down(256, 512)

        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)

        self.outc = nn.Conv3d(
            32,
            4,
            kernel_size=1
        )

    def forward(self, x):

        x1 = self.inc(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        x5 = self.bottleneck(x4)

        x = self.up1(x5, x4)

        x = self.up2(x, x3)

        x = self.up3(x, x2)

        x = self.up4(x, x1)

        logits = self.outc(x)

        return logits
# ============================================================
# DICE LOSS
# ============================================================

def dice_loss(pred, target, smooth=1e-5):

    pred = torch.softmax(pred, dim=1)

    target_onehot = torch.nn.functional.one_hot(
        target,
        num_classes=4
    ).permute(0,4,1,2,3).float()

    intersection = (pred * target_onehot).sum()

    union = pred.sum() + target_onehot.sum()

    dice = (2. * intersection + smooth) / (
        union + smooth
    )

    return 1 - dice




# ============================================================
# LOAD DATASET
# ============================================================

if __name__ == "__main__":
        
    dataset = BraTSDataset(DATASET_PATH)

    print("Pacientes:", len(dataset))

    # ============================================================
    # TEST SAMPLE
    # ============================================================

    image, mask = dataset[0]

    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)

    print("Labels:", torch.unique(mask))

    # ============================================================
    # VISUALIZATION
    # ============================================================

    slice_idx = 64

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)

    plt.imshow(
        image[0, :, :, slice_idx],
        cmap="gray"
    )

    plt.title("MRI")

    plt.subplot(1,2,2)

    plt.imshow(
        mask[:, :, slice_idx]
    )

    plt.title("Mask")

    plt.show()

    
    # ============================================================
    # CREATE MODEL
    # ============================================================

    model = UNet3D().to(device)

    print(model)


    # ============================================================
    # FORWARD PASS
    # ============================================================

    image = image.unsqueeze(0).to(device)

    print("Input:", image.shape)

    with torch.no_grad():

        output = model(image)

    print("Output:", output.shape)


    # ============================================================
    # DATALOADER
    # ============================================================

    train_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True
    )

    # ============================================================
    # LOSSES
    # ============================================================

    ce_loss = nn.CrossEntropyLoss()


    # ============================================================
    # OPTIMIZER
    # ============================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )



    # ============================================================
    # TRAINING LOOP
    # ============================================================

    EPOCHS = 10

    for epoch in range(EPOCHS):

        model.train()

        epoch_loss = 0

        for batch_idx, (images, masks) in enumerate(train_loader):

            images = images.to(device)

            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss1 = dice_loss(outputs, masks)

            loss2 = ce_loss(outputs, masks)

            loss = loss1 + loss2

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

            print(
                f"Epoch [{epoch+1}/{EPOCHS}] "
                f"Batch [{batch_idx+1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )
            torch.save(
                model.state_dict(),
                rf"D:\ENZO\PFC1\models\unet_epoch_{epoch+1}.pth"
            )

            print(f"Checkpoint epoch {epoch+1} guardado")

        print(
            f"\nEpoch {epoch+1} "
            f"Average Loss: "
            f"{epoch_loss/len(train_loader):.4f}\n"
        )


    # ============================================================
    # SAVE MODEL
    # ============================================================

    torch.save(
        model.state_dict(),
        rf"D:\ENZO\PFC1\models\unet_epoch_{epoch+1}.pth"
    )

    print("Modelo guardado.")


    # ============================================================
    # INFERENCE
    # ============================================================

    model.eval()

    with torch.no_grad():

        image = image.to(device)
        output = model(image)

        pred = torch.argmax(
            output,
            dim=1
        ).squeeze(0).cpu()

    # ============================================================
    # VISUALIZATION
    # ============================================================

    slice_idx = 64

    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)

    plt.imshow(
        image[0,0,:,:,slice_idx].cpu(),
        cmap="gray"
    )

    plt.title("MRI")

    plt.subplot(1,3,2)

    plt.imshow(
        mask[:,:,slice_idx]
    )

    plt.title("Ground Truth")

    plt.subplot(1,3,3)

    plt.imshow(
        pred[:,:,slice_idx]
    )

    plt.title("Prediction")

    plt.show()