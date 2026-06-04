import os
import glob

import numpy as np
import nibabel as nib

import torch
from torch.utils.data import Dataset

CROP_SIZE = 128

NUM_CLASSES = 4

MRI_MODALITIES = ["t1", "t1ce", "t2", "flair"]

class BraTSDataset(Dataset):

    def __init__(self, root_dir, augment=False):

        self.root_dir = root_dir
        self.augment  = augment

        all_patients = sorted(os.listdir(root_dir))
        self.patients = []

        for patient in all_patients:

            patient_path = os.path.join(root_dir, patient)

            if not os.path.isdir(patient_path):
                continue

            valid = True

            for modality in MRI_MODALITIES:
                files = glob.glob(os.path.join(patient_path, f"*_{modality}*"))
                if len(files) == 0:
                    valid = False
                    break

            seg_files = glob.glob(os.path.join(patient_path, "*_seg*"))
            if len(seg_files) == 0:
                valid = False

            if valid:
                self.patients.append(patient)
            else:
                print(f"Skipping incomplete patient: {patient}")

    def __len__(self):
        return len(self.patients)

    def load_nifti(self, path):
        return nib.load(path).get_fdata().astype(np.float32)

    def normalize(self, image):
        return (image - image.mean()) / (image.std() + 1e-8)

    def center_crop(self, image, seg):
        _, D, H, W = image.shape

        tumor_voxels = np.argwhere(seg > 0)

        if len(tumor_voxels) > 0:
            center = tumor_voxels.mean(axis=0).astype(int)
            half   = CROP_SIZE // 2
            cd = int(np.clip(center[0], half, D - half))
            ch = int(np.clip(center[1], half, H - half))
            cw = int(np.clip(center[2], half, W - half))
        else:
            cd = D // 2
            ch = H // 2
            cw = W // 2

        half = CROP_SIZE // 2
        image = image[:,cd-half : cd+half, ch-half : ch+half,cw-half : cw+half,]
        seg = seg[cd-half : cd+half,ch-half : ch+half,cw-half : cw+half,]
        return image, seg

    def apply_augmentation(self, image, seg):
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=3).copy()
            seg = np.flip(seg,   axis=2).copy()
        if np.random.rand() > 0.5:
            k = np.random.randint(1, 4)
            image = np.rot90(image, k=k, axes=(2, 3)).copy()
            seg = np.rot90(seg,   k=k, axes=(1, 2)).copy()

        return image, seg
    
    def __getitem__(self, idx):

        patient = self.patients[idx]
        patient_path = os.path.join(self.root_dir, patient)

        modalities = []
        for modality in MRI_MODALITIES:
            path = glob.glob(os.path.join(patient_path, f"*_{modality}*"))[0]
            volume = self.load_nifti(path)
            modalities.append(volume)

        seg_path = glob.glob(os.path.join(patient_path, "*_seg*"))[0]
        seg = self.load_nifti(seg_path)

        seg[seg == 4] = 3

        image = np.stack(modalities, axis=0)

        image = self.normalize(image)

        image, seg = self.center_crop(image, seg)

        if self.augment:
            image, seg = self.apply_augmentation(image, seg)

        image = torch.tensor(image, dtype=torch.float32)
        seg   = torch.tensor(seg,   dtype=torch.long)

        return image, seg


if __name__ == "__main__":

    DATASET_PATH = r"D:\ENZO\PFC1\datasets\BraTS2020"
    dataset = BraTSDataset(DATASET_PATH, augment=True)

    print(f"Patients: {len(dataset)}")
    image, mask = dataset[0]

    print(f"Image shape : {image.shape}")
    print(f"Mask shape : {mask.shape}")
    print(f"Image dtype : {image.dtype}")
    print(f"Labels : {torch.unique(mask).tolist()}")
    print(f"Image RAM : {image.element_size() * image.nelement() / 1e6:.1f} MB")