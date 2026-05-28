import os
import glob
import numpy as np
import nibabel as nib

import torch
from torch.utils.data import Dataset


# Tamaño del crop 3D
CROP_SIZE = 128

# Número de clases BraTS
NUM_CLASSES = 4

# Modalidades MRI usadas
MRI_MODALITIES = [
    "t1",
    "t1ce",
    "t2",
    "flair"
]

class BraTSDataset(Dataset):

    def __init__(self, root_dir, augment=False):

        self.root_dir = root_dir

        self.augment = augment


        all_patients = sorted( os.listdir(root_dir))

        self.patients = []

        for patient in all_patients:

            patient_path = os.path.join(
                root_dir,
                patient
            )

            valid = True

            for modality in MRI_MODALITIES:

                files = glob.glob(
                    os.path.join(
                        patient_path,
                        f"*_{modality}*"
                    )
                )

                if len(files) == 0:

                    valid = False

                    break

            seg_files = glob.glob(
                os.path.join(
                    patient_path,
                    "*_seg*"
                )
            )

            if len(seg_files) == 0:

                valid = False

            if valid:

                self.patients.append(patient)

            else:

                print( f"Skipping incomplete patient: "f"{patient}")
    def __len__(self):

        return len(self.patients)

    def load_nifti(self, path):

        return nib.load(path).get_fdata()

    def normalize(self, image):

        return (image - image.mean()) / (
            image.std() + 1e-8
        )

    def center_crop(self, image, seg):

        _, D, H, W = image.shape

        start_d = (D - CROP_SIZE) // 2
        start_h = (H - CROP_SIZE) // 2
        start_w = (W - CROP_SIZE) // 2

        image = image[
            :,
            start_d:start_d + CROP_SIZE,
            start_h:start_h + CROP_SIZE,
            start_w:start_w + CROP_SIZE
        ]

        seg = seg[
            start_d:start_d + CROP_SIZE,
            start_h:start_h + CROP_SIZE,
            start_w:start_w + CROP_SIZE
        ]

        return image, seg

    def apply_augmentation(self, image, seg):

        if np.random.rand() > 0.5:

            image = np.flip(image, axis=2).copy()

            seg = np.flip(seg, axis=1).copy()


        if np.random.rand() > 0.5:

            k = np.random.randint(1, 4)

            image = np.rot90(
                image,
                k=k,
                axes=(2, 3)
            ).copy()

            seg = np.rot90(
                seg,
                k=k,
                axes=(1, 2)
            ).copy()

        return image, seg

    def __getitem__(self, idx):

        patient = self.patients[idx]

        patient_path = os.path.join(
            self.root_dir,
            patient
        )

        modalities = []

        for modality in MRI_MODALITIES:

            path = glob.glob(
                os.path.join(
                    patient_path,
                    f"*_{modality}*"
                )
            )[0]

            volume = self.load_nifti(path)

            modalities.append(volume)

        seg_path = glob.glob(
            os.path.join(
                patient_path,
                "*_seg*"
            )
        )[0]

        seg = self.load_nifti(seg_path)

        # FIX BRATS LABELS
        # 0, 1, 2, 4
        # Convertido:
        # 0, 1, 2, 3

        seg[seg == 4] = 3

        # STACK CHANNELS
        # Shape:
        # [4, D, H, W]

        image = np.stack(
            modalities,
            axis=0
        )

        image = self.normalize(image)


        image, seg = self.center_crop(
            image,
            seg
        )

        if self.augment:

            image, seg = self.apply_augmentation(
                image,
                seg
            )

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        seg = torch.tensor(
            seg,
            dtype=torch.long
        )

        return image, seg



if __name__ == "__main__":

    DATASET_PATH = (
        r"D:\ENZO\PFC1\data\BraTS_subset"
    )

    dataset = BraTSDataset(
        DATASET_PATH,
        augment=True
    )

    print("Patients:", len(dataset))

    image, mask = dataset[0]

    print("Image shape:", image.shape)
    print("Mask shape :", mask.shape)
    print("Labels:", torch.unique(mask))