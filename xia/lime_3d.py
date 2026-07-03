import numpy as np
import torch
from sklearn.linear_model import Ridge


class LIME3D:

    def __init__(
        self,
        model,
        block_size: int = 12,
        n_samples: int = 1500,
        tumor_class: int = 3,
        ridge_alpha: float = 1.0,
        seed: int = 42,
    ):
        self.model = model
        self.block_size = block_size
        self.n_samples = n_samples
        self.tumor_class = tumor_class
        self.ridge_alpha = ridge_alpha
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _create_blocks(volume_shape, block_size):
        D, H, W = volume_shape
        blocks = []
        for z in range(0, D, block_size):
            for y in range(0, H, block_size):
                for x in range(0, W, block_size):
                    blocks.append((
                        z, min(z + block_size, D),
                        y, min(y + block_size, H),
                        x, min(x + block_size, W),
                    ))
        return blocks

    @staticmethod
    def _apply_mask(input_tensor, blocks, mask_vector):
        perturbed = input_tensor.clone()
        for i, active in enumerate(mask_vector):
            if active == 0:
                z1, z2, y1, y2, x1, x2 = blocks[i]
                perturbed[:, :, z1:z2, y1:y2, x1:x2] = 0.0
        return perturbed

    def compute(self, input_tensor, mask_gt=None, verbose=False):
        self.model.eval()
        D = input_tensor.shape[2]
        H = input_tensor.shape[3]
        W = input_tensor.shape[4]

        roi_mask = None
        if mask_gt is not None:
            roi = (mask_gt > 0).float()
            if roi.sum() >= 1:
                roi_mask = roi.to(input_tensor.device)

        with torch.no_grad():
            base_output = self.model(input_tensor)
            pred_labels = torch.argmax(base_output, dim=1).squeeze(0).cpu().numpy()

        blocks = self._create_blocks((D, H, W), self.block_size)
        n_blocks = len(blocks)

        if verbose:
            print(f"  Bloques: {n_blocks}  |  Muestras: {self.n_samples}")

        mask_matrix = self.rng.integers(0, 2, size=(self.n_samples, n_blocks))
        scores = np.zeros(self.n_samples)

        for i, mask_vec in enumerate(mask_matrix):
            if verbose and (i + 1) % 300 == 0:
                print(f"  Perturbación {i+1}/{self.n_samples}")

            perturbed = self._apply_mask(input_tensor, blocks, mask_vec)

            with torch.no_grad():
                out = self.model(perturbed)
                probs = torch.softmax(out, dim=1)
                if roi_mask is not None:
                    score_map = probs[:, self.tumor_class].squeeze()
                    scores[i] = (score_map * roi_mask).sum().item() / (roi_mask.sum().item() + 1e-8)
                else:
                    scores[i] = probs[:, self.tumor_class].mean().item()

        regressor = Ridge(alpha=self.ridge_alpha)
        regressor.fit(mask_matrix, scores)
        importance = regressor.coef_ 

        importance_raw = np.zeros((D, H, W), dtype=np.float32)
        for i, (z1, z2, y1, y2, x1, x2) in enumerate(blocks):
            importance_raw[z1:z2, y1:y2, x1:x2] = importance[i]

        importance_raw = np.maximum(importance_raw, 0.0)

        i_max = importance_raw.max()
        if i_max < 1e-8:
            importance_map = np.zeros_like(importance_raw)
        else:
            importance_map = importance_raw / i_max

        return importance_map, pred_labels

    def remove_hooks(self):
        pass