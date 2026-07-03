import torch
import torch.nn.functional as F
import numpy as np


class GradCAM3D:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self._activations = {}
        self._gradients = {}

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations["value"] = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients["value"] = grad_output[0].detach()

    def compute(self, input_tensor, target_class=3, mask_gt=None):

        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor) 
        pred_labels = torch.argmax(output, dim=1).squeeze(0)
        if mask_gt is not None:
            roi = (mask_gt > 0).float().to(input_tensor.device)
            if roi.sum() >= 1:
                score = (output[0, target_class] * roi).sum()
            else:
                score = output[0, target_class].sum()
        else:
            score = output[0, target_class].sum()

        self.model.zero_grad()
        score.backward()

        act  = self._activations["value"]  
        grad = self._gradients["value"] 

        weights = grad.mean(dim=[0, 2, 3, 4])

        weighted = (act * weights[None, :, None, None, None]).sum(dim=1)
        heatmap = torch.relu(weighted).squeeze(0)

        target_size = (input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4])
        heatmap = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),
            size=target_size,
            mode="trilinear",
            align_corners=False,
        ).squeeze()

        p1  = torch.quantile(heatmap, 0.01)
        p99 = torch.quantile(heatmap, 0.99)
        denom = p99 - p1
        if denom < 1e-8:
            heatmap = torch.zeros_like(heatmap)
        else:
            heatmap = torch.clamp(heatmap, p1, p99)
            heatmap = (heatmap - p1) / denom

        return heatmap.cpu().numpy(), pred_labels.cpu().numpy()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()