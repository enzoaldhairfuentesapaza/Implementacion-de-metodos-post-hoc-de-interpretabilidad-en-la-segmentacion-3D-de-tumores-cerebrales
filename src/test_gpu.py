import torch

print("Torch:", torch.__version__)

print("CUDA disponible:", torch.cuda.is_available())

if torch.cuda.is_available():

    print("GPU:", torch.cuda.get_device_name(0))

    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory / 1e9,
            2
        ),
        "GB"
    )