import torch
import numpy as np

print(f"NumPy Version: {np.__version__}")
print(f"Is GPU (CUDA) available? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
