from pathlib import Path
import numpy as np
from .mri import read_image, cartesian_mask, simulate_acquisition


def list_images(root):
    root = Path(root)
    return sorted([p for p in root.rglob("*.png") if p.is_file()])


class ReconstructionDataset:
    """CPU numpy dataset; use as PyTorch dataset when PyTorch is available."""
    def __init__(self, root, acceleration=4, seed=2026, size=128):
        self.files = list_images(root)
        if not self.files:
            raise FileNotFoundError(f"No processed PNG images under {root}. Run prepare_dataset.py")
        self.mask = cartesian_mask((size, size), acceleration=acceleration, seed=seed)
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        truth = read_image(self.files[index], self.size)
        _, zero = simulate_acquisition(truth, self.mask)
        return zero[None].astype(np.float32), truth[None].astype(np.float32)
