"""Utilities for educational *retrospective* MRI undersampling.

These are FFTs of preprocessed magnitude images, NOT scanner-acquired raw k-space.
"""
from pathlib import Path
import numpy as np
from PIL import Image


def read_image(path, size=128):
    """Read a grayscale image as float32 in [0, 1]."""
    with Image.open(path) as image:
        image = image.convert("L")
        if size and image.size != (size, size):
            image = image.resize((size, size), Image.Resampling.BICUBIC)
        img = np.asarray(image, dtype=np.float32) / 255.0
    return img


def fft2c(image):
    """Unitary, centered 2D FFT; handles 2D arrays and batches."""
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(image, axes=(-2, -1)),
                    axes=(-2, -1), norm="ortho"), axes=(-2, -1))


def ifft2c(kspace):
    """Inverse centered 2D FFT."""
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)),
                     axes=(-2, -1), norm="ortho"), axes=(-2, -1))


def cartesian_mask(shape, acceleration=4, center_fraction=0.08, seed=2026):
    """Fixed random Cartesian phase-encode lines, with fully-sampled k-space center.

    Selects round(width / acceleration) columns in total, including central ones.
    The actual sampling fraction is the reported selected/total columns.
    """
    height, width = shape
    if acceleration < 1:
        raise ValueError("acceleration must be >= 1")
    if not (0 < center_fraction <= 1):
        raise ValueError("center_fraction must be in (0, 1]")
    selected_count = max(1, min(width, int(round(width / acceleration))))
    center_count = max(1, min(selected_count, int(round(width * center_fraction))))
    center_start = (width - center_count) // 2
    center_indices = np.arange(center_start, center_start + center_count)
    other = np.setdiff1d(np.arange(width), center_indices)
    rng = np.random.default_rng(seed)
    n_extra = selected_count - center_count
    chosen = rng.choice(other, size=n_extra, replace=False) if n_extra else np.array([], dtype=int)
    colmask = np.zeros(width, dtype=np.float32)
    colmask[np.r_[center_indices, chosen]] = 1.
    return np.broadcast_to(colmask, (height, width)).copy()


def simulate_acquisition(image, mask):
    """Return acquired k-space and magnitude zero-filled image."""
    if image.shape != mask.shape:
        raise ValueError("image and mask must have identical 2D shapes")
    y = fft2c(image) * mask
    zero_filled = np.abs(ifft2c(y)).astype(np.float32)
    return y, zero_filled


def save_gray(path, image):
    """Save image using a fixed [0, 1] display range (never per-image rescale)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    quantized = np.uint8(np.round(np.clip(image, 0., 1.) * 255))
    Image.fromarray(quantized, mode="L").save(path)
