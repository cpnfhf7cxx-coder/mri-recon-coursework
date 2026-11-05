"""Consistent evaluation metrics for [0, 1] grayscale data."""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_metrics(truth, reconstruction):
    if truth.shape != reconstruction.shape:
        raise ValueError("Ground truth and reconstruction shapes do not match")
    if not np.all(np.isfinite(reconstruction)):
        raise ValueError("Reconstruction contains NaN/Inf")
    rec = np.clip(reconstruction.astype(np.float32), 0., 1.)
    truth = truth.astype(np.float32)
    return {
        "psnr": float(peak_signal_noise_ratio(truth, rec, data_range=1.)),
        "ssim": float(structural_similarity(truth, rec, data_range=1.)),
        "mae": float(np.mean(np.abs(truth - rec))),
    }
