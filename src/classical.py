"""CPU-friendly reference reconstructions, with a method students can extend."""
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from .mri import fft2c, ifft2c


def zero_filled(kspace_under, mask=None):
    """Provided baseline. No learning or optimization."""
    return np.clip(np.abs(ifft2c(kspace_under)), 0.0, 1.0).astype(np.float32)


def tv_pocs(kspace_under, mask, steps=10, weight=0.02):
    """Example CPU reconstruction: iterative TV smoothing + data-consistency.

    Every iteration projects the Fourier coefficients at observed locations back
    to the measured data. This is a didactic TV-POCS heuristic, rather than a
    fully specified convex optimization solver. Students should investigate
    iterations/weights and implement their own improvement.
    """
    x = ifft2c(kspace_under).copy()
    observed = mask.astype(bool)
    for _ in range(steps):
        real = denoise_tv_chambolle(x.real, weight=weight, max_num_iter=20)
        imag = denoise_tv_chambolle(x.imag, weight=weight, max_num_iter=20)
        k = fft2c(real + 1j * imag)
        k[observed] = kspace_under[observed]
        x = ifft2c(k)
    return np.clip(np.abs(x), 0., 1.).astype(np.float32)
