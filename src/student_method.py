"""Your own reconstruction method. Customize this file for the classical track.

Do not read the ground-truth image, validation labels, or test labels here.
Your method can use only observed k-space, mask and zero-filled image.
"""
import numpy as np


def reconstruct(kspace_under: np.ndarray,
                mask: np.ndarray,
                zero_filled: np.ndarray) -> np.ndarray:
    """Return a 2D float array with the same shape, values normally in [0, 1]."""
    # TODO: Implement a *nontrivial* reconstruction (e.g. TV/POCS,
    # wavelet sparse regularization, ADMM, or your own optimization).
    # You may import tools from src.mri and scikit-image.
    raise NotImplementedError("Implement reconstruct() in src/student_method.py")
