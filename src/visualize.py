from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_triptych(path, truth, zero_filled, reconstruction, name, zf_metrics=None, metrics=None):
    """Required three panels: original, degraded/zero-filled and recovered."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.35), constrained_layout=True)
    texts = ["Original (GT)", "Degraded (zero-filled)", "Reconstruction"]
    if zf_metrics:
        texts[1] += f"\nPSNR {zf_metrics['psnr']:.2f} / SSIM {zf_metrics['ssim']:.3f}"
    if metrics:
        texts[2] += f"\nPSNR {metrics['psnr']:.2f} / SSIM {metrics['ssim']:.3f}"
    for ax, image, title in zip(axes, (truth, zero_filled, reconstruction), texts):
        ax.imshow(np.clip(image, 0., 1.), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.axis("off")
        ax.set_title(title, fontsize=9)
    fig.suptitle(name, fontsize=10)
    fig.savefig(path, dpi=170)
    plt.close(fig)
