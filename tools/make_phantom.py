#!/usr/bin/env python3
"""Small artificial phantoms only for smoke tests; not valid course evaluation data."""
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.data import shepp_logan_phantom
from skimage.transform import resize, rotate


def main():
    out = Path('data/phantom')
    base = resize(shepp_logan_phantom(), (128, 128), anti_aliasing=True)
    for split, count in [('train', 12), ('val', 4), ('test', 4)]:
        folder = out / split
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            angle = -10 + 2.5 * i + (0 if split == 'train' else 9)
            img = np.clip(rotate(base, angle, resize=False, preserve_range=True), 0, 1)
            img = (img ** (0.9 + (i % 5) * 0.05))
            Image.fromarray(np.uint8(np.round(img * 255)), mode='L').save(folder / f'phantom_{i:03d}.png')
    print('Generated synthetic test phantoms under data/phantom (NOT actual MRI data).')


if __name__ == '__main__':
    main()
