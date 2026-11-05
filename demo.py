#!/usr/bin/env python3
"""Unified evaluation + required 3-panel visuals for all tracks."""
import argparse
import csv
import json
import pickle
from pathlib import Path
import numpy as np
from src.mri import read_image, cartesian_mask, simulate_acquisition, save_gray
from src.classical import zero_filled, tv_pocs
from src.metrics import compute_metrics
from src.visualize import save_triptych
from src.dataset import list_images


def get_reconstructor(args):
    args.resolved_device = None
    if args.method == 'zf':
        return lambda y, mask, zero: zero_filled(y)
    if args.method == 'tv':
        return lambda y, mask, zero: tv_pocs(y, mask, steps=args.steps, weight=args.weight)
    if args.method == 'student':
        from src.student_method import reconstruct
        return reconstruct
    if args.method == 'unet':
        if not args.checkpoint:
            raise ValueError('Supply --checkpoint for --method unet')
        import torch
        from src.models import TinyUNet
        try:
            payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        except (TypeError, pickle.UnpicklingError):
            # Compatibility with old PyTorch. Only load checkpoints you trust.
            payload = torch.load(args.checkpoint, map_location='cpu')
        expected = {'acceleration': args.acceleration, 'seed': args.seed, 'size': args.size}
        for key, value in expected.items():
            if key in payload and not np.isclose(payload[key], value):
                raise ValueError(
                    f'Checkpoint {key}={payload[key]} does not match evaluation {key}={value}')
        device = ('cuda' if torch.cuda.is_available() else 'cpu') \
            if args.device == 'auto' else args.device
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable; use --device cpu')
        args.resolved_device = device
        model = TinyUNet(width=payload['width']).to(device)
        model.load_state_dict(payload['state_dict'])
        model.eval()
        def predict(y, mask, zero):
            with torch.inference_mode():
                input_tensor = torch.from_numpy(
                    zero[None, None].astype(np.float32)).to(device)
                return model(input_tensor).cpu().numpy()[0, 0]
        return predict
    raise ValueError(args.method)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='data/processed')
    p.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    p.add_argument('--file-list', default='',
                   help='Optional text file of image IDs; every ID must belong to --split')
    p.add_argument('--method', choices=['zf', 'tv', 'student', 'unet'], default='zf')
    p.add_argument('--acceleration', type=float, default=4.0)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--size', type=int, default=128)
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--weight', type=float, default=0.02)
    p.add_argument('--checkpoint', default='')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                   help='Inference device for --method unet')
    p.add_argument('--out', default='outputs/demo')
    args = p.parse_args()
    files = list_images(Path(args.data_root) / args.split)
    if not files:
        p.error('No prepared PNG files. See README: prepare_dataset.py')
    if args.file_list:
        requested = [line.strip() for line in Path(args.file_list).read_text(encoding='utf-8').splitlines()
                     if line.strip() and not line.lstrip().startswith('#')]
        available = {file.name: file for file in files}
        missing = [name for name in requested if name not in available]
        if missing:
            p.error(f'{len(missing)} IDs from --file-list are absent from split {args.split}: {missing[:3]}')
        files = [available[name] for name in requested]
    mask = cartesian_mask((args.size, args.size), acceleration=args.acceleration, seed=args.seed)
    infer = get_reconstructor(args)
    out = Path(args.out)
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    (out / 'reconstructions').mkdir(parents=True, exist_ok=True)
    np.save(out / 'mask.npy', mask)
    rows = []
    for file in files:
        truth = read_image(file, args.size)
        measured, degraded = simulate_acquisition(truth, mask)
        output = np.asarray(infer(measured, mask, degraded), dtype=np.float32)
        if output.shape != truth.shape:
            raise ValueError(f'{file.name}: method returned shape {output.shape}, expected {truth.shape}')
        scores = compute_metrics(truth, output)
        base_scores = compute_metrics(truth, degraded)
        rows.append(dict(image=file.name, **scores,
                         zf_psnr=base_scores['psnr'], zf_ssim=base_scores['ssim'],
                         zf_mae=base_scores['mae']))
        save_gray(out / 'reconstructions' / file.name, output)
        save_triptych(out / 'figures' / file.name, truth, degraded, output,
                      file.name, zf_metrics=base_scores, metrics=scores)
    with (out / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    means = {k: float(np.mean([row[k] for row in rows]))
             for k in ('psnr', 'ssim', 'mae', 'zf_psnr', 'zf_ssim', 'zf_mae')}
    summary = {'method': args.method, 'split': args.split, 'file_list': args.file_list or None,
               'count': len(rows),
               'acceleration_requested': args.acceleration,
               'acceleration_actual': float(mask.shape[1] / mask[0].sum()),
               'seed': args.seed, 'device': args.resolved_device,
               'checkpoint': args.checkpoint or None, 'mean': means}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Figures / CSV / images saved under {out.resolve()}')


if __name__ == '__main__':
    main()
