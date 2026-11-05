#!/usr/bin/env python3
"""Optional minimal trainable baseline for students with a GPU (CPU also works)."""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.dataset import ReconstructionDataset
from src.models import TinyUNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='data/processed')
    p.add_argument('--epochs', type=int, default=30,
                   help='Maximum epochs; validation early stopping may finish sooner')
    p.add_argument('--patience', type=int, default=5,
                   help='Stop after this many epochs without validation improvement')
    p.add_argument('--min-delta', type=float, default=1e-4,
                   help='Minimum validation L1 improvement to reset patience')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--width', type=int, default=16)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--acceleration', type=float, default=4)
    p.add_argument('--size', type=int, default=128)
    p.add_argument('--out', default='checkpoints/unet.pt')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    args = p.parse_args()
    if args.epochs < 1:
        p.error('--epochs must be at least 1')
    if args.patience < 1:
        p.error('--patience must be at least 1')
    if args.min_delta < 0:
        p.error('--min-delta must be nonnegative')
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA is unavailable. Use --device cpu or choose the classical track.')
    train_set = ReconstructionDataset(Path(args.data_root) / 'train',
                                      acceleration=args.acceleration, seed=args.seed, size=args.size)
    val_set = ReconstructionDataset(Path(args.data_root) / 'val',
                                    acceleration=args.acceleration, seed=args.seed, size=args.size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = TinyUNet(width=args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()
    best = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for degraded, target in train_loader:
            degraded, target = degraded.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(degraded), target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_losses = []
            for degraded, target in val_loader:
                degraded, target = degraded.to(device), target.to(device)
                val_losses.append(float(loss_fn(model(degraded), target).item()))
        train = float(np.mean(losses))
        val = float(np.mean(val_losses))
        history.append({'epoch': epoch + 1, 'train_l1': train, 'val_l1': val})
        print(f'Epoch {epoch+1}/{args.epochs}  train_L1={train:.5f}  '
              f'val_L1={val:.5f}  device={device}', flush=True)
        if val < best - args.min_delta:
            best = val
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save({'width': args.width, 'state_dict': model.state_dict(),
                        'acceleration': args.acceleration, 'seed': args.seed,
                        'size': args.size, 'best_epoch': best_epoch,
                        'val_l1': best}, destination)
        else:
            epochs_without_improvement += 1
            print(f'  no validation improvement: '
                  f'{epochs_without_improvement}/{args.patience}', flush=True)
            if epochs_without_improvement >= args.patience:
                print(f'Early stopping at epoch {epoch+1}; '
                      f'best epoch was {best_epoch}.', flush=True)
                break
    history_path = destination.with_name(f'{destination.stem}-history.csv')
    with history_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['epoch', 'train_l1', 'val_l1'])
        writer.writeheader()
        writer.writerows(history)
    print(f'Saved best validation checkpoint to {destination} '
          f'(epoch={best_epoch}, val_L1={best:.5f})')
    print(f'Saved training history to {history_path}')


if __name__ == '__main__':
    main()
