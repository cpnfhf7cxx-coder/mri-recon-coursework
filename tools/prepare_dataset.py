#!/usr/bin/env python3
"""Prepare the course dataset or reproduce an instructor-published split."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
FIELDNAMES = ['image_id', 'split', 'source_path', 'sha256_image']


def scan_source(source, out, size):
    files = sorted(p for p in source.rglob('*') if p.is_file()
                   and p.suffix.lower() in EXTENSIONS and out not in p.parents)
    if not files:
        raise ValueError('No PNG/JPG/TIF/BMP images found in source')
    seen, records = set(), []
    for path in files:
        try:
            with Image.open(path) as image:
                image = image.convert('L').resize((size, size), Image.Resampling.BICUBIC)
                gray = np.asarray(image, dtype=np.uint8)
        except Exception:
            continue
        if gray.max() <= gray.min():
            continue
        fingerprint = hashlib.sha256(gray.tobytes()).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        records.append({'source_path': path.relative_to(source).as_posix(),
                        'gray': gray, 'sha256_image': fingerprint})
    return records


def assign_splits(records, seed):
    """Create a deterministic 70/15/15 image-level split per source folder."""
    groups = defaultdict(list)
    for record in records:
        groups[Path(record['source_path']).parts[0]].append(record)
    rng = np.random.default_rng(seed)
    for group in sorted(groups):
        items = groups[group]
        rng.shuffle(items)
        train_end, val_end = int(len(items) * .70), int(len(items) * .85)
        for i, record in enumerate(items):
            record['split'] = 'train' if i < train_end else ('val' if i < val_end else 'test')
    rng.shuffle(records)
    for record in records:
        record['image_id'] = f"mri_{record['sha256_image'][:16]}.png"


def read_protocol(path):
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not set(FIELDNAMES).issubset(rows[0]):
        raise ValueError(f'{path} must contain: {", ".join(FIELDNAMES)}')
    if any(row['split'] not in {'train', 'val', 'test'} for row in rows):
        raise ValueError(f'{path} contains an invalid split')
    return rows


def apply_protocol(records, protocol_rows):
    scanned = {record['source_path']: record for record in records}
    selected, errors = [], []
    for row in protocol_rows:
        record = scanned.get(row['source_path'])
        if record is None:
            errors.append(f"missing: {row['source_path']}")
            continue
        if record['sha256_image'] != row['sha256_image']:
            errors.append(f"hash mismatch: {row['source_path']}")
            continue
        record['split'], record['image_id'] = row['split'], row['image_id']
        selected.append(record)
    if errors:
        raise ValueError('Dataset does not match the official manifest:\n  ' + '\n  '.join(errors[:10]))
    return selected


def write_manifest(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in FIELDNAMES})


def balanced_subset(records, split, count, seed):
    groups = defaultdict(list)
    for record in records:
        if record['split'] == split:
            groups[Path(record['source_path']).parts[0]].append(record['image_id'])
    rng = np.random.default_rng(seed)
    for values in groups.values():
        rng.shuffle(values)
    chosen, names = [], sorted(groups)
    while len(chosen) < count and any(groups.values()):
        for name in names:
            if groups[name] and len(chosen) < count:
                chosen.append(groups[name].pop())
    return chosen


def write_protocol_dir(protocol_dir, records, size, seed):
    protocol_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(protocol_dir / 'split_manifest.csv', records)
    for filename, split, count in (
            ('classical_val.txt', 'val', 20), ('classical_test.txt', 'test', 30)):
        names = balanced_subset(records, split, count, seed)
        (protocol_dir / filename).write_text('\n'.join(names) + '\n', encoding='utf-8')
    counts = {split: sum(record['split'] == split for record in records)
              for split in ('train', 'val', 'test')}
    metadata = {'seed': seed, 'image_size': size,
                'split': {'train': 0.70, 'val': 0.15, 'test': 0.15},
                'actual_counts': counts,
                'classical_val_count': 20, 'classical_test_count': 30,
                'note': 'Image-level split; patient IDs are unavailable.'}
    (protocol_dir / 'protocol.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='Extracted Kaggle dataset directory')
    parser.add_argument('--out', default='data/processed')
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--manifest', help='Official split_manifest.csv to reproduce')
    parser.add_argument('--publish-protocol',
                        help='Instructor only: directory for official protocol files')
    args = parser.parse_args()
    source = Path(args.source).resolve()
    out = Path(args.out).resolve()
    if not source.is_dir():
        parser.error(f'Source directory not found: {source}')
    if source == out:
        parser.error('Source and output directories must be different')
    if out.exists() and any(out.rglob('*.png')):
        parser.error(f'Output already contains PNG files: {out}. Use a new/empty --out directory.')
    try:
        records = scan_source(source, out, args.size)
        if args.manifest:
            records = apply_protocol(records, read_protocol(Path(args.manifest)))
        else:
            assign_splits(records, args.seed)
        if args.publish_protocol:
            write_protocol_dir(Path(args.publish_protocol), records, args.size, args.seed)
    except ValueError as error:
        parser.error(str(error))
    if len(records) < 3:
        parser.error('Need at least three nonblank, nonduplicate images')
    for record in records:
        destination = out / record['split'] / record['image_id']
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(record['gray'], mode='L').save(destination)
    write_manifest(out / 'manifest.csv', records)
    counts = {split: sum(record['split'] == split for record in records)
              for split in ('train', 'val', 'test')}
    print(f'Prepared {len(records)} images at {out}; exact duplicates removed. Splits: {counts}')
    print('CAUTION: No patient IDs: patient-independent split cannot be established.')


if __name__ == '__main__':
    main()
