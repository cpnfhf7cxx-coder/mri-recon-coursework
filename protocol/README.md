# Official evaluation protocol

These files are part of the coursework and must be distributed unchanged.

- `split_manifest.csv`: official image-level train/val/test membership and
  post-processing SHA-256 hashes.
- `classical_val.txt`: 20 fixed validation images for tuning classical methods.
- `classical_test.txt`: 30 fixed test images for the final classical evaluation.
- `protocol.json`: image size, seed, split counts, and subset sizes.

Students reproduce the split with:

```bash
python tools/prepare_dataset.py --source "<extracted dataset>" --out data/processed --size 128 --manifest protocol/split_manifest.csv
```

The source dataset has no patient identifiers. This is therefore an image-level,
not patient-independent, split.
