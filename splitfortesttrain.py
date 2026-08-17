"""
Create train/test CSV files from a preprocessed dataset folder.

Usage examples:
  python train_test_split.py --input-dir processed_output --test-size 0.2
  python train_test_split.py --meta processed_output/metadata.csv --test-size 0.25 --out-dir processed_output

The script writes `train.csv` and `test.csv` into `--out-dir` (defaults to `--input-dir`).
"""
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import sys


def stratified_split(indices, labels, test_size, rng):
    # labels: array-like of class labels
    labels = np.asarray(labels)
    unique = np.unique(labels)
    train_idx = []
    test_idx = []
    for u in unique:
        idx = np.where(labels == u)[0]
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        n_test = int(np.ceil(len(idx) * test_size)) if test_size < 1.0 else int(test_size) if isinstance(test_size, int) else 0
        test_sub = idx[:n_test]
        train_sub = idx[n_test:]
        test_idx.extend(test_sub.tolist())
        train_idx.extend(train_sub.tolist())
    return np.array(train_idx), np.array(test_idx)


def main(argv=None):
    p = argparse.ArgumentParser(description='Split preprocessed dataset into train and test CSVs')
    p.add_argument('--input-dir', required=False, help='Directory with preprocessed .npz files')
    p.add_argument('--meta', required=False, help='Path to metadata.csv linking preproc files (optional)')
    p.add_argument('--out-dir', required=False, help='Where to write train.csv and test.csv (defaults to input-dir or metadata folder)')
    p.add_argument('--test-size', type=float, default=0.2, help='Fraction (0-1) or integer count for test set')
    p.add_argument('--random-state', type=int, default=42)
    p.add_argument('--stratify', required=False, help='Column name in metadata to stratify by')
    args = p.parse_args(args=argv)

    rng = np.random.RandomState(args.random_state)

    meta_df = None
    if args.meta:
        meta_path = Path(args.meta)
        if not meta_path.exists():
            print('Metadata file not found:', meta_path)
            return
        meta_df = pd.read_csv(meta_path)
        base_out = args.out_dir or str(meta_path.parent)
    else:
        if not args.input_dir:
            print('Either --meta or --input-dir must be provided')
            return
        base_out = args.out_dir or args.input_dir

    out_dir = Path(base_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if meta_df is None:
        # build metadata from .npz files
        inp = Path(args.input_dir)
        if not inp.exists():
            print('Input directory not found:', inp)
            return
        npz_files = sorted([p for p in inp.glob('*.npz')])
        if len(npz_files) == 0:
            print('No .npz files found in', inp)
            return
        meta_df = pd.DataFrame({'preproc_file': [str(p) for p in npz_files], 'image_stem': [p.stem for p in npz_files]})

    n = len(meta_df)
    if isinstance(args.test_size, float) and 0.0 < args.test_size < 1.0:
        test_size = int(np.round(n * args.test_size))
    elif isinstance(args.test_size, float) and args.test_size >= 1.0:
        test_size = int(args.test_size)
    else:
        test_size = int(args.test_size)

    indices = np.arange(n)

    if args.stratify and args.stratify in meta_df.columns:
        train_idx, test_idx = stratified_split(indices, meta_df[args.stratify].values, args.test_size, rng)
    else:
        rng.shuffle(indices)
        if test_size <= 0:
            print('Test size is 0 or invalid; writing only train.csv')
            train_idx = indices
            test_idx = np.array([], dtype=int)
        else:
            test_idx = indices[:test_size]
            train_idx = indices[test_size:]

    train_df = meta_df.iloc[train_idx].reset_index(drop=True)
    test_df = meta_df.iloc[test_idx].reset_index(drop=True)

    train_path = out_dir / 'train.csv'
    test_path = out_dir / 'test.csv'
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f'Wrote {len(train_df)} train rows to', train_path)
    print(f'Wrote {len(test_df)} test rows to', test_path)


if __name__ == '__main__':
    try:
        if hasattr(sys, 'ps1'): # Running in interactive mode
            print("Running in interactive mode. Providing default arguments for direct cell execution.")
            main(['--input-dir', 'processed_output', '--out-dir', 'model_output', '--test-size', '0.2'])
        else:
            main() # When run as an external script, argv=None is passed
    except SystemExit as e:
        if e.code != 0:
            raise
