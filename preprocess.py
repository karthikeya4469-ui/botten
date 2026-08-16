#!/usr/bin/env python3
"""
Fast parallel preprocessing for MRI images and clinical CSV.

Saves per-sample `.npz` files by default for low-memory, fast I/O.

Usage (example):
  python preprocess.py --image-dir data/images --csv clinical.csv --out-dir preproc --workers 8

Features:
- Supports NIfTI (.nii, .nii.gz), DICOM (.dcm), and common image types (.png, .jpg)
- Uses multiprocessing to process images in parallel
- Handles 3D volumes by taking central axial slice
- Normalizes images (0-1) and z-score standardization
"""
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm

try:
    import nibabel as nib
except Exception:
    nib = None

try:
    import pydicom
except Exception:
    pydicom = None

from PIL import Image


IMG_EXTS = ('.nii', '.nii.gz', '.dcm', '.png', '.jpg', '.jpeg', '.tiff')
IGNORE_CSV_NAMES = {'housing.csv'}


def find_image_files(image_dir):
    p = Path(image_dir)
    files = [str(x) for x in p.rglob('*') if x.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tiff', '.dcm'} or x.name.lower().endswith('.nii.gz') or x.suffix.lower() == '.nii']
    return files


def load_image(path):
    path = str(path)
    lower = path.lower()
    if lower.endswith('.nii') or lower.endswith('.nii.gz'):
        if nib is None:
            raise ImportError('nibabel required to read NIfTI files')
        img = nib.load(path).get_fdata()
        return img
    if lower.endswith('.dcm'):
        if pydicom is None:
            raise ImportError('pydicom required to read DICOM files')
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array
        return arr
    # fallback to PIL for 2D images
    img = Image.open(path).convert('L')
    return np.array(img)


def center_slice(volume):
    if volume.ndim == 2:
        return volume
    # pick central slice along the last axis or first depending on shape
    # prefer axial (z) dimension if present
    z = volume.shape[2] if volume.ndim >= 3 else 0
    if z == 0:
        # fallback to middle of first axis
        idx = volume.shape[0] // 2
        return volume[idx]
    idx = volume.shape[2] // 2
    return volume[:, :, idx]


def process_one(path, out_dir, img_size, id_key=None, save_per_sample=True):
    try:
        arr = load_image(path)
    except Exception as e:
        return {'path': path, 'error': str(e)}

    if arr is None:
        return {'path': path, 'error': 'empty'}

    if arr.ndim >= 3:
        arr = center_slice(arr)

    # convert to float32
    arr = arr.astype('float32')

    # resize using PIL
    img = Image.fromarray(np.nan_to_num(arr).astype('float32'))
    if img_size is not None:
        img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(img)

    # normalize 0-1
    mn = arr.min()
    mx = arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = arr * 0.0

    # z-score
    mean = arr.mean()
    std = arr.std() if arr.std() > 0 else 1.0
    arr = (arr - mean) / std

    # prepare saving
    fname = Path(path).stem
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_per_sample:
        out_path = out_dir / f"{fname}.npz"
        np.savez_compressed(out_path, image=arr, mean=mean, std=std, src=path)
        return {'path': path, 'out': str(out_path)}
    else:
        # return in-memory
        return {'path': path, 'image': arr, 'mean': mean, 'std': std}


def group_and_save_in_memory(results, out_dir, meta_df=None):
    images = []
    paths = []
    for r in results:
        if 'image' in r:
            images.append(r['image'])
            paths.append(r['path'])
    images = np.stack(images)
    out_path = Path(out_dir) / 'dataset.npz'
    np.savez_compressed(out_path, images=images, paths=np.array(paths))
    return str(out_path)


def merge_with_csv(out_dir, csv_path, id_column='image_id', file_stem_column=None):
    # Creates a metadata CSV linking preprocessed files to clinical data
    csv_name = Path(csv_path).name.lower()
    if csv_name in IGNORE_CSV_NAMES:
        print(f'Ignored CSV "{csv_name}" — skipping merge (irrelevant file).')
        return ''
    df = pd.read_csv(csv_path)
    out_dir = Path(out_dir)
    records = []
    for p in out_dir.glob('*.npz'):
        stem = p.stem
        # try to match by stem
        match = df[df[id_column].astype(str) == stem]
        if match.empty and file_stem_column:
            match = df[df[file_stem_column].astype(str) == stem]
        row = match.to_dict(orient='records')[0] if not match.empty else {}
        row.update({'preproc_file': str(p), 'image_stem': stem})
        records.append(row)
    meta = pd.DataFrame(records)
    meta_path = out_dir / 'metadata.csv'
    meta.to_csv(meta_path, index=False)
    return str(meta_path)


def parse_args():
    p = argparse.ArgumentParser(description='Fast parallel preprocessing')
    p.add_argument('--image-dir', required=True)
    p.add_argument('--csv', required=False, help='Clinical CSV to merge')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--save-per-sample', action='store_true')
    p.add_argument('--id-column', default='image_id', help='column in CSV matching file stem')
    return p.parse_args()


def main():
    args = parse_args()
    files = find_image_files(args.image_dir)
    if not files:
        print('No image files found in', args.image_dir)
        return

    process = partial(process_one, out_dir=args.out_dir, img_size=args.img_size, save_per_sample=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process, f): f for f in files}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                r = fut.result()
            except Exception as e:
                r = {'path': str(futures[fut]), 'error': str(e)}
            results.append(r)

    errors = [r for r in results if 'error' in r]
    print(f'Processed: {len(results)-len(errors)}; Errors: {len(errors)}')
    if args.csv:
        meta = merge_with_csv(args.out_dir, args.csv, id_column=args.id_column)
        print('Wrote metadata to', meta)


if __name__ == '__main__':
    main()
