# Preprocessing for Batten Disease Project

This folder contains a fast, parallel preprocessing script to prepare MRI images and clinical CSVs for training.

Quick start

1. Install requirements:

```bash
pip install -r requirements.txt
```

2. Run preprocessing:

```bash
python preprocess.py --image-dir /path/to/images --csv clinical_synthetic_dataset.csv --out-dir preproc --workers 8
```

Notes
- The script supports NIfTI, DICOM, and common image files.
- It produces one `.npz` file per image in `--out-dir` and a `metadata.csv` if `--csv` is provided.
- For 3D volumes the script extracts the central axial slice.

Note about irrelevant CSVs
- Any CSV named `Housing.csv` will be ignored by the merge step to prevent accidental processing of unrelated files.

If you want a single packed dataset for training, we can add an option to merge all arrays into one large file or TFRecord.
