import os
import re
import argparse
import warnings
from pathlib import Path

import numpy as np
import nibabel as nib
from PIL import Image
from skimage.transform import resize


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slicing parameters")

    parser.add_argument('--data_folder', type=str, required=True,
                        help="name of the data folder with sliced data, eg data/prediction/best_epoch/val")
    parser.add_argument('--dest_folder', type=str, required=True,
                        help="name of the destination folder with stitched data, eg val/pred")
    parser.add_argument('--num_classes', type=int, default=5,
                        help="number of classes (for SEGTHOR it should be 5: 0–4)")
    parser.add_argument('--grp_regex', type=str, default="(Patient_\\d\\d)_\\d\\d\\d\\d",
                        help="pattern for the filename")
    parser.add_argument("--source_scan_pattern", type=str,
                        default="data/train/train/{id_}/GT.nii.gz",
                        help="pattern to the original scans to get original size (with {id_} replaced by the PatientID)")

    args = parser.parse_args()
    print(args)
    return args


def load_png_as_array(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")  # grayscale
    return np.array(img, dtype=np.uint8)


def stitch_patient(patient_id: str, slice_files: list[str],
                   source_scan_pattern: str, dest_folder: str, num_classes: int) -> None:
    # Load and stack -> (Z, H, W)
    slices = [load_png_as_array(f) for f in sorted(slice_files)]
    if len(slices) == 0:
        warnings.warn(f"No slices for {patient_id}")
        return
    vol = np.stack(slices, axis=0)

    # Reorder axes -> (H, W, Z) == (X, Y, Z)
    vol = np.transpose(vol, (1, 2, 0))

    # map back to 0..4 
    uniq = set(np.unique(vol).tolist())
    if uniq.issubset({0, 63, 126, 189, 252}):
        mapper = {0: 0, 63: 1, 126: 2, 189: 3, 252: 4}
        vol = np.vectorize(mapper.get)(vol).astype(np.uint8)

    # Match GT shape & affine
    gt_path = source_scan_pattern.replace("{id_}", patient_id)
    if not os.path.exists(gt_path):
        raise RuntimeError(f"GT not found for {patient_id}")
    else:
        gt_img = nib.load(gt_path)
        gt_shape = gt_img.shape  # (X, Y, Z)
        affine = gt_img.affine

        if vol.shape != gt_shape:
            # Resize label volume with order 0
            vol = resize(vol, gt_shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)

    # Clamp and save
    vol = np.clip(vol, 0, num_classes - 1).astype(np.uint8)

    os.makedirs(dest_folder, exist_ok=True)
    out_path = os.path.join(dest_folder, f"{patient_id}.nii.gz")
    nib.save(nib.Nifti1Image(vol, affine), out_path)
    print(f"Saved {out_path} | shape={vol.shape} dtype={vol.dtype} unique={np.unique(vol)}")


def main(args):
    data_folder = Path(args.data_folder)
    regex = re.compile(args.grp_regex)

    slice_files = list(data_folder.glob("*.png"))
    if not slice_files:
        raise RuntimeError(f"No .png files found in {data_folder}")

    # Group by patient ID captured by grp_regex
    groups: dict[str, list[str]] = {}
    for f in slice_files:
        m = regex.match(f.name)
        if m:
            pid = m.group(1)
            groups.setdefault(pid, []).append(str(f))
        else:
            warnings.warn(f"File {f} did not match regex, skipping")

    for pid, files in groups.items():
        stitch_patient(pid, files, args.source_scan_pattern, args.dest_folder, args.num_classes)


if __name__ == "__main__":
    main(get_args())
