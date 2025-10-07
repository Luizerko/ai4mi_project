#!/usr/bin/env python3.7

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pickle
import random
import argparse
import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from typing import Callable

import numpy as np
import nibabel as nib
from skimage.io import imsave
from skimage.transform import resize

from utils import map_, tqdm_
import numpy as np
import scipy.ndimage as ndi

def _body_mask(vol_hu: np.ndarray) -> np.ndarray:
    """Return a binary body mask: voxels above -500 HU, followed by morphological closing."""
    m = (vol_hu > -500).astype(np.uint8)
    return ndi.binary_closing(m, iterations=2).astype(np.uint8)

def _lung_mask(vol_hu: np.ndarray) -> np.ndarray:
    """Return a binary lung mask: voxels below -320 HU, with opening and hole filling."""
    m = (vol_hu < -320).astype(np.uint8)
    m = ndi.binary_opening(m, iterations=1).astype(np.uint8)
    m = ndi.binary_fill_holes(m).astype(np.uint8)
    return m

def _mediastinum_mask(body: np.ndarray, lung: np.ndarray) -> np.ndarray:
    """Return mediastinum ROI as body minus lung, with a light morphological opening."""
    med = (body.astype(bool) & (~lung.astype(bool))).astype(np.uint8)
    return ndi.binary_opening(med, iterations=1).astype(np.uint8)

def dilate_by_mm(mask, spacing_xyz, margin_mm):
    """Dilate a 3D boolean mask by an isotropic physical margin in millimeters."""
    dz, dy, dx = spacing_xyz
    dist = ndi.distance_transform_edt(~mask.astype(bool), sampling=(dz, dy, dx))
    return (dist <= margin_mm)

def top_percentile_artifact_mask(vol_hu, roi_mask, p=99.97, hu_abs=1800,
                                 min_area_px=20, dilate_mm_val=2.0,
                                 spacing_xyz=(1.0, 1.0, 1.0)):
    """
    Detect very bright artifacts: threshold at max(percentile p inside ROI, absolute HU),
    remove small components, then optionally dilate by a physical margin.
    Returns uint8 binary mask.
    """
    hu = vol_hu.astype(np.float32)
    roi = (roi_mask > 0)
    thr_p = np.percentile(hu[roi], p) if roi.any() else np.percentile(hu, p)
    cand = (hu >= max(thr_p, hu_abs))
    lab, n = ndi.label(cand)
    if n > 0:
        sizes = ndi.sum(cand, lab, np.arange(1, n + 1))
        keep = (np.where(sizes >= min_area_px)[0] + 1)
        cand = np.isin(lab, keep)
    if dilate_mm_val > 0:
        cand = dilate_by_mm(cand, spacing_xyz, dilate_mm_val)
    return cand.astype(np.uint8)

def replace_artifacts(vol_hu, art_mask, roi_mask=None, mode="roi_median"):
    """
    Replace artifact voxels either with air (-1000 HU) or with the per-slice median
    inside the provided ROI (fallback to slice median if ROI is empty).
    """
    out = vol_hu.copy().astype(np.float32)
    Z = out.shape[0]
    for z in range(Z):
        m_art = art_mask[z] > 0
        if not m_art.any():
            continue
        if roi_mask is not None and (roi_mask[z] > 0).any():
            med = float(np.median(out[z][roi_mask[z] > 0]))
        else:
            med = float(np.median(out[z]))
        out[z][m_art] = med
    return out

def normalize_soft_roi_u8(vol, roi_mask, soft_win=(-150, 250), bone_thr=200):
    """
    Robust uint8 normalization focused on soft tissue within ROI:
    clip to window, z-score within soft-tissue subset, then percentile clamp [2,98] and scale to [0,255].
    """
    hu = vol.astype(np.float32)
    hu = np.clip(hu, soft_win[0], soft_win[1])
    roi = roi_mask > 0
    soft = roi & (hu <= bone_thr) & (hu >= -300)
    if soft.sum() < 100:
        soft = roi if roi.any() else np.ones_like(roi, bool)
    mu = hu[soft].mean()
    sd = hu[soft].std() + 1e-6
    z = (hu - mu) / sd
    if soft.any():
        zmin, zmax = np.percentile(z[soft], (2, 98))
    else:
        zmin, zmax = z.min(), z.max()
    norm = ((np.clip(z, zmin, zmax) - zmin) / (zmax - zmin + 1e-6) * 255).round().astype(np.uint8)
    return norm

def compress_bone(u8, hu, roi_mask, bone_thr=200, alpha=0.6):
    """
    Bone compression: blend high-HU voxels towards the ROI median in the uint8 image.
    alpha controls the blend strength.
    """
    out = u8.astype(np.float32).copy()
    m = (hu > bone_thr) & (roi_mask > 0)
    if not m.any():
        return u8
    med = float(np.median(u8[roi_mask > 0])) if (roi_mask > 0).any() else 128.0
    out[m] = alpha * out[m] + (1.0 - alpha) * med
    return np.clip(out, 0, 255).astype(np.uint8)

def trachea_mask_from_air(hu2d, med2d, air_thr=-800):
    """
    Estimate the trachea as the largest air component within the mediastinum,
    favoring components near the midline.
    """
    air = (hu2d < air_thr) & (med2d > 0)
    if not air.any():
        return np.zeros_like(air, bool)
    lab, n = ndi.label(air)
    if n == 0:
        return np.zeros_like(air, bool)
    H, W = air.shape
    xs = ndi.center_of_mass(np.ones_like(air, float), labels=lab, index=np.arange(1, n + 1))
    midx = W / 2.0
    areas = ndi.sum(air, lab, index=np.arange(1, n + 1))
    scores = []
    for i, ((cy, cx), a) in enumerate(zip(xs, areas)):
        scores.append((a / (1.0 + abs(cx - midx)), i + 1))
    _, best_label = max(scores, key=lambda t: t[0])
    return lab == best_label

def esophageal_band_from_trachea(trach2d, spacing_xy, r_min_mm=4.0, r_max_mm=25.0, posterior_only=True):
    """
    Create an annular band around the trachea based on Euclidean distance in mm.
    """
    if not trach2d.any():
        return np.zeros_like(trach2d, bool)
    dy, dx = spacing_xy
    dist = ndi.distance_transform_edt(~trach2d.astype(bool), sampling=(dy, dx))
    band = (dist >= r_min_mm) & (dist <= r_max_mm)
    if posterior_only:
        H, W = trach2d.shape
        ys, xs = np.nonzero(trach2d)
        cy = ys.mean() if ys.size else H / 2.0
        post = np.zeros_like(band, bool)
        post[int(cy):, :] = True
        band = band & post
    return band

def enhance_esophagus_slice(slice_, band_mask, feather_sigma_px=2.0,
                            clahe_clip=2.0, clahe_tile=(8, 8),
                            unsharp_sigma_px=1.0, unsharp_amount=0.7):
    """
    Apply CLAHE and unsharp masking only within the band mask, with soft blending (feathering).
    """
    import cv2
    base = slice_.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=tuple(clahe_tile))
    eq = clahe.apply(base)
    blur = cv2.GaussianBlur(eq, ksize=(0, 0), sigmaX=float(unsharp_sigma_px))
    sharp = cv2.addWeighted(eq, 1.0 + float(unsharp_amount), blur, -float(unsharp_amount), 0)
    if feather_sigma_px and feather_sigma_px > 0:
        w = ndi.gaussian_filter(band_mask.astype(np.float32), sigma=float(feather_sigma_px))
        w = np.clip(w / (w.max() + 1e-6), 0, 1)
    else:
        w = band_mask.astype(np.float32)
    out = (1.0 - w) * base.astype(np.float32) + w * sharp.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def enhance_esophagus(u8_vol, hu_vol, med_mask, spacing_xyz,
                      r_min_mm=4.0, r_max_mm=25.0):
    """
    Slice-wise esophageal enhancement:
    detect tracheal air within mediastinum, build a posterior annular band,
    and apply local enhancement inside that band.
    """
    dz, dy, dx = spacing_xyz
    out = u8_vol.copy()
    for z in range(u8_vol.shape[0]):
        roi = med_mask[z] > 0
        if not roi.any():
            continue
        trach = trachea_mask_from_air(hu_vol[z], med_mask[z], air_thr=-800)
        if not trach.any():
            continue
        band = esophageal_band_from_trachea(trach, spacing_xy=(dy, dx),
                                            r_min_mm=r_min_mm, r_max_mm=r_max_mm,
                                            posterior_only=True)
        if not band.any():
            continue
        
        out[z] = enhance_esophagus_slice(out[z], band, feather_sigma_px=2.0,
                                             clahe_clip=2.0, clahe_tile=(8, 8),
                                             unsharp_sigma_px=1.0, unsharp_amount=0.7) 
                     
    return out

def _apply_mask(u8_vol: np.ndarray, roi_mask: np.ndarray, feather, sigma_px=4, outside_value=0) -> np.ndarray:
    """
    Apply a binary mask to a uint8 volume. With feather=True, softly blend boundaries using a Gaussian.
    """
    if not feather:
        out = u8_vol.copy()
        out[roi_mask == 0] = outside_value
        return out
    m = roi_mask.astype(np.float32)
    m = ndi.gaussian_filter(m, sigma=(0, sigma_px, sigma_px))
    m = m / (m.max() + 1e-6)
    out = (u8_vol.astype(np.float32) * m).clip(0, 255).astype(np.uint8)
    return out

def zero_from_col_173(img: np.ndarray) -> np.ndarray:
    """
    Zero out all columns x >= 173 for 2D uint8 grayscale or multi-channel images of width >= 174.
    """
    if img.shape[1] < 174:
        return img
    if img.ndim == 2:
        img[:, 173:] = 0
    else:
        img[:, 173:, :] = 0
    return img




def sanity_ct(ct, x, y, z, dx, dy, dz) -> bool:
    assert ct.dtype in [np.int16, np.int32], ct.dtype
    assert -1000 <= ct.min(), ct.min()
    assert ct.max() <= 31743, ct.max()

    assert 0.896 <= dx <= 1.37, dx  # Rounding error
    assert dx == dy
    assert 2 <= dz <= 3.7, dz

    assert (x, y) == (512, 512)
    assert x == y
    assert 135 <= z <= 284, z

    return True


def sanity_gt(gt, ct) -> bool:
    assert gt.shape == ct.shape
    assert gt.dtype in [np.uint8], gt.dtype

    # Do the test on 3d: assume all organs are present..
    assert set(np.unique(gt)) == set(range(5))

    return True


resize_: Callable = partial(resize, mode="constant", preserve_range=True, anti_aliasing=False)


def slice_patient(id_: str, dest_path: Path, source_path: Path, shape: tuple[int, int],
                  test_mode: bool = False) -> tuple[float, float, float]:
    id_path: Path = source_path / ("train" if not test_mode else "test") / id_

    ct_path: Path = (id_path / f"{id_}.nii.gz") if not test_mode else (source_path / "test" / f"{id_}.nii.gz")
    nib_obj = nib.load(str(ct_path))
    ct: np.ndarray = np.asarray(nib_obj.dataobj)
    # dx, dy, dz = nib_obj.header.get_zooms()
    x, y, z = ct.shape
    dx, dy, dz = nib_obj.header.get_zooms()

    assert sanity_ct(ct, *ct.shape, *nib_obj.header.get_zooms())

    body = _body_mask(ct)
    lungs = _lung_mask(ct)
    med = _mediastinum_mask(body, lungs)
    dz,dy,dx = nib_obj.header.get_zooms()[2], nib_obj.header.get_zooms()[1], nib_obj.header.get_zooms()[0]
    art = top_percentile_artifact_mask(ct, med, p=99.97, hu_abs=1800,
                                    min_area_px=20, dilate_mm_val=2.0,
                                    spacing_xyz=(dz,dy,dx))

    # 2) Sustituye artefacto por valor neutro (mediana ROI por slice)
    ct= replace_artifacts(ct, art, roi_mask=med, mode="roi_median")

    gt: np.ndarray
    if not test_mode:
        gt_path: Path = id_path / "GT.nii.gz"
        gt_nib = nib.load(str(gt_path))
        # print(nib_obj.affine, gt_nib.affine)
        gt = np.asarray(gt_nib.dataobj)
        assert sanity_gt(gt, ct)
    else:
        gt = np.zeros_like(ct, dtype=np.uint8)

    norm_ct = normalize_soft_roi_u8(ct, med)
    norm_ct = compress_bone(norm_ct, ct, med, bone_thr=200, alpha=0.6)

# ---------- Realce dirigido del esófago ----------
    norm_ct = enhance_esophagus(norm_ct, ct, med, spacing_xyz=(dz,dy,dx),
                        r_min_mm=4.0, r_max_mm=25.0)

    to_slice_ct = _apply_mask(norm_ct, med, feather=False, sigma_px=4, outside_value=0)

    to_slice_gt = gt


    for idz in range(z):
        img_slice = resize_(to_slice_ct[:, :, idz], shape).astype(np.uint8)
        gt_slice = resize_(to_slice_gt[:, :, idz], shape, order=0).astype(np.uint8)
        img_slice = zero_from_col_173(img_slice).astype(np.uint8)
        assert img_slice.shape == gt_slice.shape
        gt_slice *= 63
        assert gt_slice.dtype == np.uint8, gt_slice.dtype
        # assert set(np.unique(gt_slice)) <= set(range(5))
        assert set(np.unique(gt_slice)) <= set([0, 63, 126, 189, 252]), np.unique(gt_slice)

        arrays: list[np.ndarray] = [img_slice, gt_slice]

        subfolders: list[str] = ["img", "gt"]
        assert len(arrays) == len(subfolders)
        for save_subfolder, data in zip(subfolders,
                                        arrays):
            filename = f"{id_}_{idz:04d}.png"

            save_path: Path = Path(dest_path, save_subfolder)
            save_path.mkdir(parents=True, exist_ok=True)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                imsave(str(save_path / filename), data)

    return dx, dy, dz


def get_splits(src_path: Path, retains: int, fold: int) -> tuple[list[str], list[str], list[str]]:
    ids: list[str] = sorted(map_(lambda p: p.name, (src_path / 'train').glob('*')))
    print(f"Founds {len(ids)} in the id list")
    print(ids[:10])
    assert len(ids) > retains

    random.shuffle(ids)  # Shuffle before to avoid any problem if the patients are sorted in any way
    validation_slice = slice(fold * retains, (fold + 1) * retains)
    validation_ids: list[str] = ids[validation_slice]
    assert len(validation_ids) == retains

    training_ids: list[str] = [e for e in ids if e not in validation_ids]
    assert (len(training_ids) + len(validation_ids)) == len(ids)

    test_ids: list[str] = sorted(map_(lambda p: Path(p.stem).stem, (src_path / 'test').glob('*')))
    print(f"Founds {len(test_ids)} test ids")
    print(test_ids[:10])

    return training_ids, validation_ids, test_ids


def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)

    # Assume the clean up is done before calling the script
    assert src_path.exists()
    assert not dest_path.exists()

    training_ids: list[str]
    validation_ids: list[str]
    test_ids: list[str]
    training_ids, validation_ids, test_ids = get_splits(src_path, args.retains, args.fold)

    resolution_dict: dict[str, tuple[float, float, float]] = {}

    split_ids: list[str]
    for mode, split_ids in zip(["train", "val", "test"], [training_ids, validation_ids, test_ids]):
        dest_mode: Path = dest_path / mode
        print(f"Slicing {len(split_ids)} pairs to {dest_mode}")

        pfun: Callable = partial(slice_patient,
                                 dest_path=dest_mode,
                                 source_path=src_path,
                                 shape=tuple(args.shape),
                                 test_mode=mode == 'test')
        resolutions: list[tuple[float, float, float]]
        iterator = tqdm_(split_ids)
        match args.process:
            case 1:
                resolutions = list(map(pfun, iterator))
            case -1:
                resolutions = Pool().map(pfun, iterator)
            case _ as p:
                resolutions = Pool(p).map(pfun, iterator)

        for key, val in zip(split_ids, resolutions):
            resolution_dict[key] = val

    with open(dest_path / "spacing.pkl", 'wb') as f:
        pickle.dump(resolution_dict, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved spacing dictionnary to {f}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Slicing parameters')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)

    parser.add_argument('--shape', type=int, nargs="+", default=[256, 256])
    parser.add_argument('--retains', type=int, default=25, help="Number of retained patient for the validation data")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--process', '-p', type=int, default=1,
                        help="The number of cores to use for processing")
    args = parser.parse_args()
    random.seed(args.seed)

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
