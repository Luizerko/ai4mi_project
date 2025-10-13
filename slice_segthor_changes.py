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
import cv2

#PREPROCESSING CHANGES

#1. MEDIASTINUM MASKING

def body_mask(volume_original: np.ndarray) -> np.ndarray:  
    body_mask = (volume_original > -500).astype(np.uint8) #We set the threshold to -500 to remove background
    body_mask = ndi.binary_closing(body_mask, iterations=2).astype(np.uint8) #We apply binary closing to reduce background
    return body_mask

def lung_mask(volume_original: np.ndarray) -> np.ndarray:
    mask_lungs = (volume_original < -320).astype(np.uint8) #We set the threshold to -320 to avoid possible lungs soft tissue
    mask_lungs = ndi.binary_fill_holes(ndi.binary_opening(mask_lungs, iterations=1).astype(np.uint8)).astype(np.uint8) #We apply binary opening to include possible soft tissue within the mask
    return mask_lungs

def mediastinum_mask(body_mask: np.ndarray, lung_mask: np.ndarray) -> np.ndarray:
    mediastinum_mask = (body_mask.astype(bool) & (~lung_mask.astype(bool))).astype(np.uint8) #To include body and exclude lung within the mask
    mediastinum_mask = ndi.binary_opening(mediastinum_mask, iterations=1).astype(np.uint8) #We apply opening to include more soft tissue inside the mediastinum_mask
    return mediastinum_mask

def dilate(mediastinum_mask, spacing, margin):
    dz, dy, dx = spacing
    dist = ndi.distance_transform_edt(~mediastinum_mask.astype(bool), sampling=(dz, dy, dx)) #Calculate the eucledian distance from every voxel outside the mediastinum mask to the rest of the voxels
    dilated_mask = dist <= margin #Return the dilated_mask
    return dilated_mask

def apply_mask(volume: np.ndarray, mediastinum_mask: np.ndarray) -> np.ndarray: 
    out = volume.copy()
    out[mediastinum_mask == 0] = 0 #We set everything outside the mediastinum mask to 0
    return out

#2. SOFT TISSUE NORMALIZATION

def normalize_soft_tissue(volume, mediastinum_mask, soft_tissue_window=(-150, 250), bone_thr=200):

    volume = volume.astype(np.float32) #We set values to float32 to allow for accurate calculations
    volume = np.clip(volume, soft_tissue_window[0], soft_tissue_window[1]) #1. Clipping to soft tissue window
    mask_norm = mediastinum_mask > 0 
    soft = mask_norm & (volume <= bone_thr) & (volume >= -300) #We make a soft tissue mask (inlcuding the mediastinum_mask and excluding bones and background) 
    if soft.sum() < 100: #If the sum of the soft tissue mask is very low to avoid wrong normalization we set the soft mask tissue to the entire mediastinum 
        soft = mask_norm if mask_norm.any() else np.ones_like(mask_norm, bool)
    # 2. Z-score normalization
    mu = volume[soft].mean()
    sd = volume[soft].std() + 1e-6
    z = (volume - mu) / sd
    if soft.any():
        zmin, zmax = np.percentile(z[soft], (2, 98)) # We set the zmin and zmax to the percentiles 2 and 98 to avoid extreme values
    else:
        zmin, zmax = z.min(), z.max()
    
    #3. Normalization
    norm = ((np.clip(z, zmin, zmax) - zmin) / (zmax - zmin + 1e-6) * 255).round().astype(np.uint8) #We include 1e-6 terminus in the denominator to avoid 0 division error
    return norm

#3. CONTRAST ENHANCEMENT

def compress_bone(volume_norm, volume_original, mediastinum_mask, bone_thr=200, alpha=0.6):
    out = volume_norm.astype(np.float32).copy()
    mask_meadistinum_bones = (volume_original > bone_thr) & (mediastinum_mask > 0) #We create a mask for the mediastinum bones (bones inside mediastinum mask)
    if not mask_meadistinum_bones.any():
        return volume_norm
    med = float(np.median(volume_norm[mediastinum_mask > 0])) if (mediastinum_mask > 0).any() else 128.0 #Median of the mediastinum values
    out[mask_meadistinum_bones] = alpha * out[mask_meadistinum_bones] + (1.0 - alpha) * med #We move the value of the meadistinum_bones towards the median
    out=np.clip(out, 0, 255).astype(np.uint8)
    return out

def find_trachea(volume_slices, mediastinum_mask_slices, air_thr=-800):

    air = (volume_slices < air_thr) & (mediastinum_mask_slices > 0)
    if not air.any():
        return np.zeros_like(air, bool)
    label, n = ndi.label(air) #We find elements of air inside the mediastinum
    if n == 0:
        return np.zeros_like(air, bool) 
    H, W = air.shape
    xs = ndi.center_of_mass(np.ones_like(air, float), labels=label, index=np.arange(1, n + 1))
    midx = W / 2.0
    areas = ndi.sum(air, label, index=np.arange(1, n + 1)) #We compute the sum of each elements within the air mask
    scores = []
    for i, ((cy, cx), a) in enumerate(zip(xs, areas)):
        scores.append((a / (1.0 + abs(cx - midx)), i + 1)) #We favour the components which are close to the middle line of the mediastinum
    _, best_label = max(scores, key=lambda t: t[0]) #We identify the best label via the maximum score
    trachea = (label == best_label) #Trachea is the best label
    return trachea

def esophageal_band_from_trachea(trachea_slice, spacing, r_min_mm=4.0, r_max_mm=25.0):

    if not trachea_slice.any(): 
        return np.zeros_like(trachea_slice, bool) #If there are no trachea slices found return a zero mask
    dy, dx = spacing
    dist = ndi.distance_transform_edt(~trachea_slice.astype(bool), sampling=(dy, dx)) #We compute the eucledian distance fron the trachea to zero elements
    band = (dist >= r_min_mm) & (dist <= r_max_mm) #We compute the band from 4mm to 25mm from trachea
    #We identify the elements which are posterior to the trachea
    H, W = trachea_slice.shape
    ys, xs = np.nonzero(trachea_slice)
    cy = ys.mean() if ys.size else H / 2.0
    post = np.zeros_like(band, bool)
    post[int(cy):, :] = True
    band = band & post #Final band is 4mm to 25mm posterior to trachea
    return band

def enhance_esophagus_slice(slice_, band_mask, feather_sigma_px=2.0, clahe_clip=2.0, clahe_tile=(8, 8), unsharp_sigma_px=1.0, unsharp_amount=0.7): 
    base = slice_.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=tuple(clahe_tile))
    equalized = clahe.apply(base) #Apply CLAHE enhancement to base slices 
    blur = cv2.GaussianBlur(equalized, ksize=(0, 0), sigmaX=float(unsharp_sigma_px)) #Apply blurring to the equilized model
    sharp = cv2.addWeighted(equalized, 1.0 + float(unsharp_amount), blur, -float(unsharp_amount), 0)
    if feather_sigma_px and feather_sigma_px > 0:
        w = ndi.gaussian_filter(band_mask.astype(np.float32), sigma=float(feather_sigma_px)) #Apply gaussian_filtering 
        w = np.clip(w / (w.max() + 1e-6), 0, 1)
    else:
        w = band_mask.astype(np.float32)
    out = (1.0 - w) * base.astype(np.float32) + w * sharp.astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def enhance_esophagus(volume_norm, volume_original, med_mask, spacing, r_min_mm=4.0, r_max_mm=25.0):
    dz, dy, dx = spacing
    out = volume_norm.copy()
    for z in range(volume_norm.shape[0]):
        roi = med_mask[z] > 0
        if not roi.any():
            continue
        trach = find_trachea(volume_original[z], med_mask[z], air_thr=-800) #Find the trachea
        if not trach.any():
            continue
        band = esophageal_band_from_trachea(trach, spacing=(dy, dx), r_min_mm=r_min_mm, r_max_mm=r_max_mm) #Find the esophageal band posterior to trachea
        if not band.any():
            continue
        out[z] = enhance_esophagus_slice(out[z], band, feather_sigma_px=2.0, clahe_clip=2.0, clahe_tile=(8, 8), unsharp_sigma_px=1.0, unsharp_amount=0.7) #Enhance the esophagus     
    return out

#4. NOISE AND ARTIFACTS REMOVAL

def artifact_mask(volume, mediastinum_mask, p=99.97, upper_thr=1800, min_area_px=20, dilate_mm_val=2.0, spacing=(1.0, 1.0, 1.0)):
    volume = volume.astype(np.float32)
    mask_artifact = (mediastinum_mask > 0)
    upper_value_inside_mask = np.percentile(volume[mask_artifact], p) if mask_artifact.any() else np.percentile(volume, p)
    Art_surface = (volume >= max(upper_value_inside_mask, upper_thr)) #Mask with the possible artifact volume (maximun of 1800--very bright or the upper percentile)
    lab, n = ndi.label(Art_surface)
    if n > 0: #If there are high-intensity components 
        sizes = ndi.sum(Art_surface, lab, np.arange(1, n + 1)) #We compute the area of those components
        keep = (np.where(sizes >= min_area_px)[0] + 1) #We keep the components whose area is greater than 20 pixels (exclude minor artifacts)
        Art_surface = np.isin(lab, keep)
    if dilate_mm_val > 0:
        art_surface = dilate(Art_surface, spacing, dilate_mm_val) #We dillate a little bit the artifact mask
    return art_surface.astype(np.uint8)

def replace_artifacts(volume, art_mask, mediastinum_mask=None):
    out = volume.copy().astype(np.float32)
    Z = out.shape[0]
    for z in range(Z):
        m_art = art_mask[z] > 0
        if not m_art.any():
            continue #If there are no artifacts, continue
        if mediastinum_mask is not None and (mediastinum_mask[z] > 0).any(): 
            median = float(np.median(out[z][mediastinum_mask[z] > 0])) #If mediastinum is not None 
        else:
            median = float(np.median(out[z]))
        out[z][m_art] = median #We replace the bright artifacts with the median 
    return out 

def ct_artifacts(img: np.ndarray) -> np.ndarray:
    img[:, 173:] = 0 #Set to 0 all the values from pixel 173 onwards
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

    body = body_mask(ct)
    lungs = lung_mask(ct)
    med = mediastinum_mask(body, lungs)
    dz,dy,dx = nib_obj.header.get_zooms()[2], nib_obj.header.get_zooms()[1], nib_obj.header.get_zooms()[0]
    art = artifact_mask(ct, med, p=99.97, upper_thr=1800,
                                    min_area_px=20, dilate_mm_val=2.0,
                                    spacing=(dz,dy,dx))

    ct= replace_artifacts(ct, art, mediastinum_mask=med)

    gt: np.ndarray
    if not test_mode:
        gt_path: Path = id_path / "GT.nii.gz"
        gt_nib = nib.load(str(gt_path))
        # print(nib_obj.affine, gt_nib.affine)
        gt = np.asarray(gt_nib.dataobj)
        assert sanity_gt(gt, ct)
    else:
        gt = np.zeros_like(ct, dtype=np.uint8)

    norm_ct = normalize_soft_tissue(ct, med)
    norm_ct = compress_bone(norm_ct, ct, med, bone_thr=200, alpha=0.6)

    norm_ct = enhance_esophagus(norm_ct, ct, med, spacing=(dz,dy,dx),
                        r_min_mm=4.0, r_max_mm=25.0)

    to_slice_ct = apply_mask(norm_ct, med)

    to_slice_gt = gt


    for idz in range(z):
        img_slice = resize_(to_slice_ct[:, :, idz], shape).astype(np.uint8)
        gt_slice = resize_(to_slice_gt[:, :, idz], shape, order=0).astype(np.uint8)
        img_slice = ct_artifacts(img_slice).astype(np.uint8)
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
