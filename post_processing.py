import numpy as np
import os
import argparse
import scipy.ndimage as ndi
import nibabel as nib
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter, binary_dilation


def keep_components(labelmap, min_size=800, collapse_to_single_component=False):
    out = np.zeros_like(labelmap, dtype=labelmap.dtype)
    for label in np.unique(labelmap):
        if label == 0:
            continue
        mask = (labelmap == label)
        labeled, n = ndi.label(mask)
        if n == 0:
            continue
        # sizes: array indexed 1..n
        sizes = np.bincount(labeled.ravel())[1:]
        keep_idx = []
        for i, s in enumerate(sizes, start=1):
            if s >= min_size:
                keep_idx.append((i, s))
        if not keep_idx:
            largest_idx = int(np.argmax(sizes) + 1)
            keep_idx = [(largest_idx, sizes[largest_idx - 1])]

        if collapse_to_single_component:
            keep_idx.sort(key=lambda p: p[1], reverse=True)
            keep_idx = keep_idx[:1]
        for idx, _ in keep_idx:
            out[labeled == idx] = label
    return out


def smooth_prediction(volume, structure_size=2):
    smoothed = np.zeros_like(volume)
    structure = np.ones((structure_size,)*3)  # 3D kernel
    for label in np.unique(volume):
        if label == 0:
            continue
        mask = (volume == label)
        mask = binary_fill_holes(mask)
        mask = binary_closing(mask, structure=structure)
        smoothed[mask] = label
    return smoothed


def expand_volume(volume, iterations=1, structure_size=2):
    expanded = np.zeros_like(volume)
    structure = np.ones((structure_size,) * 3)  # 3x3x3 cube by default
    for label in np.unique(volume):
        if label == 0:
            continue
        mask = (volume == label)
        dilated = binary_dilation(mask, structure=structure, iterations=iterations)
        expanded[dilated] = label
    return expanded

def strong_smooth_and_expand(volume, structure_size=3, expand_iters=1):
    closed = np.zeros_like(volume)
    structure = np.ones((structure_size,) * 3)
    for label in np.unique(volume):
        if label == 0:
            continue
        mask = (volume == label)
        mask = binary_fill_holes(mask)
        mask = binary_closing(mask, structure=structure)
        mask = binary_dilation(mask, structure=structure, iterations=expand_iters)
        closed[mask] = label
    return closed



def gaussian_smooth_expand(volume, sigma=1.5, threshold=0.4):
    smoothed = np.zeros_like(volume)
    labels = sorted(np.unique(volume))
    labels_ordered = [labels[3], labels[1], labels[4], labels[2]]  #trachea, esophagus, aorta, heart
    for label in labels_ordered:
        if label == 0:
            continue
        mask = (volume == label).astype(np.float32)
        blurred = gaussian_filter(mask, sigma=sigma)
        new_mask = blurred > threshold
        smoothed[new_mask] = label
    return smoothed


def postprocess_predicted_labelmap(pred_labelmap,
                                   min_size=800,
                                   collapse_to_single_component=False,
                                   gaussian_sigma=1.5,
                                   gaussian_threshold=0.4):
    step1 = keep_components(pred_labelmap,
                            min_size=min_size,
                            collapse_to_single_component=False)
    step2 = smooth_prediction(step1)
    final = gaussian_smooth_expand(step2, sigma=gaussian_sigma, threshold=gaussian_threshold)

    if collapse_to_single_component:
        final = keep_components(final,
                                min_size=0,
                                collapse_to_single_component=True)
    return final


def process_file(pred_file, dest_path, mode):
    pred_img = nib.load(pred_file)
    pred_data = pred_img.get_fdata().astype(np.uint8)

    if mode == "full":
        print(f"Running full postprocessing on {os.path.basename(pred_file)}...")
        proc = postprocess_predicted_labelmap(pred_data)

    elif mode == "strong":
        print(f"Running keep_components + strong_smooth_and_expand on {os.path.basename(pred_file)}...")
        comp = keep_components(pred_data)
        proc = strong_smooth_and_expand(comp)

    elif mode == "gaussian":
        print(f"Running keep_components + gaussian_smooth_expand on {os.path.basename(pred_file)}...")
        comp = keep_components(pred_data)
        proc = gaussian_smooth_expand(comp, sigma=1.5, threshold=0.4)

    elif mode == "expand_gaussian":
        print(f"Running keep_components + expand + gaussian_smooth_expand on {os.path.basename(pred_file)}...")
        comp = keep_components(pred_data)
        expanded = expand_volume(comp, iterations=1, structure_size=2)
        proc = gaussian_smooth_expand(expanded, sigma=1.5, threshold=0.4)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    new_img = nib.Nifti1Image(proc, affine=pred_img.affine, header=pred_img.header)
    fname = os.path.basename(pred_file)
    nib.save(new_img, os.path.join(dest_path, fname))
    print(f"Saved: {fname}")


def main():
    parser = argparse.ArgumentParser(description="Postprocessing for .nii.gz predictions")
    parser.add_argument("--predictions_path", required=True, help="Path to predictions folder")
    parser.add_argument("--dest_path", required=True, help="Path to save processed .nii.gz files")
    parser.add_argument("--mode", choices=["full", "strong", "gaussian", "expand_gaussian"], required=True,
                        help="Pipeline mode to run")
    args = parser.parse_args()

    os.makedirs(args.dest_path, exist_ok=True)

    pred_files = sorted([f for f in os.listdir(args.predictions_path) if f.endswith(".nii.gz")])

    for pred_fname in pred_files:
        pred_file = os.path.join(args.predictions_path, pred_fname)
        process_file(pred_file, args.dest_path, args.mode)


if __name__ == "__main__":
    main()