import argparse
import os
import re
import shutil

import nibabel as nib
import numpy as np

from collections import defaultdict
from pathlib import Path
from PIL import Image
from skimage.transform import resize
from tqdm import tqdm

###########################################################################################################################
##                                                 I RUN IT LOCALLY WITH                                                 ##
##  python stitch.py --data_folder data/SEGTHOR_CLEAN/training_output/best_epoch/val/ --dest_folder data/stitches/val/ --num_classes 5 --grp_regex "(Patient_\d\d)_\d\d\d\d" --source_scan_pattern "data/segthor_train/train/{id_}/GT.nii.gz"  ##
##  SO IT SHOULD BE HOW THE ASSIGNMENT REQUIRES US TO DO IT, BUT NOTICE THAT I HAVE GT_FIXED FILES INSIDE SEGTHOR_TRAIN  ##
#########################################################################################################

# Function that transforms 2D slices (either from predictions or ground-truth) to 3D object
def main(args: argparse.Namespace):
    data_folder = Path(args.data_folder)
    dest_folder = Path(args.dest_folder)
    patient_pattern = re.compile(args.grp_regex)
    source_metadata_pattern = args.source_scan_pattern

    # Getting matching files and splitting them into different patients
    patients_dict = defaultdict(list)
    for file in sorted(os.listdir(data_folder)):
        patient = patient_pattern.match(file)
        if patient:
            patients_dict[patient.group(1)].append(data_folder / file)

    # Building 3D objects from slices
    for patient_id, slices in tqdm(patients_dict.items()):
        reconstruct_3d = []
        for slice in slices:
            slice = Image.open(slice).convert("L")
            reconstruct_3d.append(np.array(slice, dtype=np.uint8))
        reconstruct_3d = np.stack(reconstruct_3d, axis=0).transpose(1, 2, 0)
        reconstruct_3d = reconstruct_3d // 63

        if not args.test:
            # Getting original metadata
            gt_path = Path(source_metadata_pattern.replace("{id_}", patient_id).replace("GT", patient_id))
            gt_volume = nib.load(gt_path)

            # Projecting prediction into original space and saving stitched volume
            reconstruct_3d = resize(reconstruct_3d, gt_volume.shape, anti_aliasing=True, preserve_range=True, order=0)
            reconstruct_3d = nib.Nifti1Image(reconstruct_3d, gt_volume.affine, gt_volume.header)
        else:
            img_path = Path(source_metadata_pattern.replace("{id_}", patient_id))
            img_volume = nib.load(img_path)
            reconstruct_3d = nib.Nifti1Image(reconstruct_3d, img_volume.affine)
        
        # Saving predictions and GT on proper folders
        nib.save(reconstruct_3d, (dest_folder / "pred" / f"{patient_id}.nii.gz"))
        if not args.test:
          shutil.copy(Path(str(gt_path).replace("GT", "GT_fixed")), dest_folder / "gt" / f"{patient_id}.nii.gz")
    

# Function to get the arguments passed on the command line
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True)
    parser.add_argument('--dest_folder', type=str, required=True)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--grp_regex', type=str, required=True)
    parser.add_argument('--source_scan_pattern', type=str, required=True)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    return args

# Directly calling the main when running the script
if __name__ == "__main__":
    main(get_args())