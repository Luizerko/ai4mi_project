import os, glob, nibabel as nib, numpy as np
from pathlib import Path
import shutil

BASE_GT = "/gpfs/home5/sdijke/ai4mi_project/data/segthor_fixed/train"
ROOT    = "/gpfs/home5/sdijke/final_project/data/SEGTHOR_CLEAN"

for i in range(1,6):
    run_dir  = os.path.join(ROOT, f"training_{i}")
    pred_dir = os.path.join(run_dir, "stitches", "val", "pp_pred")   # or "pred"
    out_gt   = os.path.join(run_dir, "gt")
    os.makedirs(out_gt, exist_ok=True)

    for pred_path in sorted(glob.glob(os.path.join(pred_dir, "Patient_*.nii.gz"))):
        name = Path(pred_path).name
        pid  = name[:-7] if name.endswith(".nii.gz") else Path(pred_path).stem  # <-- fix
        src  = os.path.join(BASE_GT, pid, "GT_fixed.nii.gz")
        dst  = os.path.join(out_gt, f"{pid}.nii.gz")
        if os.path.exists(src):
            if not os.path.exists(dst):
                try:
                    os.link(src, dst)     # hardlink to save space
                except OSError:
                    shutil.copyfile(src, dst)
        else:
            print(f"[WARN] missing GT for {pid}: {src}")

print("Done assembling GT folders.")