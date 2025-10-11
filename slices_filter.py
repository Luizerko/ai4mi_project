"""
Filter a mirrored dataset using ONLY cohort priors.
Keeps slices whose normalized index lies in [START_P10, END_P90] mapped to [0, Z-1].
Does not use intensities, HU, span median, or expansion.

Inputs:
  --in_root  : directory with {train,val[,test]}/{img,gt}
  --out_root : destination directory where the same structure is written, filtered
"""

import re
import math
import shutil
from pathlib import Path
from typing import Dict, List
import numpy as np
from slices_noise_analysis import analyze_statistics

IDX_RE = re.compile(r"(\d+)\.(png|gif|tiff)$", re.IGNORECASE)
PID_RE = re.compile(r"^(.+?)_(\d{4})\.(png|gif|tiff)$", re.IGNORECASE)

def sort_by_index(files: List[Path]) -> List[Path]:
    """Return files sorted by the integer slice index extracted from the filename."""
    return sorted(files, key=lambda f: int(IDX_RE.search(f.name).group(1)))

def safe_copy(src: Path, dst: Path):
    """Create parent folders if needed and copy file preserving metadata."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def group_split(in_root: Path, split: str) -> Dict[str, Dict[str, List[Path]]]:
    """
    Group per patient within <split>/{img,gt}.
    Returns a dict: { pid: {"img":[...], "gt":[...] (possibly empty)} }.
    """
    groups: Dict[str, Dict[str, List[Path]]] = {}
    img_dir = in_root / split / "img"
    gt_dir  = in_root / split / "gt"

    if not img_dir.exists():
        return {}

    for f in img_dir.glob("*.png"):
        m = PID_RE.match(f.name)
        if not m:
            continue
        pid = m.group(1)
        groups.setdefault(pid, {"img": [], "gt": []})
        groups[pid]["img"].append(f)

    if gt_dir.exists():
        for f in gt_dir.glob("*.png"):
            m = PID_RE.match(f.name)
            if not m:
                continue
            pid = m.group(1)
            if pid in groups:
                groups[pid]["gt"].append(f)

    for pid in list(groups.keys()):
        groups[pid]["img"] = sort_by_index(groups[pid]["img"])
        if groups[pid]["gt"]:
            groups[pid]["gt"] = sort_by_index(groups[pid]["gt"])
            assert len(groups[pid]["img"]) == len(groups[pid]["gt"]), (
                f"{split}/{pid}: #img ({len(groups[pid]['img'])}) != #gt ({len(groups[pid]['gt'])})"
            )
            for fi, fg in zip(groups[pid]["img"], groups[pid]["gt"]):
                ii = int(IDX_RE.search(fi.name).group(1))
                ig = int(IDX_RE.search(fg.name).group(1))
                assert ii == ig, f"{split}/{pid}: misaligned indices img({ii}) != gt({ig})"
    return groups

def priors_only_mask(Z: int, start_p10: float, end_p90: float) -> np.ndarray:
    """
    Build a boolean mask of length Z keeping only slices whose normalized index
    is in [START_P10, END_P90] when mapping indices to [0, Z-1].
    """
    denom = max(Z - 1, 1)
    z_min = int(math.floor(start_p10 * denom))
    z_max = int(math.ceil(end_p90  * denom))
    keep = np.zeros(Z, dtype=bool)
    keep[z_min:z_max + 1] = True
    return keep

def process_split(in_root: Path, out_root: Path, split: str, start_p10: float, end_p90: float):
    """
    Apply the priors-only mask to all patients in the given split and
    copy {img,gt} files for kept slices to the output structure.
    """
    groups = group_split(in_root, split)
    if not groups:
        print(f"[{split}] not found, skipping.")
        return

    print(f"[{split}] patients: {len(groups)}")
    out_img_dir = out_root / split / "img"
    out_gt_dir  = out_root / split / "gt"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    has_gt_any = any(len(g["gt"]) > 0 for g in groups.values())
    if has_gt_any:
        out_gt_dir.mkdir(parents=True, exist_ok=True)

    for pid, g in sorted(groups.items()):
        imgs = g["img"]
        Z = len(imgs)
        if Z == 0:
            continue

        keep = priors_only_mask(Z, start_p10, end_p90)
        kept_count = int(keep.sum())

        for i, fimg in enumerate(imgs):
            if not keep[i]:
                continue
            dst_img = out_img_dir / fimg.name
            safe_copy(fimg, dst_img)
            if g["gt"]:
                fgt = g["gt"][i]
                dst_gt = out_gt_dir / fgt.name
                safe_copy(fgt, dst_gt)

        first = int(np.argmax(keep))
        last  = int(Z - 1 - np.argmax(keep[::-1]))
        denom = max(Z - 1, 1)
        start_prop = first / denom
        end_prop   = last  / denom
        keep_prop  = kept_count / Z
        def pct(x): return f"{100.0*x:6.2f}"
        print(f"  {pid:<18} Z={Z:<4} kept={kept_count:<4} "
              f"first={first:<4} last={last:<4} "
              f"start%={pct(start_prop)} end%={pct(end_prop)} keep%={pct(keep_prop)}")

    print(f"[{split}] output → {out_root / split}")

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Filter {train,val[,test]}/{img,gt} by prior range learned from GT."
    )
    parser.add_argument("--in_root", type=Path, required=True, help="Input root with splits.")
    parser.add_argument("--out_root", type=Path, required=True, help="Output root.")
    parser.add_argument("--stats_gt_root", type=Path, required=True,
                        help="Root with GT PNGs to compute priors (recursive).")
    parser.add_argument("--grp_regex", type=str, required=True,
                        help='Regex to group by patient (group 1 = pid), e.g. "(Patient_\\d\\d)_\\d\\d\\d\\d".')
    parser.add_argument("--nonzero_threshold", type=int, default=0,
                        help="Pixels > threshold count as GT>0 for statistics.")
    parser.add_argument("--verbose_stats", action="store_true", help="Print per-patient stats while computing priors.")
    args = parser.parse_args()

    stats = analyze_statistics(
        gt_root=args.stats_gt_root,
        grp_regex=args.grp_regex,
        nonzero_threshold=args.nonzero_threshold,
        verbose=args.verbose_stats,
    )

    if any(math.isnan(stats[k]) for k in ("start_p10", "end_p90")):
        raise SystemExit("Invalid priors: could not compute START_P10/END_P90 from provided GT set.")

    start_p10 = stats["start_p10"]
    end_p90 = stats["end_p90"]
    print(f"Priors: START_P10={start_p10:.6f}  END_P90={end_p90:.6f}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        process_split(args.in_root, args.out_root, split)

if __name__ == "__main__":
    main()
