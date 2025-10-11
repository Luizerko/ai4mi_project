
"""
Recover missing slices by filling them with blank 256x256 uint8 images.

Arguments:
  --dir         : directory with FILTERED slices (recursive)
  --base_dir    : directory with COMPLETE slices (recursive)
  --dir_recover : flat output directory with the recovered slices

Filename pattern (PNG only):
  (Patient\\d\\d)_(\\d\\d\\d\\d).png

For each patient in --base_dir, copy present slices from --dir and
create blank 256x256 images for the missing indices into --dir_recover.
Files not matching the pattern are ignored.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
import shutil
import numpy as np
from PIL import Image

PID_RE = re.compile(r"^(Patient_\d{2})_(\d{4})\.png$", re.IGNORECASE)

def find_pngs(root: Path) -> List[Path]:
    """Return all PNG files under root, recursively."""
    return list(root.rglob("*.png"))

def index_by_pid_and_idx(files: List[Path]) -> Dict[str, Dict[int, Path]]:
    """
    Index files by patient and slice index following the expected pattern.
    Returns a mapping: { pid: { idx: Path } }.
    """
    table: Dict[str, Dict[int, Path]] = {}
    for f in files:
        m = PID_RE.match(f.name)
        if not m:
            continue
        pid = m.group(1)
        idx = int(m.group(2))
        table.setdefault(pid, {})
        table[pid].setdefault(idx, f)
    return table

def ensure_dir(p: Path):
    """Ensure a directory exists."""
    p.mkdir(parents=True, exist_ok=True)

def save_blank_png(out_path: Path, size: Tuple[int, int] = (256, 256)):
    """Write a uint8 0-valued grayscale PNG of the given size."""
    arr = np.zeros(size, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(out_path)

def main():
    ap = argparse.ArgumentParser(
        description="Fill missing slices (per patient/index) with 256x256 blank PNGs."
    )
    ap.add_argument("--dir", required=True, type=Path, help="Directory with FILTERED slices (recursive).")
    ap.add_argument("--base_dir", required=True, type=Path, help="Directory with COMPLETE slices (recursive).")
    ap.add_argument("--dir_recover", required=True, type=Path, help="Flat output directory for recovered PNGs.")
    ap.add_argument("--blank_size", type=int, nargs=2, metavar=("H", "W"), default=(256, 256),
                    help="Blank PNG size H W (default 256 256).")
    args = ap.parse_args()

    dir_filtered: Path = args.dir
    dir_base: Path = args.base_dir
    dir_out: Path = args.dir_recover
    blank_hw = (int(args.blank_size[0]), int(args.blank_size[1]))
    ensure_dir(dir_out)

    print("Scanning --base_dir ...")
    base_files = find_pngs(dir_base)
    base_index = index_by_pid_and_idx(base_files)

    print("Scanning --dir ...")
    filtered_files = find_pngs(dir_filtered)
    filtered_index = index_by_pid_and_idx(filtered_files)

    total_written = 0
    total_copied = 0
    total_blank = 0

    for pid in sorted(base_index.keys()):
        base_idxs: Set[int] = set(base_index[pid].keys())
        filt_idxs: Set[int] = set(filtered_index.get(pid, {}).keys())

        missing = sorted(base_idxs - filt_idxs)
        present = sorted(base_idxs & filt_idxs)

        for idx in present:
            src = filtered_index[pid][idx]
            dst = dir_out / f"{pid}_{idx:04d}.png"
            shutil.copy2(src, dst)
            total_copied += 1
            total_written += 1

        for idx in missing:
            dst = dir_out / f"{pid}_{idx:04d}.png"
            save_blank_png(dst, size=blank_hw)
            total_blank += 1
            total_written += 1

if __name__ == "__main__":
    main()
