"""
Analyze GT PNGs to find the occupied slice range per patient in the same order used by a stitch/build step.
Groups files by patient via a user-provided regex (group 1 = patient id), computes first/last slices with GT>threshold,
and reports per-patient proportions plus global summary statistics.
"""

import re
import math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image

def _fmt_pct(x: float) -> str:
    """Format a proportion in [0,1] as percentage with fixed width; return 'nan' when undefined."""
    return "  nan" if math.isnan(x) else f"{100.0*x:6.2f}"

def _print_stats(label: str, vals: List[float]):
    """Print count, mean, std, min, percentiles, and max for a list of proportions."""
    vals = np.asarray([v for v in vals if not math.isnan(v)], dtype=float)
    if vals.size == 0:
        print(f"{label}: no data.")
        return
    stats = {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p25": float(np.percentile(vals, 25)),
        "p50": float(np.percentile(vals, 50)),
        "p75": float(np.percentile(vals, 75)),
        "p90": float(np.percentile(vals, 90)),
        "max": float(np.max(vals)),
    }
    def pp(k): return _fmt_pct(stats[k])
    print(f"\n{label} (proportion in [0,1]):")
    print(f"  n={stats['n']}  mean={pp('mean')}  std={pp('std')}")
    print(f"  min={pp('min')}  p10={pp('p10')}  p25={pp('p25')}  median={pp('p50')}  p75={pp('p75')}  p90={pp('p90')}  max={pp('max')}")

def analyze_gt_pngs_stitch_style(
    gt_root: Path,
    grp_regex: str,
    nonzero_threshold: int = 0,
):
    """
    Scan GT PNGs under gt_root, group them by patient using grp_regex (group 1 is the patient id),
    sort slices by numeric index as in stitch/build, and compute first/last/after indices and proportions.
    """
    pat_pid = re.compile(grp_regex)
    idx_re  = re.compile(r"(\d+)\.(png|gif|tiff)$", re.IGNORECASE)

    groups: Dict[str, List[Path]] = {}
    for f in gt_root.rglob("*.png"):
        m = pat_pid.match(f.name)
        if not m:
            continue
        pid = m.group(1)
        groups.setdefault(pid, []).append(f)

    header = f"{'patient_id':<20} {'Z':>4} {'first':>6} {'last':>6} {'after':>6} {'start%':>8} {'end%':>8} {'span%':>8}"
    print(header)
    print("-" * len(header))

    rows: List[Tuple[str,int,int,int,int,float,float,float]] = []

    for pid, files in sorted(groups.items()):
        if not files:
            continue
        try:
            files_sorted = sorted(files, key=lambda f: int(idx_re.search(f.name).group(1)))
        except Exception as e:
            print(f"[WARN] {pid}: could not extract an index from some file ({e}); skipping patient.")
            continue

        Z = len(files_sorted)
        has_gt = np.zeros(Z, dtype=bool)

        contiguous = True
        for i, f in enumerate(files_sorted):
            znum = int(idx_re.search(f.name).group(1))
            if znum != i:
                contiguous = False
            gt = np.array(Image.open(f).convert("L"))
            has_gt[i] = (gt > nonzero_threshold).any()

        if not contiguous:
            print(f"[INFO] {pid}: file indices are not strict 0..Z-1; using sorted order as-is.")

        if not has_gt.any():
            first_idx = last_idx = -1
            after_idx = 0
            start_prop = float("nan")
            end_prop   = float("nan")
            span_prop  = 0.0
        else:
            first_idx = int(np.argmax(has_gt))
            last_idx  = int(Z - 1 - np.argmax(has_gt[::-1]))
            after_idx = min(last_idx + 1, Z)
            denom = max(Z - 1, 1)
            start_prop = first_idx / denom
            end_prop   = last_idx  / denom
            span_prop  = has_gt.sum() / Z

        rows.append((pid, Z, first_idx, last_idx, after_idx, start_prop, end_prop, span_prop))

        print(f"{pid:<20} {Z:>4} {first_idx:>6} {last_idx:>6} {after_idx:>6} "
              f"{_fmt_pct(start_prop):>8} {_fmt_pct(end_prop):>8} {_fmt_pct(span_prop):>8}")

    start_vals = [r[5] for r in rows if not math.isnan(r[5])]
    end_vals   = [r[6] for r in rows if not math.isnan(r[6])]
    span_vals  = [r[7] for r in rows]

    print("\nGlobal summary (proportions):")
    _print_stats("start_prop", start_vals)
    _print_stats("end_prop",   end_vals)
    _print_stats("span_prop",  span_vals)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Slice range with GT≠0 (PNG) with full statistics and stitch-like ordering.")
    ap.add_argument("--gt_folder", type=Path, required=True, help="Root folder containing GT PNGs (recursive search).")
    ap.add_argument("--grp_regex", type=str, required=True,
                    help='Regex to group by patient (group 1 = pid). Example: "(Patient_\\d\\d)_\\d\\d\\d\\d"')
    ap.add_argument("--nonzero_threshold", type=int, default=0, help="Pixels > threshold count as GT≠0.")
    args = ap.parse_args()

    analyze_gt_pngs_stitch_style(
        gt_root=args.gt_folder,
        grp_regex=args.grp_regex,
        nonzero_threshold=args.nonzero_threshold,
    )
