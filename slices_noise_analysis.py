import re
import math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image

def analyze_statistics(
    gt_root: Path,
    grp_regex: str,
    nonzero_threshold: int = 0,
    verbose: bool = False,
):
    """
    Compute cohort priors from GT PNGs grouped by patient.

    Returns:
      dict with keys: start_p10, end_p90, span_median, n_patients, n_with_gt
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

    rows: List[Tuple[str,int,int,int,int,float,float,float]] = []
    for pid, files in sorted(groups.items()):
        if not files:
            continue
        try:
            files_sorted = sorted(files, key=lambda f: int(idx_re.search(f.name).group(1)))
        except Exception:
            if verbose:
                print(f"[WARN] {pid}: index parse error; skipped.")
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

        if verbose and not contiguous:
            print(f"[INFO] {pid}: non 0..Z-1 indices; using sorted order.")

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

        if verbose:
            def pct(x): return "  nan" if math.isnan(x) else f"{100.0*x:6.2f}"
            print(f"{pid:<20} {Z:>4} {first_idx:>6} {last_idx:>6} {after_idx:>6} "
                  f"{pct(start_prop):>8} {pct(end_prop):>8} {pct(span_prop):>8}")

    start_vals = [r[5] for r in rows if not math.isnan(r[5])]
    end_vals   = [r[6] for r in rows if not math.isnan(r[6])]
    span_vals  = [r[7] for r in rows]

    return {
        "start_p10": float(np.percentile(start_vals, 10)) if start_vals else float("nan"),
        "end_p90": float(np.percentile(end_vals, 90)) if end_vals else float("nan"),
        "span_median": float(np.median(span_vals)) if span_vals else float("nan"),
        "n_patients": len(groups),
        "n_with_gt": sum(1 for r in rows if not math.isnan(r[5])),
    }
