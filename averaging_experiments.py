import argparse
import os
import sys
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Summarizing 3D metrics across runs.")
    parser.add_argument("--arch", default="baseline")
    parser.add_argument("--base_dir", default="data/SEGTHOR_CLEAN")
    parser.add_argument("--metrics", nargs="+", default=["3d_dice", "3d_hd95"])
    args = parser.parse_args()

    arch = args.arch
    base_dir = args.base_dir
    metrics = args.metrics
    runs = 5

    # Collecting results for each metric
    results = {}
    for metric in metrics:
        per_run = []
        for i in range(1, runs + 1):
            # Expecting files at data/SEGTHOR_CLEAN/<arch>_training_output_i/stitches/val/<metric>.npy
            metric_path = os.path.join(
                base_dir, f"{arch}_training_output_{i}", "stitches", "val", f"{metric}.npy"
            )
            arr = np.load(metric_path)
            arr = np.asarray(arr).squeeze()
            per_run.append(arr)

        stacked = np.concatenate(per_run, axis=0)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0, ddof=1)

        # Saving results
        results[metric] = {
            "mean": mean,
            "std": std
        }
        df = pd.DataFrame({
            "class_id": np.arange(5),
            f"{metric}_mean": mean,
            f"{metric}_std": std,
        })
        out_csv = os.path.join(base_dir, f"{arch}_{metric}_summary.csv")
        df.to_csv(out_csv, index=False)

if __name__ == "__main__":
    main()