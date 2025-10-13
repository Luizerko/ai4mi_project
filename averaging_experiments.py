import argparse
import os
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Summarizing 3D metrics across runs.")
    parser.add_argument("--arch", default="segthor_augmented__run")
    parser.add_argument("--base_dir", default="/scratch-shared/scur1645/results")
    parser.add_argument("--metrics", nargs="+", default=["3d_dice", "3d_hd95"])
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    arch = args.arch
    base_dir = args.base_dir
    metrics = args.metrics
    runs = args.runs

    results = {}
    for metric in metrics:
        per_run = []
        for i in range(1, runs + 1):
            # Match your actual folder structure:
            metric_path = os.path.join(
                base_dir,
                f"{arch}_{i}",
                "best_epoch",
                "val_pred_stitched",
                f"{metric}.npz"
            )

            if not os.path.exists(metric_path):
                print(f"[Warning] Missing file: {metric_path}")
                continue

            # npz can contain multiple arrays; assume main data is under 'arr_0'
            arr = np.load(metric_path)
            key = list(arr.keys())[0]
            data = np.asarray(arr[key]).squeeze()
            per_run.append(data)

        if not per_run:
            print(f"[Error] No data found for metric {metric}, skipping.")
            continue

        stacked = np.stack(per_run, axis=0)
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0, ddof=1)

        results[metric] = {"mean": mean, "std": std}

        # Save summary CSV
        df = pd.DataFrame({
            "class_id": np.arange(len(mean)),
            f"{metric}_mean": mean,
            f"{metric}_std": std,
        })
        out_csv = os.path.join(base_dir, f"{arch}_{metric}_summary.csv")
        df.to_csv(out_csv, index=False)
        print(f"Saved summary to {out_csv}")

if __name__ == "__main__":
    main()
