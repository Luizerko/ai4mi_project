#!/usr/bin/env bash

ARCH="${1:-baseline}"
BASE_DIR="data/SEGTHOR_CLEAN"
RUNS=5

# Training runs
echo "[1/6] Training ${RUNS} runs for arch=${ARCH}"
for i in $(seq 1 "$RUNS"); do
  OUT_DIR="${BASE_DIR}/${ARCH}_training_output_${i}"
  echo "Training run ${i}"
  mkdir -p "${OUT_DIR}"
  python -O main.py \
    --dataset SEGTHOR_CLEAN \
    --mode full \
    --epoch 25 \
    --dest "${OUT_DIR}/" \
    --gpu
done

# Plotting per run
echo "[2/6] Generating plots for each run for arch=${ARCH}"
for i in $(seq 1 "$RUNS"); do
  OUT_DIR="${BASE_DIR}/${ARCH}_training_output_${i}"
  echo "Plots for run ${i}"

  python plot.py \
    --metric_file "${OUT_DIR}/loss_tra.npy" \
    --dest "${OUT_DIR}/loss_tra.png" \
    --headless

  python plot.py \
    --metric_file "${OUT_DIR}/loss_val.npy" \
    --dest "${OUT_DIR}/loss_val.png" \
    --headless

  python plot.py \
    --metric_file "${OUT_DIR}/dice_tra.npy" \
    --dest "${OUT_DIR}/dice_tra.png" \
    --headless

  python plot.py \
    --metric_file "${OUT_DIR}/dice_val.npy" \
    --dest "${OUT_DIR}/dice_val.png" \
    --headless
done

# Making test predictions
echo "[3/6] Making test predictions for arch=${ARCH}"
for i in $(seq 1 "$RUNS"); do
  OUT_DIR="${BASE_DIR}/${ARCH}_training_output_${i}/best_epoch/test/"
  mkdir -p "${OUT_DIR}"
  echo "Testing run ${i}"
  python inference.py --dataset data/SEGTHOR_TRAIN_TEST/ --split test --outdir "${OUT_DIR}" --bweights "${BASE_DIR}/${ARCH}_training_output_${i}/" --batch 8 --gpu
done

# Stitching per run
echo "[4/6] Stitching validation outputs for arch=${ARCH}"
for i in $(seq 1 "$RUNS"); do
  OUT_DIR="${BASE_DIR}/${ARCH}_training_output_${i}"
  mkdir -p "${OUT_DIR}/stitches/val/pred/"
  mkdir -p "${OUT_DIR}/stitches/val/gt/"
  mkdir -p "${OUT_DIR}/stitches/test/pred/"
  echo "Stitch for run ${i}"

  python stitch.py \
    --data_folder "${OUT_DIR}/best_epoch/val/" \
    --dest_folder "${OUT_DIR}/stitches/val/" \
    --num_classes 5 \
    --grp_regex '(Patient_\d\d)_\d\d\d\d' \
    --source_scan_pattern "data/segthor_train/train/{id_}/GT.nii.gz"
done

# Stitching tests too
echo "[5/7] Stitching test for arch=${ARCH}"
for i in $(seq 1 "$RUNS"); do
  OUT_DIR="${BASE_DIR}/${ARCH}_training_output_${i}"
  mkdir -p "${OUT_DIR}/stitches/test/pred/"
  echo "Stitch for run ${i}"

  python stitch.py \
    --data_folder "${OUT_DIR}/best_epoch/test/" \
    --dest_folder "${OUT_DIR}/stitches/test/" \
    --num_classes 5 \
    --grp_regex '(Patient_\d\d)_\d\d\d\d' \
    --source_scan_pattern "data/segthor_train/test/{id_}.nii.gz" \
    --test
done

# Computing metrics per run
echo "[6/7] Computing metrics for arch=${ARCH}"
cd distorch/
for i in $(seq 1 "$RUNS"); do
  OUT_DIR="${BASE_DIR}/${ARCH}_training_output_${i}"
  echo "Metrics for run ${i}"

  python compute_metrics.py \
    --ref_folder "../data/SEGTHOR_CLEAN/${ARCH}_training_output_${i}/stitches/val/gt/" \
    --pred_folder "../data/SEGTHOR_CLEAN/${ARCH}_training_output_${i}/stitches/val/pred/" \
    --ref_extension ".nii.gz" \
    --pred_extension ".nii.gz" \
    -K 5 \
    --metrics 3d_hd95 3d_dice \
    --background_class 0 \
    --save_folder "../data/SEGTHOR_CLEAN/${ARCH}_training_output_${i}/stitches/val/" \
    --overwrite
done

# Computing statistics for the metrics
echo "[7/7] Computing metrics average and standard deviation for arch=${ARCH}"
cd ../
python averaging_experiments.py --arch "${ARCH}"\
 --base_dir "${BASE_DIR}" \
 --metrics 3d_hd95 3d_dice