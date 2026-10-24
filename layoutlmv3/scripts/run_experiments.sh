#!/usr/bin/env bash
# Run baseline and Boundary Loss experiments across FUNSD & CORD (base & large)
# and compile an aggregated evaluation summary table, syncing results to Google Drive.
#
# Usage (from layoutlmv3/ with the layoutlmv3 conda env active):
#   DRIVE_MOUNT=/content/drive/MyDrive bash scripts/run_experiments.sh
#   DATASETS="funsd cord" SIZES="base large" VARIANTS="baseline boundary" bash scripts/run_experiments.sh
set -euo pipefail

DATASETS="${DATASETS:-funsd cord}"
SIZES="${SIZES:-base large}"
VARIANTS="${VARIANTS:-baseline boundary}"
SEED="${SEED:-42}"
OUT="${OUT:-$HOME/runs}"
BOUNDARY_LOSS_WEIGHT="${BOUNDARY_LOSS_WEIGHT:-0.2}"
BOUNDARY_POS_WEIGHT="${BOUNDARY_POS_WEIGHT:-5.0}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$OUT"

for dataset in $DATASETS; do
  for size in $SIZES; do
    for variant in $VARIANTS; do
      echo "================================================================================"
      echo "Running: dataset=$dataset | size=$size | variant=$variant | seed=$SEED"
      echo "================================================================================"
      if [ "$variant" = "boundary" ]; then
        DATASET="$dataset" SIZE="$size" SEED="$SEED" OUT="$OUT" \
          USE_BOUNDARY_LOSS=1 \
          BOUNDARY_LOSS_WEIGHT="$BOUNDARY_LOSS_WEIGHT" \
          BOUNDARY_POS_WEIGHT="$BOUNDARY_POS_WEIGHT" \
          bash "$SCRIPT_DIR/run_baseline.sh" train
      else
        DATASET="$dataset" SIZE="$size" SEED="$SEED" OUT="$OUT" \
          USE_BOUNDARY_LOSS=0 \
          bash "$SCRIPT_DIR/run_baseline.sh" train
      fi
    done
  done
done

# Aggregate evaluation metrics across all completed runs into CSV and Markdown tables
SUMMARY_DIR="$OUT/summary"
mkdir -p "$SUMMARY_DIR"

python - "$OUT" "$SUMMARY_DIR" <<'EOF'
import csv
import glob
import json
import os
import sys

out_dir, summary_dir = sys.argv[1], sys.argv[2]
rows = []

for eval_path in sorted(glob.glob(os.path.join(out_dir, "*", "eval_results.json"))):
    run_dir = os.path.dirname(eval_path)
    run_name = os.path.basename(run_dir)
    if run_name == "summary":
        continue
    with open(eval_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    info = {}
    info_path = os.path.join(run_dir, "run_info.txt")
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            for line in f:
                for token in line.strip().split():
                    if "=" in token:
                        k, v = token.split("=", 1)
                        info[k] = v
    rows.append({
        "run_name": run_name,
        "dataset": info.get("dataset", ""),
        "size": info.get("size", ""),
        "use_boundary_loss": info.get("use_boundary_loss", "0"),
        "boundary_loss_weight": info.get("boundary_loss_weight", "0.0"),
        "boundary_pos_weight": info.get("boundary_pos_weight", "1.0"),
        "precision": metrics.get("eval_precision", 0.0),
        "recall": metrics.get("eval_recall", 0.0),
        "f1": metrics.get("eval_f1", 0.0),
        "accuracy": metrics.get("eval_accuracy", 0.0),
        "loss": metrics.get("eval_loss", 0.0),
    })

csv_path = os.path.join(summary_dir, "summary_metrics.csv")
md_path = os.path.join(summary_dir, "summary_metrics.md")

fieldnames = [
    "run_name", "dataset", "size", "use_boundary_loss",
    "boundary_loss_weight", "boundary_pos_weight",
    "precision", "recall", "f1", "accuracy", "loss"
]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

with open(md_path, "w", encoding="utf-8") as f:
    f.write("# LayoutLMv3 Boundary Loss Evaluation Summary\n\n")
    f.write("| Run | Dataset | Size | Boundary Loss | Weight (λ) | Pos Weight | Precision | Recall | F1 | Accuracy |\n")
    f.write("|---|---|---|---|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        f.write(
            f"| `{r['run_name']}` | {r['dataset']} | {r['size']} | {r['use_boundary_loss']} | "
            f"{r['boundary_loss_weight']} | {r['boundary_pos_weight']} | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | **{r['f1']:.4f}** | {r['accuracy']:.4f} |\n"
        )

print(open(md_path, "r", encoding="utf-8").read())
EOF

if [ -n "${DRIVE_REMOTE:-}" ] || [ -n "${DRIVE_MOUNT:-}" ]; then
  bash "$SCRIPT_DIR/sync_to_drive.sh" "$SUMMARY_DIR"
fi
