#!/bin/bash
# Train CIFAR-100 ResNet18-32x32 variants under OpenOOD's standard 100-epoch
# baseline recipe (SGD, lr 0.1, cosine, base_preprocessor), one seed each.
#
#   (a) maxsep_resnet18_32x32 (fixed simplex-ETF classifier) at wd 5e-4/1e-3/2e-3
#   (b) standard  resnet18_32x32                              at wd 1e-3/2e-3
#       (the wd 5e-4 standard baseline already exists on the OpenOOD leaderboard)
#
# Output dirs are made distinct via --mark (folded into exp_name by
# baseline.yml: <dataset>_<network>_<trainer>_e<ep>_lr<lr>_<mark>/s<seed>).
#
# Usage: bash run_maxsep.sh
set -euo pipefail

SEED=0

run () {
  local network_cfg=$1
  local wd=$2
  local mark=$3
  echo "=== training ${network_cfg} wd=${wd} mark=${mark} ==="
  PYTHONPATH='.':${PYTHONPATH:-} python main.py \
    --config configs/datasets/cifar100/cifar100.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/networks/${network_cfg}.yml \
    configs/pipelines/train/baseline.yml \
    --optimizer.weight_decay "${wd}" \
    --mark "${mark}" \
    --seed "${SEED}"
}

# (a) max-separation ETF classifier
run maxsep_resnet18_32x32 0.0005 maxsep_wd5e-4
run maxsep_resnet18_32x32 0.001  maxsep_wd1e-3
run maxsep_resnet18_32x32 0.002  maxsep_wd2e-3

# (b) standard learned classifier (wd 5e-4 baseline already on leaderboard)
run resnet18_32x32 0.001 baseline_wd1e-3
run resnet18_32x32 0.002 baseline_wd2e-3
