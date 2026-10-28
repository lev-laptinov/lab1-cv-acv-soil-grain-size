#!/usr/bin/env bash
# Push the GPU kernel that fine-tunes the given configs, e.g.
#   bash kaggle/push.sh configs/exp/C1_resnet18_ft.yaml configs/exp/C2_resnet18_bn_train.yaml
# then, when it has finished:  uvx kaggle kernels output <user>/soilgsd-finetune -p build/kaggle_out
set -euo pipefail
cd "$(dirname "$0")/.."
USER=$(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
OUT=build/kaggle_kernel
rm -rf "$OUT" && mkdir -p "$OUT"
LIST=$(printf '"%s", ' "$@")
sed "s|^CONFIGS = .*|CONFIGS = [${LIST%, }]|" kaggle/kernel/run.py > "$OUT/run.py"
cat > "$OUT/kernel-metadata.json" <<JSON
{"id": "$USER/soilgsd-finetune", "title": "soilgsd-finetune", "code_file": "run.py", "language": "python",
 "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": true,
 "dataset_sources": ["$USER/soilgsd-bundle"], "competition_sources": ["soil-grain-size-from-photos"]}
JSON
uvx kaggle kernels push -p "$OUT"
