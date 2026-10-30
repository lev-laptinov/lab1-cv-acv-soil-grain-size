"""Kaggle GPU entry point: the local repository and cached tiles from the soilgsd-bundle dataset.

Runs the experiment configs listed in CONFIGS: configs/exp/* with scripts/run_finetune.py, part2_vit/configs/*
with `python -m part2_vit.train`. outputs/ and reports/results.csv end up in /kaggle/working/repo and are
downloaded with `kaggle kernels output`. W&B logs online if the kernel has a WANDB_API_KEY secret, else offline.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

CONFIGS = ["configs/exp/C1_resnet18_ft.yaml"]  # set by kaggle/push.sh

# Located by content, not by mount path: Kaggle's input layout differs between competitions and datasets.
INPUT = Path("/kaggle/input")
BUNDLE = next(INPUT.rglob("repo/data/soil_boxes.csv")).parents[2]
COMP = next(INPUT.rglob("Training_labels_updated.csv")).parent
REPO = Path("/kaggle/working/repo")

shutil.copytree(BUNDLE / "repo", REPO, dirs_exist_ok=True)
if not (REPO / "data/raw").exists():
    (REPO / "data/raw").symlink_to(COMP)
# The tile directory stays writable (test tiles are cut here on first use); the cached files are linked in.
(REPO / "data/cache/tiles").mkdir(parents=True, exist_ok=True)
for f in (BUNDLE / "tiles").glob("*.npz"):
    if not (REPO / "data/cache/tiles" / f.name).exists():
        (REPO / "data/cache/tiles" / f.name).symlink_to(f)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm==1.0.30", "lightning==2.6.6", "wandb==0.31.0"],
               check=True)
# Keep Kaggle's environment (CUDA library paths); only add our packages to the import path.
env = {**os.environ, "PYTHONPATH": f"{REPO}:{REPO / 'src'}"}
try:
    from kaggle_secrets import UserSecretsClient

    env["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
except Exception as e:  # no secret attached to the kernel: W&B runs offline into outputs/wandb/
    print(f"W&B offline: {e}")
subprocess.run(["nvidia-smi"], check=False)
subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__, torch.cuda.get_device_name())"],
               check=True, env=env)
# One process per config: a failing experiment does not stop the ones after it.
for config in CONFIGS:
    entry = ["-m", "part2_vit.train"] if config.startswith("part2_vit/") else ["scripts/run_finetune.py"]
    done = subprocess.run([sys.executable, *entry, *config.split()], cwd=REPO, env=env)
    print(f"{config}: exit code {done.returncode}", flush=True)
