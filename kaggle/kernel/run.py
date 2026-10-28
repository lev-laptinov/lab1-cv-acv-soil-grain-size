"""Kaggle GPU entry point: the local repository and cached tiles from the soilgsd-bundle dataset.

Runs the experiment configs listed in CONFIGS with scripts/run_finetune.py; outputs/ and
reports/results.csv end up in /kaggle/working/repo and are downloaded with `kaggle kernels output`.
"""

import shutil
import subprocess
import sys
from pathlib import Path

CONFIGS = ["configs/exp/C1_resnet18_ft.yaml"]  # set by kaggle/push.sh

# Located by content, not by mount path: Kaggle's input layout differs between competitions and datasets.
INPUT = Path("/kaggle/input")
BUNDLE = next(INPUT.rglob("train_t50_p4.5.npz")).parent.parent
COMP = next(INPUT.rglob("Training_labels_updated.csv")).parent
REPO = Path("/kaggle/working/repo")

shutil.copytree(BUNDLE / "repo", REPO, dirs_exist_ok=True)
(REPO / "data/cache").mkdir(parents=True, exist_ok=True)
for link, target in [(REPO / "data/raw", COMP), (REPO / "data/cache/tiles", BUNDLE / "tiles")]:
    if not link.exists():
        link.symlink_to(target)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm==1.0.30"], check=True)
env = {"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/local/bin:/usr/bin:/bin"}
subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__, torch.cuda.get_device_name())"],
               check=True, env=env)
# One process per config: a failing experiment does not stop the ones after it.
for config in CONFIGS:
    done = subprocess.run([sys.executable, "scripts/run_finetune.py", config], cwd=REPO, env=env)
    print(f"{config}: exit code {done.returncode}", flush=True)
