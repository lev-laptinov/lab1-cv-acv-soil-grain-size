from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "configs/base.yaml") -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(rel: str | Path) -> Path:
    """Paths in configs are relative to the repository root."""
    rel = Path(rel)
    return rel if rel.is_absolute() else ROOT / rel
