import subprocess
import sys

import pandas as pd

# Each process writes 20 rows of its own; without the lock, concurrent read-modify-writes lose some of them.
WRITER = """
import sys
from pathlib import Path
import soilgsd.results as r
r.RESULTS = Path(sys.argv[1])
for i in range(20):
    r.save_result({"name": f"{sys.argv[2]}_{i}", "emd": float(i)})
"""


def test_parallel_runs_keep_every_row_of_results_csv(tmp_path):
    path = tmp_path / "results.csv"
    procs = [subprocess.Popen([sys.executable, "-c", WRITER, str(path), f"run{k}"]) for k in range(4)]
    assert all(p.wait() == 0 for p in procs)
    names = pd.read_csv(path)["name"]
    assert len(names) == 80 and names.is_unique
