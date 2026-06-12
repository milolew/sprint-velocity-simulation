"""Run only the multi-sprint extended sweeps (Ext 1-4).

Kept separate from ``run_experiments.main()`` so it does NOT rerun (and
overwrite) the single-sprint Exp 1-3 that feed the existing single-sprint
study. Writes ``results/ext{1..4}_*.csv``; analyse with ``analyze_extended.py``.
"""

import os
import sys

from run_experiments import OUT_DIR, _run_extended_sweeps

LEVELS_RS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


if __name__ == "__main__":
    # Optional sweep names to run a subset, e.g. `python run_extended.py ext2 ext3`.
    names = sys.argv[1:] or None
    os.makedirs(OUT_DIR, exist_ok=True)
    _run_extended_sweeps(LEVELS_RS, names=names)
    print("extended sweeps done")
