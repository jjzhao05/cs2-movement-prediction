"""Run the whole pipeline: demos -> processed/ -> mirage.db -> baselines.

Steps, in order (each one is a script you can also run on its own):

    extract    extract_rars.py      .rar archives in ~/Downloads -> data/*.dem (opt-in, moves files)
    dataset    build_dataset.py     data/*.dem -> processed/*.npz + manifest.json
    sql        build_sql.py         processed/ -> mirage.db
    baselines  baselines.ipynb      run the baseline models and write their results into the notebook

Usage:
    python main.py                             # dataset, sql, baselines
    python main.py --extract                   # also extract new .rar archives first
    python main.py --only dataset sql          # just these steps
    python main.py --skip baselines            # everything except the models
    python main.py --overwrite --rebuild       # reparse every demo and rebuild the database
                                               # (needed after changing features.py)

The baselines step reuses the trained GRU in checkpoints/ when the training data
is unchanged; set RETRAIN_GRU = True in the notebook to train it again.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
STEPS = ["extract", "dataset", "sql", "baselines"]


def run(name, args):
    print(f"\n=== {name}: {' '.join(args)}", flush=True)
    start = time.perf_counter()
    result = subprocess.run([sys.executable, *args], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"=== {name} failed (exit code {result.returncode})")
    print(f"=== {name} done in {time.perf_counter() - start:.0f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extract", action="store_true", help="also run the extract step (moves .rar files out of ~/Downloads)")
    parser.add_argument("--only", nargs="+", choices=STEPS, help="run only these steps")
    parser.add_argument("--skip", nargs="+", choices=STEPS, default=[], help="skip these steps")
    parser.add_argument("--workers", type=int, default=6, help="demos parsed in parallel (dataset step)")
    parser.add_argument("--overwrite", action="store_true", help="reparse demos that already have an .npz (dataset step)")
    parser.add_argument("--rebuild", action="store_true", help="delete and rebuild mirage.db (sql step)")
    args = parser.parse_args()

    steps = args.only or [s for s in STEPS if s != "extract" or args.extract]
    steps = [s for s in STEPS if s in steps and s not in args.skip]
    print(f"Steps: {', '.join(steps) or 'none'}")

    start = time.perf_counter()
    if "extract" in steps:
        run("extract", ["extract_rars.py"])
    if "dataset" in steps:
        run("dataset", ["build_dataset.py", "--workers", str(args.workers)] + (["--overwrite"] if args.overwrite else []))
    if "sql" in steps:
        run("sql", ["build_sql.py"] + (["--rebuild"] if args.rebuild else []))
    if "baselines" in steps:
        run("baselines", ["run_notebook.py", "baselines.ipynb"])
    print(f"\nAll done in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
