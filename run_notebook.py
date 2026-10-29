"""Execute a notebook in place (what `jupyter nbconvert --execute --inplace` does), without its warnings.

    python run_notebook.py baselines.ipynb

- Cells without an id (e.g. added by a script) get one before validation.
- On Windows the kernel's messaging (zmq) needs the selector event loop, not the default
  proactor loop, so that is set first.
The "Kernel is running over TCP without encryption" warning stays: Windows has no IPC
transport, and the kernel only listens on 127.0.0.1.
The kernel runs from the notebook's folder with no time limit; outputs are saved even if a cell
fails, and the exit code is 1 then.
"""

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebook")
    args = parser.parse_args()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    path = Path(args.notebook)
    raw = json.loads(path.read_text(encoding="utf-8"))
    for cell in raw["cells"]:  # cell ids exist from nbformat 4.5 on
        cell.setdefault("id", uuid.uuid4().hex[:8])
    raw["nbformat_minor"] = max(raw.get("nbformat_minor", 0), 5)
    nb = nbformat.reads(json.dumps(raw), as_version=4)

    client = NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(path.parent.resolve())}})
    failed = None
    try:
        client.execute()
    except CellExecutionError as e:
        failed = e
    nbformat.write(nb, path)
    if failed:
        print(failed, file=sys.stderr)
    # Everything is saved; skip interpreter teardown, where libzmq on Windows can abort with
    # "Assertion failed: Connection reset by peer" while closing the (already shut down) kernel's sockets.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1 if failed else 0)


if __name__ == "__main__":
    main()
