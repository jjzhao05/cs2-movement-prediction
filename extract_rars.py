"""Extract .rar archives and keep only files with "mirage" in the name.

Each archive is extracted to a temporary directory; any file whose name
contains "mirage" (case-insensitive) is moved into the data folder, and
everything else is discarded.

Only archives modified/downloaded in the last 2 days are processed
(change with --days). Each archive that extracts successfully is then moved
into the rars/ folder (change with --archive-dir).

Usage:
    python extract_rars.py                     # recent .rar files in ~/Downloads
    python extract_rars.py path/to/archives    # all .rar files in a directory
    python extract_rars.py a.rar b.rar         # specific archives
    python extract_rars.py raw/ --out data     # choose the output folder
    python extract_rars.py --days 7            # look back 7 days instead
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

KEYWORD = "mirage"
DOWNLOADS = Path.home() / "Downloads"


def find_extractor():
    """Return a function that extracts (archive, dest) using an available tool."""
    # Windows 10/11 ships bsdtar, which can read RAR and RAR5.
    candidates = [
        ("tar", lambda a, d: ["tar", "-xf", str(a), "-C", str(d)]),
        ("7z", lambda a, d: ["7z", "x", "-y", f"-o{d}", str(a)]),
        ("unrar", lambda a, d: ["unrar", "x", "-o+", "-y", str(a), str(d) + "/"]),
    ]
    known_paths = {
        # Prefer Windows' bsdtar over e.g. Git Bash's GNU tar, which can't read RAR.
        "tar": [r"C:\Windows\System32\tar.exe"],
        "7z": [r"C:\Program Files\7-Zip\7z.exe"],
        "unrar": [r"C:\Program Files\WinRAR\UnRAR.exe"],
    }
    for name, build_cmd in candidates:
        exe = next((p for p in known_paths.get(name, []) if Path(p).exists()), None)
        if exe is None and name != "tar":
            exe = shutil.which(name)
        if exe is None:
            continue

        def extract(archive, dest, exe=exe, build_cmd=build_cmd):
            cmd = build_cmd(archive, dest)
            cmd[0] = exe
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

        return name, extract
    sys.exit("No extractor found. Install 7-Zip or WinRAR, or use Windows' built-in tar.")


def collect_archives(inputs, max_age_days):
    cutoff = time.time() - max_age_days * 86400
    archives = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            archives.extend(sorted(path.glob("*.rar")))
        elif path.suffix.lower() == ".rar" and path.exists():
            archives.append(path)
        else:
            print(f"Skipping {item}: not a .rar file or directory")
    # On Windows st_ctime is the creation time, i.e. when the file was downloaded.
    return [a for a in archives if max(a.stat().st_mtime, a.stat().st_ctime) >= cutoff]


def unique_path(path):
    """Avoid overwriting: foo.dem -> foo_1.dem, foo_2.dem, ..."""
    if not path.exists():
        return path
    i = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def format_duration(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {seconds:.1f}s" if minutes else f"{seconds:.1f}s"


def process_archive(archive, out_dir, extract):
    """Return the number of files kept, or None if extraction failed."""
    kept = 0
    with tempfile.TemporaryDirectory() as tmp:
        try:
            extract(archive, Path(tmp))
        except subprocess.CalledProcessError as e:
            print(f"  Failed to extract {archive.name} (exit code {e.returncode})")
            return None
        for file in Path(tmp).rglob("*"):
            if file.is_file() and KEYWORD in file.name.lower():
                dest = unique_path(out_dir / file.name)
                shutil.move(str(file), dest)
                kept += 1
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", default=[str(DOWNLOADS)], help=".rar files or directories containing them (default: ~/Downloads)")
    parser.add_argument("--out", default=str(Path(__file__).parent / "data"), help="output folder (default: data/ next to this script)")
    parser.add_argument("--archive-dir", default=str(Path(__file__).parent / "rars"), help="where processed .rar files are moved (default: rars/ next to this script)")
    parser.add_argument("--days", type=float, default=2, help="only process archives at most this many days old (default: 2)")
    args = parser.parse_args()

    archives = collect_archives(args.inputs, args.days)
    if not archives:
        sys.exit(f"No .rar files from the last {args.days:g} day(s) found.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = Path(args.archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)

    tool, extract = find_extractor()
    print(f"Using {tool} to extract {len(archives)} archive(s) into {out_dir}/")

    total = 0
    start = time.perf_counter()
    for archive in archives:
        print(f"{archive.name}...")
        archive_start = time.perf_counter()
        kept = process_archive(archive, out_dir, extract)
        if kept is None:
            continue  # leave failed archives where they are
        print(f"  kept {kept} file(s) in {format_duration(time.perf_counter() - archive_start)}")
        total += kept
        shutil.move(str(archive), unique_path(archive_dir / archive.name))
    print(f"Done. {total} file(s) with '{KEYWORD}' in the name saved to {out_dir}/")
    print(f"Processed archives moved to {archive_dir}/")
    print(f"Total time: {format_duration(time.perf_counter() - start)}")


if __name__ == "__main__":
    main()
