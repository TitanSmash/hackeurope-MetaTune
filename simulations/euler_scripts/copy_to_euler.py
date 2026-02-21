from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List


def _run(cmd: List[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rsync project to Euler and normalize line endings.")
    parser.add_argument("--local-dir", default=".", help="Local repo path")
    parser.add_argument("--remote-user", required=True, help="Euler username")
    parser.add_argument("--remote-host", default="euler.ethz.ch", help="Euler host")
    parser.add_argument("--remote-dir", required=True, help="Remote destination directory")
    parser.add_argument("--exclude-output", action="store_true", help="Exclude outputs/ directory")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    local_dir = Path(args.local_dir).resolve()
    source = str(local_dir) + "/"
    remote = f"{args.remote_user}@{args.remote_host}:{args.remote_dir}"

    rsync_cmd = [
        "rsync",
        "-av",
        "--exclude=.git/",
        "--exclude=.github/",
    ]
    if args.exclude_output:
        rsync_cmd.append("--exclude=outputs/")
    gitignore = local_dir / ".gitignore"
    if gitignore.exists():
        rsync_cmd.append(f"--exclude-from={gitignore}")
    rsync_cmd.extend([source, remote])
    _run(rsync_cmd)

    normalize_cmd = [
        "ssh",
        f"{args.remote_user}@{args.remote_host}",
        (
            "find "
            f"{args.remote_dir} "
            "-type d -name outputs -prune -o "
            "-type f \\( -name '*.sh' -o -name '*.sbatch' -o -name '*.py' -o -name '*.yaml' \\) "
            "-exec dos2unix {} +"
        ),
    ]
    _run(normalize_cmd)


if __name__ == "__main__":
    main()

