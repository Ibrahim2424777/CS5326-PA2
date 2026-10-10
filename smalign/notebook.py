"""Helpers for the Kaggle notebooks.

``!python script.py`` in a notebook does NOT stop a "Save & Run All" commit when
the script fails, so every stage is launched through :func:`sh`, which streams
the output into the cell and raises on a non-zero exit code.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import time
from pathlib import Path

QUIET_ENV = {
    "PYTHONUNBUFFERED": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_DATASETS_DISABLE_PROGRESS_BARS": "1",
    "TQDM_MININTERVAL": "60",
    "TOKENIZERS_PARALLELISM": "false",
}


def _env(extra: dict | None = None) -> dict:
    env = dict(os.environ)
    env.update(QUIET_ENV)
    env.update(extra or {})
    return env


def sh(cmd: str, log_path: str | None = None, check: bool = True, env: dict | None = None) -> int:
    """Run a shell command, stream its output, optionally tee to a log file, raise on failure."""

    print(f"$ {cmd}", flush=True)
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "a", encoding="utf-8") if log_path else None
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=_env(env))
    for line in proc.stdout:
        print(line, end="", flush=True)
        if log:
            log.write(line)
            log.flush()
    code = proc.wait()
    if log:
        log.close()
    if check and code != 0:
        raise RuntimeError(f"command failed with exit code {code}: {cmd}")
    return code


def start_bg(cmd: str, log_path: str, env: dict | None = None) -> subprocess.Popen:
    """Start a command in the background, writing all output to ``log_path``."""

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"$ (background) {cmd}\n  log -> {log_path}", flush=True)
    handle = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, shell=True, stdout=handle, stderr=subprocess.STDOUT, env=_env(env))
    proc._log_path = log_path  # type: ignore[attr-defined]
    return proc


def wait_bg(*procs: subprocess.Popen, poll_seconds: int = 300, tail_lines: int = 3) -> None:
    """Wait for background jobs, printing the tail of their logs every few minutes; raise if any failed."""

    pending = list(procs)
    while pending:
        for proc in list(pending):
            if proc.poll() is not None:
                pending.remove(proc)
        for proc in procs:
            lines = Path(proc._log_path).read_text(encoding="utf-8", errors="replace").splitlines()  # type: ignore[attr-defined]
            state = "running" if proc.poll() is None else f"exit {proc.returncode}"
            print(f"--- {proc._log_path} [{state}]", *lines[-tail_lines:], sep="\n", flush=True)  # type: ignore[attr-defined]
        if pending:
            time.sleep(poll_seconds)
    failed = [proc._log_path for proc in procs if proc.returncode != 0]  # type: ignore[attr-defined]
    if failed:
        raise RuntimeError(f"background job(s) failed, see logs: {failed}")


def find_input(relative: str, root: str = "/kaggle/input") -> str:
    """Locate ``relative`` (e.g. ``outputs/sft/best/config.json``) inside any attached Kaggle input."""

    hits = sorted(glob.glob(f"{root}/**/{relative}", recursive=True))
    if not hits:
        raise FileNotFoundError(
            f"could not find '{relative}' under {root}. Attach the previous notebook's output "
            f"(Add Input -> Your Work -> Notebooks) and make sure that version finished."
        )
    if len(hits) > 1:
        print(f"[find_input] several matches for {relative}, using the first:\n  " + "\n  ".join(hits))
    print(f"[find_input] {relative} -> {hits[0]}")
    return hits[0]


def copy_run_metadata(src_run_dir: str, dst_run_dir: str) -> None:
    """Copy a run's small files (metrics, configs, summaries) without the multi-GB checkpoints."""

    src, dst = Path(src_run_dir), Path(dst_run_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)
    for sub in ("best", "final"):
        state = src / sub / "checkpoint_state.json"
        if state.exists():
            (dst / sub).mkdir(exist_ok=True)
            shutil.copy2(state, dst / sub / "checkpoint_state.json")


def zip_dir(src: str, zip_path: str, exclude_dirs: tuple[str, ...] = ("best", "final", "adapter", "trainer")) -> str:
    """Zip ``src`` skipping checkpoint folders; returns the zip path."""

    import zipfile

    src_path = Path(src)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in src_path.rglob("*"):
            if path.is_file() and not any(part in exclude_dirs for part in path.relative_to(src_path).parts):
                archive.write(path, path.relative_to(src_path.parent))
    size = Path(zip_path).stat().st_size / 2**20
    print(f"[zip] {zip_path} ({size:.1f} MiB)")
    return zip_path


def disk_report(path: str = "/kaggle/working") -> None:
    total, used, free = shutil.disk_usage(path)
    print(f"[disk] {path}: used {used / 2**30:.1f} GiB, free {free / 2**30:.1f} GiB")
