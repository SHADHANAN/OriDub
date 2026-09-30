# -*- coding: utf-8 -*-
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
Resumable PyTorch 2.6 (cu124) wheel downloader + installer
===========================================================
pip cannot resume large downloads and drops the connection for the ~2.5 GB
torch wheel.  This script downloads in 8 MB chunks with automatic resume
on every timeout/disconnect, then installs with pip once complete.

Run:  python download_torch.py
"""
import os
import subprocess
import time
import requests
from pathlib import Path

# ── Wheel details ────────────────────────────────────────────────────────────
# torch 2.6.0 + CUDA 12.4, Python 3.10, Windows 64-bit
TORCH_WHL_URL = (
    "https://download.pytorch.org/whl/cu124/"
    "torch-2.6.0%2Bcu124-cp310-cp310-win_amd64.whl"
)
TORCH_WHL_NAME   = "torch-2.6.0+cu124-cp310-cp310-win_amd64.whl"
TORCH_WHL_SIZE   = 2_532_341_760   # ~2.36 GB (approximate; script adapts)

TORCHVISION_URL  = (
    "https://download.pytorch.org/whl/cu124/"
    "torchvision-0.21.0%2Bcu124-cp310-cp310-win_amd64.whl"
)
TORCHVISION_NAME = "torchvision-0.21.0+cu124-cp310-cp310-win_amd64.whl"

TORCHAUDIO_URL   = (
    "https://download.pytorch.org/whl/cu124/"
    "torchaudio-2.6.0%2Bcu124-cp310-cp310-win_amd64.whl"
)
TORCHAUDIO_NAME  = "torchaudio-2.6.0+cu124-cp310-cp310-win_amd64.whl"

DOWNLOAD_DIR = Path(__file__).parent / "torch_wheels"
CHUNK_SIZE   = 8 * 1024 * 1024   # 8 MB
READ_TIMEOUT = 120                # seconds per chunk read

WHEELS = [
    (TORCH_WHL_URL,      TORCH_WHL_NAME,     TORCH_WHL_SIZE),
    (TORCHVISION_URL,    TORCHVISION_NAME,    None),
    (TORCHAUDIO_URL,     TORCHAUDIO_NAME,     None),
]


def sizeof_fmt(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024.0:
            return f"{num:6.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def get_remote_size(url: str) -> int | None:
    """HEAD request to get Content-Length."""
    try:
        r = requests.head(url, timeout=15, allow_redirects=True)
        cl = r.headers.get("Content-Length")
        return int(cl) if cl else None
    except Exception:
        return None


def download_with_resume(url: str, dest: Path, expected_size: int | None) -> None:
    """Download *url* to *dest*, resuming any partial download."""
    tmp = dest.with_suffix(dest.suffix + ".part")

    resume_from = tmp.stat().st_size if tmp.exists() else 0

    # Resolve actual size via HEAD if not given
    if expected_size is None:
        expected_size = get_remote_size(url) or 0

    if dest.exists() and (expected_size == 0 or dest.stat().st_size >= expected_size * 0.99):
        print(f"[SKIP] {dest.name} already downloaded.")
        return

    if resume_from:
        print(f"[->>] Resuming {dest.name} from {sizeof_fmt(resume_from)} ...")
    else:
        size_str = sizeof_fmt(expected_size) if expected_size else "unknown size"
        print(f"[DL]  Downloading {dest.name} ({size_str}) ...")

    attempt = 0
    while True:
        attempt += 1
        headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
        try:
            resp = requests.get(url, headers=headers, stream=True,
                                timeout=(15, READ_TIMEOUT))
            if resp.status_code == 416:
                # Range not satisfiable → already complete
                print(f"[OK]  {dest.name} already fully downloaded (416).")
                tmp.rename(dest)
                return
            if resp.status_code not in (200, 206):
                print(f"[!]   HTTP {resp.status_code} on attempt {attempt}. Retrying in 10s ...")
                time.sleep(10)
                continue

            mode = "ab" if (resume_from and resp.status_code == 206) else "wb"
            if mode == "wb":
                resume_from = 0

            written = resume_from
            t0 = time.time()
            with tmp.open(mode) as fh:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    written += len(chunk)
                    elapsed = time.time() - t0 or 0.001
                    speed   = (written - resume_from) / elapsed
                    pct     = (written / expected_size * 100) if expected_size else 0
                    bar_w   = 38
                    filled  = int(bar_w * written / expected_size) if expected_size else 0
                    bar     = "#" * filled + "." * (bar_w - filled)
                    print(
                        f"\r  [{bar}] {pct:5.1f}%  "
                        f"{sizeof_fmt(written)}"
                        + (f" / {sizeof_fmt(expected_size)}" if expected_size else "")
                        + f"  @ {sizeof_fmt(speed)}/s   ",
                        end="", flush=True,
                    )
            print()  # newline
            break

        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError) as exc:
            resume_from = tmp.stat().st_size if tmp.exists() else 0
            print(f"\n[!]   Attempt {attempt} cut short ({exc.__class__.__name__}). "
                  f"Saved {sizeof_fmt(resume_from)}. Retrying in 5s ...")
            headers = {"Range": f"bytes={resume_from}-"}
            time.sleep(5)

    tmp.rename(dest)
    print(f"[OK]  Saved: {dest}")


def install_wheel(whl: Path) -> None:
    print(f"\n[PIP] Installing {whl.name} ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", str(whl), "--no-deps", "-q"],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"[!]  pip install failed for {whl.name}")
        sys.exit(1)
    print(f"[OK]  {whl.name} installed.")


def main() -> None:
    DOWNLOAD_DIR.mkdir(exist_ok=True)

    # ── Lock file – prevents two instances corrupting the same .part file ─
    lock = DOWNLOAD_DIR / ".lock"
    if lock.exists():
        print("[!] Another instance is already running (lock file exists).")
        print(f"    If this is stale, delete: {lock}")
        sys.exit(1)
    lock.write_text(str(os.getpid()))
    try:
        _run_downloads()
    finally:
        lock.unlink(missing_ok=True)


def _run_downloads() -> None:
    # ── Download all three wheels ─────────────────────────────────────────
    for url, name, size in WHEELS:
        dest = DOWNLOAD_DIR / name
        download_with_resume(url, dest, size)

    # ── Install torch first (torchvision/torchaudio depend on it) ─────────
    for _, name, _ in WHEELS:
        install_wheel(DOWNLOAD_DIR / name)

    print("\n[OK]  All done! Verifying ...")
    result = subprocess.run(
        [sys.executable, "-c",
         "import torch; print('torch', torch.__version__); "
         "print('CUDA available:', torch.cuda.is_available()); "
         "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"],
        capture_output=False,
    )
    if result.returncode == 0:
        print("\n[OK]  PyTorch 2.6 + CUDA 12.4 ready. Restart the server.")
    else:
        print("\n[!]   Verification failed — check output above.")


if __name__ == "__main__":
    main()
