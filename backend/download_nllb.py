# -*- coding: utf-8 -*-
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
Resumable downloader for facebook/nllb-200-distilled-600M pytorch_model.bin
===========================================================================
Uses HTTP Range requests to resume interrupted downloads automatically.
Saves directly into the HuggingFace cache blob store so transformers
picks it up without any re-downloading.
"""
import glob
import os
import shutil
import sys
import time
import requests
from pathlib import Path

# ── Target paths ────────────────────────────────────────────────────────────
CACHE_DIR   = Path.home() / ".cache" / "huggingface" / "hub"
MODEL_DIR   = CACHE_DIR / "models--facebook--nllb-200-distilled-600M"
BLOBS_DIR   = MODEL_DIR / "blobs"
SNAPSHOT    = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
SNAP_DIR    = MODEL_DIR / "snapshots" / SNAPSHOT

BLOB_SHA    = "c266c2cfd19758b6d09c1fc31ecdf1e485509035f6b51dfe84f1ada83eefcc42"
BLOB_FILE   = BLOBS_DIR / BLOB_SHA
TMP_FILE    = BLOBS_DIR / f"{BLOB_SHA}.incomplete"
SYMLINK     = SNAP_DIR / "pytorch_model.bin"

URL = (
    "https://huggingface.co/facebook/nllb-200-distilled-600M"
    "/resolve/main/pytorch_model.bin"
)

CHUNK_SIZE   = 8 * 1024 * 1024   # 8 MB per chunk
READ_TIMEOUT = 60                # seconds
TOTAL_BYTES  = 2_460_457_927     # Exact remote Content-Length


def sizeof_fmt(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024.0:
            return f"{num:6.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def prepare_incomplete_file() -> None:
    """Find the largest existing incomplete file and consolidate to TMP_FILE."""
    BLOBS_DIR.mkdir(parents=True, exist_ok=True)
    SNAP_DIR.mkdir(parents=True, exist_ok=True)

    candidates = glob.glob(str(BLOBS_DIR / f"{BLOB_SHA}*incomplete*"))
    if not candidates:
        return

    # Find the candidate with max size
    largest = max(candidates, key=lambda p: os.path.getsize(p))
    largest_size = os.path.getsize(largest)
    print(f"[NLLB Download] Found existing download: {Path(largest).name} ({sizeof_fmt(largest_size)})")

    # If it's not already TMP_FILE, rename it to TMP_FILE
    if Path(largest).resolve() != TMP_FILE.resolve():
        if TMP_FILE.exists():
            TMP_FILE.unlink()
        shutil.move(largest, str(TMP_FILE))

    # Remove any other leftover incomplete files to reclaim disk space
    for c in candidates:
        p = Path(c)
        if p.exists() and p.resolve() != TMP_FILE.resolve():
            try:
                p.unlink()
                print(f"[NLLB Download] Cleaned up older temp file: {p.name}")
            except Exception:
                pass


def download_with_resume() -> None:
    BLOBS_DIR.mkdir(parents=True, exist_ok=True)
    SNAP_DIR.mkdir(parents=True, exist_ok=True)

    # If final blob already exists and is full size
    if BLOB_FILE.exists() and BLOB_FILE.stat().st_size >= TOTAL_BYTES:
        print(f"[OK] pytorch_model.bin already complete in cache ({sizeof_fmt(BLOB_FILE.stat().st_size)})")
        _make_symlink()
        return

    prepare_incomplete_file()

    resume_from = TMP_FILE.stat().st_size if TMP_FILE.exists() else 0
    if resume_from >= TOTAL_BYTES:
        print(f"[OK] Download already complete ({sizeof_fmt(resume_from)}). Finalizing...")
        if TMP_FILE.exists():
            TMP_FILE.rename(BLOB_FILE)
        _make_symlink()
        return

    if resume_from:
        print(f"[->] Resuming from {sizeof_fmt(resume_from)} / {sizeof_fmt(TOTAL_BYTES)} ({(resume_from/TOTAL_BYTES)*100:.1f}%) …")
    else:
        print(f"[DL] Starting fresh download of pytorch_model.bin ({sizeof_fmt(TOTAL_BYTES)}) …")

    attempt = 0
    session = requests.Session()

    while True:
        attempt += 1
        headers = {}
        resume_from = TMP_FILE.stat().st_size if TMP_FILE.exists() else 0
        if resume_from >= TOTAL_BYTES:
            break

        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"

        try:
            resp = session.get(URL, headers=headers, stream=True, timeout=(15, READ_TIMEOUT))
            if resp.status_code not in (200, 206):
                print(f"[!] HTTP {resp.status_code}. Retrying in 5 s …")
                time.sleep(5)
                continue

            mode = "ab" if (resume_from and resp.status_code == 206) else "wb"
            if mode == "wb" and resume_from > 0:
                print("[!] Server did not honor Range request; restarting from 0")
                resume_from = 0

            written = resume_from
            t0 = time.time()
            with TMP_FILE.open(mode) as fh:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    written += len(chunk)
                    elapsed = max(time.time() - t0, 0.001)
                    speed = (written - resume_from) / elapsed
                    pct = (written / TOTAL_BYTES) * 100
                    bar_w = 30
                    filled = min(int(bar_w * written / TOTAL_BYTES), bar_w)
                    bar = "=" * filled + "." * (bar_w - filled)
                    print(
                        f"\r  [{bar}] {pct:5.1f}%  "
                        f"{sizeof_fmt(written)} / {sizeof_fmt(TOTAL_BYTES)}  "
                        f"@ {sizeof_fmt(speed)}/s   ",
                        end="", flush=True,
                    )
            print()
            if written >= TOTAL_BYTES:
                break
        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError) as exc:
            cur = TMP_FILE.stat().st_size if TMP_FILE.exists() else 0
            print(f"\n[!] Attempt {attempt} interrupted ({exc.__class__.__name__}). "
                  f"Saved {sizeof_fmt(cur)}. Retrying in 3 s …")
            time.sleep(3)
        except Exception as exc:
            print(f"\n[!] Unexpected error: {exc}. Retrying in 5 s …")
            time.sleep(5)

    # Move .incomplete → final blob
    if TMP_FILE.exists():
        if BLOB_FILE.exists():
            BLOB_FILE.unlink()
        TMP_FILE.rename(BLOB_FILE)
    print(f"\n[OK] Saved to cache blob: {BLOB_FILE} ({sizeof_fmt(BLOB_FILE.stat().st_size)})")
    _make_symlink()


def _make_symlink() -> None:
    """Create the snapshot entry so transformers sees the file."""
    if not SYMLINK.exists():
        try:
            SYMLINK.symlink_to(BLOB_FILE)
            print(f"[OK] Symlink created: {SYMLINK}")
        except OSError:
            print("[~] Symlink not supported. Copying file into snapshot directory …")
            shutil.copy2(BLOB_FILE, SYMLINK)
            print(f"[OK] Copied to: {SYMLINK}")
    else:
        # Check if symlink is broken or 0 bytes
        if SYMLINK.stat().st_size < TOTAL_BYTES:
            try:
                SYMLINK.unlink()
                _make_symlink()
            except Exception:
                pass
        else:
            print(f"[OK] Snapshot entry ready: {SYMLINK}")


if __name__ == "__main__":
    download_with_resume()
    print("\n[OK] NLLB model weights downloaded and linked successfully.")
