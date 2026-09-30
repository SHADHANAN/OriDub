# -*- coding: utf-8 -*-
"""
Resumable & Multi-threaded Whisper Model Downloader
===================================================
Uses URLs and SHA256 hashes directly from OpenAI Whisper's registry (`whisper._MODELS`).
Downloads in parallel 1 MB chunks with streaming I/O, automatic resume,
and full SHA256 integrity verification directly into ~/.cache/whisper.

Run: python download_whisper.py [tiny|base|small]
"""
import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import whisper

# Unbuffered UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

CACHE_DIR = Path(os.path.expanduser("~")) / ".cache" / "whisper"
CHUNK_SIZE = 1024 * 1024  # 1 MB per chunk
NUM_WORKERS = 4


def sizeof_fmt(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024.0:
            return f"{num:6.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def verify_sha256(file_path: Path, expected: str) -> bool:
    if not file_path.is_file():
        return False
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest().lower() == expected.lower()


def download_model(model_name: str = "tiny") -> Path:
    """Download and verify a Whisper model using parallel chunked streaming.

    Does NOT call sys.exit(); raises RuntimeError on unrecoverable failure.
    """
    if model_name not in whisper._MODELS:
        raise ValueError(f"Unknown model: '{model_name}'. Options: {list(whisper._MODELS.keys())}")

    url = whisper._MODELS[model_name]
    expected_sha = url.split("/")[-2]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = CACHE_DIR / f"{model_name}.pt"
    parts_dir = CACHE_DIR / f"{model_name}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Check if complete target already exists
    if target.is_file():
        if verify_sha256(target, expected_sha):
            print(f"[OK] Whisper model '{model_name}' already complete and verified ({sizeof_fmt(target.stat().st_size)})")
            return target
        else:
            print(f"[!] Target '{target.name}' exists but has checksum mismatch. Rebuilding from parts...")

    # 2. Get remote file size
    total_size = 0
    for _ in range(5):
        try:
            head_resp = requests.head(url, timeout=15)
            total_size = int(head_resp.headers.get("content-length", 0))
            if total_size > 0:
                break
        except Exception:
            time.sleep(1)

    if not total_size:
        known_sizes = {
            "tiny": 75572083,
            "base": 145262807,
            "small": 483785139,
        }
        total_size = known_sizes.get(model_name, 0)

    if not total_size:
        raise RuntimeError(f"Could not determine file size for Whisper model '{model_name}'.")

    # 3. Build chunk list
    chunks = []
    start = 0
    c_idx = 0
    while start < total_size:
        end = min(start + CHUNK_SIZE - 1, total_size - 1)
        chunks.append((c_idx, start, end))
        c_idx += 1
        start = end + 1

    # Seed parts from existing target if valid size
    if target.is_file() and target.stat().st_size > 0:
        existing_size = target.stat().st_size
        with target.open("rb") as tf:
            for idx, c_start, c_end in chunks:
                expected_len = c_end - c_start + 1
                part_file = parts_dir / f"part_{idx:04d}.bin"
                if c_end < existing_size:
                    if not part_file.is_file() or part_file.stat().st_size != expected_len:
                        tf.seek(c_start)
                        data = tf.read(expected_len)
                        with part_file.open("wb") as pf:
                            pf.write(data)
                else:
                    break

    # Count completed parts
    done_parts = {
        idx for idx, c_start, c_end in chunks
        if (parts_dir / f"part_{idx:04d}.bin").is_file()
        and (parts_dir / f"part_{idx:04d}.bin").stat().st_size == (c_end - c_start + 1)
    }

    needed_chunks = [c for c in chunks if c[0] not in done_parts]
    size_label = sizeof_fmt(total_size)
    print(f"[Whisper] Model '{model_name}' ({size_label}): {len(done_parts)}/{len(chunks)} parts ready on disk.")

    def _download_single_chunk(item):
        idx, c_start, c_end = item
        expected_len = c_end - c_start + 1
        part_file = parts_dir / f"part_{idx:04d}.bin"
        part_tmp  = parts_dir / f"part_{idx:04d}.tmp"

        headers = {"Range": f"bytes={c_start}-{c_end}"}
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(max_retries=5)
        session.mount("https://", adapter)

        for attempt in range(1, 30):
            try:
                with session.get(url, headers=headers, stream=True, timeout=(15, 60)) as resp:
                    if resp.status_code in (200, 206):
                        bytes_read = 0
                        with part_tmp.open("wb") as pf:
                            for blk in resp.iter_content(chunk_size=32768):
                                if blk:
                                    pf.write(blk)
                                    bytes_read += len(blk)
                        if bytes_read == expected_len:
                            if part_file.exists():
                                part_file.unlink()
                            part_tmp.rename(part_file)
                            return idx, expected_len
                time.sleep(1 + attempt % 4)
            except Exception:
                time.sleep(1 + attempt % 4)

        raise RuntimeError(f"Chunk {idx} ({c_start}-{c_end}) failed after 30 attempts.")

    completed = len(done_parts)
    start_time = time.time()
    initial_bytes = sum(c[2] - c[1] + 1 for c in chunks if c[0] in done_parts)
    completed_bytes = initial_bytes

    if needed_chunks:
        print(f"[Whisper] Downloading {len(needed_chunks)} remaining chunks using {NUM_WORKERS} workers...")
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(_download_single_chunk, c): c for c in needed_chunks}
            for future in as_completed(futures):
                idx, nbytes = future.result()
                completed += 1
                completed_bytes += nbytes
                elapsed = time.time() - start_time
                speed = (completed_bytes - initial_bytes) / (elapsed if elapsed > 0 else 1)
                pct = (completed_bytes / total_size) * 100
                print(
                    f"\r  [{completed:02d}/{len(chunks)}] {pct:5.1f}%  "
                    f"{sizeof_fmt(completed_bytes)} / {size_label}  @{sizeof_fmt(speed)}/s   ",
                    end="",
                    flush=True,
                )
        print()

    print(f"[Whisper] Assembling '{model_name}.pt' from {len(chunks)} parts...")
    tmp_assembled = CACHE_DIR / f"{model_name}.pt.downloading"
    with tmp_assembled.open("wb") as out:
        for idx, _, _ in chunks:
            part_file = parts_dir / f"part_{idx:04d}.bin"
            with part_file.open("rb") as pf:
                out.write(pf.read())

    print("[Whisper] Verifying SHA256 checksum...")
    if verify_sha256(tmp_assembled, expected_sha):
        if target.exists():
            target.unlink()
        tmp_assembled.rename(target)
        print(f"[OK] Whisper model '{model_name}' verified and ready at: {target}")
        # Clean up parts directory
        for idx, _, _ in chunks:
            (parts_dir / f"part_{idx:04d}.bin").unlink(missing_ok=True)
        parts_dir.rmdir()
        return target
    else:
        tmp_assembled.unlink(missing_ok=True)
        raise RuntimeError(
            f"Whisper model '{model_name}' checksum verification failed. Please retry."
        )


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    download_model(name)
