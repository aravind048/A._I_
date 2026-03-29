"""
PHASE 0 — download_dataset.py
================================
Downloads a BALANCED set of real MRI images from all 4 label classes:
  Label 0 — no_tumor
  Label 1 — glioma
  Label 2 — meningioma
  Label 3 — pituitary

Scans the dataset in batches until TARGET_PER_CLASS images are
collected from every class. Never generates synthetic data.

Run:
    python download_dataset.py
"""

import os, requests, numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path

DATASET    = "PranomVignesh/MRI-Images-of-Brain-Tumor"
SAVE_DIR   = "images"
API_BASE   = "https://datasets-server.huggingface.co"

TARGET_PER_CLASS = 15   # images per label class
MAX_SCAN_ROWS    = 3000
BATCH_SIZE       = 50

LABEL_MAP = {0:"no_tumor", 1:"glioma", 2:"meningioma", 3:"pituitary"}


def get_parquet_urls():
    url = f"{API_BASE}/parquet?dataset={DATASET.replace('/','%2F')}"
    r   = requests.get(url, timeout=30)
    if r.status_code != 200:
        print(f"  ERROR {r.status_code}: {r.text[:200]}")
        return []
    files = r.json().get("parquet_files", [])
    urls  = [f["url"] for f in files if f.get("split") == "train"]
    if not urls:
        urls = [f["url"] for f in files]
    print(f"  Found {len(urls)} parquet file(s)")
    for f in files:
        mb = f.get("size", 0) / 1024 / 1024
        print(f"  [{f.get('split'):12s}] {f.get('filename','')[:45]}  ({mb:.1f} MB)")
    return urls


def download_balanced(parquet_urls):
    os.makedirs(SAVE_DIR, exist_ok=True)

    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pip install pandas pyarrow")
        return

    collected = {k: 0 for k in LABEL_MAP}
    saved = 0

    for url in parquet_urls:
        if all(v >= TARGET_PER_CLASS for v in collected.values()):
            break
        print(f"\n  Loading: {Path(url).name}")
        try:
            df = pd.read_parquet(url)
        except Exception as e:
            print(f"  ERROR loading parquet: {e}")
            continue

        # Shuffle so we don't always get same-class runs at the top
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"  Rows: {len(df)}")

        for _, row in df.iterrows():
            if all(v >= TARGET_PER_CLASS for v in collected.values()):
                break

            raw = row.get("label", None)
            if raw is None:
                continue
            lbl = int(raw)

            # Handle labels outside our map
            if lbl not in LABEL_MAP:
                LABEL_MAP[lbl] = f"class_{lbl}"
                collected[lbl] = 0

            if collected[lbl] >= TARGET_PER_CLASS:
                continue

            # Extract image bytes
            img_data = row.get("image", None)
            if img_data is None:
                continue
            if isinstance(img_data, dict):
                img_bytes = img_data.get("bytes")
            elif isinstance(img_data, bytes):
                img_bytes = img_data
            else:
                continue
            if not img_bytes:
                continue

            try:
                img   = Image.open(BytesIO(img_bytes)).convert("RGB")
                n     = collected[lbl]
                fname = f"{SAVE_DIR}/{LABEL_MAP[lbl]}_{n:04d}.png"
                img.save(fname)
                collected[lbl] += 1
                saved += 1
                if saved % 10 == 0:
                    print(f"  [{saved:3d}] saved so far | " +
                          " | ".join(f"{LABEL_MAP[k]}:{v}" for k,v in sorted(collected.items())))
            except Exception as e:
                print(f"  Row error: {e}")

    print(f"\n  Download complete: {saved} images saved")
    print(f"\n  Class distribution:")
    for lbl, name in sorted(LABEL_MAP.items()):
        cnt = collected.get(lbl, 0)
        bar = "█" * cnt + "░" * max(0, TARGET_PER_CLASS - cnt)
        print(f"    [{lbl}] {name:12s}: {cnt:3d}/{TARGET_PER_CLASS}  {bar}")


def verify():
    files = sorted(Path(SAVE_DIR).glob("*.png"))
    print(f"\n  Total images: {len(files)}")
    counts = {}
    for f in files:
        cls = "_".join(f.stem.split("_")[:-1])
        counts[cls] = counts.get(cls, 0) + 1
    for cls, cnt in sorted(counts.items()):
        print(f"  {cls:15s}: {cnt}")
    return len(files) > 0


if __name__ == "__main__":
    print("\n PHASE 0 — BALANCED DATASET DOWNLOAD")
    print("=" * 55)
    print(f"  Target: {TARGET_PER_CLASS} per class × {len(LABEL_MAP)} classes")
    print(f"  Classes: {list(LABEL_MAP.values())}")
    print()

    # Check splits
    r = requests.get(f"{API_BASE}/splits?dataset={DATASET.replace('/','%2F')}", timeout=30)
    splits = [s["split"] for s in r.json().get("splits",[])] if r.status_code==200 else []
    print(f"  Splits: {splits}")

    urls = get_parquet_urls()
    download_balanced(urls)
    verify()
    print("\n  Next: python preprocessing.py\n")
