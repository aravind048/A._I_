"""
utils.py
=========
Shared helper used by both compare.py and evaluation.py.

Core function: get_balanced_npy_files()
  Returns image .npy files with EQUAL representation from every class.
  Interleaves classes so even a small --n value covers all of them.

WHY THIS IS NEEDED:
  Alphabetical sorting gives: glioma, glioma, glioma, meningioma,
  meningioma, meningioma, no_tumor, pituitary ...
  With n=8, no_tumor gets 2 slots and pituitary gets 0.

  Interleaved gives: glioma_0, meningioma_0, no_tumor_0, pituitary_0,
                     glioma_1, meningioma_1, no_tumor_1, pituitary_1 ...
  Every class always represented, even with small n.
"""

from pathlib import Path
from collections import defaultdict

# All valid tumor classes in the dataset
ALL_CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary", "tumor"]

# Prefixes to always skip — these are masks or old synthetic files, NOT input images
SKIP_PREFIXES = ("mask_", "mri_synth_", "mask_synth_")


def _get_class(stem):
    """Identify a file's class from its stem (e.g. 'glioma_0003' → 'glioma')."""
    for cls in ALL_CLASSES:
        if stem.startswith(cls + "_"):
            return cls
    return None   # unknown / not an image file


def get_balanced_npy_files(npy_dir, per_class=4):
    """
    Return a list of image .npy paths with equal representation per class.

    Algorithm:
      1. Scan npy_dir for *.npy files
      2. Skip mask_*, mri_synth_*, mask_synth_*, *_preview.npy
      3. Bucket remaining files by class
      4. Take up to per_class files from each class
      5. INTERLEAVE by class:
            round 0: glioma_0000, meningioma_0000, no_tumor_0001, pituitary_0000
            round 1: glioma_0001, meningioma_0001, no_tumor_0003, pituitary_0001
            ...
         so that slicing [:n] always contains all classes

    Args:
        npy_dir   : path to results/preprocessed/npy/
        per_class : max images taken from each class (default 4)

    Returns:
        list of Path — interleaved, all classes represented
    """
    npy_dir = Path(npy_dir)

    # ── Bucket files by class ────────────────────────────────────
    by_class = defaultdict(list)
    skipped  = []

    for f in sorted(npy_dir.glob("*.npy")):
        # Skip preview files
        if "_preview" in f.name:
            continue
        # Skip mask / synthetic files
        if any(f.name.startswith(p) for p in SKIP_PREFIXES):
            skipped.append(f.name)
            continue
        cls = _get_class(f.stem)
        if cls is None:
            continue   # unrecognised prefix — skip silently
        by_class[cls].append(f)

    # ── Report ───────────────────────────────────────────────────
    found_classes = [c for c in ALL_CLASSES if by_class[c]]
    missing       = [c for c in ALL_CLASSES if not by_class[c]]

    print(f"\n  Image files per class (quota: {per_class} each):")
    for cls in ALL_CLASSES:
        found = len(by_class[cls])
        used  = min(found, per_class)
        if found == 0:
            print(f"    {cls:15s}:   0 found  ← not downloaded yet")
        else:
            bar = "█" * used + "░" * max(0, per_class - used)
            print(f"    {cls:15s}: {found:3d} found  →  {used} used  [{bar}]")

    if missing:
        print(f"\n  NOTE: {missing} not found.")
        print(f"  Run python download_dataset.py to get all classes.\n")

    # ── Interleave across classes ────────────────────────────────
    # Take up to per_class from each class
    pools  = {cls: by_class[cls][:per_class] for cls in found_classes}
    result = []

    # Round-robin: one from each class per round
    max_rounds = per_class
    for round_i in range(max_rounds):
        for cls in found_classes:          # fixed class order (not alphabetical of files)
            if round_i < len(pools[cls]):
                result.append(pools[cls][round_i])

    return result
