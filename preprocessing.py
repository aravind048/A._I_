"""
PHASE 1 — preprocessing.py
============================
Pipeline per image:
  1. Load as grayscale
  2. Gaussian blur       → reduce noise
  3. Bilateral filter    → denoise while PRESERVING edges
  4. CLAHE               → enhance local contrast
  5. Percentile normalize → scale to [0,1] without outlier crush

KEY FIX — percentile normalization (p1/p99):
  Min-max normalization lets a few skull-ring pixels (pushed to 255
  by CLAHE) dominate and compress the tumor region to ~0.85.
  All segmentation methods then find the skull rim instead of the tumor.
  Percentile clipping fixes this by treating the top 1% as outliers.

Run:
    python preprocessing.py
"""

import os, argparse, cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_DIR  = "images"
OUTPUT_DIR = "results/preprocessed"
MAX_PROCESS = 100   # process all images

def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  WARNING: could not load {path}")
    return img

def apply_gaussian_blur(img):
    return cv2.GaussianBlur(img, (5, 5), sigmaX=0)

def apply_bilateral_filter(img):
    """Smooth noise but keep edges sharp — critical for tumor boundary detection."""
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

def apply_clahe(img):
    """Local contrast enhancement — makes tumor visible against brain tissue."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def normalize(img):
    """
    Percentile (p1/p99) normalization.

    WHY NOT MIN-MAX:
    CLAHE pushes a few skull pixels to 255. Min-max maps those to 1.0
    and compresses the tumor region to ~0.85. Segmentation methods then
    find the skull rim (intensity=1.0) instead of the tumor (0.85).

    Percentile clipping discards those outliers so the tumor becomes
    the genuine intensity peak in the preprocessed image.
    """
    img_f = img.astype(np.float32)
    lo    = float(np.percentile(img_f, 1))
    hi    = float(np.percentile(img_f, 99))
    if hi - lo < 1e-6:
        return np.zeros_like(img_f)
    return np.clip((img_f - lo) / (hi - lo), 0.0, 1.0)

def preprocess(filepath):
    original = load_image(filepath)
    if original is None:
        return None
    blurred   = apply_gaussian_blur(original)
    bilateral = apply_bilateral_filter(original)   # applied to original, not blurred
    enhanced  = apply_clahe(bilateral)
    normed    = normalize(enhanced)
    return {
        "original":   original,
        "blurred":    blurred,
        "bilateral":  bilateral,
        "clahe":      enhanced,
        "normalized": normed,
        "filepath":   str(filepath),
        "name":       Path(filepath).stem,
    }

def save_normalized_for_segmentation(results, output_dir):
    npy_dir = Path(output_dir) / "preprocessed" / "npy"
    npy_dir.mkdir(parents=True, exist_ok=True)
    for res in results:
        np.save(str(npy_dir / f"{res['name']}.npy"), res["normalized"])
        cv2.imwrite(
            str(npy_dir / f"{res['name']}_preview.png"),
            (res["normalized"] * 255).astype(np.uint8)
        )
    print(f"  Saved {len(results)} .npy files → {npy_dir}/")

def save_comparison(result, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    fig.suptitle(f"Preprocessing — {result['name']}", fontsize=12, fontweight="bold")
    stages = [
        ("original",   "1. Original",          result["original"],   [0,255]),
        ("blurred",    "2. Gaussian blur",      result["blurred"],    [0,255]),
        ("bilateral",  "3. Bilateral (edges preserved)", result["bilateral"],[0,255]),
        ("clahe",      "4. CLAHE",              result["clahe"],      [0,255]),
        ("normalized", "5. Normalized [0,1]",   result["normalized"], [0,1]),
    ]
    colors = {"original":"#888780","blurred":"#378ADD","bilateral":"#1D9E75",
              "clahe":"#D85A30","normalized":"#534AB7"}
    for col,(key,title,data,xlim) in enumerate(stages):
        axes[0][col].imshow(data, cmap="gray", vmin=xlim[0], vmax=xlim[1])
        axes[0][col].set_title(title, fontsize=9)
        axes[0][col].axis("off")
        flat = data.flatten()
        axes[1][col].hist(flat, bins=64, color=colors[key], alpha=0.8, density=True)
        axes[1][col].set_xlim(xlim)
        axes[1][col].set_xlabel("Intensity", fontsize=8)
        axes[1][col].tick_params(labelsize=7)
        axes[1][col].spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    out = os.path.join(out_dir, f"{result['name']}_preprocessing.png")
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

def main(single=None):
    print("\n PHASE 1 — PREPROCESSING")
    print("=" * 50)
    if single:
        paths = [Path(single)]
    else:
        paths = sorted(Path(INPUT_DIR).glob("*.png"))[:MAX_PROCESS]
    print(f"  Images to process: {len(paths)}")
    if not paths:
        print(f"  ERROR: No .png files in {INPUT_DIR}/")
        print("  Run download_dataset.py first.")
        return
    results = []
    for i, p in enumerate(paths):
        r = preprocess(p)
        if r:
            results.append(r)
            if (i+1) % 10 == 0 or i+1 == len(paths):
                print(f"  [{i+1}/{len(paths)}] {p.name}")
    save_normalized_for_segmentation(results, "results")
    # Save comparison for first image of each class
    seen_classes = set()
    comp_dir = os.path.join(OUTPUT_DIR, "comparisons")
    for res in results:
        cls = "_".join(res["name"].split("_")[:-1])
        if cls not in seen_classes:
            save_comparison(res, comp_dir)
            seen_classes.add(cls)
    print(f"\n  Done. {len(results)} images preprocessed.")
    print(f"  NPY files: results/preprocessed/npy/")
    print(f"  Next: run each method in methods/ or run compare.py\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None)
    args = parser.parse_args()
    main(single=args.image)
