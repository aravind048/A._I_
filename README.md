# Automated Medical Image Boundary Detection
### MRI Brain Tumor Segmentation using Classical Image Processing

---

## Overview

This project implements and compares four classical segmentation methods to automatically detect tumor boundaries in MRI brain images. Each method approaches the problem differently — from energy-minimising contours to density-based clustering — allowing a direct comparison of their strengths on real clinical data.

**Dataset:** [PranomVignesh/MRI-Images-of-Brain-Tumor](https://huggingface.co/datasets/PranomVignesh/MRI-Images-of-Brain-Tumor) (HuggingFace)

**Labels:**

| Label | Class | Description |
|---|---|---|
| 0 | `glioma` | Malignant tumor arising from glial cells |
| 1 | `meningioma` | Tumor arising from the meninges |
| 2 | `no_tumor` | Healthy brain — no tumor present |
| 3 | `pituitary` | Tumor in the pituitary gland region |

---

## Methods Implemented

### Phase 2 — Active Contours (Snakes)
An initial circular contour placed around the suspected tumor region evolves iteratively under two competing forces: an internal smoothness force keeps the contour elastic, and an external edge force pulls it toward strong intensity gradients. The contour converges on the tumor boundary when the two forces balance.

**Key parameters:** `alpha` (elasticity), `beta` (rigidity), `w_edge` (edge attraction)

### Phase 3 — Level Sets
Two models implemented:
- **Chan-Vese** (region-based): Partitions the image into two intensity regions by minimising a functional that penalises long boundaries and non-uniform regions. Works even without sharp edges.
- **Geodesic Active Contour** (edge-based): Evolves a level-set function stopped by an edge-indicator function. Better at high-contrast boundaries.

**Key parameters:** `num_iter`, `smoothing`, `lambda1/lambda2` (Chan-Vese); `balloon`, `threshold` (Geodesic)

### Phase 4 — Mean Shift
Runs `cv2.pyrMeanShiftFiltering` to flatten the image into distinct colour blobs by mode-seeking in joint spatial-colour space. K-means (K=4) then labels the resulting clusters by intensity. The tumor candidate is selected as the brightest cluster weighted by area × centrality — preventing small peripheral skull hotspots from outscoring the larger central tumour blob.

**Key parameters:** `sp` (spatial radius), `sr` (colour radius), `K_CLUSTERS`

### Phase 5 — Split and Merge (Quadtree)
Built entirely from scratch without any segmentation library.

- **Split phase:** Recursively divides regions whose intensity variance exceeds a threshold into four quadrants (quadtree). Stops when all regions are uniform or too small.
- **Merge phase:** Union-Find algorithm merges adjacent leaf nodes whose mean intensities differ by less than a threshold.
- **Selection:** Picks the segment with the highest `mean_intensity × area × centrality` score.

**Key parameters:** `SPLIT_VAR_THRESH`, `MIN_REGION_SIZE`, `MERGE_MEAN_THRESH`

---

## Project Structure

```
project/
│
├── download_dataset.py     Phase 0 — Balanced 4-class download from HuggingFace
├── preprocessing.py        Phase 1 — Bilateral filter + CLAHE + percentile normalisation
│
├── methods/
│   ├── active_contours.py  Phase 2 — Snakes (scikit-image)
│   ├── level_sets.py       Phase 3 — Chan-Vese + Geodesic (scikit-image)
│   ├── mean_shift.py       Phase 4 — Mean shift + K-means (OpenCV)
│   └── split_merge.py      Phase 5 — Quadtree + Union-Find (from scratch)
│
├── compare.py              Phase 6 — Side-by-side comparison of all 4 methods
├── evaluation.py           Phase 7 — Accuracy metrics vs ground truth (Dice, IoU, HD95)
├── main.py                 Phase 8 — Full pipeline runner + final report
└── utils.py                Shared — balanced class sampling for all scripts
│
├── images/                 Downloaded MRI images (created by download_dataset.py)
└── results/
    ├── preprocessed/npy/   Normalised .npy arrays used by all methods
    ├── active_contours/    Snake boundary visualisations
    ├── level_sets/         Chan-Vese and Geodesic visualisations
    ├── mean_shift/         K-means cluster and watershed visualisations
    ├── split_merge/        Quadtree grid and segment visualisations
    ├── comparison/         Side-by-side grids + master poster
    ├── evaluation/         Error maps, radar chart, metrics bars, CSV
    └── report/             Final summary figures + summary.json
```

---

## Setup

### Requirements

```bash
pip install opencv-python scikit-image numpy matplotlib scipy Pillow pandas pyarrow
```

### Python version
Python 3.9 or later recommended.

---

## Running the Pipeline

### Option A — Full pipeline (one command)

```bash
# Download data, preprocess, run all methods, compare, evaluate, report
python main.py --n 4
```

`--n 4` means 4 images per class (glioma, meningioma, no_tumor, pituitary) = 16 images total.

### Option B — Step by step

```bash
# Phase 0 — Download balanced dataset (15 images × 4 classes)
python download_dataset.py

# Phase 1 — Preprocess (bilateral filter + CLAHE + p99 normalise)
python preprocessing.py

# Phases 2–5 — Run individual methods
python methods/active_contours.py --all
python methods/level_sets.py --all
python methods/mean_shift.py --all
python methods/split_merge.py --all

# Phase 6 — Side-by-side comparison
python compare.py --n 4

# Phase 7 — Accuracy evaluation with ground truth
python evaluation.py --n 12 --csv

# Phase 8 — Final report
python main.py --skip-preprocess --report-only
```

### Useful flags

| Script | Flag | Effect |
|---|---|---|
| `main.py` | `--skip-preprocess` | Skip Phase 1 if images already in `npy/` |
| `main.py` | `--report-only` | Regenerate report from existing CSV |
| `main.py` | `--n N` | Images per class (default: 4) |
| `evaluation.py` | `--n N` | Max evaluation pairs |
| `evaluation.py` | `--csv` | Save results to `evaluation_results.csv` |
| `evaluation.py` | `--list` | Print discovered pairs and exit |
| `compare.py` | `--n N` | Images per class |
| `compare.py` | `--timing` | Measure and plot runtime per method |
| Any method | `--all` | Run on all preprocessed images |
| Any method | `--image path` | Run on a single `.npy` file |

---

## Preprocessing Pipeline

Each image goes through four stages before reaching any segmentation method:

```
Raw PNG
  ↓  Grayscale conversion
  ↓  Gaussian blur (5×5)            — coarse noise removal
  ↓  Bilateral filter (d=9, σ=75)   — edge-preserving denoising
  ↓  CLAHE (clipLimit=2.0, 8×8)     — local contrast enhancement
  ↓  Percentile normalisation (p1/p99) — scale to [0, 1]
Preprocessed .npy
```

**Why percentile normalisation (not min-max)?**
CLAHE can push a few skull-ring pixels to intensity 255. Min-max normalization maps those to 1.0 and compresses the tumor region to ~0.85, causing all methods to detect the skull rim instead of the tumor. Clipping at the 1st and 99th percentile discards those outliers and preserves the tumor as the genuine intensity peak.

---

## Evaluation Metrics

| Metric | Formula | What it measures |
|---|---|---|
| **Dice / F1** | 2·TP / (2·TP + FP + FN) | Overlap quality — primary metric |
| **IoU** | TP / (TP + FP + FN) | Stricter overlap (always ≤ Dice) |
| **Pixel Accuracy** | (TP + TN) / total | Can be misleading with class imbalance |
| **Precision** | TP / (TP + FP) | How much detected region is real tumour |
| **Recall** | TP / (TP + FN) | How much of the real tumour was found |
| **Hausdorff-95** | 95th percentile boundary distance (px) | Spatial boundary accuracy |
| **Specificity** | TN / (TN + FP) | For `no_tumor` images — false alarm rate |

**Clinical benchmarks:**
- Dice ≥ 0.7 → clinically acceptable
- IoU ≥ 0.5 → good overlap
- HD95 ≤ 10 px → boundary within ~10 pixels of ground truth

**Ground truth generation:**
For glioma / meningioma / pituitary images, ground truth masks are auto-generated from the preprocessed image: Otsu threshold → morphological erosion (removes skull ring) → central connected component (= tumour blob).

For `no_tumor` images, ground truth is all-zeros. Specificity is reported instead of Dice.

---

## Output Files

### Per-method outputs (`results/<method>/`)
Each method saves a multi-panel PNG for every processed image showing:
- The original MRI
- Intermediate processing stages specific to the method
- The final detected region overlay (coloured tint + yellow boundary)

### Comparison (`results/comparison/`)

| File | Description |
|---|---|
| `<name>_sidebyside.png` | All 5 method results + agreement map in one row |
| `<name>_boundaries.png` | Boundary-only overlay — all methods on one image |
| `metrics_comparison.png` | Area%, compactness, and perimeter bar charts |
| `master_grid.png` | Poster: all images × all methods |

### Evaluation (`results/evaluation/`)

| File | Description |
|---|---|
| `<name>_errors.png` | TP/FP/FN colour map per method |
| `radar_chart.png` | Spider chart across all 5 metrics |
| `metrics_bars.png` | Dice, IoU, HD95, Precision vs Recall bars |
| `evaluation_results.csv` | Full metric table — importable for further analysis |

### Report (`results/report/`)

| File | Description |
|---|---|
| `final_performance_summary.png` | 4-panel: Dice, IoU, HD95, Prec vs Recall |
| `class_method_heatmap.png` | Dice per class × method (Specificity for no_tumor) |
| `radar_chart.png` | Method performance radar |
| `pipeline_timing.png` | Runtime per phase |
| `summary.json` | Machine-readable metrics summary |

---

## Method Comparison Summary

| Method | Strength | Limitation |
|---|---|---|
| Active Contours | Smoothest boundaries, high compactness | Needs good initialisation, slow (2500 iters) |
| Level Sets (CV) | Works without strong edges, handles topology | Assumes two uniform regions |
| Level Sets (GAC) | Precise at sharp edges | Can leak through weak boundaries |
| Mean Shift | Fully automatic, no initial contour | K must be tuned per image type |
| Split & Merge | Most interpretable, fastest, built from scratch | Produces staircase boundaries |

---

## Key Design Decisions

**Balanced class sampling (`utils.py`)**
Both `compare.py` and `evaluation.py` use `get_balanced_npy_files()` which interleaves classes in round-robin order:
```
Round 0: glioma_0 → meningioma_0 → no_tumor_0 → pituitary_0
Round 1: glioma_1 → meningioma_1 → no_tumor_1 → pituitary_1
```
This guarantees every class appears even with small `--n` values.

**Centrality-weighted tumour selection**
Both Mean Shift and Split & Merge score candidates as:
```
score = mean_intensity × area × centrality
centrality = 1 / (1 + distance_from_centre / 60)
```
This prevents tiny bright skull-rim spots from scoring higher than the larger, more central tumour blob.

---

## Acknowledgements

Dataset: PranomVignesh — MRI Images of Brain Tumor (HuggingFace)

Libraries: OpenCV, scikit-image, NumPy, Matplotlib, SciPy, Pillow, pandas
