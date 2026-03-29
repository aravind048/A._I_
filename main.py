"""
PHASE 8 — main.py
==================
Master runner — executes the complete pipeline end to end and
produces a final summary report.

PIPELINE ORDER:
  Phase 1 → preprocessing.py       (bilateral + CLAHE + p99 normalize)
  Phase 2 → active_contours.py     (snakes)
  Phase 3 → level_sets.py          (Chan-Vese + Geodesic)
  Phase 4 → mean_shift.py          (K-means on mean-shifted image)
  Phase 5 → split_merge.py         (quadtree + union-find)
  Phase 6 → compare.py             (side-by-side comparison)
  Phase 7 → evaluation.py          (Dice, IoU, HD95)
  Phase 8 → main.py                (this file — full report)

Usage:
    python main.py                  full pipeline on all images
    python main.py --skip-preprocess  if images already preprocessed
    python main.py --n 8            limit to N images per class
    python main.py --report-only    just generate report from existing results
"""

import os, sys, time, argparse, csv, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, ".")
from utils       import get_balanced_npy_files, ALL_CLASSES


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMAGES_DIR  = "images"
NPY_DIR     = "results/preprocessed/npy"
REPORT_DIR  = "results/report"

METHODS = {
    "snake":       "Active Contours",
    "chan_vese":   "Level Sets (CV)",
    "geodesic":    "Level Sets (GAC)",
    "mean_shift":  "Mean Shift",
    "split_merge": "Split & Merge",
}

METHOD_COLORS = {
    "snake":       "#E24B4A",
    "chan_vese":   "#1D9E75",
    "geodesic":    "#0F6E56",
    "mean_shift":  "#534AB7",
    "split_merge": "#BA7517",
}


# ─────────────────────────────────────────────
# PHASE RUNNERS
# ─────────────────────────────────────────────
def run_phase(name, fn, *args, **kwargs):
    """Run one phase, print timing, return result."""
    print(f"\n{'═'*60}")
    print(f"  {name}")
    print(f"{'═'*60}")
    t0  = time.perf_counter()
    res = fn(*args, **kwargs)
    dt  = time.perf_counter() - t0
    print(f"\n  ✓ {name} completed in {dt:.1f}s")
    return res, dt


def phase1_preprocess(image_dir, max_images=100):
    """Preprocess all images in images/."""
    from preprocessing import preprocess, save_normalized_for_segmentation

    paths = sorted(Path(image_dir).glob("*.png"))[:max_images]
    if not paths:
        print(f"  ERROR: No images in {image_dir}/")
        print("  Run python download_dataset.py first.")
        sys.exit(1)

    print(f"  Processing {len(paths)} images...")
    results = []
    for i, p in enumerate(paths):
        r = preprocess(str(p))
        if r:
            results.append(r)
            if (i+1) % 10 == 0 or i+1 == len(paths):
                print(f"  [{i+1}/{len(paths)}] done")

    save_normalized_for_segmentation(results, "results")
    return len(results)


def phase2to5_segment(npy_files):
    """
    Run all 4 segmentation methods on every file.
    Returns nested dict: results[stem][method] = {mask, metrics}
    """
    from methods.active_contours import run_on_image as run_snake
    from methods.level_sets      import run_on_image as run_ls
    from methods.mean_shift      import run_on_image as run_ms
    from methods.split_merge     import run_on_image as run_sm

    all_results = {}
    total = len(npy_files)

    for i, f in enumerate(npy_files):
        stem = Path(f).stem
        print(f"  [{i+1}/{total}] {stem}")
        r1 = run_snake(str(f))
        r2 = run_ls(str(f))
        r3 = run_ms(str(f))
        r4 = run_sm(str(f))

        all_results[stem] = {
            "image":   r1["image"],
            "snake":   {"mask": r1["mask"],         "metrics": r1["metrics"]},
            "chan_vese":{"mask": r2["mask_cv"],      "metrics": r2["metrics"]["cv"]},
            "geodesic": {"mask": r2["mask_gac"],     "metrics": r2["metrics"]["gac"]},
            "mean_shift":{"mask":r3["tumor_mask"],   "metrics": r3["metrics"]},
            "split_merge":{"mask":r4["tumor_mask"],  "metrics": r4["metrics"]},
        }
    return all_results


def phase6_compare(npy_files):
    """Run comparison visualizations."""
    from compare import compare
    compare([str(f) for f in npy_files])


def phase7_evaluate(max_pairs):
    """Run accuracy evaluation, return metrics dict."""
    from evaluation import discover_pairs, evaluate
    pairs = discover_pairs(max_pairs=max_pairs)
    return evaluate(pairs, save_csv=True)


# ─────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────
def load_evaluation_csv():
    """Load evaluation_results.csv into a nested dict."""
    csv_path = Path("results/evaluation/evaluation_results.csv")
    if not csv_path.exists():
        return {}

    data = defaultdict(lambda: defaultdict(dict))
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            pair   = row["pair"]
            method = row["method"]
            data[pair][method] = {
                k: float(v) if v not in ("", "-1.0") else None
                for k, v in row.items()
                if k not in ("pair", "method", "gt_type")
            }
    return data


def compute_summary_stats(data):
    """
    Compute per-method averages across all non-zero-GT pairs.
    Returns dict: stats[method] = {dice_mean, iou_mean, hd95_mean, ...}
    """
    stats = {m: defaultdict(list) for m in METHODS}

    for pair, methods in data.items():
        for method, m in methods.items():
            if method not in METHODS:
                continue
            # Skip no_tumor pairs for Dice/IoU averages
            if m.get("dice") == 0 and m.get("recall") in (0, None):
                continue
            for key in ["dice", "iou", "precision", "recall", "pixel_acc"]:
                if m.get(key) is not None and m[key] >= 0:
                    stats[method][key].append(m[key])
            if m.get("hausdorff_95") and m["hausdorff_95"] > 0:
                stats[method]["hausdorff_95"].append(m["hausdorff_95"])

    return {
        method: {k: round(np.mean(v), 4) if v else 0
                 for k, v in vals.items()}
        for method, vals in stats.items()
    }


# ─────────────────────────────────────────────
# FIGURE 1: Final performance summary
# ─────────────────────────────────────────────
def plot_final_summary(summary_stats, out_dir):
    """
    4-panel summary chart:
      Panel 1 — Dice score bar chart per method
      Panel 2 — IoU bar chart per method
      Panel 3 — Hausdorff-95 bar chart per method (lower = better)
      Panel 4 — Precision vs Recall scatter plot
    """
    os.makedirs(out_dir, exist_ok=True)

    methods = list(METHODS.keys())
    labels  = [METHODS[m] for m in methods]
    colors  = [METHOD_COLORS[m] for m in methods]
    x       = np.arange(len(methods))

    dice  = [summary_stats.get(m, {}).get("dice", 0)       for m in methods]
    iou   = [summary_stats.get(m, {}).get("iou", 0)        for m in methods]
    hd95  = [summary_stats.get(m, {}).get("hausdorff_95", 0) for m in methods]
    prec  = [summary_stats.get(m, {}).get("precision", 0)  for m in methods]
    rec   = [summary_stats.get(m, {}).get("recall", 0)     for m in methods]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle("Phase 8 — Final Performance Summary\n(averaged over all evaluated image pairs)",
                 fontsize=13, fontweight="bold")

    def _bar(ax, values, ylabel, title, ref=None, low_better=False):
        bars = ax.bar(x, values, color=colors, alpha=0.87, width=0.55)
        for bar, v in zip(bars, values):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=9, fontweight="500")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="500")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        if ref:
            ax.axhline(ref, color="#888780", lw=1.2, ls="--",
                       alpha=0.7, label=f"target ≥{ref}")
            ax.legend(fontsize=8)
        if not low_better:
            ax.set_ylim(0, 1.15)
        # Highlight best bar
        if values:
            best_idx = (values.index(min(values)) if low_better
                        else values.index(max(values)))
            bars[best_idx].set_edgecolor("black")
            bars[best_idx].set_linewidth(2)

    _bar(axes[0], dice, "Dice Score", "Dice (F1)\n★ higher is better", ref=0.7)
    _bar(axes[1], iou,  "IoU",        "IoU (Jaccard)\n★ higher is better", ref=0.5)
    _bar(axes[2], hd95, "Distance (px)",
         "Hausdorff-95\n★ lower is better", low_better=True)

    # Precision vs Recall scatter
    for i, m in enumerate(methods):
        p_val = prec[i]; r_val = rec[i]
        axes[3].scatter(r_val, p_val, color=colors[i], s=180,
                        zorder=5, label=METHODS[m], edgecolors="white", lw=1)
        axes[3].annotate(
            METHODS[m].split()[0],
            (r_val, p_val),
            textcoords="offset points", xytext=(8, 4),
            fontsize=7, color=colors[i]
        )

    axes[3].set_xlabel("Recall (sensitivity)", fontsize=10)
    axes[3].set_ylabel("Precision (specificity)", fontsize=10)
    axes[3].set_title("Precision vs Recall\n★ top-right = best balance", fontsize=11)
    axes[3].set_xlim(-0.05, 1.1); axes[3].set_ylim(-0.05, 1.1)
    axes[3].axhline(0.5, color="#D3D1C7", lw=0.8, ls="--")
    axes[3].axvline(0.5, color="#D3D1C7", lw=0.8, ls="--")
    axes[3].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(out_dir, "final_performance_summary.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ─────────────────────────────────────────────
# FIGURE 2: Per-class performance heatmap
# ─────────────────────────────────────────────
def plot_class_heatmap(eval_data, out_dir):
    """
    Heatmap: rows = ALL classes (including no_tumor), columns = methods.

    For tumor classes (glioma / meningioma / pituitary / tumor):
        Cell = mean Dice score  (higher = better, green)

    For no_tumor class:
        Cell = mean Specificity = TN / (TN + FP)
        = fraction of background pixels correctly left undetected
        (higher = better, method is not over-detecting on healthy brain)
        Shown with a different label so the reader knows the metric differs.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Collect per-class per-method scores
    tumor_dice    = defaultdict(lambda: defaultdict(list))   # cls → method → [dice]
    notumor_spec  = defaultdict(list)                        # method → [specificity]

    for pair, methods in eval_data.items():
        # Identify class from pair name
        cls = None
        for c in ALL_CLASSES:
            if pair.startswith(c + "_") or pair.startswith(c):
                cls = c
                break
        if cls is None:
            continue

        for method, m in methods.items():
            if method not in METHODS:
                continue

            if cls == "no_tumor":
                # Specificity = TN / (TN + FP)
                tn = m.get("TN", 0) or 0
                fp = m.get("FP", 0) or 0
                if (tn + fp) > 0:
                    spec = tn / (tn + fp)
                    notumor_spec[method].append(spec)
            else:
                dice = m.get("dice", 0)
                if dice is not None and dice >= 0:
                    tumor_dice[cls][method].append(dice)

    if not tumor_dice and not notumor_spec:
        print("  Skipping heatmap — no data available yet")
        return None

    # Build ordered class list: tumor classes first, then no_tumor
    tumor_classes_found = [c for c in ALL_CLASSES
                           if c != "no_tumor" and tumor_dice.get(c)]
    has_notumor = bool(notumor_spec)
    all_rows    = tumor_classes_found + (["no_tumor"] if has_notumor else [])

    if not all_rows:
        print("  Skipping heatmap — no rows available")
        return None

    methods_list  = list(METHODS.keys())
    method_labels = list(METHODS.values())
    n_rows = len(all_rows)
    n_cols = len(methods_list)

    # Build matrix
    matrix = np.zeros((n_rows, n_cols))
    for i, cls in enumerate(all_rows):
        for j, method in enumerate(methods_list):
            if cls == "no_tumor":
                vals = notumor_spec.get(method, [])
            else:
                vals = tumor_dice.get(cls, {}).get(method, [])
            matrix[i, j] = np.mean(vals) if vals else 0

    # ── Plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, max(4, n_rows * 1.5 + 1.5)))
    fig.suptitle("Segmentation Performance per Class × Method",
                 fontsize=13, fontweight="bold", y=1.01)

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Score (Dice for tumor | Specificity for no_tumor)", fontsize=9)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(method_labels, fontsize=10, rotation=20, ha="right")
    ax.set_yticks(np.arange(n_rows))

    # Row labels — add metric name so reader knows what they're seeing
    row_labels = []
    for cls in all_rows:
        if cls == "no_tumor":
            row_labels.append("No Tumor\n(Specificity)")
        else:
            row_labels.append(f"{cls.replace('_',' ').title()}\n(Dice)")
    ax.set_yticklabels(row_labels, fontsize=10)

    # Draw a separator line between tumor rows and no_tumor row
    if has_notumor and tumor_classes_found:
        sep_y = len(tumor_classes_found) - 0.5
        ax.axhline(sep_y, color="black", lw=1.5, ls="--", alpha=0.5)

    # Annotate each cell with the score value
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            txt_color = "white" if val < 0.35 else "black"
            ax.text(j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=11, color=txt_color, fontweight="500")

    # Highlight best method per row with a black border
    for i in range(n_rows):
        best_j = int(np.argmax(matrix[i]))
        ax.add_patch(plt.Rectangle(
            (best_j - 0.48, i - 0.48), 0.96, 0.96,
            fill=False, edgecolor="black", lw=2.5
        ))

    # Legend for the separator and border
    legend_elements = [
        mpatches.Patch(facecolor="none", edgecolor="black", lw=2.5,
                       label="Best method per class"),
    ]
    ax.legend(handles=legend_elements, fontsize=8,
              loc="upper right", bbox_to_anchor=(1.0, -0.12),
              frameon=True)

    ax.set_title("Black border = best method per row\nDashed line separates tumor classes from no-tumor",
                 fontsize=9, pad=6)

    plt.tight_layout()
    out = os.path.join(out_dir, "class_method_heatmap.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_radar(summary_stats, out_dir):
    """Spider chart: one polygon per method across 5 metrics."""
    os.makedirs(out_dir, exist_ok=True)

    axes_labels = ["Dice", "IoU", "Precision", "Recall", "Pixel\nAccuracy"]
    keys_m      = ["dice", "iou", "precision", "recall", "pixel_acc"]
    N           = len(axes_labels)
    angles      = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles     += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    for method in METHODS:
        vals = [summary_stats.get(method, {}).get(k, 0) for k in keys_m]
        vals += vals[:1]
        color = METHOD_COLORS[method]
        ax.plot(angles, vals, lw=2, color=color, label=METHODS[method])
        ax.fill(angles, vals, alpha=0.10, color=color)
        ax.scatter(angles[:-1], vals[:-1], s=50, color=color, zorder=5)

    ax.legend(loc="lower left", bbox_to_anchor=(-0.35, -0.15),
              fontsize=9, frameon=True, title="Method", title_fontsize=9)
    ax.set_title("Method performance radar\n(averaged over all GT pairs)",
                 fontsize=11, fontweight="bold", pad=20)

    plt.tight_layout()
    out = os.path.join(out_dir, "radar_chart.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ─────────────────────────────────────────────
# FIGURE 4: Pipeline timing chart
# ─────────────────────────────────────────────
def plot_timing(timings, out_dir):
    """Horizontal bar chart of time spent in each phase."""
    if not timings:
        return None

    os.makedirs(out_dir, exist_ok=True)
    labels = list(timings.keys())
    values = list(timings.values())
    colors_t = ["#534AB7","#1D9E75","#0F6E56","#BA7517","#E24B4A",
                "#D85A30","#888780","#378ADD"]

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.7)))
    bars = ax.barh(labels, values,
                   color=colors_t[:len(labels)], alpha=0.85, height=0.55)
    for bar, v in zip(bars, values):
        ax.text(v + max(values)*0.01,
                bar.get_y() + bar.get_height()/2,
                f"{v:.1f}s", va="center", fontsize=10)
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_title("Pipeline phase runtimes", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    out = os.path.join(out_dir, "pipeline_timing.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ─────────────────────────────────────────────
# TERMINAL REPORT
# ─────────────────────────────────────────────
def print_final_report(summary_stats, eval_data, timings, n_images):
    """Print a complete project summary to the terminal."""
    print(f"\n{'═'*70}")
    print("  PHASE 8 — FINAL PROJECT REPORT")
    print("  Medical Image Boundary Detection — Automated Segmentation")
    print(f"{'═'*70}")

    # Dataset summary
    class_counts = {}
    for pair in eval_data:
        for c in ALL_CLASSES:
            if pair.startswith(c):
                class_counts[c] = class_counts.get(c, 0) + 1
    print(f"\n  DATASET")
    print(f"  {'Images evaluated':30s}: {sum(class_counts.values())}")
    for cls, cnt in sorted(class_counts.items()):
        print(f"  {'  ' + cls:30s}: {cnt}")

    # Method performance table
    methods = list(METHODS.keys())
    print(f"\n  {'─'*70}")
    print(f"  METHOD PERFORMANCE (averaged over tumor-class pairs)")
    print(f"  {'─'*70}")
    print(f"  {'Method':<22} {'Dice':>7} {'IoU':>7} "
          f"{'Prec':>7} {'Recall':>8} {'HD95(px)':>10}")
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*10}")

    best = {}
    for key in ["dice","iou","precision","recall","hausdorff_95"]:
        vals = {m: summary_stats.get(m, {}).get(key, 0) for m in methods}
        best[key] = (min(vals, key=vals.get) if key == "hausdorff_95"
                     else max(vals, key=vals.get))

    for m in methods:
        s = summary_stats.get(m, {})
        star = lambda k: " ★" if best.get(k) == m else "  "
        print(f"  {METHODS[m]:<22} "
              f"{s.get('dice',0):>7.3f}{star('dice')} "
              f"{s.get('iou',0):>7.3f}{star('iou')} "
              f"{s.get('precision',0):>7.3f} "
              f"{s.get('recall',0):>8.3f} "
              f"{s.get('hausdorff_95',0):>10.1f}{star('hausdorff_95')}")

    # Recommendations
    best_dice   = best.get("dice")
    best_recall = best.get("recall")
    print(f"\n  {'─'*70}")
    print(f"  RECOMMENDATIONS")
    print(f"  {'─'*70}")
    print(f"  Best overall accuracy : {METHODS.get(best_dice,'?')} "
          f"(Dice={summary_stats.get(best_dice,{}).get('dice',0):.3f})")
    print(f"  Best tumor sensitivity: {METHODS.get(best_recall,'?')} "
          f"(Recall={summary_stats.get(best_recall,{}).get('recall',0):.3f})")
    print(f"  Most interpretable    : Split & Merge (quadtree grid visible)")
    print(f"  Fastest to run        : Split & Merge (no iterative solver)")
    print(f"  Works without edges   : Level Sets Chan-Vese (region-based)")

    # Quality thresholds
    print(f"\n  {'─'*70}")
    print(f"  QUALITY THRESHOLDS (clinical benchmark)")
    for m in methods:
        dice_val = summary_stats.get(m, {}).get("dice", 0)
        iou_val  = summary_stats.get(m, {}).get("iou",  0)
        hd_val   = summary_stats.get(m, {}).get("hausdorff_95", 999)
        dice_ok  = "✓" if dice_val >= 0.7  else "✗"
        iou_ok   = "✓" if iou_val  >= 0.5  else "✗"
        hd_ok    = "✓" if 0 < hd_val <= 10 else "✗"
        print(f"  {METHODS[m]:<22}  "
              f"Dice≥0.7:{dice_ok}  IoU≥0.5:{iou_ok}  HD95≤10px:{hd_ok}")

    # Timings
    if timings:
        total = sum(timings.values())
        print(f"\n  {'─'*70}")
        print(f"  PIPELINE TIMING")
        for phase, t in timings.items():
            print(f"  {phase:<35s}: {t:>6.1f}s")
        print(f"  {'Total':35s}: {total:>6.1f}s")

    # Output files
    print(f"\n  {'─'*70}")
    print(f"  OUTPUT FILES")
    for folder in ["results/comparison", "results/evaluation",
                   "results/active_contours", "results/level_sets",
                   "results/mean_shift", "results/split_merge",
                   "results/report"]:
        p = Path(folder)
        if p.exists():
            pngs = list(p.glob("*.png"))
            csvs = list(p.glob("*.csv"))
            if pngs or csvs:
                print(f"  {folder:<35s}: {len(pngs)} PNGs  {len(csvs)} CSVs")
    print(f"{'═'*70}\n")


# ─────────────────────────────────────────────
# SAVE JSON SUMMARY
# ─────────────────────────────────────────────
def save_json_summary(summary_stats, eval_data, timings, out_dir):
    """Save machine-readable summary for downstream use."""
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "methods":         METHODS,
        "summary_stats":   summary_stats,
        "phase_timings_s": timings,
        "n_pairs":         len(eval_data),
    }
    path = os.path.join(out_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  JSON summary: {path}")
    return path


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Phase 8: Full pipeline runner + report"
    )
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing (images already in npy/)")
    parser.add_argument("--report-only", action="store_true",
                        help="Only generate report from existing results")
    parser.add_argument("--n", type=int, default=4,
                        help="Images per class (default: 4)")
    args = parser.parse_args()

    print("\n" + "═"*60)
    print("  MEDICAL IMAGE SEGMENTATION — FULL PIPELINE")
    print("  Phase 8: Main Runner + Final Report")
    print("═"*60)

    os.makedirs(REPORT_DIR, exist_ok=True)
    timings = {}

    if not args.report_only:

        # ── Phase 1: Preprocessing ───────────────────────────────
        if not args.skip_preprocess:
            _, t = run_phase("Phase 1 — Preprocessing", phase1_preprocess, IMAGES_DIR)
            timings["Phase 1 — Preprocessing"] = t
        else:
            print("\n  Phase 1 skipped (--skip-preprocess)")

        # ── Get balanced file list ────────────────────────────────
        npy_files = get_balanced_npy_files(NPY_DIR, per_class=args.n)
        if not npy_files:
            print("\n  ERROR: No image .npy files found.")
            print("  Run: python download_dataset.py && python preprocessing.py")
            sys.exit(1)
        print(f"\n  Selected {len(npy_files)} images for segmentation")

        # ── Phases 2–5: Segmentation ──────────────────────────────
        _, t = run_phase("Phases 2–5 — All Segmentation Methods",
                         phase2to5_segment, npy_files)
        timings["Phases 2–5 — Segmentation"] = t

        # ── Phase 6: Comparison ───────────────────────────────────
        _, t = run_phase("Phase 6 — Method Comparison",
                         phase6_compare, npy_files)
        timings["Phase 6 — Comparison"] = t

        # ── Phase 7: Evaluation ───────────────────────────────────
        eval_data_raw, t = run_phase("Phase 7 — Accuracy Evaluation",
                                      phase7_evaluate, len(npy_files) * 2)
        timings["Phase 7 — Evaluation"] = t

    # ── Phase 8: Report ───────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  Phase 8 — Generating Final Report")
    print(f"{'═'*60}")
    t0 = time.perf_counter()

    # Load evaluation results from CSV
    eval_data = load_evaluation_csv()
    if not eval_data:
        print("  WARNING: No evaluation CSV found.")
        print("  Run python evaluation.py --csv first.")
        eval_data = {}

    # Compute summary stats
    summary_stats = compute_summary_stats(eval_data)

    # Generate all report figures
    print("\n  Generating figures...")
    plot_final_summary(summary_stats, REPORT_DIR)
    plot_class_heatmap(eval_data, REPORT_DIR)
    plot_radar(summary_stats, REPORT_DIR)
    if timings:
        plot_timing(timings, REPORT_DIR)

    # Save JSON
    save_json_summary(summary_stats, eval_data, timings, REPORT_DIR)

    # Print terminal report
    n_images = sum(1 for _ in Path(NPY_DIR).glob("*.npy")
                   if "_preview" not in _.name and not _.name.startswith("mask_"))
    print_final_report(summary_stats, eval_data, timings, n_images)

    timings["Phase 8 — Report"] = time.perf_counter() - t0

    print("  All done. Results in results/")
    print("  Report  : results/report/")
    print()


if __name__ == "__main__":
    main()
