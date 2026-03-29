"""
PHASE 2 — methods/active_contours.py
======================================
Active Contours (Snakes): evolves a circular contour inward until it
snaps onto tumor boundaries driven by edge energy.

Run:
    python methods/active_contours.py
    python methods/active_contours.py --all
"""

import os, sys, argparse, numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from skimage.segmentation import active_contour
from skimage.filters import gaussian

NPY_DIR    = "results/preprocessed/npy"
OUTPUT_DIR = "results/active_contours"

ALPHA   = 0.015   # elasticity  (lower = more stretchy)
BETA    = 10.0    # rigidity    (lower = bends more freely)
W_EDGE  = 1.0     # edge pull   (higher = snaps harder to edges)
N_ITER  = 2500
SIGMA   = 3.0     # pre-smooth sigma
RADIUS_FRAC = 0.38


def load_image(path):
    img = np.load(path).astype(np.float32)
    if img.ndim == 3: img = img[:,:,0]
    return img

def make_initial_contour(img):
    H, W = img.shape
    r = min(H, W) * RADIUS_FRAC
    t = np.linspace(0, 2*np.pi, 400, endpoint=False)
    return np.column_stack([H/2 + r*np.sin(t), W/2 + r*np.cos(t)])

def run_snake(img):
    smooth = gaussian(img, sigma=SIGMA)
    init   = make_initial_contour(img)
    snake  = active_contour(smooth, init, alpha=ALPHA, beta=BETA,
                             w_line=0, w_edge=W_EDGE,
                             max_num_iter=N_ITER, max_px_move=1.0,
                             boundary_condition="periodic")
    return snake, init

def contour_to_mask(snake, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    pts  = np.array([[int(c), int(r)] for r,c in snake], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask

def compute_metrics(mask):
    area_px  = int((mask>0).sum())
    area_pct = round(100*area_px/mask.size, 2)
    cnts,_   = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perim    = sum(cv2.arcLength(c,True) for c in cnts)
    compact  = round(4*np.pi*area_px/perim**2, 4) if perim>0 else 0
    return {"area_pct":area_pct, "perimeter_px":round(perim,1), "compactness":compact}

def visualize(img, init, snake, mask, name, out_dir=OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    # Edge energy
    sm     = gaussian(img, sigma=SIGMA)
    gx     = cv2.Sobel((sm*255).astype(np.uint8), cv2.CV_64F,1,0,ksize=3)
    gy     = cv2.Sobel((sm*255).astype(np.uint8), cv2.CV_64F,0,1,ksize=3)
    energy = np.sqrt(gx**2+gy**2); energy/=(energy.max()+1e-8)
    # Overlay
    ov = np.stack([img]*3, axis=-1).copy()
    m  = mask>0
    ov[m,0]=ov[m,0]*0.4; ov[m,1]=ov[m,1]*0.4+0.4; ov[m,2]=ov[m,2]*0.4
    bnd = cv2.Canny(mask,10,200)>0
    ov_u8 = (ov*255).astype(np.uint8)
    cv2.polylines(ov_u8,[np.array([[int(c),int(r)] for r,c in snake],np.int32)],
                  True,(229,36,74),2)

    fig,axes = plt.subplots(1,4,figsize=(20,5))
    fig.suptitle(f"Active Contours — {name}", fontsize=12, fontweight="bold")
    axes[0].imshow(img,cmap="gray"); axes[0].plot(init[:,1],init[:,0],'--',color="#378ADD",lw=1.5)
    axes[0].set_title("Original + initial contour"); axes[0].axis("off")
    axes[1].imshow(energy,cmap="hot")
    axes[1].set_title("Edge energy field"); axes[1].axis("off")
    axes[2].imshow(img,cmap="gray")
    axes[2].plot(init[:,1],init[:,0],'--',color="#378ADD",lw=1,alpha=0.4)
    axes[2].plot(snake[:,1],snake[:,0],color="#E24B4A",lw=2)
    axes[2].set_title("Final snake boundary"); axes[2].axis("off")
    axes[3].imshow(ov_u8)
    axes[3].set_title(f"Segmented region"); axes[3].axis("off")
    plt.tight_layout()
    out = os.path.join(out_dir, f"{name}_snake.png")
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)

def run_on_image(npy_path):
    name  = Path(npy_path).stem
    img   = load_image(npy_path)
    snake, init = run_snake(img)
    mask  = contour_to_mask(snake, img.shape)
    m     = compute_metrics(mask)
    visualize(img, init, snake, mask, name)
    print(f"  {name}: area={m['area_pct']}%  compact={m['compactness']}")
    return {"name":name,"image":img,"snake":snake,"mask":mask,"metrics":m}

def run_all(npy_dir=NPY_DIR, max_images=20):
    files = sorted(Path(npy_dir).glob("*.npy"))
    files = [f for f in files if not f.name.endswith("_preview.npy")][:max_images]
    if not files:
        print(f"  ERROR: No .npy files in {npy_dir}/"); sys.exit(1)
    results = []
    for f in files:
        results.append(run_on_image(str(f)))
    return results

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    print("\n PHASE 2 — ACTIVE CONTOURS"); print("="*50)
    if args.image:
        run_on_image(args.image)
    elif args.all:
        run_all()
    else:
        files = sorted(Path(NPY_DIR).glob("*.npy"))
        files = [f for f in files if not f.name.endswith("_preview.npy")]
        if not files: print(f"  ERROR: No .npy files in {NPY_DIR}/"); sys.exit(1)
        run_on_image(str(files[0]))
    print("\n  Next: python methods/level_sets.py\n")
