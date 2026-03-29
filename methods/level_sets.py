"""
PHASE 3 — methods/level_sets.py
=================================
Two level-set models:
  Chan-Vese  — region-based  (works even without strong edges)
  Geodesic   — edge-based    (better at sharp, clear boundaries)

Run:
    python methods/level_sets.py
    python methods/level_sets.py --all
"""

import os, sys, argparse, numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from skimage.segmentation import (morphological_chan_vese,
    morphological_geodesic_active_contour,
    checkerboard_level_set, disk_level_set, inverse_gaussian_gradient)
from skimage.filters import gaussian

NPY_DIR    = "results/preprocessed/npy"
OUTPUT_DIR = "results/level_sets"


def load_image(path):
    img = np.load(path).astype(np.float32)
    if img.ndim == 3: img = img[:,:,0]
    return img

def run_chan_vese(img):
    init = checkerboard_level_set(img.shape, square_size=6)
    mask = morphological_chan_vese(img, num_iter=200, init_level_set=init,
                                    smoothing=3, lambda1=1.0, lambda2=1.0)
    if mask.mean() > 0.5: mask = ~mask
    return mask.astype(np.uint8)*255

def run_geodesic(img):
    sm     = gaussian(img, sigma=1.5)
    gimage = inverse_gaussian_gradient(sm, alpha=100, sigma=1.5)
    init   = disk_level_set(img.shape,
                             center=(img.shape[0]//2, img.shape[1]//2),
                             radius=int(min(img.shape)*0.40))
    mask = morphological_geodesic_active_contour(
        gimage, num_iter=300, init_level_set=init,
        smoothing=1, threshold=0.60, balloon=-1)
    return mask.astype(np.uint8)*255

def compute_metrics(mask_cv, mask_gac):
    def _m(mask):
        area = int((mask>0).sum())
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perim  = sum(cv2.arcLength(c,True) for c in cnts)
        compact= round(4*np.pi*area/perim**2,4) if perim>0 else 0
        return {"area_pct":round(100*area/mask.size,2),
                "perimeter_px":round(perim,1), "compactness":compact}
    m1, m2 = _m(mask_cv), _m(mask_gac)
    cv_b  = mask_cv>0; gac_b = mask_gac>0
    inter = (cv_b&gac_b).sum(); union = (cv_b|gac_b).sum()
    iou   = round(float(inter/union),4) if union>0 else 0
    return {"cv":m1, "gac":m2, "inter_method_iou":iou}

def visualize(img, mask_cv, mask_gac, name, out_dir=OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    sm   = gaussian(img, sigma=1.5)
    gimg = inverse_gaussian_gradient(sm, alpha=100, sigma=1.5)

    def overlay(img, mask, color):
        ov = np.stack([img]*3, axis=-1).copy()
        m  = mask>0
        if color=="green":
            ov[m,0]*=0.4; ov[m,1]=ov[m,1]*0.4+0.4; ov[m,2]*=0.4
        else:
            ov[m,0]*=0.4; ov[m,1]*=0.4; ov[m,2]=ov[m,2]*0.4+0.5
        bnd = cv2.Canny(mask,10,200)>0
        ov[bnd]=[1,0.9,0]
        return np.clip(ov,0,1)

    fig,axes=plt.subplots(1,5,figsize=(25,5))
    fig.suptitle(f"Level Sets — {name}", fontsize=12, fontweight="bold")
    axes[0].imshow(img,cmap="gray"); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(gimg,cmap="RdYlGn"); axes[1].set_title("Edge-stopping function g")
    axes[1].axis("off")
    axes[2].imshow(overlay(img,mask_cv,"green"))
    axes[2].set_title(f"Chan-Vese\narea={100*(mask_cv>0).mean():.1f}%"); axes[2].axis("off")
    axes[3].imshow(overlay(img,mask_gac,"blue"))
    axes[3].set_title(f"Geodesic\narea={100*(mask_gac>0).mean():.1f}%"); axes[3].axis("off")
    # Difference map
    diff=np.zeros((*img.shape,3))
    both=(mask_cv>0)&(mask_gac>0); cv_only=(mask_cv>0)&~(mask_gac>0)
    gac_only=(mask_gac>0)&~(mask_cv>0)
    diff[both]=[1,0.9,0]; diff[cv_only]=[0,0.7,0]; diff[gac_only]=[0,0,0.8]
    bg=np.stack([img*0.3]*3,axis=-1)
    axes[4].imshow(np.clip(diff+bg,0,1))
    m1=(mask_cv>0);m2=(mask_gac>0)
    iou=float((m1&m2).sum()/(m1|m2).sum()) if (m1|m2).sum()>0 else 0
    axes[4].set_title(f"Difference\nIoU={iou:.3f}"); axes[4].axis("off")
    plt.tight_layout()
    out=os.path.join(out_dir,f"{name}_levelsets.png")
    plt.savefig(out,dpi=120,bbox_inches="tight"); plt.close(fig)

def run_on_image(npy_path):
    name     = Path(npy_path).stem
    img      = load_image(npy_path)
    mask_cv  = run_chan_vese(img)
    mask_gac = run_geodesic(img)
    m        = compute_metrics(mask_cv, mask_gac)
    visualize(img, mask_cv, mask_gac, name)
    print(f"  {name}: CV area={m['cv']['area_pct']}%  "
          f"GAC area={m['gac']['area_pct']}%  IoU={m['inter_method_iou']}")
    return {"name":name,"image":img,"mask_cv":mask_cv,"mask_gac":mask_gac,"metrics":m}

def run_all(npy_dir=NPY_DIR, max_images=20):
    files = [f for f in sorted(Path(npy_dir).glob("*.npy"))
             if not f.name.endswith("_preview.npy")][:max_images]
    if not files: print(f"ERROR: No .npy in {npy_dir}"); sys.exit(1)
    return [run_on_image(str(f)) for f in files]

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--image",default=None)
    parser.add_argument("--all",action="store_true")
    args=parser.parse_args()
    print("\n PHASE 3 — LEVEL SETS"); print("="*50)
    if args.image: run_on_image(args.image)
    elif args.all: run_all()
    else:
        files=[f for f in sorted(Path(NPY_DIR).glob("*.npy")) if not f.name.endswith("_preview.npy")]
        if not files: print(f"ERROR: No .npy in {NPY_DIR}"); sys.exit(1)
        run_on_image(str(files[0]))
    print("\n  Next: python methods/mean_shift.py\n")
