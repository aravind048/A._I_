"""
PHASE 4 — methods/mean_shift.py
=================================
Mean Shift segmentation:
  Step A — cv2.pyrMeanShiftFiltering  (flatten image into colour blobs)
  Step B — K-means (K=4) on the filtered output (label intensity clusters)
  Step C — Select tumor: brightest cluster weighted by area × centrality

KEY FIX — centrality-weighted scoring:
  Pure brightness picks tiny skull-rim hotspots over the larger tumor.
  Multiplying by area × centrality ensures the central tumour blob wins.

KEY FIX — K=4 clusters:
  K=3 lumps all bright tissue into one cluster (19% of image).
  K=4 separates it into bright-tissue vs tumour-blob (~3%).

Run:
    python methods/mean_shift.py
    python methods/mean_shift.py --all
"""

import os, sys, argparse, numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.color import label2rgb

NPY_DIR    = "results/preprocessed/npy"
OUTPUT_DIR = "results/mean_shift"

SP          = 21    # spatial window radius
SR          = 51    # colour range radius
K_CLUSTERS  = 4     # K=4 separates: background / tissue / bright-tissue / tumour
WATERSHED_MIN_DIST = 15


def load_image(path):
    img = np.load(path).astype(np.float32)
    if img.ndim == 3: img = img[:,:,0]
    img_u8  = (img*255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    return img, img_u8, img_bgr

def run_mean_shift_filter(img_bgr):
    shifted      = cv2.pyrMeanShiftFiltering(img_bgr, sp=SP, sr=SR)
    shifted_gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    return shifted, shifted_gray

def run_kmeans(shifted_gray, K=K_CLUSTERS):
    Z = shifted_gray.reshape(-1,1).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    _, lf, centers = cv2.kmeans(Z, K, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    H, W = shifted_gray.shape
    sorted_idx = np.argsort(centers.flatten())
    remap = np.zeros(K, dtype=np.uint8)
    for new_lbl, old_lbl in enumerate(sorted_idx):
        remap[old_lbl] = new_lbl
    label_map = remap[lf.reshape(H,W)]
    # Brightest cluster = tumour candidate
    tumor_mask = np.where(label_map==(K-1), 255, 0).astype(np.uint8)
    return label_map, tumor_mask

def run_watershed(shifted_bgr, shifted_gray):
    _,fg = cv2.threshold(shifted_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    dist = ndi.distance_transform_edt(fg)
    coords = peak_local_max(dist, min_distance=WATERSHED_MIN_DIST, labels=fg.astype(bool))
    sm = np.zeros(dist.shape, bool)
    if coords.shape[0]>0: sm[tuple(coords.T)]=True
    markers,_ = ndi.label(sm)
    ws = watershed(-dist, markers, mask=fg.astype(bool))
    bnd = np.zeros_like(ws, bool)
    bnd[:-1,:]|=(ws[:-1,:]!=ws[1:,:]); bnd[:,:-1]|=(ws[:,:-1]!=ws[:,1:])
    return ws, bnd

def find_tumor_region(tumor_mask_raw, img_gray, min_area_pct=0.5, max_area_pct=40.0):
    """
    From the brightest K-means cluster, pick the single best region.

    Scoring = mean_intensity × area × centrality
    Centrality penalises regions far from the image centre.
    This prevents tiny peripheral skull hotspots from winning over
    the larger, central tumour blob.
    """
    total = img_gray.size
    min_px, max_px = total*min_area_pct/100, total*max_area_pct/100
    H, W = img_gray.shape
    cx, cy = W/2.0, H/2.0

    n, cc, stats, _ = cv2.connectedComponentsWithStats(tumor_mask_raw, connectivity=8)
    best_mask, best_score = None, -1

    for lbl in range(1, n):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if not (min_px < area < max_px):
            continue
        reg   = cc==lbl
        mu    = float(img_gray[reg].mean())
        bx    = stats[lbl,cv2.CC_STAT_LEFT]+stats[lbl,cv2.CC_STAT_WIDTH]/2
        by_   = stats[lbl,cv2.CC_STAT_TOP]+stats[lbl,cv2.CC_STAT_HEIGHT]/2
        dist  = ((bx-cx)**2+(by_-cy)**2)**0.5
        centr = 1.0/(1.0+dist/60.0)
        score = mu * area * centr
        if score > best_score:
            best_score = score
            best_mask  = (reg.astype(np.uint8)*255)

    return best_mask if best_mask is not None else np.zeros_like(tumor_mask_raw)

def compute_metrics(mask, img_gray, ws_labels):
    m    = mask>0
    area = int(m.sum())
    pct  = round(100*area/mask.size,2)
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    perim=sum(cv2.arcLength(c,True) for c in cnts)
    compact=round(4*np.pi*area/perim**2,4) if perim>0 else 0
    mu_in = float(img_gray[m].mean()) if m.sum()>0 else 0
    mu_out= float(img_gray[~m].mean()) if (~m).sum()>0 else 0
    contrast=round(mu_in/mu_out,3) if mu_out>0 else 0
    return {"area_pct":pct,"perimeter_px":round(perim,1),"compactness":compact,
            "contrast_ratio":contrast,"n_ws_segments":int(ws_labels.max())}

def visualize(img_f, shifted_gray, label_map, ws_labels, tumor_mask, ws_bnd, name, out_dir=OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    img_u8 = (img_f*255).astype(np.uint8)
    fig,axes=plt.subplots(2,3,figsize=(18,11))
    fig.suptitle(f"Mean Shift — {name}", fontsize=12, fontweight="bold")

    axes[0][0].imshow(img_f,cmap="gray"); axes[0][0].set_title("1. Original"); axes[0][0].axis("off")
    axes[0][1].imshow(shifted_gray,cmap="gray")
    axes[0][1].set_title(f"2. After mean shift\n{len(np.unique(shifted_gray))} unique intensities"); axes[0][1].axis("off")

    K=label_map.max()+1
    cmap=plt.cm.Set1(np.linspace(0,0.8,K))
    rgb=np.zeros((*label_map.shape,3),np.float32)
    for k in range(K): rgb[label_map==k]=cmap[k][:3]
    axes[0][2].imshow(rgb); axes[0][2].set_title(f"3. K-means (K={K})"); axes[0][2].axis("off")

    if ws_labels.max()>0:
        ws_rgb=label2rgb(ws_labels,image=img_f,kind="avg",bg_label=0)
        ws_rgb[ws_bnd]=[0,0,0]
        axes[1][0].imshow(np.clip(ws_rgb,0,1))
    else:
        axes[1][0].imshow(img_f,cmap="gray")
    axes[1][0].set_title(f"4. Watershed ({ws_labels.max()} segs)"); axes[1][0].axis("off")

    ov=np.stack([img_f]*3,axis=-1).copy()
    m=tumor_mask>0
    ov[m,0]=ov[m,0]*0.3+0.6; ov[m,1]*=0.3; ov[m,2]*=0.3
    bnd=cv2.Canny(tumor_mask,10,200)>0; ov[bnd]=[1,0.9,0]
    axes[1][1].imshow(np.clip(ov,0,1))
    axes[1][1].set_title(f"5. Detected region\narea={100*m.mean():.1f}%"); axes[1][1].axis("off")

    ax=axes[1][2]
    ax.hist(img_u8.flatten(),bins=64,color="#888780",alpha=0.6,density=True,label="Before")
    ax.hist(shifted_gray.flatten(),bins=64,color="#534AB7",alpha=0.8,density=True,label="After")
    ax.set_title("6. Histogram before/after"); ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out=os.path.join(out_dir,f"{name}_meanshift.png")
    plt.savefig(out,dpi=120,bbox_inches="tight"); plt.close(fig)

def run_on_image(npy_path):
    name = Path(npy_path).stem
    img_f, img_u8, img_bgr = load_image(npy_path)
    shifted, shifted_gray   = run_mean_shift_filter(img_bgr)
    label_map, tm           = run_kmeans(shifted_gray)
    ws_labels, ws_bnd       = run_watershed(shifted, shifted_gray)
    tumor_mask              = find_tumor_region(tm, img_u8)
    m                       = compute_metrics(tumor_mask, img_u8, ws_labels)
    visualize(img_f, shifted_gray, label_map, ws_labels, tumor_mask, ws_bnd, name)
    print(f"  {name}: area={m['area_pct']}%  contrast={m['contrast_ratio']}  compact={m['compactness']}")
    return {"name":name,"image":img_f,"shifted":shifted_gray,"label_map":label_map,
            "tumor_mask":tumor_mask,"ws_labels":ws_labels,"metrics":m}

def run_all(npy_dir=NPY_DIR, max_images=20):
    files=[f for f in sorted(Path(npy_dir).glob("*.npy")) if not f.name.endswith("_preview.npy")][:max_images]
    if not files: print(f"ERROR: No .npy in {npy_dir}"); sys.exit(1)
    return [run_on_image(str(f)) for f in files]

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--image",default=None)
    parser.add_argument("--all",action="store_true")
    args=parser.parse_args()
    print("\n PHASE 4 — MEAN SHIFT"); print("="*50)
    if args.image: run_on_image(args.image)
    elif args.all: run_all()
    else:
        files=[f for f in sorted(Path(NPY_DIR).glob("*.npy")) if not f.name.endswith("_preview.npy")]
        if not files: print(f"ERROR: No .npy in {NPY_DIR}"); sys.exit(1)
        run_on_image(str(files[0]))
    print("\n  Next: python methods/split_merge.py\n")
