"""
PHASE 5 — methods/split_merge.py
==================================
Split and Merge segmentation built from scratch using a quadtree.

SPLIT: recursively divide regions with variance > threshold
MERGE: union-find merging of adjacent leaves with similar mean intensity
SELECT: pick the best tumor candidate using brightness × area × centrality

KEY FIX — centrality-weighted scoring:
  Pure brightness picks tiny bright skull spots over the tumour.
  score = mean_intensity × area × centrality prevents this.

Run:
    python methods/split_merge.py
    python methods/split_merge.py --all
"""

import os, sys, argparse, numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import deque

NPY_DIR    = "results/preprocessed/npy"
OUTPUT_DIR = "results/split_merge"

SPLIT_VAR_THRESH = 0.005
MIN_REGION_SIZE  = 8
MAX_DEPTH        = 6
MERGE_MEAN_THRESH= 0.08


class QuadNode:
    __slots__ = ('r','c','h','w','depth','mean','var','children','parent','label')
    def __init__(self,r,c,h,w,depth,mean,var):
        self.r=r; self.c=c; self.h=h; self.w=w
        self.depth=depth; self.mean=mean; self.var=var
        self.children=[]; self.parent=None; self.label=-1
    @property
    def is_leaf(self): return len(self.children)==0
    @property
    def area(self): return self.h*self.w


def load_image(path):
    img = np.load(path).astype(np.float32)
    if img.ndim==3: img=img[:,:,0]
    return img

def split(img, var_thresh=SPLIT_VAR_THRESH, min_size=MIN_REGION_SIZE, max_depth=MAX_DEPTH):
    H, W = img.shape
    def _rec(r,c,h,w,depth,parent):
        patch = img[r:r+h,c:c+w]
        node  = QuadNode(r,c,h,w,depth,float(patch.mean()),float(patch.var()))
        node.parent = parent
        if patch.var()>var_thresh and h>min_size and w>min_size and depth<max_depth:
            hh,hw = h//2, w//2
            node.children = [
                _rec(r,   c,   hh,   hw,   depth+1,node),
                _rec(r,   c+hw,hh,   w-hw, depth+1,node),
                _rec(r+hh,c,   h-hh, hw,   depth+1,node),
                _rec(r+hh,c+hw,h-hh, w-hw, depth+1,node),
            ]
        return node
    root   = _rec(0,0,H,W,0,None)
    leaves = []
    q = deque([root])
    while q:
        n=q.popleft()
        if n.is_leaf: leaves.append(n)
        else: q.extend(n.children)
    return root, leaves

def merge(img, leaves, thresh=MERGE_MEAN_THRESH):
    N = len(leaves)
    uf = list(range(N))
    def find(i):
        while uf[i]!=i: uf[i]=uf[uf[i]]; i=uf[i]
        return i
    def union(i,j):
        ri,rj=find(i),find(j)
        if ri!=rj: uf[ri]=rj
    for i in range(N):
        A=leaves[i]; A_r2=A.r+A.h; A_c2=A.c+A.w
        for j in range(i+1,N):
            B=leaves[j]; B_r2=B.r+B.h; B_c2=B.c+B.w
            horiz=(A_c2==B.c or B_c2==A.c) and (A.r<B_r2 and B.r<A_r2)
            vert =(A_r2==B.r or B_r2==A.r) and (A.c<B_c2 and B.c<A_c2)
            if (horiz or vert) and abs(A.mean-B.mean)<thresh:
                union(i,j)
    root_to_lbl={}; nxt=[0]
    for i,node in enumerate(leaves):
        rid=find(i)
        if rid not in root_to_lbl: root_to_lbl[rid]=nxt[0]; nxt[0]+=1
        node.label=root_to_lbl[rid]
    lmap=np.full(img.shape,-1,dtype=np.int32)
    for node in leaves:
        lmap[node.r:node.r+node.h,node.c:node.c+node.w]=node.label
    return nxt[0], lmap

def find_tumor_segment(label_map, img, min_area_pct=0.5, max_area_pct=40.0):
    """
    Select the tumour segment using brightness × area × centrality score.
    Prevents tiny skull hotspots from beating the larger tumour blob.
    """
    total = img.size
    min_px, max_px = total*min_area_pct/100, total*max_area_pct/100
    H, W = img.shape
    cx, cy = W/2.0, H/2.0

    best_lbl, best_score = -1, -1
    for lbl in np.unique(label_map[label_map>=0]):
        mask  = label_map==lbl
        area  = int(mask.sum())
        if not (min_px<area<max_px): continue
        mu    = float(img[mask].mean())
        rows,cols = np.where(mask)
        dist  = ((cols.mean()-cx)**2+(rows.mean()-cy)**2)**0.5
        centr = 1.0/(1.0+dist/60.0)
        score = mu*area*centr
        if score>best_score: best_score=score; best_lbl=lbl

    tumor_mask = np.where(label_map==best_lbl,255,0).astype(np.uint8) if best_lbl>=0 \
                 else np.zeros(img.shape,np.uint8)
    return tumor_mask, best_lbl

def draw_qt_grid(img, leaves):
    ov = np.stack([img]*3,axis=-1).copy()
    cmap = plt.cm.plasma
    max_d = max(n.depth for n in leaves) if leaves else 1
    for node in leaves:
        t = node.depth/max(max_d,1)
        col = cmap(t)[:3]
        r,c,h,w = node.r,node.c,node.h,node.w
        ov[r,c:c+w]=ov[r+h-1,c:c+w]=ov[r:r+h,c]=ov[r:r+h,c+w-1]=col
    return np.clip(ov,0,1)

def compute_metrics(mask, img, n_segs, n_leaves):
    m    = mask>0; area=int(m.sum()); pct=round(100*area/mask.size,2)
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    perim=sum(cv2.arcLength(c,True) for c in cnts)
    compact=round(4*np.pi*area/perim**2,4) if perim>0 else 0
    mu_in=float(img[m].mean()) if m.sum()>0 else 0
    mu_out=float(img[~m].mean()) if (~m).sum()>0 else 0
    contrast=round(mu_in/mu_out,3) if mu_out>0 else 0
    return {"area_pct":pct,"perimeter_px":round(perim,1),"compactness":compact,
            "contrast_ratio":contrast,"n_segments":n_segs,"n_leaves":n_leaves}

def visualize(img, leaves, label_map, tumor_mask, n_segs, name, out_dir=OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(18,11))
    fig.suptitle(f"Split & Merge — {name}", fontsize=12, fontweight="bold")

    axes[0][0].imshow(img,cmap="gray"); axes[0][0].set_title("1. Original"); axes[0][0].axis("off")
    axes[0][1].imshow(draw_qt_grid(img,leaves)); axes[0][1].set_title(f"2. Quadtree ({len(leaves)} leaves)"); axes[0][1].axis("off")

    mean_map=np.zeros_like(img)
    for node in leaves: mean_map[node.r:node.r+node.h,node.c:node.c+node.w]=node.mean
    im=axes[0][2].imshow(mean_map,cmap="hot",vmin=0,vmax=1)
    axes[0][2].set_title("3. Leaf mean intensity"); axes[0][2].axis("off")
    plt.colorbar(im,ax=axes[0][2],fraction=0.046,pad=0.04)

    K=int(label_map.max())+1
    try: cmap=plt.colormaps["tab20"].resampled(max(K,2))
    except: cmap=plt.cm.get_cmap("tab20",max(K,2))
    seg_rgb=np.zeros((*img.shape,3),np.float32)
    for lbl in range(K):
        m=(label_map==lbl)
        if m.sum()>0: seg_rgb[m]=cmap(lbl%20)[:3]
    axes[1][0].imshow(np.clip(seg_rgb*0.7+np.stack([img]*3,axis=-1)*0.3,0,1))
    axes[1][0].set_title(f"4. Segments after merge ({n_segs})"); axes[1][0].axis("off")

    ov=np.stack([img]*3,axis=-1).copy(); m=tumor_mask>0
    ov[m,0]=ov[m,0]*0.3+0.6; ov[m,1]*=0.3; ov[m,2]*=0.3
    bnd=cv2.Canny(tumor_mask,10,200)>0; ov[bnd]=[1,0.9,0]
    axes[1][1].imshow(np.clip(ov,0,1))
    axes[1][1].set_title(f"5. Tumour region\narea={100*m.mean():.1f}%"); axes[1][1].axis("off")

    areas=[n.area for n in leaves]; means=[n.mean for n in leaves]
    sc=axes[1][2].scatter(areas,means,c=[n.depth for n in leaves],cmap="plasma",alpha=0.7,s=20)
    plt.colorbar(sc,ax=axes[1][2],label="Depth")
    axes[1][2].set_xlabel("Leaf area (px²)"); axes[1][2].set_ylabel("Mean intensity")
    axes[1][2].set_title("6. Area vs intensity\n(small+bright = tumour)")
    axes[1][2].spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out=os.path.join(out_dir,f"{name}_splitmerge.png")
    plt.savefig(out,dpi=120,bbox_inches="tight"); plt.close(fig)

def run_on_image(npy_path):
    name    = Path(npy_path).stem
    img     = load_image(npy_path)
    _,leaves= split(img)
    n_segs,lmap = merge(img,leaves)
    tumor_mask,_ = find_tumor_segment(lmap,img)
    m       = compute_metrics(tumor_mask,img,n_segs,len(leaves))
    visualize(img,leaves,lmap,tumor_mask,n_segs,name)
    print(f"  {name}: area={m['area_pct']}%  segs={m['n_segments']}  contrast={m['contrast_ratio']}")
    return {"name":name,"image":img,"leaves":leaves,"label_map":lmap,
            "tumor_mask":tumor_mask,"metrics":m}

def run_all(npy_dir=NPY_DIR, max_images=20):
    files=[f for f in sorted(Path(npy_dir).glob("*.npy")) if not f.name.endswith("_preview.npy")][:max_images]
    if not files: print(f"ERROR: No .npy in {npy_dir}"); sys.exit(1)
    return [run_on_image(str(f)) for f in files]

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--image",default=None)
    parser.add_argument("--all",action="store_true")
    args=parser.parse_args()
    print("\n PHASE 5 — SPLIT AND MERGE"); print("="*50)
    if args.image: run_on_image(args.image)
    elif args.all: run_all()
    else:
        files=[f for f in sorted(Path(NPY_DIR).glob("*.npy")) if not f.name.endswith("_preview.npy")]
        if not files: print(f"ERROR: No .npy in {NPY_DIR}"); sys.exit(1)
        run_on_image(str(files[0]))
    print("\n  Next: python compare.py\n")
