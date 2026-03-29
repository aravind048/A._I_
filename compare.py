"""
PHASE 6 — compare.py
======================
Runs all 4 methods on every preprocessed image and produces:
  1. Side-by-side grid per image (+ agreement map)
  2. Boundary overlay (all 5 methods on one image)
  3. Metrics bar chart (area%, compactness, Hausdorff)
  4. Master grid poster (all images × all methods)

Run:
    python compare.py
    python compare.py --all
    python compare.py --timing
"""

import os, sys, argparse, time, numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.path.insert(0, ".")
from utils                   import get_balanced_npy_files
from methods.active_contours import run_on_image as run_snake
from methods.level_sets      import run_on_image as run_ls
from methods.mean_shift      import run_on_image as run_ms
from methods.split_merge     import run_on_image as run_sm

NPY_DIR    = "results/preprocessed/npy"
OUTPUT_DIR = "results/comparison"
MAX_IMAGES = 8

METHODS = {
    "snake":      {"label":"Active Contours",  "color":"#E24B4A"},
    "chan_vese":  {"label":"Level Sets (CV)",   "color":"#1D9E75"},
    "geodesic":  {"label":"Level Sets (GAC)",  "color":"#0F6E56"},
    "mean_shift":{"label":"Mean Shift",         "color":"#534AB7"},
    "split_merge":{"label":"Split & Merge",     "color":"#BA7517"},
}
TINTS = {
    "snake":      (0.55,0,0),
    "chan_vese":  (0,0.45,0),
    "geodesic":   (0,0.35,0),
    "mean_shift": (0.25,0,0.5),
    "split_merge":(0.55,0.35,0),
}


def run_all_methods(npy_path, measure_time=False):
    name = Path(npy_path).stem
    masks, metrics, times = {}, {}, {}
    def _run(fn, key, *a):
        t0=time.perf_counter(); r=fn(*a); dt=time.perf_counter()-t0
        times[key]=round(dt,2); return r
    r1 = (_run(run_snake,"snake",npy_path) if measure_time else run_snake(npy_path))
    r2 = (_run(run_ls,"chan_vese",npy_path) if measure_time else run_ls(npy_path))
    r3 = (_run(run_ms,"mean_shift",npy_path) if measure_time else run_ms(npy_path))
    r4 = (_run(run_sm,"split_merge",npy_path) if measure_time else run_sm(npy_path))
    masks["snake"]       = r1["mask"]
    masks["chan_vese"]   = r2["mask_cv"]
    masks["geodesic"]   = r2["mask_gac"]
    masks["mean_shift"]  = r3["tumor_mask"]
    masks["split_merge"] = r4["tumor_mask"]
    img = r1["image"]
    for key in ["snake","mean_shift","split_merge"]:
        metrics[key] = (r1 if key=="snake" else r3 if key=="mean_shift" else r4)["metrics"]
    metrics["chan_vese"] = {"area_pct":r2["metrics"]["cv"]["area_pct"],
                            "compactness":r2["metrics"]["cv"]["compactness"],
                            "perimeter_px":r2["metrics"]["cv"]["perimeter_px"]}
    metrics["geodesic"]  = {"area_pct":r2["metrics"]["gac"]["area_pct"],
                            "compactness":r2["metrics"]["gac"]["compactness"],
                            "perimeter_px":r2["metrics"]["gac"]["perimeter_px"]}
    return {"name":name,"image":img,"masks":masks,"metrics":metrics,"times":times}


def _make_overlay(img, mask, key):
    tr,tg,tb = TINTS[key]
    ov = np.stack([img]*3,axis=-1).copy()
    m  = mask>0
    ov[m,0]=ov[m,0]*0.35+tr; ov[m,1]=ov[m,1]*0.35+tg; ov[m,2]=ov[m,2]*0.35+tb
    bnd=cv2.Canny(mask,10,200)>0; ov[bnd]=[1,0.92,0]
    return np.clip(ov,0,1)


def plot_sidebyside(result, out_dir=OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    img=result["image"]; masks=result["masks"]; name=result["name"]
    ncols=len(METHODS)+2
    fig,axes=plt.subplots(1,ncols,figsize=(ncols*3.8,4.4))
    fig.suptitle(f"All Methods — {name}", fontsize=11, fontweight="bold", y=1.01)
    axes[0].imshow(img,cmap="gray"); axes[0].set_title("Original"); axes[0].axis("off")
    for col,key in enumerate(METHODS,start=1):
        ov=_make_overlay(img,masks[key],key)
        axes[col].imshow(ov)
        area=100*(masks[key]>0).mean()
        compact=result["metrics"].get(key,{}).get("compactness",0)
        axes[col].set_title(f"{METHODS[key]['label']}\narea={area:.1f}%",fontsize=8,linespacing=1.4)
        axes[col].text(0.03,0.03,f"c={compact:.3f}",transform=axes[col].transAxes,
                       fontsize=7,color="white",va="bottom",bbox=dict(fc="black",alpha=0.5,boxstyle="round,pad=0.2"))
        axes[col].axis("off")
    agree=sum((masks[k]>0).astype(float) for k in METHODS)/len(METHODS)
    axes[-1].imshow(img,cmap="gray",alpha=0.4)
    im=axes[-1].imshow(agree,cmap="hot",vmin=0,vmax=1,alpha=0.75)
    plt.colorbar(im,ax=axes[-1],fraction=0.046,pad=0.04)
    axes[-1].set_title("Agreement map",fontsize=8); axes[-1].axis("off")
    plt.tight_layout()
    out=os.path.join(out_dir,f"{name}_sidebyside.png")
    plt.savefig(out,dpi=130,bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")


def plot_boundaries(result, out_dir=OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    img=result["image"]; masks=result["masks"]; name=result["name"]
    ov=np.stack([img]*3,axis=-1).copy()
    patches=[]
    for key,info in METHODS.items():
        bnd=cv2.Canny(masks[key],10,200)>0
        bnd2=cv2.dilate(bnd.astype(np.uint8),np.ones((2,2),np.uint8)).astype(bool)
        h=info["color"].lstrip("#")
        rf,gf,bf=int(h[:2],16)/255,int(h[2:4],16)/255,int(h[4:],16)/255
        ov[bnd2]=[rf,gf,bf]
        patches.append(mpatches.Patch(color=info["color"],label=info["label"]))
    fig,ax=plt.subplots(1,1,figsize=(7,7))
    ax.imshow(np.clip(ov,0,1))
    ax.legend(handles=patches,fontsize=8,loc="upper right",framealpha=0.85,
              title="Method",title_fontsize=8)
    ax.set_title(f"Boundary comparison — {name}",fontsize=10,fontweight="bold"); ax.axis("off")
    plt.tight_layout()
    out=os.path.join(out_dir,f"{name}_boundaries.png")
    plt.savefig(out,dpi=140,bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")


def plot_metrics_bars(all_results, out_dir=OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    keys=list(METHODS.keys()); n_m=len(keys); colors=[METHODS[k]["color"] for k in keys]
    names=[r["name"][:12] for r in all_results]; n_i=len(all_results)
    x=np.arange(n_i); bw=0.14
    offsets=np.linspace(-(n_m-1)/2,(n_m-1)/2,n_m)*bw
    fig,axes=plt.subplots(1,3,figsize=(20,6))
    fig.suptitle("Method Comparison Metrics",fontsize=13,fontweight="bold")
    for ax,(metric,ylabel,ref) in zip(axes,[
        ("area_pct","Area (%)",""),("compactness","Compactness",0.7),("perimeter_px","Perimeter (px)","")]):
        for i,(key,off) in enumerate(zip(keys,offsets)):
            vals=[r["metrics"].get(key,{}).get(metric,0) for r in all_results]
            bars=ax.bar(x+off,vals,width=bw,color=colors[i],alpha=0.85,label=METHODS[key]["label"])
            for bar,v in zip(bars,vals):
                if v>0: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,
                                f"{v:.1f}",ha="center",va="bottom",fontsize=6,rotation=45)
        ax.set_xticks(x); ax.set_xticklabels(names,fontsize=7,rotation=15)
        ax.set_ylabel(ylabel,fontsize=9); ax.spines[["top","right"]].set_visible(False)
        if ref: ax.axhline(ref,color="#888780",lw=1,ls="--",alpha=0.7)
    handles=[mpatches.Patch(color=METHODS[k]["color"],label=METHODS[k]["label"]) for k in keys]
    fig.legend(handles=handles,loc="lower center",ncol=n_m,fontsize=8,bbox_to_anchor=(0.5,-0.06))
    plt.tight_layout()
    out=os.path.join(out_dir,"metrics_comparison.png")
    plt.savefig(out,dpi=120,bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")


def plot_master_grid(all_results, out_dir=OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    keys=list(METHODS.keys()); n_i=len(all_results); ncols=len(keys)+1
    fig=plt.figure(figsize=(ncols*2.8,n_i*2.9+0.8))
    gs=gridspec.GridSpec(n_i,ncols,figure=fig,hspace=0.06,wspace=0.04,
                         top=0.93,bottom=0.02,left=0.02,right=0.98)
    for col,(title,color) in enumerate([("Original","#444441")]+
                                        [(METHODS[k]["label"].replace("\n"," "),METHODS[k]["color"])
                                         for k in keys]):
        fig.text((col+0.5)/ncols*0.96+0.02,0.96,title,ha="center",va="center",
                 fontsize=9,fontweight="500",color=color)
    for row,res in enumerate(all_results):
        ax0=fig.add_subplot(gs[row,0])
        ax0.imshow(res["image"],cmap="gray"); ax0.set_ylabel(res["name"][:14],fontsize=7)
        ax0.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
        for col,key in enumerate(keys,start=1):
            ax=fig.add_subplot(gs[row,col])
            ov=_make_overlay(res["image"],res["masks"][key],key)
            ax.imshow(ov)
            ax.text(0.03,0.03,f"{100*(res['masks'][key]>0).mean():.1f}%",
                    transform=ax.transAxes,fontsize=7,color="white",va="bottom",
                    bbox=dict(fc="black",alpha=0.45,boxstyle="round,pad=0.1"))
            ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
            for sp in ax.spines.values():
                sp.set_linewidth(0.4); sp.set_edgecolor(METHODS[key]["color"])
    fig.suptitle("Segmentation — All Methods × All Images",fontsize=12,fontweight="bold",y=0.995)
    out=os.path.join(out_dir,"master_grid.png")
    plt.savefig(out,dpi=140,bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")


def compare(npy_paths, measure_time=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results=[]
    for p in npy_paths:
        print(f"\n  Running all methods: {Path(p).stem}")
        all_results.append(run_all_methods(str(p), measure_time))
    print("\n  Saving figures...")
    for res in all_results:
        plot_sidebyside(res)
        plot_boundaries(res)
    plot_metrics_bars(all_results)
    plot_master_grid(all_results)
    saved=sorted(Path(OUTPUT_DIR).glob("*.png"))
    print(f"\n  {len(saved)} output files in {OUTPUT_DIR}/")
    return all_results


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--all",action="store_true")
    parser.add_argument("--timing",action="store_true")
    parser.add_argument("--n",type=int,default=MAX_IMAGES)
    args=parser.parse_args()
    print("\n PHASE 6 — COMPARISON"); print("="*50)
    files = get_balanced_npy_files(NPY_DIR, per_class=args.n)
    if not files: print(f"ERROR: No image .npy files in {NPY_DIR}"); sys.exit(1)
    compare(files, measure_time=args.timing)
    print("\n  Next: python evaluation.py --csv\n")
