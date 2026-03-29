"""
PHASE 7 — evaluation.py
========================
Accuracy evaluation against ground truth.

LABELS IN DATASET:
  0 = no_tumor   → GT is all-zeros (false-positive test)
  1 = glioma     → GT auto-generated from preprocessed image
  2 = meningioma → GT auto-generated from preprocessed image
  3 = pituitary  → GT auto-generated from preprocessed image

GT GENERATION (for tumor classes):
  Otsu threshold on preprocessed .npy → erode to remove skull ring
  → keep central connected component = tumour blob.

METRICS:
  Dice, IoU, Pixel Accuracy, Precision, Recall, Hausdorff-95

Run:
    python evaluation.py
    python evaluation.py --n 12
    python evaluation.py --csv
"""

import os, sys, csv, argparse, numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from collections import defaultdict

sys.path.insert(0, ".")
from utils                   import get_balanced_npy_files
from methods.active_contours import run_on_image as run_snake
from methods.level_sets      import run_on_image as run_ls
from methods.mean_shift      import run_on_image as run_ms
from methods.split_merge     import run_on_image as run_sm

NPY_DIR    = "results/preprocessed/npy"
OUTPUT_DIR = "results/evaluation"

TUMOR_CLASSES = ["glioma", "meningioma", "pituitary", "tumor"]

METHODS = {
    "snake":      {"label":"Active Contours", "color":"#E24B4A"},
    "chan_vese":  {"label":"Level Sets (CV)",  "color":"#1D9E75"},
    "geodesic":  {"label":"Level Sets (GAC)", "color":"#0F6E56"},
    "mean_shift":{"label":"Mean Shift",        "color":"#534AB7"},
    "split_merge":{"label":"Split & Merge",    "color":"#BA7517"},
}


# ─────────────────────────────────────────────
# GT generation
# ─────────────────────────────────────────────
def generate_gt_from_npy(npy_path):
    """
    Generate binary GT mask from the preprocessed .npy image.
    Works in the same intensity space as all segmentation methods.

    Algorithm:
      1. Otsu threshold → separates bright (tumour+skull) from dark
      2. Erode ×3 with 5×5 kernel → removes thin skull ring
      3. Keep component with highest (area × centrality) score → tumour
    """
    img = np.load(npy_path).astype(np.float32)
    if img.ndim==3: img=img[:,:,0]
    H,W = img.shape

    img_u8 = (img*255).astype(np.uint8)
    _,bw   = cv2.threshold(img_u8,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel = np.ones((5,5),np.uint8)
    eroded = cv2.erode(bw,kernel,iterations=3)

    n,labels,stats,_ = cv2.connectedComponentsWithStats(eroded)
    if n<=1:
        thresh = float(np.percentile(img,85))
        return (img>thresh).astype(bool)

    cx,cy = W/2.0, H/2.0
    best_lbl, best_score = -1, -1
    for lbl in range(1,n):
        area = stats[lbl,cv2.CC_STAT_AREA]
        if area<30: continue
        bx = stats[lbl,cv2.CC_STAT_LEFT]+stats[lbl,cv2.CC_STAT_WIDTH]/2
        by_= stats[lbl,cv2.CC_STAT_TOP]+stats[lbl,cv2.CC_STAT_HEIGHT]/2
        dist  = ((bx-cx)**2+(by_-cy)**2)**0.5
        score = area*(1.0/(1.0+dist/60.0))
        if score>best_score: best_score=score; best_lbl=lbl

    if best_lbl<0: return None
    return (labels==best_lbl).astype(bool)


# ─────────────────────────────────────────────
# Pair discovery
# ─────────────────────────────────────────────
def discover_pairs(npy_dir=NPY_DIR, max_pairs=12):
    """
    Build evaluation pairs using balanced class sampling.

    Uses get_balanced_npy_files() from utils.py so the same
    balanced logic applies to both compare.py and evaluation.py.

    For each image .npy:
      - tumor classes (glioma/meningioma/pituitary/tumor):
          GT auto-generated from the preprocessed image
          (Otsu + erode to find the central bright blob)
      - no_tumor class:
          GT = all-zeros  (evaluates false-positive rate)
    """
    # Get balanced files — equal quota per class, interleaved
    per_class = max(2, max_pairs // len([c for c in
                    ["glioma","meningioma","no_tumor","pituitary","tumor"]]))
    image_files = get_balanced_npy_files(npy_dir, per_class=per_class)

    if not image_files:
        print("  ERROR: No image .npy files found.")
        print("  Run: python download_dataset.py")
        print("  Then: python preprocessing.py")
        sys.exit(1)

    TUMOR_CLASSES = ["glioma", "meningioma", "pituitary", "tumor"]
    pairs = []

    for npy_path in image_files[:max_pairs]:
        stem = Path(npy_path).stem
        cls  = "_".join(stem.split("_")[:-1])   # e.g. "glioma" from "glioma_0000"

        if cls == "no_tumor":
            # GT = all-zeros for no_tumor images
            img     = np.load(str(npy_path))
            gt_mask = np.zeros(img.shape[:2], dtype=bool)
            gt_type = "no_tumor_zero_gt"
            gt_area = 0.0
        else:
            # GT = auto-generated from preprocessed image
            gt_mask = generate_gt_from_npy(str(npy_path))
            if gt_mask is None:
                print(f"  WARNING: Could not generate GT for {stem} — skipping")
                continue
            gt_area = round(100 * gt_mask.mean(), 2)
            if gt_area < 0.3 or gt_area > 45:
                print(f"  WARNING: GT area {gt_area:.1f}% out of range for {stem} — skipping")
                continue
            gt_type = f"auto_gt_{cls}"

        pairs.append({
            "name":        stem,
            "img_npy":     str(npy_path),
            "gt_mask":     gt_mask,
            "gt_type":     gt_type,
            "gt_area_pct": gt_area,
            "label_class": cls,
        })

    if not pairs:
        print("  ERROR: No valid pairs after GT generation.")
        sys.exit(1)

    # Summary
    from collections import Counter
    class_counts = Counter(p["label_class"] for p in pairs)
    print(f"  Evaluation pairs ready: {len(pairs)}")
    for cls, cnt in sorted(class_counts.items()):
        print(f"    {cls:15s}: {cnt}")
    print()
    for p in pairs:
        print(f"  [{p['gt_type']:22s}] {p['name']:35s} GT={p['gt_area_pct']:.1f}%")
    return pairs

def compute_metrics(pred_u8, gt_bool):
    pred  = pred_u8>0; gt=gt_bool; total=pred.size
    TP=int((pred& gt).sum()); TN=int((~pred&~gt).sum())
    FP=int((pred&~gt).sum()); FN=int((~pred& gt).sum())
    if gt.sum()==0 and pred.sum()==0:
        return {"dice":1.0,"iou":1.0,"pixel_acc":1.0,"precision":1.0,
                "recall":1.0,"f1":1.0,"hausdorff_95":0.0,
                "TP":0,"TN":total,"FP":0,"FN":0}
    if gt.sum()==0 and pred.sum()>0:
        return {"dice":0.0,"iou":0.0,"pixel_acc":round(TN/total,4),
                "precision":0.0,"recall":1.0,"f1":0.0,"hausdorff_95":-1.0,
                "TP":0,"TN":TN,"FP":FP,"FN":0}
    dice=round(2*TP/(2*TP+FP+FN),4) if (2*TP+FP+FN)>0 else 0
    iou =round(TP/(TP+FP+FN),4)     if (TP+FP+FN)>0   else 0
    prec=round(TP/(TP+FP),4)         if (TP+FP)>0      else 0
    rec =round(TP/(TP+FN),4)         if (TP+FN)>0      else 0
    pacc=round((TP+TN)/total,4)
    hd95=_hd95(pred,gt_bool)
    return {"dice":dice,"iou":iou,"pixel_acc":pacc,"precision":prec,
            "recall":rec,"f1":dice,"hausdorff_95":hd95,
            "TP":TP,"TN":TN,"FP":FP,"FN":FN}

def _hd95(pred_bool, gt_bool):
    pb=cv2.Canny(pred_bool.astype(np.uint8)*255,10,200)>0
    gb=cv2.Canny(gt_bool.astype(np.uint8)*255, 10,200)>0
    if pb.sum()==0 or gb.sum()==0: return -1.0
    dg=distance_transform_edt(~gb); dp=distance_transform_edt(~pb)
    return round(max(float(np.percentile(dg[pb],95)),
                     float(np.percentile(dp[gb],95))),2)


# ─────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────
def make_error_map(pred_u8, gt_bool, img):
    pred=pred_u8>0; gt=gt_bool
    rgb=np.stack([img*0.55]*3,axis=-1)
    TP_m=pred& gt; FP_m=pred&~gt; FN_m=~pred&gt
    rgb[TP_m,0]=0.1; rgb[TP_m,1]=0.8;  rgb[TP_m,2]=0.1
    rgb[FP_m,0]=0.85;rgb[FP_m,1]=0.1;  rgb[FP_m,2]=0.1
    rgb[FN_m,0]=0.1; rgb[FN_m,1]=0.1;  rgb[FN_m,2]=0.85
    return np.clip(rgb,0,1)

def plot_error_maps(mri, gt_mask, preds, metrics_all, pair_name, gt_type, out_dir=OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    keys=list(METHODS.keys()); ncols=len(keys)+1
    fig,axes=plt.subplots(1,ncols,figsize=(ncols*3.6,4.8))
    fig.suptitle(f"Accuracy — {pair_name} [{gt_type}]",fontsize=10,fontweight="bold",y=1.02)
    # GT panel
    gt_bnd=cv2.Canny(gt_mask.astype(np.uint8)*255,10,200)>0
    gt_vis=np.stack([mri]*3,axis=-1).copy()
    gt_vis[gt_mask,1]=gt_vis[gt_mask,1]*0.4+0.4
    gt_vis[gt_mask,0]*=0.4; gt_vis[gt_mask,2]*=0.4
    gt_vis[gt_bnd]=[1,1,0]
    axes[0].imshow(np.clip(gt_vis,0,1))
    axes[0].set_title(f"GT\narea={100*gt_mask.mean():.1f}%",fontsize=9); axes[0].axis("off")
    # Method panels
    for col,key in enumerate(keys,start=1):
        m=metrics_all[key]
        emap=make_error_map(preds[key],gt_mask,mri)
        axes[col].imshow(emap)
        axes[col].set_title(f"{METHODS[key]['label']}\nDice={m['dice']:.3f}  IoU={m['iou']:.3f}",
                             fontsize=8,linespacing=1.5)
        axes[col].text(0.03,0.03,f"P={m['precision']:.2f}\nR={m['recall']:.2f}\nHD95={m['hausdorff_95']:.1f}",
                       transform=axes[col].transAxes,fontsize=6.5,color="white",va="bottom",
                       bbox=dict(fc="black",alpha=0.55,boxstyle="round,pad=0.2"))
        axes[col].axis("off")
    axes[-1].legend(handles=[
        mpatches.Patch(color="#1DBF1D",label="TP"),
        mpatches.Patch(color="#D91616",label="FP"),
        mpatches.Patch(color="#1616DB",label="FN"),
    ],fontsize=7,loc="lower right",framealpha=0.85)
    plt.tight_layout()
    out=os.path.join(out_dir,f"{pair_name[:40]}_errors.png")
    plt.savefig(out,dpi=130,bbox_inches="tight"); plt.close(fig)
    print(f"  Error map: {out}")

def plot_radar(all_metrics, out_dir=OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    axes_labels=["Dice","IoU","Precision","Recall","Pixel\nAccuracy"]
    keys_m=["dice","iou","precision","recall","pixel_acc"]
    N=len(axes_labels); angles=np.linspace(0,2*np.pi,N,endpoint=False).tolist(); angles+=angles[:1]
    fig,ax=plt.subplots(figsize=(8,8),subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(axes_labels,fontsize=11)
    ax.set_ylim(0,1); ax.set_yticks([0.2,0.4,0.6,0.8,1.0])
    ax.yaxis.grid(True,linestyle="--",alpha=0.5)
    for key,info in METHODS.items():
        vals=[]
        for mk in keys_m:
            sc=[all_metrics[p][key][mk] for p in all_metrics
                if key in all_metrics[p] and all_metrics[p][key].get(mk,-1)>=0]
            vals.append(np.mean(sc) if sc else 0)
        vals+=vals[:1]
        ax.plot(angles,vals,lw=2,color=info["color"],label=info["label"])
        ax.fill(angles,vals,alpha=0.1,color=info["color"])
        ax.scatter(angles[:-1],vals[:-1],s=45,color=info["color"],zorder=5)
    ax.legend(loc="lower left",bbox_to_anchor=(-0.32,-0.12),fontsize=9,
              frameon=True,title="Method",title_fontsize=9)
    ax.set_title("Method performance radar\n(averaged over evaluated pairs)",
                 fontsize=11,fontweight="bold",pad=20)
    plt.tight_layout()
    out=os.path.join(out_dir,"radar_chart.png")
    plt.savefig(out,dpi=140,bbox_inches="tight"); plt.close(fig)
    print(f"  Radar: {out}")

def plot_metric_bars(all_metrics, out_dir=OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    keys=list(METHODS.keys()); pairs=list(all_metrics.keys())
    n_p=len(pairs); n_m=len(keys)
    colors=[METHODS[k]["color"] for k in keys]
    x=np.arange(n_p); bw=0.14
    offsets=np.linspace(-(n_m-1)/2,(n_m-1)/2,n_m)*bw
    fig,axes=plt.subplots(1,4,figsize=(22,6))
    fig.suptitle("Evaluation Metrics — All Methods vs Ground Truth",fontsize=12,fontweight="bold")
    for ax,(metric,ylabel,ref) in zip(axes,[
        ("dice","Dice Score","0.7"),("iou","IoU","0.5"),
        ("hausdorff_95","Hausdorff 95 (px)",""),("precision","Precision","")]):
        for i,(key,off) in enumerate(zip(keys,offsets)):
            vals=[all_metrics[p].get(key,{}).get(metric,0) for p in pairs]
            bars=ax.bar(x+off,vals,width=bw,color=colors[i],alpha=0.85,label=METHODS[key]["label"])
            for bar,v in zip(bars,vals):
                if v>0: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,
                                f"{v:.2f}",ha="center",va="bottom",fontsize=6,rotation=45)
        ax.set_xticks(x); ax.set_xticklabels([p[:10] for p in pairs],fontsize=7,rotation=20)
        ax.set_ylabel(ylabel,fontsize=9); ax.spines[["top","right"]].set_visible(False)
        if ref: ax.axhline(float(ref),color="#888780",lw=1,ls="--",alpha=0.7)
    handles=[mpatches.Patch(color=METHODS[k]["color"],label=METHODS[k]["label"]) for k in keys]
    fig.legend(handles=handles,loc="lower center",ncol=n_m,fontsize=8,bbox_to_anchor=(0.5,-0.06))
    plt.tight_layout()
    out=os.path.join(out_dir,"metrics_bars.png")
    plt.savefig(out,dpi=120,bbox_inches="tight"); plt.close(fig)
    print(f"  Metrics bars: {out}")


# ─────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────
def print_report(all_metrics):
    keys=list(METHODS.keys()); pairs=list(all_metrics.keys())
    print(f"\n{'═'*80}")
    print("  PHASE 7 — EVALUATION REPORT")
    print(f"{'═'*80}")
    for pair in pairs:
        gt_type=next(iter(all_metrics[pair].values())).get("gt_type","")
        print(f"\n  ── {pair}  [{gt_type}]")
        print(f"  {'Method':<22} {'Dice':>6} {'IoU':>6} {'Prec':>6} {'Rec':>6} {'HD95':>8}")
        print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
        for key in keys:
            m=all_metrics[pair].get(key,{})
            print(f"  {METHODS[key]['label']:<22} "
                  f"{m.get('dice',0):>6.3f} {m.get('iou',0):>6.3f} "
                  f"{m.get('precision',0):>6.3f} {m.get('recall',0):>6.3f} "
                  f"{m.get('hausdorff_95',-1):>8.1f}")
    # Averages (skip no-tumor pairs)
    active=[p for p in pairs if not any(v.get("gt_type")=="no_tumor_zero_gt"
                                         for v in all_metrics[p].values())]
    if not active: active=pairs
    print(f"\n{'─'*80}")
    print(f"  AVERAGES over {len(active)} tumor pairs")
    print(f"  {'Method':<22} {'Dice':>6} {'IoU':>6} {'Prec':>6} {'Rec':>6} {'HD95':>8}")
    print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
    avg={}
    for key in keys:
        sc={mk:[] for mk in ["dice","iou","precision","recall","hausdorff_95"]}
        for p in active:
            for mk in sc:
                v=all_metrics[p].get(key,{}).get(mk,-1)
                if v>=0: sc[mk].append(v)
        avg[key]={mk:np.mean(v) if v else 0 for mk,v in sc.items()}
        print(f"  {METHODS[key]['label']:<22} "
              f"{avg[key]['dice']:>6.3f} {avg[key]['iou']:>6.3f} "
              f"{avg[key]['precision']:>6.3f} {avg[key]['recall']:>6.3f} "
              f"{avg[key]['hausdorff_95']:>8.1f}")
    print(f"\n{'─'*80}")
    print("  BEST METHOD PER METRIC:")
    for mk,lbl,hi in [("dice","Dice",True),("iou","IoU",True),
                       ("precision","Precision",True),("recall","Recall",True),
                       ("hausdorff_95","Hausdorff95",False)]:
        vals={k:avg[k][mk] for k in keys if avg[k][mk]>0}
        if not vals: continue
        best=max(vals,key=vals.get) if hi else min(vals,key=vals.get)
        print(f"  {lbl:<14} → {METHODS[best]['label']:<22} ({vals[best]:.3f})")
    print(f"\n{'─'*80}")
    print("  GUIDE: Dice≥0.7 good  |  IoU≥0.5 good  |  HD95≤10px precise")
    print("         Prec>Rec = under-segments  |  Rec>Prec = over-segments")
    print(f"{'═'*80}\n")


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────
def evaluate(pairs, save_csv=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_metrics={}
    for pi in pairs:
        pair_name=pi["name"]; img_npy=pi["img_npy"]
        gt_mask=pi["gt_mask"]; gt_type=pi["gt_type"]
        print(f"\n  ── {pair_name}  [{gt_type}]")
        mri=np.load(img_npy).astype(np.float32)
        if mri.ndim==3: mri=mri[:,:,0]
        # Run all methods
        r1=run_snake(img_npy); r2=run_ls(img_npy)
        r3=run_ms(img_npy);    r4=run_sm(img_npy)
        preds={"snake":r1["mask"],"chan_vese":r2["mask_cv"],
               "geodesic":r2["mask_gac"],"mean_shift":r3["tumor_mask"],
               "split_merge":r4["tumor_mask"]}
        pair_m={}
        for key,pred in preds.items():
            m=compute_metrics(pred,gt_mask); m["gt_type"]=gt_type
            pair_m[key]=m
            print(f"  {METHODS[key]['label']:<22} Dice={m['dice']:.3f}  "
                  f"IoU={m['iou']:.3f}  HD95={m['hausdorff_95']:.1f}px")
        all_metrics[pair_name]=pair_m
        plot_error_maps(mri,gt_mask,preds,pair_m,pair_name,gt_type)
    print("\n  Generating summary charts...")
    plot_radar(all_metrics)
    plot_metric_bars(all_metrics)
    print_report(all_metrics)
    if save_csv:
        _save_csv(all_metrics)
    saved=sorted(Path(OUTPUT_DIR).glob("*.png"))
    print(f"  {len(saved)} output files in {OUTPUT_DIR}/")
    return all_metrics

def _save_csv(all_metrics):
    path=os.path.join(OUTPUT_DIR,"evaluation_results.csv")
    fields=["pair","method","gt_type","dice","iou","pixel_acc",
            "precision","recall","f1","hausdorff_95","TP","FP","FN","TN"]
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for pair,pm in all_metrics.items():
            for method,m in pm.items():
                w.writerow({"pair":pair,"method":method,
                            "gt_type":m.get("gt_type",""),
                            **{k:m.get(k,"") for k in fields[3:]}})
    print(f"  CSV: {path}")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--n",type=int,default=12)
    parser.add_argument("--csv",action="store_true")
    parser.add_argument("--list",action="store_true")
    args=parser.parse_args()
    print("\n PHASE 7 — EVALUATION"); print("="*50)
    pairs=discover_pairs(max_pairs=args.n)
    if args.list: sys.exit(0)
    evaluate(pairs, save_csv=args.csv)
    print(" All done!\n")
