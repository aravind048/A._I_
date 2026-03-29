"""
app.py — Streamlit Demo
========================
Interactive web app for MRI brain tumor boundary detection.

Upload an MRI image → choose a method → see the segmentation result live.

Run:
    pip install streamlit
    streamlit run app.py

NOTE: This uses classical image processing methods (not deep learning).
      No model weights to load — the algorithm runs fresh on each image.
      Works entirely on CPU, no GPU needed.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import time
import sys
import os

# ── Make sure methods/ folder is on the path ─────────────────────────
sys.path.insert(0, os.path.dirname(__file__))


# ─────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MRI Tumor Boundary Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Metric cards — dark background, white text, coloured value */
    [data-testid="stMetric"] {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 14px 18px;
        border-left: 4px solid #7c6af7;
    }
    [data-testid="stMetricLabel"] > div {
        color: #a0a0b8 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricValue"] > div {
        color: #ffffff !important;
        font-size: 22px !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricDelta"] > div {
        color: #7c6af7 !important;
    }
    /* Per-method metric card colours in "All Methods" view */
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 10px 14px;
        margin: 4px 0;
        border-left: 4px solid #7c6af7;
    }
    .metric-card .label {
        color: #a0a0b8;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    .metric-card .value {
        color: #ffffff;
        font-size: 18px;
        font-weight: 700;
    }
    .method-header {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 8px;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Preprocessing (same pipeline as preprocessing.py)
# ─────────────────────────────────────────────────────────────────────
def preprocess_image(img_array):
    """
    Full preprocessing pipeline:
      Grayscale → Bilateral filter → CLAHE → Percentile normalise

    Input : RGB uint8 numpy array
    Output: float32 [0,1] numpy array
    """
    # Convert to grayscale
    if img_array.ndim == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.copy()

    # Bilateral filter — edge-preserving denoising
    bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # CLAHE — local contrast enhancement
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced  = clahe.apply(bilateral)

    # Percentile normalisation (p1/p99) — prevents skull-rim from dominating
    img_f = enhanced.astype(np.float32)
    lo    = float(np.percentile(img_f, 1))
    hi    = float(np.percentile(img_f, 99))
    if hi - lo < 1e-6:
        return np.zeros_like(img_f)
    return np.clip((img_f - lo) / (hi - lo), 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────
# Segmentation methods (inline — no import from methods/ needed)
# ─────────────────────────────────────────────────────────────────────

def run_active_contours(img):
    from skimage.segmentation import active_contour
    from skimage.filters import gaussian

    smooth = gaussian(img, sigma=3.0)
    H, W   = img.shape
    r      = min(H, W) * 0.38
    t      = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    init   = np.column_stack([H/2 + r*np.sin(t), W/2 + r*np.cos(t)])

    snake  = active_contour(
        smooth, init, alpha=0.015, beta=10.0,
        w_line=0, w_edge=1.0,
        max_num_iter=2500, max_px_move=1.0,
        boundary_condition="periodic"
    )

    mask = np.zeros(img.shape, dtype=np.uint8)
    pts  = np.array([[int(c), int(r)] for r, c in snake], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask, snake


def run_chan_vese(img):
    from skimage.segmentation import morphological_chan_vese, checkerboard_level_set

    init = checkerboard_level_set(img.shape, square_size=6)
    mask = morphological_chan_vese(
        img, num_iter=200, init_level_set=init,
        smoothing=3, lambda1=1.0, lambda2=1.0
    )
    if mask.mean() > 0.5:
        mask = ~mask
    return mask.astype(np.uint8) * 255


def run_geodesic(img):
    from skimage.segmentation import (
        morphological_geodesic_active_contour,
        disk_level_set, inverse_gaussian_gradient
    )
    from skimage.filters import gaussian

    sm     = gaussian(img, sigma=1.5)
    gimage = inverse_gaussian_gradient(sm, alpha=100, sigma=1.5)
    init   = disk_level_set(img.shape,
                             center=(img.shape[0]//2, img.shape[1]//2),
                             radius=int(min(img.shape) * 0.40))
    mask = morphological_geodesic_active_contour(
        gimage, num_iter=300, init_level_set=init,
        smoothing=1, threshold=0.60, balloon=-1
    )
    return mask.astype(np.uint8) * 255


def run_mean_shift(img):
    img_u8  = (img * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    shifted = cv2.pyrMeanShiftFiltering(img_bgr, sp=21, sr=51)
    sg      = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)

    Z    = sg.reshape(-1, 1).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    _, lf, centers = cv2.kmeans(Z, 4, None, crit, 5, cv2.KMEANS_PP_CENTERS)

    H, W = img.shape
    sorted_idx = np.argsort(centers.flatten())
    remap      = np.zeros(4, dtype=np.uint8)
    for new_lbl, old_lbl in enumerate(sorted_idx):
        remap[old_lbl] = new_lbl
    label_map  = remap[lf.reshape(H, W)]
    raw_mask   = np.where(label_map == 3, 255, 0).astype(np.uint8)

    # Centrality-weighted selection
    cx, cy = W / 2.0, H / 2.0
    total  = img.size
    n, cc, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    best_mask, best_score = None, -1
    for lbl in range(1, n):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if not (total * 0.005 < area < total * 0.40):
            continue
        mu   = float(img_u8[cc == lbl].mean())
        bx   = stats[lbl, cv2.CC_STAT_LEFT] + stats[lbl, cv2.CC_STAT_WIDTH]  / 2
        by_  = stats[lbl, cv2.CC_STAT_TOP]  + stats[lbl, cv2.CC_STAT_HEIGHT] / 2
        dist = ((bx - cx)**2 + (by_ - cy)**2) ** 0.5
        score = mu * area * (1.0 / (1.0 + dist / 60.0))
        if score > best_score:
            best_score = score
            best_mask  = (cc == lbl).astype(np.uint8) * 255

    return best_mask if best_mask is not None else raw_mask


def run_split_merge(img):
    from collections import deque

    SPLIT_VAR  = 0.005
    MIN_SIZE   = 8
    MAX_DEPTH  = 6
    MERGE_THR  = 0.08

    class QN:
        def __init__(self, r, c, h, w, depth, mean, var):
            self.r=r; self.c=c; self.h=h; self.w=w
            self.depth=depth; self.mean=mean; self.var=var
            self.children=[]; self.parent=None; self.label=-1
        @property
        def is_leaf(self): return len(self.children) == 0

    def _split(img):
        def _rec(r, c, h, w, depth, parent):
            patch = img[r:r+h, c:c+w]
            n = QN(r, c, h, w, depth, float(patch.mean()), float(patch.var()))
            n.parent = parent
            if patch.var() > SPLIT_VAR and h > MIN_SIZE and w > MIN_SIZE and depth < MAX_DEPTH:
                hh, hw = h//2, w//2
                n.children = [
                    _rec(r,    c,    hh,    hw,    depth+1, n),
                    _rec(r,    c+hw, hh,    w-hw,  depth+1, n),
                    _rec(r+hh, c,    h-hh,  hw,    depth+1, n),
                    _rec(r+hh, c+hw, h-hh,  w-hw,  depth+1, n),
                ]
            return n
        root   = _rec(0, 0, *img.shape, 0, None)
        leaves = []
        q = deque([root])
        while q:
            nd = q.popleft()
            if nd.is_leaf: leaves.append(nd)
            else: q.extend(nd.children)
        return leaves

    def _merge(img, leaves):
        N  = len(leaves)
        uf = list(range(N))
        def find(i):
            while uf[i] != i: uf[i] = uf[uf[i]]; i = uf[i]
            return i
        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj: uf[ri] = rj
        for i in range(N):
            A = leaves[i]; A_r2 = A.r+A.h; A_c2 = A.c+A.w
            for j in range(i+1, N):
                B = leaves[j]; B_r2 = B.r+B.h; B_c2 = B.c+B.w
                horiz = (A_c2==B.c or B_c2==A.c) and (A.r < B_r2 and B.r < A_r2)
                vert  = (A_r2==B.r or B_r2==A.r) and (A.c < B_c2 and B.c < A_c2)
                if (horiz or vert) and abs(A.mean - B.mean) < MERGE_THR:
                    union(i, j)
        root_to_lbl = {}; nxt = [0]
        for i, nd in enumerate(leaves):
            rid = find(i)
            if rid not in root_to_lbl: root_to_lbl[rid] = nxt[0]; nxt[0] += 1
            nd.label = root_to_lbl[rid]
        lmap = np.full(img.shape, -1, dtype=np.int32)
        for nd in leaves:
            lmap[nd.r:nd.r+nd.h, nd.c:nd.c+nd.w] = nd.label
        return lmap

    leaves = _split(img)
    lmap   = _merge(img, leaves)

    H, W  = img.shape
    cx, cy = W/2.0, H/2.0
    total  = img.size
    best_lbl, best_score = -1, -1
    for lbl in np.unique(lmap[lmap >= 0]):
        mask = lmap == lbl
        area = int(mask.sum())
        if not (total * 0.005 < area < total * 0.40): continue
        mu   = float(img[mask].mean())
        rows, cols = np.where(mask)
        dist  = ((cols.mean()-cx)**2 + (rows.mean()-cy)**2)**0.5
        score = mu * area * (1.0 / (1.0 + dist/60.0))
        if score > best_score: best_score = score; best_lbl = lbl

    return np.where(lmap == best_lbl, 255, 0).astype(np.uint8) if best_lbl >= 0 \
           else np.zeros(img.shape, np.uint8)


# ─────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────

TINTS = {
    "Active Contours":    (0.6, 0.0, 0.0),
    "Level Sets CV":      (0.0, 0.5, 0.0),
    "Level Sets Geodesic":(0.0, 0.35,0.0),
    "Mean Shift":         (0.3, 0.0, 0.6),
    "Split & Merge":      (0.6, 0.4, 0.0),
}

# Hex accent colours — used for method name badges and metric card borders
METHOD_COLORS = {
    "Active Contours":    "#E24B4A",
    "Level Sets CV":      "#1D9E75",
    "Level Sets Geodesic":"#0F6E56",
    "Mean Shift":         "#534AB7",
    "Split & Merge":      "#BA7517",
}

def make_overlay(img_f, mask, method_name):
    tr, tg, tb = TINTS.get(method_name, (0.5, 0.5, 0.0))
    ov = np.stack([img_f]*3, axis=-1).copy()
    m  = mask > 0
    ov[m, 0] = ov[m, 0] * 0.35 + tr
    ov[m, 1] = ov[m, 1] * 0.35 + tg
    ov[m, 2] = ov[m, 2] * 0.35 + tb
    bnd = cv2.Canny(mask, 10, 200) > 0
    ov[bnd] = [1.0, 0.92, 0.0]          # yellow boundary
    return np.clip(ov, 0, 1)

def mask_to_metrics(mask, img_f):
    """Compute self-evaluation metrics (no ground truth needed)."""
    m        = mask > 0
    area_px  = int(m.sum())
    area_pct = round(100 * area_px / mask.size, 1)
    cnts, _  = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perim    = round(sum(cv2.arcLength(c, True) for c in cnts), 1)
    compact  = round(4 * np.pi * area_px / perim**2, 3) if perim > 0 else 0.0
    mu_in    = round(float(img_f[m].mean()), 3)  if m.sum()  > 0 else 0.0
    mu_out   = round(float(img_f[~m].mean()), 3) if (~m).sum()> 0 else 0.0
    contrast = round(mu_in / mu_out, 2)           if mu_out   > 0 else 0.0
    return {
        "Area": f"{area_pct}%",
        "Perimeter": f"{perim:.0f}px",
        "Compactness": f"{compact:.3f}",
        "Contrast": f"{contrast:.2f}×",
    }


def to_display(arr_f):
    """Convert float32 [0,1] array to uint8 RGB for st.image()."""
    return (np.clip(arr_f, 0, 1) * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    # st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Meningioma.jpg/320px-Meningioma.jpg",
    #          caption="Sample MRI (Wikipedia)", use_container_width=True)

    st.markdown("## 🧠 MRI Tumor Segmentation")
    st.markdown("Upload an MRI image and select a method to detect the tumor boundary.")

    st.markdown("---")
    st.markdown("### Method")
    method = st.radio(
        "Select segmentation method",
        [
            "Active Contours",
            "Level Sets CV",
            "Level Sets Geodesic",
            "Mean Shift",
            "Split & Merge",
            "All Methods",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### About the methods")
    method_info = {
        "Active Contours":     "Evolves a contour under edge forces. Smooth boundaries.",
        "Level Sets CV":       "Region-based. Works even without sharp edges.",
        "Level Sets Geodesic": "Edge-based level set. Better at clear boundaries.",
        "Mean Shift":          "Clusters pixels by colour+position. Fully automatic.",
        "Split & Merge":       "Quadtree + Union-Find. Built from scratch.",
    }
    for m, desc in method_info.items():
        st.markdown(f"**{m}** — {desc}")

    # st.markdown("---")
    # st.caption("Classical CV methods — no GPU needed.")
    # st.caption("All processing on CPU in real time.")


# ─────────────────────────────────────────────────────────────────────
# Main panel
# ─────────────────────────────────────────────────────────────────────
st.title("🧠 MRI Brain Tumor Boundary Detection")
st.markdown(
    "Upload an MRI brain scan. The selected segmentation method will highlight the "
    "detected tumor region in real time. No model training required — classical "
    "image processing runs fresh on each image."
)

uploaded = st.file_uploader(
    "Upload MRI image (PNG, JPG, BMP)",
    type=["png", "jpg", "jpeg", "bmp"],
    help="Grayscale or RGB MRI brain scan"
)

if uploaded is None:
    st.info("Upload an MRI image using the file uploader above to get started.")

    # Show example layout
    st.markdown("### What you will see")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Original MRI**\nPreprocessed with bilateral filter + CLAHE")
    c2.markdown("**Detected boundary**\nYellow line = tumor edge")
    c3.markdown("**Metrics**\nArea %, compactness, contrast ratio")
    st.stop()

# ── Load and preprocess ───────────────────────────────────────────────
img_pil   = Image.open(uploaded).convert("RGB")
img_arr   = np.array(img_pil)
img_f     = preprocess_image(img_arr)

col_orig, col_pre = st.columns(2)
with col_orig:
    st.markdown("**Original upload**")
    st.image(img_arr, use_container_width=True)
with col_pre:
    st.markdown("**After preprocessing** (bilateral + CLAHE + p99 normalise)")
    st.image(to_display(np.stack([img_f]*3, axis=-1)), use_container_width=True)

st.markdown("---")

# ── Run selected method(s) ────────────────────────────────────────────
RUNNERS = {
    "Active Contours":     lambda img: (run_active_contours(img)[0], None),
    "Level Sets CV":       lambda img: (run_chan_vese(img), None),
    "Level Sets Geodesic": lambda img: (run_geodesic(img), None),
    "Mean Shift":          lambda img: (run_mean_shift(img), None),
    "Split & Merge":       lambda img: (run_split_merge(img), None),
}

methods_to_run = (
    list(RUNNERS.keys()) if method == "All Methods" else [method]
)

if method == "All Methods":
    st.markdown("## All Methods Comparison")
    cols = st.columns(len(methods_to_run))

    for col, mname in zip(cols, methods_to_run):
        with col:
            hex_color = METHOD_COLORS.get(mname, "#7c6af7")
            st.markdown(
                f'<div class="method-header" '
                f'style="background:{hex_color};">{mname}</div>',
                unsafe_allow_html=True
            )
            with st.spinner(f"Running {mname}..."):
                t0       = time.perf_counter()
                mask, _  = RUNNERS[mname](img_f)
                dt       = time.perf_counter() - t0
                if mask is not None:
                    ov = make_overlay(img_f, mask, mname)
                    st.image(to_display(ov), use_container_width=True)
                    m  = mask_to_metrics(mask, img_f)
                    # Coloured metric cards with readable contrast
                    for label, value in [
                        ("Area",        m["Area"]),
                        ("Compactness", m["Compactness"]),
                        ("Contrast",    m["Contrast"]),
                        ("Time",        f"{dt:.1f}s"),
                    ]:
                        st.markdown(
                            f'<div class="metric-card" '
                            f'style="border-left-color:{hex_color};">'
                            f'<div class="label">{label}</div>'
                            f'<div class="value">{value}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.warning("Method failed on this image.")
else:
    st.markdown(f"## {method}")
    with st.spinner(f"Running {method}..."):
        t0      = time.perf_counter()
        mask, _ = RUNNERS[method](img_f)
        dt      = time.perf_counter() - t0

    if mask is None:
        st.error("Segmentation failed. Try a different method or image.")
        st.stop()

    ov = make_overlay(img_f, mask, method)

    col_result, col_metrics = st.columns([2, 1])

    with col_result:
        st.markdown("**Detected region** (tint = detected, yellow = boundary)")
        st.image(to_display(ov), use_container_width=True)

    with col_metrics:
        hex_color = METHOD_COLORS.get(method, "#7c6af7")
        st.markdown(
            f'<div class="method-header" '
            f'style="background:{hex_color}; margin-bottom:12px;">'
            f'Metrics</div>',
            unsafe_allow_html=True
        )
        # Inject per-card border colour via a dynamic style override
        st.markdown(
            f"""<style>
            [data-testid="stMetric"] {{
                border-left-color: {hex_color} !important;
            }}
            </style>""",
            unsafe_allow_html=True
        )
        m = mask_to_metrics(mask, img_f)
        st.metric("Detected area",  m["Area"])
        st.metric("Perimeter",      m["Perimeter"])
        st.metric("Compactness",    m["Compactness"],
                  help="4πA/P² — 1.0 = perfect circle. "
                       "Higher = rounder, more tumour-like shape.")
        st.metric("Contrast ratio", m["Contrast"],
                  help="Mean intensity inside / outside. "
                       "Higher = detected region is brighter than surroundings.")
        st.markdown(
            f'<div class="metric-card" style="border-left-color:{hex_color}; margin-top:8px;">'
            f'<div class="label">Runtime</div>'
            f'<div class="value">{dt:.2f}s</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.markdown("**Preprocessing stages**")
        st.markdown("1. Grayscale conversion")
        st.markdown("2. Bilateral filter (edge-preserving denoise)")
        st.markdown("3. CLAHE (local contrast enhance)")
        st.markdown("4. Percentile normalise (p1/p99)")

    # ── Preprocessing detail ──────────────────────────────────────────
    with st.expander("Show preprocessing stages side by side"):
        gray      = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced  = clahe_obj.apply(bilateral)

        s1, s2, s3, s4 = st.columns(4)
        s1.image(gray,      caption="1. Grayscale",     use_container_width=True, clamp=True)
        s2.image(bilateral, caption="2. Bilateral",     use_container_width=True, clamp=True)
        s3.image(enhanced,  caption="3. CLAHE",         use_container_width=True, clamp=True)
        s4.image(to_display(np.stack([img_f]*3, axis=-1)),
                              caption="4. Normalised",  use_container_width=True)

    # ── Download result ───────────────────────────────────────────────
    st.markdown("---")
    result_img = Image.fromarray(to_display(ov))
    buf = BytesIO()
    result_img.save(buf, format="PNG")
    st.download_button(
        label="Download result image",
        data=buf.getvalue(),
        file_name=f"segmentation_{method.lower().replace(' ','_')}.png",
        mime="image/png",
    )

# ─────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Methods: Active Contours · Level Sets (Chan-Vese + Geodesic) · "
    "Mean Shift · Split & Merge  |  "
    "Dataset: PranomVignesh/MRI-Images-of-Brain-Tumor (HuggingFace)  |  "
    "Classical image processing — no deep learning"
)
