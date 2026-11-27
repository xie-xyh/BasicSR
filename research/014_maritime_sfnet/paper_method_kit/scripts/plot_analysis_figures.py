#!/usr/bin/env python3
"""
Generate all 5 publication-grade train-free analytical figures (300 DPI)
and exact NumPy/SciPy t-SNE + K-Means manifold metrics for MaritimeSFNet (014d).

Palette: Semantic Morandi-IEEE (aligned 1-to-1 with arch_v3.html & visual_design_spec.md)
  - Deep Maritime Teal: #2F6B60 (014d / Ours / CD / Mist)
  - Muted Terracotta:   #A95035 (014b / Sigmoid / Dense)
  - Steel Navy:         #3B5978 (014a / Plain / Attenuate)
  - Muted Brass:        #9B7432 (Moderate / HD / FCSF)
  - Cool Slate:         #64748B (PW / Secondary / Reference)
"""
import json
import os
import math
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

BASE = "research/014_maritime_sfnet/paper_method_kit"
ANALYSIS = os.path.join(BASE, "analysis")
FIGDIR = os.path.join(BASE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

C_TEAL = "#2F6B60"
C_TEAL_BG = "#E9F3F0"
C_TERRA = "#A95035"
C_TERRA_BG = "#F9ECE7"
C_NAVY = "#3B5978"
C_NAVY_BG = "#EDF2F7"
C_BRASS = "#9B7432"
C_BRASS_BG = "#FAF3E6"
C_SLATE = "#64748B"
C_INK = "#1F2933"
C_GRID = "#E2E8F0"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "stix",
    "axes.edgecolor": "#475569",
    "axes.linewidth": 0.9,
    "axes.labelcolor": C_INK,
    "axes.titlecolor": C_INK,
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.labelsize": 9.4,
    "axes.titlesize": 10.3,
    "legend.fontsize": 8.0,
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "savefig.facecolor": "#FFFFFF",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def style_ax(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#475569")
    ax.spines["bottom"].set_color("#475569")
    if grid_axis:
        ax.grid(axis=grid_axis, linestyle="--", linewidth=0.65, color=C_GRID, alpha=0.85, zorder=0)
        ax.set_axisbelow(True)


# =============================================================================
# Exact Pure-NumPy/SciPy t-SNE, K-Means++, Silhouette, CH, ARI, NMI, Purity
# =============================================================================
def exact_tsne(X, n_components=2, perplexity=25.0, n_iter=1000, lr=200.0, seed=42):
    rng = np.random.RandomState(seed)
    X = np.asarray(X, dtype=np.float64)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    k_pca = min(30, X.shape[1], X.shape[0] - 1)
    Xp = U[:, :k_pca] * S[:k_pca]

    N = Xp.shape[0]
    D2 = squareform(pdist(Xp, "sqeuclidean"))

    P = np.zeros((N, N), dtype=np.float64)
    target_H = np.log(perplexity)
    for i in range(N):
        betamin, betamax = -np.inf, np.inf
        beta = 1.0
        di = np.delete(D2[i], i)
        for _ in range(50):
            Pi = np.exp(-di * beta)
            sumP = np.sum(Pi) + 1e-12
            H = np.log(sumP) + beta * np.sum(di * Pi) / sumP
            Hdiff = H - target_H
            if abs(Hdiff) < 1e-5:
                break
            if Hdiff > 0:
                betamin = beta
                beta = beta * 2.0 if np.isinf(betamax) else 0.5 * (beta + betamax)
            else:
                betamax = beta
                beta = beta / 2.0 if np.isinf(betamin) else 0.5 * (beta + betamin)
        Pi = Pi / (np.sum(Pi) + 1e-12)
        P[i, np.arange(N) != i] = Pi

    P = (P + P.T) / (2.0 * N)
    P = np.maximum(P, 1e-12)
    P_exag = P * 4.0

    Y = Xp[:, :n_components].copy()
    Y = Y / (np.std(Y) + 1e-8) * 1e-2 + rng.normal(0, 1e-4, (N, n_components))
    iY = np.zeros_like(Y)
    gains = np.ones_like(Y)

    for it in range(n_iter):
        P_cur = P_exag if it < 120 else P
        sum_Y = np.sum(Y ** 2, axis=1)
        num = 1.0 / (1.0 + np.maximum(sum_Y[:, None] + sum_Y[None, :] - 2.0 * (Y @ Y.T), 0.0))
        np.fill_diagonal(num, 0.0)
        Q = np.maximum(num / (np.sum(num) + 1e-12), 1e-12)
        PQ = (P_cur - Q) * num
        dY = 4.0 * (np.diag(np.sum(PQ, axis=1)) - PQ) @ Y

        mom = 0.5 if it < 250 else 0.8
        same_sign = (dY > 0.0) == (iY > 0.0)
        gains = np.where(same_sign, gains * 0.8, gains + 0.2)
        gains = np.maximum(gains, 0.01)
        iY = mom * iY - lr * (gains * dY)
        Y = Y + iY
        Y -= Y.mean(axis=0)
    return Y


def kmeans_pp(X, K=3, n_init=80, max_iter=200, seed=42):
    rng = np.random.RandomState(seed)
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    best_inertia = np.inf
    best_labels = None
    best_centers = None
    for _ in range(n_init):
        centers = np.empty((K, X.shape[1]), dtype=np.float64)
        centers[0] = X[rng.randint(N)]
        for c in range(1, K):
            d2 = np.min(cdist(X, centers[:c], "sqeuclidean"), axis=1)
            probs = d2 / (d2.sum() + 1e-12)
            centers[c] = X[rng.choice(N, p=probs)]
        for _ in range(max_iter):
            labels = np.argmin(cdist(X, centers, "sqeuclidean"), axis=1)
            new_centers = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k) else centers[k]
                for k in range(K)
            ])
            if np.allclose(new_centers, centers, atol=1e-7):
                centers = new_centers
                break
            centers = new_centers
        inertia = np.sum(np.min(cdist(X, centers, "sqeuclidean"), axis=1))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()
    return best_labels, best_centers, best_inertia


def silhouette_score_np(X, labels):
    D = squareform(pdist(X, "euclidean"))
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return 0.0
    s = np.zeros(len(labels))
    for i in range(len(labels)):
        same = (labels == labels[i])
        same[i] = False
        a = D[i, same].mean() if np.any(same) else 0.0
        b = np.min([D[i, labels == u].mean() for u in uniq if u != labels[i]])
        s[i] = (b - a) / max(a, b, 1e-12)
    return float(np.mean(s))


def calinski_harabasz_np(X, labels):
    N = X.shape[0]
    uniq = np.unique(labels)
    K = len(uniq)
    mean_all = X.mean(axis=0)
    bgss = sum(np.sum(labels == u) * np.sum((X[labels == u].mean(axis=0) - mean_all) ** 2) for u in uniq)
    wgss = sum(np.sum((X[labels == u] - X[labels == u].mean(axis=0)) ** 2) for u in uniq)
    return float((bgss / (K - 1)) / (wgss / (N - K) + 1e-12))


def adjusted_rand_index(y_true, y_pred):
    classes = np.unique(y_true)
    clusters = np.unique(y_pred)
    nij = np.zeros((len(classes), len(clusters)), dtype=np.int64)
    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            nij[i, j] = np.sum((y_true == c) & (y_pred == k))
    comb2 = lambda n: n * (n - 1) / 2.0
    sum_comb_nij = np.sum(comb2(nij))
    ai = nij.sum(axis=1)
    bj = nij.sum(axis=0)
    N = len(y_true)
    expected = np.sum(comb2(ai)) * np.sum(comb2(bj)) / comb2(N)
    max_idx = 0.5 * (np.sum(comb2(ai)) + np.sum(comb2(bj)))
    return float((sum_comb_nij - expected) / (max_idx - expected + 1e-12))


def normalized_mutual_info(y_true, y_pred):
    classes = np.unique(y_true)
    clusters = np.unique(y_pred)
    mi = 0.0
    for c in classes:
        pc = np.mean(y_true == c)
        for k in clusters:
            pk = np.mean(y_pred == k)
            pck = np.mean((y_true == c) & (y_pred == k))
            if pck > 0:
                mi += pck * np.log(pck / (pc * pk))
    hc = -sum(np.mean(y_true == c) * np.log(np.mean(y_true == c) + 1e-12) for c in classes)
    hk = -sum(np.mean(y_pred == k) * np.log(np.mean(y_pred == k) + 1e-12) for k in clusters)
    return float(2.0 * mi / (hc + hk + 1e-12))


def hungarian_accuracy(y_true, y_pred):
    K = max(int(y_true.max()), int(y_pred.max())) + 1
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    r, c = linear_sum_assignment(-cm)
    return float(cm[r, c].sum() / len(y_true))


def ridge_5fold_acc(X, y, alpha=15.0, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, 5)
    K = int(y.max()) + 1
    correct = 0
    for f in range(5):
        te = folds[f]
        tr = np.concatenate([folds[j] for j in range(5) if j != f])
        mu = X[tr].mean(axis=0)
        sd = X[tr].std(axis=0) + 1e-8
        Xtr = np.hstack([(X[tr] - mu) / sd, np.ones((len(tr), 1))])
        Xte = np.hstack([(X[te] - mu) / sd, np.ones((len(te), 1))])
        Ytr = np.eye(K)[y[tr]]
        reg = alpha * np.eye(Xtr.shape[1])
        reg[-1, -1] = 0.0
        W = np.linalg.solve(Xtr.T @ Xtr + reg, Xtr.T @ Ytr)
        correct += int(np.sum(np.argmax(Xte @ W, axis=1) == y[te]))
    return float(correct / len(y))


# =============================================================================
# Load Verified d533 Data
# =============================================================================
with open(os.path.join(ANALYSIS, "trainfree_metrics.json")) as f:
    metrics = json.load(f)
with open(os.path.join(ANALYSIS, "paired_014d_vs_014b_8000", "summary.json")) as f:
    paired = json.load(f)

arr_main = np.load(os.path.join(ANALYSIS, "trainfree_arrays.npz"))
arr_emb = np.load(os.path.join(ANALYSIS, "embeddings_and_erf.npz"))


# =============================================================================
# FIGURE 2: Maritime Spectrum + Rep5-CD Spectral Prior & Stage Allocation
# =============================================================================
def make_fig2():
    fig = plt.figure(figsize=(14.4, 4.45))
    gs = GridSpec(1, 3, width_ratios=[1.12, 1.16, 1.05], wspace=0.27, left=0.055, right=0.985, top=0.87, bottom=0.18)

    # (a) Maritime Sea-Fog Radial Power Spectrum
    ax = fig.add_subplot(gs[0, 0])
    style_ax(ax, "both")
    r = np.array(metrics["physics"]["r_centers"])
    mask = r <= 0.35
    r_sub = r[mask]

    ax.axvspan(0.0, 0.025, color=C_TEAL_BG, alpha=0.95, zorder=0)
    ax.axvspan(0.025, 0.15, color=C_BRASS_BG, alpha=0.65, zorder=0)

    for sub, col, ls in [("Dense", C_TERRA, "-"), ("Moderate", C_BRASS, "--"), ("Mist", C_NAVY, "-.")]:
        prof = np.array(metrics["physics"]["per_subset"][sub]["mean_profile"])[mask]
        lf = metrics["physics"]["per_subset"][sub]["low_frac_lt002"] * 100
        ax.plot(r_sub, prof, color=col, linestyle=ls, linewidth=1.9, label=f"{sub} ({lf:.1f}% < 0.02)", zorder=3)

    ax.set_yscale("log")
    ax.set_xlim(0, 0.35)
    ax.set_ylim(6e-4, 1.3)

    for val, col in [(0.005, C_TERRA), (0.010, C_BRASS), (0.025, C_TEAL)]:
        ax.axvline(val, color=col, linestyle=":", linewidth=1.35, zorder=2)

    ax.annotate("Haze Veil\n86–87%\n(DC–0.025)", xy=(0.012, 0.22), xytext=(0.058, 0.34),
                fontsize=7.7, fontweight="bold", color="#1A3B35",
                arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=1.0),
                bbox=dict(boxstyle="round,pad=0.24", fc="#FFFFFF", ec=C_TEAL, lw=0.8))
    ax.text(0.088, 0.068, "Vessel / Horizon\nMid-Band (8.4–9.2%)", fontsize=7.5, color="#6E5220",
            ha="center", fontweight="bold", bbox=dict(boxstyle="round,pad=0.22", fc="#FFFDF7", ec=C_BRASS, lw=0.7))
    ax.annotate("$s$=4 cutoff\n($c_{\\mathrm{img}}$=0.025)", xy=(0.025, 0.0045), xytext=(0.085, 0.0014),
                fontsize=7.5, color=C_TEAL, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=1.0))

    ax.set_xlabel("Normalized Spatial Frequency $r_{\\mathrm{img}}$ (Nyquist)")
    ax.set_ylabel("Radial Residual Power Fraction (log)")
    ax.set_title("(a) Maritime Haze Spectrum & Scale Alignment", fontweight="bold", loc="left", pad=8)
    ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", loc="upper right", fontsize=7.6)

    # (b) Learned Rep5-CD 5-Branch Radial FFT Response + 2D Insets
    ax = fig.add_subplot(gs[0, 1])
    style_ax(ax, "both")
    rf = np.array(metrics["rep5_cd"]["r_centers"])
    profs = metrics["rep5_cd"]["branch_radial_profile"]

    styles = [
        ("cd", "Center Diff (CD, DC=5.8e-19)", C_TEAL, "-", 2.3),
        ("hd", "Horiz Diff (HD, DC=0.0)", C_BRASS, "--", 1.8),
        ("vd", "Vert Diff (VD, DC=0.0)", C_TERRA, ":", 1.9),
        ("plain", "Plain 3$\\times$3 (Low-Pass)", C_NAVY, "-", 1.8),
        ("pw", "Pointwise 1$\\times$1 (All-Pass)", C_SLATE, "-.", 1.4),
    ]
    for k, lbl, col, ls, lw in styles:
        ax.plot(rf, profs[k], label=lbl, color=col, linestyle=ls, linewidth=lw, zorder=3)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.002, 0.088)
    ax.set_xlabel("Feature Radial Frequency $r_f$ (Nyquist)")
    ax.set_ylabel("Normalized Branch Spectral Energy")
    ax.set_title("(b) Learned Rep5-CD Branch Frequency Responses", fontweight="bold", loc="left", pad=8)
    ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", loc="upper right",
              fontsize=7.1, ncol=1, handlelength=2.1)

    cmap_fft = LinearSegmentedColormap.from_list("morandi_fft", ["#FFFFFF", "#D8E8E4", "#5C9B8E", "#2F6B60", "#16352F"])
    for key_in, lbl_in, x0 in [("cd", "CD 2D", 0.56), ("hd", "HD 2D", 0.71), ("vd", "VD 2D", 0.86)]:
        iax = ax.inset_axes([x0, 0.035, 0.13, 0.21])
        im = arr_main[f"fft2d_{key_in}"]
        im = im / (im.max() + 1e-12)
        iax.imshow(im, cmap=cmap_fft, interpolation="bilinear")
        iax.set_xticks([])
        iax.set_yticks([])
        for sp in iax.spines.values():
            sp.set_edgecolor("#64748B")
            sp.set_linewidth(0.7)
        iax.set_title(lbl_in, fontsize=6.6, pad=1.5, color=C_INK, fontweight="bold")

    # (c) Per-Stage Energy Share Stacked Bars
    ax = fig.add_subplot(gs[0, 2])
    style_ax(ax, "y")
    stage_order = ["enc1", "enc2", "bot_seg1", "bot_seg2", "bot_seg3", "dec2", "dec1"]
    stage_labels = ["Enc1\n(1$\\times$)", "Enc2\n(1/2)", "Bot1\n(1/4)", "Bot2\n(1/4)", "Bot3\n(1/4)", "Dec2\n(1/2)", "Dec1\n(1$\\times$)"]
    se = metrics["rep5_cd"]["stage_energy_share"]
    x = np.arange(len(stage_order))
    bottom = np.zeros(len(stage_order))

    for bk, blbl, col in [
        ("cd", "CD (60.9%)", C_TEAL),
        ("hd", "HD (13.0%)", C_BRASS),
        ("vd", "VD (13.1%)", C_TERRA),
        ("plain", "Plain (8.9%)", C_NAVY),
        ("pw", "PW (4.1%)", "#94A3B8"),
    ]:
        vals = np.array([se[s][bk] * 100.0 for s in stage_order])
        ax.bar(x, vals, bottom=bottom, width=0.62, color=col, edgecolor="#FFFFFF", linewidth=0.6, label=blbl, zorder=3)
        if bk == "cd":
            for xi, v in zip(x, vals):
                ax.text(xi, v * 0.5, f"{v:.0f}%", ha="center", va="center", color="#FFFFFF", fontsize=7.3, fontweight="bold", zorder=4)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, fontsize=7.8)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Frobenius Kernel Energy Share (%)")
    ax.set_title("(c) Stage-Wise Branch Allocation & Folding", fontweight="bold", loc="left", pad=8)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=7.1, columnspacing=0.8, handlelength=1.1)

    ax.text(3.0, -23.5, "Lossless fold_reparam(): 7.84M $\\rightarrow$ 4.55M ($-$41.9%) | Max Err = 0.00 | 26.5 FPS @ 512$^2$ (RTX 3090)",
            ha="center", va="top", fontsize=7.35, color="#1A3B35", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.28", fc=C_TEAL_BG, ec=C_TEAL, lw=0.8))

    out = os.path.join(FIGDIR, "fig2_spectral_cd_folding.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved {out}")


# =============================================================================
# FIGURE 3: Effective Receptive Field (ERF) & Anisotropic Profiles
# =============================================================================
def make_fig3():
    fig = plt.figure(figsize=(14.4, 4.4))
    gs = GridSpec(1, 4, width_ratios=[1.0, 1.0, 1.0, 1.28], wspace=0.24, left=0.045, right=0.985, top=0.85, bottom=0.22)

    cmap_erf = LinearSegmentedColormap.from_list(
        "morandi_erf", ["#FFFFFF", "#EDF2F7", "#A9C4BE", "#4B867B", "#2F6B60", "#9B7432", "#A95035"]
    )

    e_b_a = arr_emb["erf_bot_014a"]
    e_b_d = arr_emb["erf_bot_014d"]
    e_d_a = arr_emb["erf_dec_014a"]
    e_d_d = arr_emb["erf_dec_014d"]
    extent = [-128, 128, -128, 128]
    xs = np.linspace(-128, 128, 256)

    # Smooth ONLY contour lines (raw ERF heatmaps remain 100% unsmoothed)
    c_b_a = gaussian_filter(e_b_a, sigma=1.6)
    c_b_d = gaussian_filter(e_b_d, sigma=1.6)
    c_d_a = gaussian_filter(e_d_a, sigma=3.0)
    c_d_d = gaussian_filter(e_d_d, sigma=3.0)

    # (a) 014a Bottleneck ERF
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(e_b_a, cmap=cmap_erf, vmin=0.0, vmax=0.85, extent=extent, origin="lower")
    ax0.contour(xs, xs, c_b_a, levels=[0.40], colors=[C_INK], linewidths=1.35, linestyles="--")
    ax0.set_title("(a) 014a Bottleneck ERF\n(No AnisoLK, Cross/Diag = 1.34)", fontweight="bold", fontsize=9.2, pad=6)
    ax0.set_xlabel("Horizontal Offset $\\Delta x$ (px)")
    ax0.set_ylabel("Vertical Offset $\\Delta y$ (px)")
    ax0.set_xticks([-100, 0, 100])
    ax0.set_yticks([-100, 0, 100])
    ax0.text(-120, -118, "Area>40%: 13.1%\nH$\\times$V: 103$\\times$96 px", fontsize=7.5, color=C_INK, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.22", fc="#FFFFFF", ec="#94A3B8", alpha=0.93, lw=0.75))

    # (b) 014d Bottleneck ERF
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(e_b_d, cmap=cmap_erf, vmin=0.0, vmax=0.85, extent=extent, origin="lower")
    ax1.contour(xs, xs, c_b_d, levels=[0.40], colors=[C_INK], linewidths=1.45, linestyles="-")
    ax1.set_title("(b) 014d Bottleneck ERF\n(+ASFB, Cross/Diag = 1.56)", fontweight="bold", fontsize=9.2, pad=6)
    ax1.set_xlabel("Horizontal Offset $\\Delta x$ (px)")
    ax1.set_xticks([-100, 0, 100])
    ax1.set_yticks([-100, 0, 100])
    ax1.set_yticklabels([])
    ax1.text(-120, -118, "Area>40%: 18.8% (+44%)\nH$\\times$V: 160$\\times$148 px", fontsize=7.5, color="#1A3B35", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.22", fc="#FFFFFF", ec=C_TEAL, alpha=0.95, lw=0.85))

    # (c) Pre-BASR Decoder (dec1) ERF with clean smoothed 25% contours
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(e_d_d, cmap=cmap_erf, vmin=0.0, vmax=0.75, extent=extent, origin="lower")
    ax2.contour(xs, xs, c_d_a, levels=[0.25], colors=[C_TERRA], linewidths=1.7, linestyles="--")
    ax2.contour(xs, xs, c_d_d, levels=[0.25], colors=[C_INK], linewidths=1.6, linestyles="-")
    ax2.set_title("(c) 014d Pre-BASR Dec1 ERF\n(25% Contour: 014d vs 014a)", fontweight="bold", fontsize=9.2, pad=6)
    ax2.set_xlabel("Horizontal Offset $\\Delta x$ (px)")
    ax2.set_xticks([-100, 0, 100])
    ax2.set_yticks([-100, 0, 100])
    ax2.set_yticklabels([])
    ax2.text(-120, -118, "Area>25%: 43.7% vs 16.2%\n(+169% | H$\\times$V: 213$\\times$165 px)", fontsize=7.3, color="#1A3B35", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.22", fc="#FFFFFF", ec=C_TEAL, alpha=0.95, lw=0.85))
    leg_c = [
        Line2D([0], [0], color=C_INK, lw=1.6, ls="-", label="014d (25%)"),
        Line2D([0], [0], color=C_TERRA, lw=1.6, ls="--", label="014a (25%)"),
    ]
    ax2.legend(handles=leg_c, loc="upper right", fontsize=7.1, frameon=True, facecolor="#FFFFFF", framealpha=0.94, edgecolor="#CBD5E1")

    # Shared horizontal colorbar cleanly positioned below (a)-(c)
    cax = fig.add_axes([0.045, 0.045, 0.615, 0.028])
    cb = fig.colorbar(im1, cax=cax, orientation="horizontal")
    cb.set_label("Normalized Log-ERF Response  $\\log(1 + |\\partial F / \\partial I|)$", fontsize=8.3, labelpad=2.5)
    cb.ax.tick_params(labelsize=7.6)

    # (d) 1D Horizontal & Vertical Profiles (pre-BASR dec1)
    ax3 = fig.add_subplot(gs[0, 3])
    style_ax(ax3, "both")
    off = np.arange(-128, 128)
    c = 128
    h_d = e_d_d[c - 3:c + 4, :].mean(axis=0)
    v_d = e_d_d[:, c - 3:c + 4].mean(axis=1)
    h_a = e_d_a[c - 3:c + 4, :].mean(axis=0)
    v_a = e_d_a[:, c - 3:c + 4].mean(axis=1)

    ax3.fill_between(off, h_a, h_d, where=(h_d > h_a), color=C_TEAL_BG, alpha=0.9, zorder=1)
    ax3.plot(off, h_d, color=C_TEAL, lw=2.1, label="014d Horiz (1$\\times$31 / 1$\\times$15)", zorder=4)
    ax3.plot(off, v_d, color=C_BRASS, lw=1.8, ls="-.", label="014d Vert (31$\\times$1 / 15$\\times$1)", zorder=3)
    ax3.plot(off, h_a, color=C_TERRA, lw=1.6, ls="--", label="014a Horiz (No AnisoLK)", zorder=2)
    ax3.plot(off, v_a, color=C_NAVY, lw=1.4, ls=":", label="014a Vert (No AnisoLK)", zorder=2)
    ax3.axhline(0.25, color="#94A3B8", lw=0.95, ls="--")
    ax3.text(122, 0.265, "25% thresh", ha="right", fontsize=7.3, color="#64748B")

    ax3.set_xlim(-128, 128)
    ax3.set_ylim(0.05, 0.92)
    ax3.set_xticks([-120, -60, 0, 60, 120])
    ax3.set_xlabel("Spatial Offset from Center (px)")
    ax3.set_ylabel("Normalized Log-ERF (Pre-BASR Dec1)")
    ax3.set_title("(d) Directional Horizon & Depth ERF Profiles", fontweight="bold", fontsize=9.2, pad=6)
    ax3.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", loc="upper right", fontsize=7.2)

    out = os.path.join(FIGDIR, "fig3_erf_anisotropy.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved {out}")


# =============================================================================
# FIGURE 4: C2-BASR Sensitivity & Local Jensen Proof, Real Gates, & Paired CIs
# =============================================================================
def make_fig4():
    fig = plt.figure(figsize=(15.4, 4.55))
    gs = GridSpec(1, 3, width_ratios=[1.08, 1.10, 1.18], wspace=0.40, left=0.048, right=0.985, top=0.87, bottom=0.15)

    # --- Panel (a): Gain Slope Sensitivity G'(k) & Zero-Curvature Tangent at k=0 ---
    ax = fig.add_subplot(gs[0, 0])
    style_ax(ax, "both")
    k = np.linspace(-1.4, 2.6, 500)
    sig = 1.0 / (1.0 + np.exp(-(k - math.log(8.0))))
    gp_sig = 4.5 * sig * (1.0 - sig)
    gp_sym = np.where(k >= 0, 1.0 / np.cosh(k / 3.0) ** 2, 1.0 / np.cosh(2.0 * k) ** 2)

    ax.axvline(0.0, color="#64748B", lw=0.95, ls=":")
    ax.axvspan(-0.5, 0.0, color=C_NAVY_BG, alpha=0.85, zorder=0)
    ax.plot(k, gp_sig, color=C_TERRA, lw=2.1, ls="--", label="Convex Sigmoid $G_{\\mathrm{sig}}'(k)$", zorder=3)
    ax.plot(k, gp_sym, color=C_TEAL, lw=2.3, label="$C^2$-BASR $G_{\\mathrm{sym}}'(k)$", zorder=4)

    # Tangent at k=0: flat for C2-BASR (G''(0)=0), steep positive slope +0.346 for Sigmoid!
    kt = np.linspace(-0.42, 0.42, 40)
    ax.plot(kt, 1.0 + 0.0 * kt, color="#1A3B35", lw=1.3, ls="-", alpha=0.8, zorder=5)
    ax.plot(kt, (4.0 / 9.0) + (28.0 / 81.0) * kt, color="#783520", lw=1.4, ls="-", alpha=0.85, zorder=5)
    ax.scatter([0.0], [1.0], color=C_TEAL, s=46, edgecolor="#FFFFFF", lw=0.8, zorder=6)
    ax.scatter([0.0], [4.0 / 9.0], color=C_TERRA, s=44, edgecolor="#FFFFFF", lw=0.8, zorder=6)

    ax.annotate("Zero curvature $G''_{\\mathrm{sym}}(0)=0$\n$G'(0)=1$ (2.2$\\times$ stronger $k<0$ pullback)",
                xy=(-0.12, 0.94), xytext=(-1.36, 1.18),
                fontsize=7.1, fontweight="bold", color="#1A3B35",
                arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=0.95),
                bbox=dict(boxstyle="round,pad=0.22", fc=C_TEAL_BG, ec=C_TEAL, lw=0.75))
    ax.annotate("Convex tilt $G''_{\\mathrm{sig}}(0)=+0.346$\n($2.16\\times$ upward bias)",
                xy=(0.0, 4.0 / 9.0), xytext=(-1.36, 0.56),
                fontsize=7.1, fontweight="bold", color="#783520",
                arrowprops=dict(arrowstyle="->", color=C_TERRA, lw=0.95),
                bbox=dict(boxstyle="round,pad=0.22", fc=C_TERRA_BG, ec=C_TERRA, lw=0.75))

    ax.set_xlim(-1.4, 2.6)
    ax.set_ylim(0.0, 1.48)
    ax.set_xlabel("Logit Activation $k$ (Around Identity $k=0$)")
    ax.set_ylabel("Gain Sensitivity Derivative $G'(k)$")
    ax.set_title("(a) $C^2$ Zero-Curvature & Gradient Symmetry", fontweight="bold", loc="left", pad=8)
    ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", loc="upper right", fontsize=7.2)

    # Clean inset in lower-right (fully clear of Sigmoid peak and legend!)
    iax = ax.inset_axes([0.54, 0.085, 0.43, 0.34])
    iax.set_facecolor("#FFFFFF")
    iax.set_zorder(8)
    rng = np.random.RandomState(0)
    z = rng.normal(0, 1.0, 80000)
    sigmas = np.linspace(0.0, 0.16, 25)
    b_sig = [np.mean(0.5 + 4.5 / (1.0 + np.exp(-(s * z - math.log(8.0))))) - 1.0 for s in sigmas]
    b_sym = [np.mean(np.where(s * z >= 0, 1.0 + 3.0 * np.tanh(s * z / 3.0), 1.0 + 0.5 * np.tanh(2.0 * s * z))) - 1.0 for s in sigmas]
    iax.axvspan(0.04, 0.10, color=C_TEAL_BG, alpha=0.85)
    iax.plot(sigmas, np.array(b_sig) * 1000, color=C_TERRA, lw=1.7, ls="--")
    iax.plot(sigmas, np.array(b_sym) * 1000, color=C_TEAL, lw=1.9)
    iax.text(0.07, 3.55, "Mist std", fontsize=6.1, color="#1A3B35", ha="center", fontweight="bold")
    iax.set_xlim(0.0, 0.16)
    iax.set_ylim(0.0, 4.8)
    iax.set_xticks([0.0, 0.08, 0.16])
    iax.set_yticks([0, 2, 4])
    iax.tick_params(labelsize=6.3, pad=1.2)
    iax.set_xlabel("Mist $\\sigma_k$", fontsize=6.5, labelpad=1)
    iax.set_ylabel("$\\times 10^{-3}$", fontsize=6.5, labelpad=1)
    iax.set_title("Local Jensen Bias", fontsize=6.9, fontweight="bold", pad=2, color=C_INK)
    iax.spines["top"].set_visible(False)
    iax.spines["right"].set_visible(False)

    # --- Panel (b): Real 283-Image Learned Scene-FCSF & BASR Gain Range ---
    ax = fig.add_subplot(gs[0, 1])
    style_ax(ax, "y")
    subs = ["Dense", "Moderate", "Mist"]
    rms_vals = [metrics["gate_summary"]["014d"][s]["img_rms"]["mean"] for s in subs]
    x = np.arange(3)
    w = 0.21

    a3 = [metrics["gate_summary"]["014d"][s]["fcsf3_alpha"]["mean"] for s in subs]
    a2 = [metrics["gate_summary"]["014d"][s]["fcsf2_alpha"]["mean"] for s in subs]
    b1 = [metrics["gate_summary"]["014d"][s]["fcsf1_beta"]["mean"] for s in subs]

    ax.bar(x - w, a3, width=w * 0.92, color=C_TEAL, label="FCSF3 $\\alpha_3$ (1/4 Low)", zorder=3)
    ax.bar(x, a2, width=w * 0.92, color=C_BRASS, label="FCSF2 $\\alpha_2$ (1/2 Low)", zorder=3)
    ax.bar(x + w, b1, width=w * 0.92, color=C_NAVY, label="FCSF1 $\\beta_1$ (1$\\times$ Detail)", zorder=3)
    for xi, v3, v2, v1 in zip(x, a3, a2, b1):
        ax.text(xi - w, v3 + 0.012, f"{v3:.2f}", ha="center", va="bottom", fontsize=6.9, color=C_TEAL, fontweight="bold")
        ax.text(xi, v2 + 0.012, f"{v2:.2f}", ha="center", va="bottom", fontsize=6.9, color="#6E5220", fontweight="bold")
        ax.text(xi + w, v1 + 0.012, f"{v1:.2f}", ha="center", va="bottom", fontsize=6.9, color=C_NAVY, fontweight="bold")

    ax.set_ylim(0.0, 0.74)
    ax.set_ylabel("Scene-FCSF Mean Gate Value ($\\alpha, \\beta$)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(RMS={r:.3f})" for s, r in zip(subs, rms_vals)], fontsize=8.1)

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(C_TERRA)
    ax2.tick_params(axis="y", colors="#783520", labelsize=8.0)
    ax2.axhline(1.0, color="#94A3B8", lw=0.95, ls=":")

    for model_name, offset, col, mkr, lbl in [
        ("014b", -0.10, C_TERRA, "s", "014b BASR $G$ [p05,p95]"),
        ("014d", +0.10, "#1A3B35", "o", "014d BASR $G$ [p05,p95]"),
    ]:
        means = np.array([metrics["gate_summary"][model_name][s]["basr_gain_mean"]["mean"] for s in subs])
        p05 = np.array([metrics["gate_summary"][model_name][s]["basr_gain_p05"]["mean"] for s in subs])
        p95 = np.array([metrics["gate_summary"][model_name][s]["basr_gain_p95"]["mean"] for s in subs])
        ax2.errorbar(x + offset, means, yerr=[means - p05, p95 - means], fmt=mkr, color=col,
                     ecolor=col, elinewidth=1.5, capsize=3.5, markersize=5.0, label=lbl, zorder=5)

    ax2.set_ylim(0.60, 1.40)
    ax2.set_ylabel("BASR Gain $G(x)$ (Mean & 5–95%)", color="#783520", fontsize=8.8)
    ax.set_title("(b) Learned Visibility-Adaptive Gating (283 Imgs)", fontweight="bold", loc="left", pad=8)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=6.6, frameon=True, facecolor="#FFFFFF",
              edgecolor="#CBD5E1", ncol=2, columnspacing=0.6, handlelength=1.2)

    # --- Panel (c): Paired % MSE Reduction (014d vs 014b, 95% Bootstrap CIs) ---
    ax = fig.add_subplot(gs[0, 2])
    style_ax(ax, "x")
    metric_keys = [
        ("overshoot_mse", "Overshoot MSE"),
        ("mid_mse", "Mid-Band MSE\n(0.02–0.15)"),
        ("bottom_smooth_mse", "Near-Sea\nSmooth MSE"),
        ("low_mse", "Low-Band MSE\n(<0.02)"),
        ("undershoot_mse", "Undershoot MSE"),
    ]
    sub_map = [
        ("OWH-Dense-internal-v2", "Dense (+0.44 dB*)", C_TERRA),
        ("OWH-Moderate-internal-v2", "Moderate (+0.60 dB*)", C_BRASS),
        ("OWH-Mist-internal-v2", "Mist (+0.48 dB*)", C_TEAL),
    ]

    y = np.arange(len(metric_keys))
    bh = 0.25
    ax.axvline(0.0, color=C_INK, lw=0.95)

    for si, (skey, slbl, col) in enumerate(sub_map):
        d = paired[skey]
        vals, lo_err, hi_err, sigs = [], [], [], []
        for mk, _ in metric_keys:
            ctrl = d[mk]["control"]
            delta = d[mk]["delta"]
            ci = d[mk]["ci95"]
            pct = -delta / ctrl * 100.0
            pct_lo = -ci[1] / ctrl * 100.0
            pct_hi = -ci[0] / ctrl * 100.0
            vals.append(pct)
            lo_err.append(max(pct - pct_lo, 0.0))
            hi_err.append(max(pct_hi - pct, 0.0))
            sigs.append(ci[1] < 0)
        ys = y + (si - 1) * bh
        ax.barh(ys, vals, height=bh * 0.88, color=col, alpha=0.92, label=slbl, zorder=3)
        ax.errorbar(vals, ys, xerr=[lo_err, hi_err], fmt="none", ecolor="#334155", elinewidth=0.95, capsize=2.0, zorder=4)
        for j, (yi, v, sg) in enumerate(zip(ys, vals, sigs)):
            if sg:
                ax.text(v + hi_err[j] + 1.0, yi, "*", va="center", ha="left",
                        fontsize=8.8, color=col, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([lbl for _, lbl in metric_keys], fontsize=8.0)
    ax.invert_yaxis()
    ax.set_xlim(-18, 66)
    ax.set_xlabel("Relative MSE Reduction: 014d vs 014b (%)")
    ax.set_title("(c) Paired Error Attribution (95% Bootstrap CI)", fontweight="bold", loc="left", pad=8)
    ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", loc="upper right", fontsize=7.2)

    out = os.path.join(FIGDIR, "fig4_basr_curvature_gates.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved {out}")


# =============================================================================
# FIGURE 5: Spatial Restoration & Interpretability Heatmaps (3 Real OWH Scenes)
# =============================================================================
def make_fig5():
    rows_meta = [
        ("dense", "Dense", "Dense_06924.jpg"),
        ("moderate", "Moderate", "Moderate_08167.jpg"),
        ("mist", "Mist", "Mist_05403.jpg"),
    ]
    meta = metrics["sample_maritime_meta"]

    cmap_gain = LinearSegmentedColormap.from_list(
        "morandi_gain", ["#2C4B6E", "#6C91B5", "#F8FAFC", "#D49B6A", "#A95035"]
    )
    cmap_alpha = LinearSegmentedColormap.from_list(
        "morandi_alpha", ["#F8FAFC", "#D5E6E1", "#6BA396", "#2F6B60", "#16352F"]
    )
    cmap_beta = LinearSegmentedColormap.from_list(
        "morandi_beta", ["#FBF8F1", "#EFE0BC", "#C79C4A", "#9B7432", "#593E12"]
    )

    fig = plt.figure(figsize=(15.4, 6.75))
    gs = GridSpec(3, 7, wspace=0.045, hspace=0.14, left=0.032, right=0.99, top=0.915, bottom=0.09)

    col_titles = [
        "Hazy Input $I$",
        "014b (Convex BASR)",
        "014d (MaritimeSFNet)",
        "Ground Truth $J$",
        "$C^2$-BASR Gain $G(x)$",
        "FCSF2 Low-Gate $\\alpha_2(x)$",
        "FCSF1 Detail $\\beta_1(x)$",
    ]

    im_g = im_a = im_b = None
    for ri, (tag, sub, fn) in enumerate(rows_meta):
        z = np.load(os.path.join(ANALYSIS, "sample_maritime_maps", f"{tag}_maps.npz"))
        lq = np.clip(z["lq"], 0, 1)
        gt = np.clip(z["gt"], 0, 1)
        o_b = np.clip(z["out_014b"], 0, 1)
        o_d = np.clip(z["out_014d"], 0, 1)
        g_map = z["basr_gain_map"]
        a_map = z["fcsf2_alpha_map"]
        b_map = z["fcsf1_beta_map"]

        pb = meta[sub]["psnr_014b"]
        pd = meta[sub]["psnr_014d"]
        rms = float(np.std(lq))

        imgs = [lq, o_b, o_d, gt, g_map, a_map, b_map]
        for ci, img in enumerate(imgs):
            ax = fig.add_subplot(gs[ri, ci])
            if ci < 4:
                ax.imshow(img)
            elif ci == 4:
                im_g = ax.imshow(img, cmap=cmap_gain, norm=TwoSlopeNorm(vmin=0.82, vcenter=1.0, vmax=1.28))
            elif ci == 5:
                im_a = ax.imshow(img, cmap=cmap_alpha, vmin=0.18, vmax=0.72)
            else:
                im_b = ax.imshow(img, cmap=cmap_beta, vmin=0.14, vmax=0.58)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#CBD5E1")
                sp.set_linewidth(0.75)

            if ri == 0:
                ax.set_title(col_titles[ci], fontsize=9.2, fontweight="bold", pad=6,
                             color=(C_TEAL if ci == 2 else C_INK))
            if ci == 0:
                ax.set_ylabel(f"{sub}\n({fn.split('_')[1].split('.')[0]})", fontsize=8.8, fontweight="bold", labelpad=5)
                ax.text(0.04, 0.06, f"RMS={rms:.3f}", transform=ax.transAxes, fontsize=7.2, color="#FFFFFF",
                        fontweight="bold", bbox=dict(boxstyle="round,pad=0.18", fc="#1F2933", alpha=0.74, lw=0))
            elif ci == 1:
                ax.text(0.04, 0.06, f"{pb:.2f} dB", transform=ax.transAxes, fontsize=7.4, color="#FFFFFF",
                        fontweight="bold", bbox=dict(boxstyle="round,pad=0.18", fc=C_TERRA, alpha=0.86, lw=0))
            elif ci == 2:
                ax.text(0.04, 0.06, f"{pd:.2f} dB (+{pd - pb:.2f})", transform=ax.transAxes, fontsize=7.4, color="#FFFFFF",
                        fontweight="bold", bbox=dict(boxstyle="round,pad=0.18", fc=C_TEAL, alpha=0.90, lw=0))
            elif ci == 4:
                ax.text(0.04, 0.06, f"$\\mu$={g_map.mean():.2f}", transform=ax.transAxes, fontsize=7.2, color="#1F2933",
                        fontweight="bold", bbox=dict(boxstyle="round,pad=0.16", fc="#FFFFFF", alpha=0.85, lw=0))
            elif ci == 5:
                ax.text(0.04, 0.06, f"$\\mu$={a_map.mean():.2f}", transform=ax.transAxes, fontsize=7.2, color="#FFFFFF",
                        fontweight="bold", bbox=dict(boxstyle="round,pad=0.16", fc="#1A3B35", alpha=0.78, lw=0))
            elif ci == 6:
                ax.text(0.04, 0.06, f"$\\mu$={b_map.mean():.2f}", transform=ax.transAxes, fontsize=7.2, color="#FFFFFF",
                        fontweight="bold", bbox=dict(boxstyle="round,pad=0.16", fc="#593E12", alpha=0.78, lw=0))

    for c_idx, (im_obj, lbl, ticks) in zip(
        [4, 5, 6],
        [
            (im_g, "Gain $G(x)$  (<1 Attenuate | >1 Boost)", [0.85, 1.00, 1.25]),
            (im_a, "Low-Band Skip $\\alpha_2(x)$", [0.20, 0.45, 0.70]),
            (im_b, "High-Pass Detail $\\beta_1(x)$", [0.15, 0.35, 0.55]),
        ],
    ):
        bbox = gs[2, c_idx].get_position(fig)
        cax = fig.add_axes([bbox.x0 + 0.005, 0.042, bbox.width - 0.01, 0.018])
        cb = fig.colorbar(im_obj, cax=cax, orientation="horizontal", ticks=ticks)
        cb.set_label(lbl, fontsize=7.6, labelpad=2)
        cb.ax.tick_params(labelsize=7.2, pad=1.2)

    out = os.path.join(FIGDIR, "fig5_spatial_heatmaps.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved {out}")


# =============================================================================
# FIGURE 6: Pure-NumPy/SciPy t-SNE & K-Means Visibility Manifold Disentanglement
# =============================================================================
def make_fig6():
    y_true = arr_emb["y_true"].astype(int)
    # Verified directly from scripts/run_trainfree_analysis_d533.py:
    # label_map = {'Dense': 0, 'Moderate': 1, 'Mist': 2}
    # Counts: 0 -> 94 (Dense), 1 -> 95 (Moderate), 2 -> 94 (Mist)
    label_map = {0: "Dense", 1: "Moderate", 2: "Mist"}
    assert np.sum(y_true == 0) == 94 and np.sum(y_true == 1) == 95 and np.sum(y_true == 2) == 94

    results = {"label_map": label_map, "bottleneck_144d": {}, "decoder_36d": {}}

    for space_key, prefix in [("bottleneck_144d", "emb_bot"), ("decoder_36d", "emb_dec")]:
        for mname in ["014a", "014b", "014d"]:
            X = arr_emb[f"{prefix}_{mname}"].astype(np.float64)
            Xz = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
            km_labels, _, _ = kmeans_pp(Xz, K=3, n_init=80, seed=42)
            sil_true = silhouette_score_np(Xz, y_true)
            sil_km = silhouette_score_np(Xz, km_labels)
            ch_true = calinski_harabasz_np(Xz, y_true)
            ari = adjusted_rand_index(y_true, km_labels)
            nmi = normalized_mutual_info(y_true, km_labels)
            purity = hungarian_accuracy(y_true, km_labels)
            lin_acc = ridge_5fold_acc(X, y_true, alpha=15.0, seed=42)
            Y_tsne = exact_tsne(X, perplexity=25.0, n_iter=1000, lr=200.0, seed=42)
            sil_2d = silhouette_score_np(Y_tsne, y_true)

            results[space_key][mname] = {
                "silhouette_true": round(sil_true, 4),
                "silhouette_kmeans": round(sil_km, 4),
                "silhouette_tsne2d": round(sil_2d, 4),
                "calinski_harabasz": round(ch_true, 2),
                "kmeans_ari": round(ari, 4),
                "kmeans_nmi": round(nmi, 4),
                "kmeans_purity": round(purity, 4),
                "linear_5fold_acc": round(lin_acc, 4),
                "_tsne": Y_tsne,
            }

    json_out = {
        "label_map": {str(k): v for k, v in label_map.items()},
        "bottleneck_144d": {m: {k: v for k, v in d.items() if not k.startswith("_")} for m, d in results["bottleneck_144d"].items()},
        "decoder_36d": {m: {k: v for k, v in d.items() if not k.startswith("_")} for m, d in results["decoder_36d"].items()},
    }
    with open(os.path.join(ANALYSIS, "manifold_tsne_kmeans.json"), "w") as f:
        json.dump(json_out, f, indent=2)

    sub_colors = {"Dense": C_TERRA, "Moderate": C_BRASS, "Mist": C_TEAL}
    sub_markers = {"Dense": "o", "Moderate": "^", "Mist": "s"}

    fig = plt.figure(figsize=(15.0, 4.25))
    gs = GridSpec(1, 4, width_ratios=[1.0, 1.0, 1.0, 1.25], wspace=0.25, left=0.042, right=0.985, top=0.84, bottom=0.15)

    titles = [
        "(a) 014a Bottleneck (No ASFB)",
        "(b) 014b Bottleneck (+ASFB)",
        "(c) 014d Bottleneck (Full)",
    ]
    for idx, mname in enumerate(["014a", "014b", "014d"]):
        ax = fig.add_subplot(gs[0, idx])
        style_ax(ax, None)
        Y = results["bottleneck_144d"][mname]["_tsne"]
        d = results["bottleneck_144d"][mname]

        for cid in [0, 1, 2]:
            sname = label_map[cid]
            mask = (y_true == cid)
            ax.scatter(Y[mask, 0], Y[mask, 1], c=sub_colors[sname], marker=sub_markers[sname],
                       s=24, alpha=0.84, edgecolors="#FFFFFF", linewidths=0.35, label=sname, zorder=3)
            cx, cy = Y[mask, 0].mean(), Y[mask, 1].mean()
            ax.scatter([cx], [cy], c=sub_colors[sname], marker="X", s=100, edgecolors=C_INK, linewidths=1.05, zorder=5)

        ax.set_title(titles[idx], fontweight="bold", fontsize=9.5, pad=7,
                     color=(C_TEAL if mname == "014d" else C_INK))
        ax.set_xlabel("t-SNE Dim 1")
        if idx == 0:
            ax.set_ylabel("t-SNE Dim 2 (1/4 Bottleneck, 144-D)")
        ax.tick_params(labelsize=7.5)
        ax.text(0.03, 0.04,
                f"CH={d['calinski_harabasz']:.1f} | Sil2D={d['silhouette_tsne2d']:.3f}\n"
                f"KM-ARI={d['kmeans_ari']:.3f} | Pur={d['kmeans_purity']*100:.1f}%",
                transform=ax.transAxes, fontsize=7.25, fontweight="bold",
                color=("#1A3B35" if mname == "014d" else C_INK),
                bbox=dict(boxstyle="round,pad=0.24", fc="#FFFFFF",
                          ec=(C_TEAL if mname == "014d" else "#CBD5E1"), alpha=0.94, lw=0.85))
        if idx == 0:
            ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", loc="upper right",
                      fontsize=7.4, markerscale=1.05, handletextpad=0.3)

    # Panel (d): Unsupervised Separation at Bottleneck (014a/b/d) vs Deliberate Haze-Invariant Collapse at Dec1!
    ax = fig.add_subplot(gs[0, 3])
    style_ax(ax, "y")
    metric_defs = [
        ("kmeans_purity", "K-Means\nPurity"),
        ("kmeans_nmi", "K-Means\nNMI"),
        ("kmeans_ari", "K-Means\nARI"),
        ("linear_5fold_acc", "Linear\n5-Fold Acc"),
    ]
    x = np.arange(len(metric_defs))
    w = 0.195
    series = [
        ("bottleneck_144d", "014a", "014a Bot", C_NAVY, ""),
        ("bottleneck_144d", "014b", "014b Bot", C_TERRA, ""),
        ("bottleneck_144d", "014d", "014d Bot", C_TEAL, ""),
        ("decoder_36d", "014d", "014d Dec1", "#CBD5E1", "///"),
    ]
    for mi, (sp, mn, lbl, col, ht) in enumerate(series):
        vals = [results[sp][mn][mk] * 100.0 for mk, _ in metric_defs]
        xs_bar = x + (mi - 1.5) * w
        ax.bar(xs_bar, vals, width=w * 0.90, color=col, edgecolor=("#475569" if ht else "#FFFFFF"),
               hatch=ht, linewidth=0.65, label=lbl, zorder=3)
        for bx, bv in zip(xs_bar, vals):
            txt_col = "#475569" if ht else col
            ax.text(bx, bv + 1.2, f"{bv:.0f}" if bv >= 10 else f"{bv:.1f}",
                    ha="center", va="bottom", fontsize=6.4, color=txt_col, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metric_defs], fontsize=7.9)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Score (%)  [283 OWH Images]")
    ax.set_title("(d) Bottleneck Separation vs Dec1 Invariance", fontweight="bold", fontsize=9.3, pad=7)
    ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", loc="upper left",
              fontsize=6.9, ncol=2, columnspacing=0.6, handlelength=1.2)

    out = os.path.join(FIGDIR, "fig6_tsne_kmeans_visibility.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    make_fig6()
    print("ALL 5 REFINED FIGURES COMPLETED CLEANLY.")
