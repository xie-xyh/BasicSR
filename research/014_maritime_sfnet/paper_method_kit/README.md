# MaritimeSFNet (`014d`) Paper Method & Figure Research Kit

Complete publication-ready methodology package for **`MaritimeSFNet` (`014d`)**, anchored in maritime/intelligent-transportation sea-fog perception (ASV/USV navigation, port surveillance, vessel silhouette recovery, real-time onboard deployment).

---

## 1. Directory Structure & Artifacts

### A. Visual Design Specification & Architecture Diagram Iterations (All Versions Preserved)
- [`notes/visual_design_spec.md`](notes/visual_design_spec.md) — Semantic 4-hue **Morandi-IEEE** palette, typography, geometry, and anti-AI rules aligned with CVPR/ICCV/ECCV/IEEE T-ITS visual standards.
- **Iterative Master Architecture Diagram (HTML+SVG → 300 DPI PNG, All Versions Retained)**:
  1. [`figures/arch_v1.html`](figures/arch_v1.html) → [`figures/arch_v1.png`](figures/arch_v1.png) — Initial 4-panel concept ([`notes/diagram_v1_critique.md`](notes/diagram_v1_critique.md)).
  2. [`figures/arch_v2.html`](figures/arch_v2.html) → [`figures/arch_v2.png`](figures/arch_v2.png) — Routing & typography pass ([`notes/diagram_v2_critique.md`](notes/diagram_v2_critique.md)).
  3. [`figures/arch_v3.html`](figures/arch_v3.html) → [`figures/arch_v3.png`](figures/arch_v3.png) — Detailed Morandi-IEEE dashboard master ([`notes/diagram_v3_audit.md`](notes/diagram_v3_audit.md)).
  4. **[`figures/arch_v4.html`](figures/arch_v4.html) → [`figures/arch_v4.png`](figures/arch_v4.png)** — **Final Publication Master (DEA-Net / IEEE TIP Flat-Vector U-Shape Split Style)** (`2980 × 1644 px`, real maritime photos, soft pastel scale strips, concise labels; embedded in `maritime_sfnet_method.tex`).
  5. **[`figures/arch_v5.html`](figures/arch_v5.html) → [`figures/arch_v5.png`](figures/arch_v5.png)** — **Alternate Publication Master (AST-v2 / CVPR Flat-Vector Horizontal Ribbon + Borderless Pastel Cards)** (`3002 × 1584 px`, [`notes/diagram_v4_v5_upgrade.md`](notes/diagram_v4_v5_upgrade.md)).
- **Maritime Motivation Teaser (HTML+SVG → 300 DPI PNG)**:
  - [`figures/fig1_maritime_motivation.html`](figures/fig1_maritime_motivation.html) → [`figures/fig1_maritime_motivation.png`](figures/fig1_maritime_motivation.png) (`2886 × 842 px`).

### B. Train-Free Mechanistic Experiments & Analytical Figures (300 DPI PNG)
All metrics are 100% real GPU measurements from `d533` checkpoints (`014a`, `014b`, `014d`) on the 283-image OWH internal validation split:
- [`figures/fig2_spectral_cd_folding.png`](figures/fig2_spectral_cd_folding.png) — Maritime sea-fog radial spectrum (86–87% veil in `DC–0.02`, scale-bandwidth cutoffs), learned `Rep5-CD` 5-branch radial + 2D FFT responses (`CD` exact zero DC `5.75e-19`), and stage-wise energy allocation (`CD 60.9%`, `HD+VD 26.1%`, lossless folding `7.84M → 4.55M`).
- [`figures/fig3_erf_anisotropy.png`](figures/fig3_erf_anisotropy.png) — Pre-BASR Effective Receptive Field (ERF) heatmaps and 1D horizon/depth profiles (`014a` vs `014d`: bottleneck cross/diag `1.34 → 1.56`, `area>40%` `+44%`; `Dec1` `area>25%` `+169%`, `213×165 px`).
- [`figures/fig4_basr_curvature_gates.png`](figures/fig4_basr_curvature_gates.png) — Analytical $C^2$ zero-curvature proof ($G''_{\mathrm{sym}}(0)=0$ vs $G''_{\mathrm{sig}}(0)=+28/81$, `2.16×` stronger $k<0$ pullback), 283-image learned `Scene-FCSF`/`BASR` gate trends, and paired `% MSE reductions` with 95% bootstrap CIs.
- [`figures/fig5_spatial_heatmaps.png`](figures/fig5_spatial_heatmaps.png) — Real maritime restoration (`Dense_06924` cruise ship, `Moderate_08167` sailing ship, `Mist_05403` coastal ferry) alongside spatial interpretability heatmaps $G(x)$, $\alpha_2(x)$, and $\beta_1(x)$.
- [`figures/fig6_tsne_kmeans_visibility.png`](figures/fig6_tsne_kmeans_visibility.png) — Unsupervised `t-SNE` & `K-Means++ (K=3)` manifold analysis showing **bottleneck visibility disentanglement** (`CH=25.26`, `ARI=0.273`, `NMI=0.319`, `Purity=61.8%`) followed by **decoder haze-invariant convergence** (`Dec1`: `NMI=0.076`, `ARI=0.065`).

### C. Reproducible Analysis Scripts & JSON Data
- [`scripts/render_html_to_png.py`](scripts/render_html_to_png.py) — Headless Chrome vector HTML/SVG → auto-cropped 300 DPI PNG renderer.
- [`scripts/run_trainfree_analysis_d533.py`](scripts/run_trainfree_analysis_d533.py) — GPU train-free extraction script (spectral physics, `Rep5-CD` folding & RTX 3090 FPS benchmark, pre-BASR ERFs, per-image gate stats, spatial maps, and feature embeddings).
- [`scripts/plot_analysis_figures.py`](scripts/plot_analysis_figures.py) — Local Matplotlib + pure-NumPy/SciPy exact `t-SNE` / `K-Means++` generator producing Figures 2–6 and `analysis/manifold_tsne_kmeans.json`.
- `analysis/*.json` — Complete quantitative JSON summaries (`trainfree_metrics.json`, `erf_stats.json`, `manifold_tsne_kmeans.json`, `paired_014d_vs_014b_8000/summary.json`). *(Large `.npz` raw arrays are kept locally in `analysis/` and excluded from Git via `.gitignore`.)*

### D. Two-Column IEEE LaTeX Manuscript
- [`latex/maritime_sfnet_method.tex`](latex/maritime_sfnet_method.tex) — Complete `IEEEtran` two-column methodology manuscript (Propositions 1–3, Theorem 1 with complete proofs, Tables I–V, and all 7 embedded figures).
- [`latex/build_pdf.sh`](latex/build_pdf.sh) — Two-pass `pdflatex` compilation + warning audit + PNG page preview script.

---

## 2. How to Compile the PDF

Once MacTeX finishes installing (`pdflatex` available in `/Library/TeX/texbin`):

```bash
bash research/014_maritime_sfnet/paper_method_kit/latex/build_pdf.sh
```

Output PDF: `research/014_maritime_sfnet/paper_method_kit/latex/maritime_sfnet_method.pdf`.
