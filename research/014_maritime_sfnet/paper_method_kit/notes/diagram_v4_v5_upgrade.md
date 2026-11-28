# Architecture Diagram v4 & v5 Visual Style Upgrade (CVPR / IEEE TIP Authentic Flat-Vector Alignment)

Following direct side-by-side comparison against five published restoration architectures (AST-v2, UCL-Dehaze, 3D-PE Dehazing, GenHaze, and DEA-Net TIP 2024), we created two publication-ready flat-vector master diagrams while preserving earlier exploratory iterations (`arch_v1`, `arch_v2`, `arch_v3`) untouched:

## Key Style Shifts Applied from Reference Papers
1. **Zero explanatory prose inside diagrams**: Replaced all sentence-length annotations with compact 1–3 token operator names (`3×3 Conv`, `Rep5-CD ×3`, `ASFB`, `FCSF2`, `C²-BASR`) and crisp mathematical symbols (`I`, `Ĵ`, `F_1`, `H×W×36`, `α`, `β`, `G''(0)=0`).
2. **Real maritime input & output photographs**: Embedded actual OWH sea-fog cruise ship imagery (`Dense_06924.jpg` input and `014d` restored output) via `figures/assets/maritime_{hazy,clear}.png`.
3. **Pure flat-vector pastel grammar on `#FFFFFF` canvas**: Soft borderless pastel level strips / rounded module cards, crisp orthogonal `#1F2933` connectors, dashed rounded logical grouping frames, and clean circular operators (`+`, `×`, `C`, `↓`, `↑`).

## Two Complementary Master Archetypes Produced
- **[`arch_v4.html`](../figures/arch_v4.html) → [`arch_v4.png`](../figures/arch_v4.png)** (`2980 × 1644 px` @ 300 DPI) — **DEA-Net (IEEE TIP 2024) Split U-Shape Master** *(Default embedded in `maritime_sfnet_method.tex`)*: Left side presents the top-down/bottom-up U-Net across three soft borderless pastel scale strips with dashed `Encoder`/`Decoder` frames; right side groups `(b) Rep5-CD`, `(c) DCA-PA`, `(d) ASFB`, `(e) Scene-FCSF`, `(f) C²-BASR`, and a clean dashed legend box.
- **[`arch_v5.html`](../figures/arch_v5.html) → [`arch_v5.png`](../figures/arch_v5.png)** (`3002 × 1584 px` @ 300 DPI) — **AST-v2 (CVPR) Horizontal Ribbon + Borderless Pastel Cards Master** *(Ready-to-swap alternate)*: Top strip presents a strict left-to-right flow with vertical flat pastel operator pillars and nested dashed stage frames; bottom 2×2 grid presents borderless soft-tinted module cards `(b)–(e)`.
