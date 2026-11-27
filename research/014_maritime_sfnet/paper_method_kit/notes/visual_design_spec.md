# MaritimeSFNet Top-Venue Visual Design Specification (CVPR / ICCV / IEEE T-ITS)

This specification governs all architecture diagrams and analytical figures in `research/014_maritime_sfnet/paper_method_kit/`. Every rule is derived from visual inspection of leading restoration papers (Restormer CVPR 2022, NAFNet ECCV 2022, DehazeFormer TIP 2023, OKNet AAAI 2024, ConvIR TPAMI 2024, AdaIR ICLR 2025) and Nature/IEEE print typography standards.

## 1. Anti-"AI Slop" Checklist (Strict Prohibitions)
- **NO** neon cyan/magenta/purple gradients or dark-mode glassmorphism.
- **NO** glowing box-shadows, thick cartoon outlines (`>2.5px`), or giant pill corners (`rx > 10px`).
- **NO** decorative emojis, clip-art icons, or unaligned floating badges.
- **NO** rainbow palettes (>4 chromatic families) or unmotivated color changes between identical operators.
- **NO** rotated vertical text along connectors (keep all typography strictly horizontal).

## 2. Semantic Morandi-IEEE Palette (4 Core Hues + Warm Ink)
All fills are low-saturation archival tints with darker harmonic strokes so grayscale print preserves clear tonal separation:

| Semantic Functional Role | Fill Tint | Border / Accent | Text / Label | Grayscale Luminance |
|---|---|---|---|---|
| **Local Texture & Reparam Backbone** (`Stem`, `Enc1/2`, `Dec1/2`, `Rep5-CD`, `Down/Up`) | `#EDF2F7` (Mist Slate) | `#3B5978` (Steel Navy) | `#1E2D3D` | Mid-cool |
| **Dual-Domain Bottleneck (`ASFB`: `MS-LFM` + `AnisoLK`, `DCA-PA`)** | `#E9F3F0` (Seafoam Sage) | `#2F6B60` (Deep Maritime Teal) | `#1A3B35` | Mid-dark |
| **Frequency Convex Skip Fusion (`Scene-FCSF`, `MIMO` Input)** | `#FAF3E6` (Parchment Ochre) | `#9B7432` (Muted Brass) | `#4F3A14` | Light-warm |
| **Physics Reconstruction (`C²-BASR` + Scene Contrast Gate)** | `#F9ECE7` (Soft Clay) | `#A95035` (Muted Terracotta) | `#542416` | Distinct warm |
| **Neutral Operators & Containers** (`⊕`, `⊗`, `©`, `GAP`, `FFT`, stage frames) | `#F8F9FA` / `#FFFFFF` | `#64748B` / `#94A3B8` | `#24292F` (Charcoal Ink) | High contrast |

## 3. Typography & Hierarchy
- **Primary UI / Module Labels**: `'Helvetica Neue', Helvetica, Arial, sans-serif` (clean optical sans-serif matching IEEE/CVPR vector illustrations).
  - Panel titles `(a)–(d)`: `13px–14px` Bold (`#1F2933`).
  - Primary module headers: `11px–12px` SemiBold (`600`).
  - Operator details (`3×3`, `1×31`, `s=2`): `9.5px–10.5px` Regular (`#3E4C59`).
- **Mathematical Variables & Tensor Dimensions**: `'Times New Roman', 'STIXGeneral', Georgia, serif` *italic* for scalars/tensors ($I, \hat{J}, F_{\mathrm{enc}}, F_{\mathrm{dec}}, G_{\mathrm{sym}}(\tilde{K}), B, \alpha, \beta, c_{\mathrm{img}}$) and upright Roman for subscripts/operators ($\mathrm{enc}, \mathrm{low}, \tanh$).

## 4. Geometry, Strokes, and Routing
- **Canvas**: Pure white `#FFFFFF` outer canvas for seamless embedding into IEEE two-column white pages.
- **Outer Panel Containers**: `rx="6"`, `stroke="#CBD5E1"`, `stroke-width="1.2"`, soft `#FBFCFD` tint only for zoomed-in detail boxes `(b)/(c)/(d)`.
- **Module Blocks**: `rx="4"`, `stroke-width="1.35"`. Repeated stacks (`×3`, `×2`) shown either via subtle offset stacked-card edge (`dx=3, dy=-3`) or crisp `×N` count badge integrated cleanly inside the right edge.
- **Arrows & Routing**:
  - Primary forward flow: `stroke="#2B3440"`, `stroke-width="1.4"`, solid, filled sharp triangular arrowhead (`6×5`).
  - Skip / modulation / MIMO routing: orthogonal Manhattan polylines with subtle rounded corners (`Q` bezier radius `5px`), `stroke-width="1.25"`, dashed (`stroke-dasharray="4 3"`) in the target semantic accent color (`#9B7432` for FCSF/MIMO, `#A95035` for raw input skip to BASR).
  - Elementwise operator circles (`⊕`, `⊗`, `©`): radius `r="8.5"`, `#FFFFFF` fill, `1.35px` dark charcoal border.
- **Aspect Ratio & Print Resolution**:
  - Double-column master architecture figure (`figure*`): logical `1400 × 820` px rendered at `device-scale-factor=2.5` (`3500 × 2050` px, ≥ 450 DPI at IEEE full textwidth `7.16 in`), guaranteeing razor-sharp legibility at 400% zoom.
