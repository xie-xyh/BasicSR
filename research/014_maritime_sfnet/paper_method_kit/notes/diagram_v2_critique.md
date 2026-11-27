# Architecture Diagram v2 Visual Critique & Final Polish (`v2 → v3`)

Visual audit of `figures/arch_v2.png` (`3694 × 2222 px @ 300 DPI`):

## Major Improvements Verified in v2
- Panel (a) U-Net topology is now **100% strictly left-to-right monotonic** with straight vertical `Up1` into `FCSF1` and clean `Up2` into `FCSF2` — zero backward loops, zero factual wiring bugs.
- All raw ASCII underscores/carets replaced with true serif italic math `<tspan>` subscripts/superscripts; unified terracotta `(L)` LoRA badge across all 4 panels.
- Panels (b) and (c) have clean padding, zero text overflow, and non-overlapping `γ`/`β` FiLM routing.

## Remaining Micro-Defects Polished in v3 (Final Print Master)
1. **[Panel (d) Top — FCSF Title Collision & Wire Crossing]** The top dashed `Detail F_enc^H` wire (`y=531`) touched the card title (`y=525`), and its vertical drop (`x=1384`) crossed behind `σ(·) → (α, β)`.
   - **Fix**: Shift FCSF inner flow down 12 px, add a clean white halo pill behind `Detail F_enc^H`, and route its drop around the right margin (`x=1442`) straight into the fusion block with zero overlap.
2. **[Panel (d) Bottom — BASR Title Collision]** The `Modulated Additive Bias B·s_b` label (`y=710`) collided with the card title (`y=705`).
   - **Fix**: Increase vertical separation between card title (`y=703`) and bias routing (`y=720`) with a crisp white label pill; tidy `2σ → (s_k, s_b)` placement.
3. **[Panel (b) Right — DCA-PA Vertical Axis Alignment]** `Interleave` (`x=382`) and `DW-Grp 7×7 PA₀` (`x=377`) were slightly offset from the `+ → 2σ → ×` central axis (`x=362`).
   - **Fix**: Snap the entire lower DCA-PA chain onto a single exact vertical spine (`x=366`) and nudge `p₁` label off the elbow line.
4. **[Panel (a) Cross-Reference Signposts]** Added subtle panel tags `(b)`, `(c)`, `(d)` directly next to the corresponding macro blocks in Panel (a), exact `x=1327` centering for the top `I` drop into `C²-BASR`, and clean `MIMO (↓2×)` / `(↓4×)` padding.
5. **[Panel (c) Completeness]** Connected the final `+` of `MS-LFM` cleanly into the `X_out` equation box and added the 1/2-scale `L2-AnisoLK (1×15, 15×1, 7×7, 3×3)` note inside the `AnisoLK` card.
