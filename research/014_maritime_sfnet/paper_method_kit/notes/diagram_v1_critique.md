# Architecture Diagram v1 Visual Critique & Revision Plan (`v1 → v2`)

Visual audit of `figures/arch_v1.png` (`3644 × 2172 px @ 300 DPI`) against CVPR / ICCV / IEEE T-ITS print publication standards:

## What Worked Well in v1
- Restrained 4-hue Morandi-IEEE semantic palette (`#EDF2F7/#3B5978`, `#E6F2EF/#2F6B60`, `#FAF2E3/#9B7432`, `#F9EAE5/#A95035`) looks calm, authentic, and completely free of neon AI artifacts.
- Crisp white background, subtle stage bands, and clean horizontal typography read well at full-width two-column scale.

## Concrete Defects Caught on Visual Inspection (Must Fix in v2)
1. **[Critical Topology Bug — Panel (a)]** `Up2` routed vertically into the bottom of `L2-AnisoLK` instead of `FCSF2`, leaving `FCSF2` visually disconnected from the bottleneck output; `Up1` also bent backward (right-to-left) from `L2-AnisoLK` into `FCSF1`.
   - **Fix**: Re-proportion horizontal column spacing (`Enc1/2` left, compact `1/4 Bottleneck` center, `FCSF2 → Dec2 → L2-AnisoLK` right-ascending, `FCSF1 → Dec1 → C²-BASR → Ĵ` top-right) so **every single forward edge flows strictly left-to-right with zero backward loops**.
2. **[Typography — Raw ASCII Subscripts Across All Panels]** Labels like `c_f = 4·c_img`, `W_eq`, `X_low`, `G_sym(K̃)`, `F_enc^L`, `s_k`, `σ_RMS(I)` used raw underscores/carets, giving a code-generated feel.
   - **Fix**: Replace **100% of math subscripts/superscripts** with true italic serif `<tspan>` subscripts/superscripts ($c_f$, $c_{\mathrm{img}}$, $W_{\mathrm{eq}}$, $X_{\mathrm{low}}$, $G_{\mathrm{sym}}(\tilde{K})$, $F_{\mathrm{enc}}^L$, $s_k$, $\sigma_{\mathrm{RMS}}$).
3. **[Symbol Inconsistency]** Panel (a) marked LoRA targets with a terracotta badge `(L)`, whereas Panels (c)–(d) used black stars `★`.
   - **Fix**: Standardize on the exact same crisp `(L)` badge across all four panels.
4. **[Text Overflow / Tight Padding]**
   - Panel (b) left: `"0 Extra Inference FLOPs | −41.9% Params"` touched the box stroke.
   - Panel (d) bottom: `"[GAP(d₁), GAP(I), σ_RMS(I)]"` overflowed the `Scene Contrast MLP` box.
   - Panel (d) top: `"F_enc^L, F_enc^H"` touched box edges.
   - **Fix**: Re-balance box widths and inner padding with ≥ 8 px horizontal breathing room everywhere.
5. **[Connector Alignment Bugs]**
   - Panel (b) right (`DCA-PA`): residual `p₁` line terminated at `y=720` instead of the `+` circle (`y=738`); right bypass of `x` hit `y=784` instead of the `×` circle (`y=792`).
   - Panel (c) left (`MS-LFM`): `β` wire overlapped the vertical `× → +` wire.
   - Panel (d) top (`FCSF`): missing explicit path showing $F_{\mathrm{enc}}^H$ entering the final fusion block.
   - Panel (d) bottom (`BASR`): additive bias $B$ exited awkwardly from the top of `×` instead of cleanly branching into `(k·s_k)` and `(B·s_b)`.
6. **[Scale Tensor Annotations]** Add explicit tensor shape tags (`3×H×W`, `36×H×W`, `72×H/2×W/2`, `144×H/4×W/4`) right on the stage transitions for instant reviewer verification.
