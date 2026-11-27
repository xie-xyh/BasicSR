# Architecture Diagram v3 Final Audit (`arch_v3.html` → `arch_v3.png`)

## Iterative Evolution Retained (All Versions Preserved)
- **[`arch_v1.html`](../figures/arch_v1.html) → [`arch_v1.png`](../figures/arch_v1.png)** (`3644 × 2172 px`): Initial Morandi-IEEE 4-panel concept. Audited in [`diagram_v1_critique.md`](./diagram_v1_critique.md) — caught `Up2` routing error, ASCII subscripts, inconsistent LoRA badges, and BASR MLP text overflow.
- **[`arch_v2.html`](../figures/arch_v2.html) → [`arch_v2.png`](../figures/arch_v2.png)** (`3694 × 2222 px`): Fixed all structural routing and math typography, unified circled terracotta `L` badges for the 4-group Sim2Real LoRA interface, added stacked-card depth cues. Audited in [`diagram_v2_critique.md`](./diagram_v2_critique.md) — caught 3 micro-collisions in Panel (d) (`Detail Skip` and `Modulated Bias` wires touching card titles) and 15 px lower spine offset in `DCA-PA`.
- **[`arch_v3.html`](../figures/arch_v3.html) → [`arch_v3.png`](../figures/arch_v3.png)** (`2962 × 1834 px` @ 300 DPI, ≥ 410 DPI at IEEE `7.16 in` full width): Final publication master.

## Visual Inspection Checklist for `arch_v3.png`
- [x] **Topology accuracy**: `Stem → Enc1(×3) → Down1 → Enc2(×3) → Down2 → Bot1(×2) → ASFB-1 → Bot2(×3) → ASFB-2 → Bot3(×3) → FCSF3 → Up2 → FCSF2 → Dec2(×3) → L2-AnisoLK → Up1 → FCSF1 → Dec1(×3) → C²-BASR → Ĵ` matches `basicsr/archs/maritime_sfnet_arch.py` (`014d`) 1-to-1.
- [x] **MIMO & skip accuracy**: Zero-init MIMO `↓2×`/`↓4×` enters `Enc2`/`Bot1` cleanly; `1×` skip explicitly annotated `α ≡ 0` (high-pass only); raw hazy radiance `I` routed via dashed terracotta line into `C²-BASR`.
- [x] **Zero collisions / text overflows**: All 4 panels `(a)–(d)` inspected at full resolution; `Detail Skip F_enc^H` and `Modulated Bias B̃ = B·s_b` sit cleanly below their card titles; `DCA-PA` central vertical spine is strictly plumb.
- [x] **Typography & anti-AI aesthetics**: Strict 4-hue Morandi-IEEE palette (`#3B5978`, `#2F6B60`, `#9B7432`, `#A95035`) on pure white `#FFFFFF` canvas, zero gradients/glows, Helvetica Neue UI labels paired with Times New Roman italic math and true lowered/raised subscripts (`c_f`, `c_img`, `F_dec^L`, `F_enc^H`, `G_sym(K̃)`, `B̃`).
- [x] **Quantitative grounding**: `Rep5-CD` summary box directly embeds measured train-free numbers (`CD 60.9%`, `HD/VD 26.1%`, `-41.9% Params`, `0.00e+00` fold error, `26.5 FPS` on RTX 3090).
