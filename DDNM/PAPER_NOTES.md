# Paper notes — Compositional Controlled Generation via NLBP on SPNN-Detector

These notes summarize the experimental setup, method, and exploration log for the "compositional generation" experiment built on top of NLBP + SPNN + class-conditional ADM. Use as input for paper / appendix writing.

---

## 1. Task

Given a real image `x_orig ∈ [0,1]^{3×256×256}` with a known object of VOC class `c` at bounding box `B` (image coords), produce a new image `x'` such that:

- **Object preservation**: `x'` contains a recognizable instance of class `c` at (approximately) the same location.
- **Background change**: outside the BB, the image is generated freely under a class-conditional natural-image prior.

The goal is **NOT reconstruction** — PSNR vs `x_orig` is *not* a meaningful metric (we explicitly want the BG to differ). Quality is judged qualitatively via side-by-side panels.

---

## 2. Method overview

We instantiate DDNM (Denoising Diffusion Null-space Model) with NLBP (Non-Linear Back-Projection) as the constraint mechanism:

```
x' = G^{-1}( G(x_cur) + λ · [ G(Ap(y_target)) − G(Ap(A(x_cur))) ] )
```

- `A`: forward operator — full SPNN-CenterNet detection pipeline (image → 24×64×64 detection tensor `y`).
- `Ap = A†`: SPNN's built-in pseudo-inverse (an auxiliary "r-network" trained alongside the detector).
- `G`: SPNN's invertible feature map (`G^{-1}` is its inverse).
- `y_target`: the target detection tensor (here: `A(x_orig)` — the detection of the real image).

The diffusion prior is a **class-conditional 256×256 ADM** (Dhariwal & Nichol). Per image, we map the VOC class to the closest ImageNet-1k class index and condition the prior on it.

---

## 3. Pipeline (per image)

1. Load `x_orig` from VOC2007 test, get first GT box `B` and class `c`.
2. Pre-compute `y_target = A(x_orig)`.
3. Map `c → c_im` via the table below; condition ADM on `c_im`.
4. Time-travel sampling schedule (T_sampling=100, travel_length=10, travel_repeat=3) → ~430 step pairs; each base `t` is visited 3 times.
5. At each descent step `(t, t_next)` with `t > t_next`:
   - Predict `ε_θ(x_t, t, c_im)`; form Tweedie estimate `x0_t = (x_t − √(1−ᾱ_t) ε_θ) / √ᾱ_t`.
   - If `t > min_bp_step`: apply the NLBP step above with `λ = lambda2`. Else `λ = lambda1` (often 0).
   - **Smooth paste**: `x0_t ← w · x0_t_hat + (1−w) · x0_t`, where `w(p) = α · exp(−½(p−c_B)² / (s·d_B)²)` is a 2-D Gaussian centered at the BB. `s = ref_paste_smoothness`, `α = ref_paste_alpha`, `d_B` is the box dimension. This concentrates the BP'd content at the BB while leaving the rest free.
   - Re-noise to `x_{t_next}` with the standard DDIM update.
6. At each re-noise step `(t, t_next)` with `t < t_next`: pure forward diffusion.

---

## 4. Best hyperparameters (final config)

| Knob | Value | Notes |
|---|---|---|
| `min_bp_step` | **650** | BP only at `t > 650` (~35% of schedule, the noisier half). Lower → too constrained, output too close to orig. Higher (800) → no localization, output ignores the BB. |
| `lambda2` | **1.0** | Full BP strength when active. |
| `lambda1` | **0.0** | No BP at low t. |
| `bp_every` | **1** | BP at every applicable step. |
| `post_bp_noise` | **0.0** | No extra noise on `x0_t_hat`. |
| `ref_paste_smoothness` | **0.5** | Gaussian σ in BB-dim units. |
| `ref_paste_alpha` | **1.0** | Full weight at BB center. |
| `imagenet_class_from_gt` | **on** | Per-image VOC→ImageNet class. |
| `T_sampling` | **100** | Base steps. |
| `travel_length` | **10** | Time-travel block length. |
| `travel_repeat` | **3** | Repeats per block. |

---

## 5. VOC → ImageNet class mapping

| VOC | ImageNet idx | ImageNet name |
|---|---|---|
| aeroplane | 404 | airliner |
| bicycle | 671 | mountain bike |
| bird | 14 | indigo bunting |
| boat | 814 | speedboat |
| bottle | 898 | water bottle |
| bus | 779 | school bus |
| car | 817 | sports car |
| cat | 281 | tabby |
| chair | 765 | rocking chair |
| cow | 345 | ox |
| diningtable | 532 | dining table |
| dog | 207 | golden retriever |
| horse | 339 | sorrel |
| motorbike | 670 | motor scooter |
| person | 981 | ballplayer |
| pottedplant | 738 | pot |
| sheep | 348 | ram |
| sofa | 831 | studio couch |
| train | 466 | bullet train |
| tvmonitor | 851 | television |

---

## 6. Architecture

### 6.1 SPNN-CenterNet detector (`A`, `Ap`)

CenterNet "Objects as Points" head built on a SPNN backbone (Surjective Pseudo-invertible NN with householder-mixed invertible layers). Outputs:

- `hmap` ∈ ℝ^{20×64×64} — per-class center heatmap (sigmoid logits).
- `regs` ∈ ℝ^{2×64×64} — sub-pixel offset for the center cell.
- `wh` ∈ ℝ^{2×64×64} — box width/height in feature-map units.

`y = concat(hmap, regs, wh)` ∈ ℝ^{24×64×64}, stride 4.

Configuration flags used:
- `head_mode="affine"`, `head_mix_type="householder"`
- `deep_det_head=True`, `deep_head_hidden=128`
- `internal_head_affine=True`
- `no_hmap_scale=True`, `no_hmap_bias=True`
- `freeze_backbone=False`

The `A()` op composes pixel-domain conversion (model trained in BGR + ImageNet stats) with the detector forward; details in `build_class_y_model.py:build_A` and `diffusion_copy_copy.py`.

### 6.2 Class-conditional ADM

256×256 ADM from Dhariwal & Nichol (2021):
- `num_channels=256`, `num_heads=4`, `num_res_blocks=2`
- `attention_resolutions=[32,16,8]`
- `learn_sigma=True`, `use_scale_shift_norm=True`
- `resblock_updown=True`, `use_fp16=True`
- `class_cond=True`, `image_size=256`

Weights: `256x256_diffusion.pt` (HuggingFace mirror `danaroth/guided_diffusion`).

---

## 7. Smooth-paste in pixel space

```python
def smooth_paste(im1, im2, topleft, h, w, smoothness, alpha):
    cy, cx = topleft[0] + h/2, topleft[1] + w/2
    sy, sx = smoothness * h, smoothness * w
    weight = alpha * exp(-0.5 * (((Y-cy)/sy)**2 + ((X-cx)/sx)**2))
    return weight * im1 + (1 - weight) * im2
```

`im1 = x0_t_hat` (post-BP), `im2 = x0_t` (pre-BP / free prior). The Gaussian weight concentrates the BP-ed signal at the BB center; tuning `smoothness` shrinks/widens it; `alpha < 1` lets the prior bleed into the BB center.

---

## 8. Exploration log — what was tried and what worked

### Worked / kept

- **Class-conditional ADM** (vs unconditional). Substantially better prior quality.
- **Per-image VOC→ImageNet class mapping**. Critical: a constant `imagenet_class` produced bad mismatches (e.g. golden retriever class on a train image).
- **`min_bp_step=650`**. Sweet spot: prior renders clean detail at low-t, BP signals object identity/location during the noisier half.
- **`post_bp_noise=0`**. Earlier 0.025 added blur via the smooth-paste blending.

### Tried and discarded

- **y-domain pasting** (paste target y inside BB, zero/BG outside; BP entire combined y). The trained pinv on a localized non-natural y produces structured "dotted/grid" adversarial artifacts. Several attempted fixes (test-time `r` fine-tuning, learnable `y_BG`, image-reconstruction loss for `r`, smooth/soft BB masks) didn't resolve this. Conclusion: structural to the SPNN+detector when fed any localized non-natural y.
- **`crop-pinv` formulation** (`x' = Ap( M ⊙ y_target + (1−M) ⊙ y_BG )`). Same adversarial artifacts.
- **Empirical class-y models** (sample from real per-class y statistics) — same artifacts.
- **`bp_only_first_visit`** (BP only on first descent through each t). Too sparse, blurry results.
- **`bp_every=2` / `bp_every=3`** (skip BP every Nth step). Under-constrained for this task.
- **`lambda1=0.025`** (small low-t BP signal) and `lambda1=0.0025` — no benefit, slowed runtime.
- **Latent-space ADM swap** — substantial code change, deferred.

### Quantitative notes from the sweep

Across 20-image VOC2007 test subsets:
- `min_bp_step=300` (default): heavy BP throughout — output close to orig, blurry at high-frequency detail.
- `min_bp_step=500`: more constrained than 650 — acceptable but less BG variation.
- `min_bp_step=550`: similar to 500.
- `min_bp_step=650`: best balance; visible BG change while object class/location preserved.
- `min_bp_step=750`: weaker localization, BG more varied.
- `min_bp_step=800`: prior dominates entirely — class is right but localization is lost.

---

## 9. Code map

- `main_copy_copy.py`: CLI driver, parses flags, builds `Diffusion` runner.
- `guided_diffusion/diffusion_copy_copy.py`: core sampler (`Diffusion.simplified_ddnm_plus`).
  - `smooth_paste()`: pixel-space Gaussian paste.
  - `VOC_TO_IMAGENET`: mapping table.
  - Class-cond model call: `model(xt, t, y=imagenet_class_for_image)`.
  - Time-travel schedule: `get_schedule_jump(T_sampling, travel_length, travel_repeat)`.
- `datasets/voc.py`: `VOCValForDDNM` test loader.
- `nets/spnn_centernet_copy.py`: detector model.
- `build_class_y_model.py`: per-class y-window stats (used in earlier ablation, no longer in pipeline).
- `make_grid_strip.py`, `make_compare_strip.py`, `make_x0_mosaic.py`: visualization helpers.
- `configs/voc_detection.yml`: ADM + diffusion config (256×256, class_cond=true, fp16, T=100, 10×3 time-travel).

---

## 10. Run command (current best)

```bash
python -u main_copy_copy.py \
  --config voc_detection.yml --path_y dummy --task detection \
  --detector_ckpt <SPNN_DETECTOR.t7> \
  --voc_data_dir <VOC_PARENT> \
  --detector_num_classes 20 --detector_deep_det_head --internal_head_affine \
  --no_hmap_scale --no_hmap_bias --y_paste_mode none \
  --imagenet_class_from_gt --lambda2 1.0 --lambda1 0.0 \
  --bp_every 1 --min_bp_step 650 \
  --post_bp_noise 0.0 \
  --ref_paste_smoothness 0.5 --ref_paste_alpha 1.0 \
  --subset_start 0 --subset_end 50 --exp exp --ni \
  -i refxp_T650_tight_noNoise_50
```

---

## 11. Open issues / next steps

- Per-image generation is ~22 s on RTX 5090 (most time in 430 ADM forward passes at fp16).
- BG quality is fundamentally bounded by 256×256 ADM. A latent-diffusion swap (SD-style) would likely raise it but requires non-trivial pipeline rewiring (pinv operates in pixel space).
- Localization for non-rigid classes (`person`, `cat`, `dog`) is weaker than rigid classes (`bus`, `train`, `tvmonitor`) — likely because class-cond ADM places these objects at canonical poses/positions that don't always match the BB.
- VOC→ImageNet mapping is an ad-hoc lexical match. Some pairs are pure category-instance ("horse"→"sorrel"), others are scene-element ("diningtable"→"dining table"). For a "scene-of" framing, an alternative table (e.g. dog→dogsled, boat→dock, tvmonitor→entertainment_center) is being considered for compositional storytelling.

---
