# SPNN - Surjective Pseudo-Neural Network

> A surjective classifier with a guaranteed pseudo-inverse, enabling attribute-guided image reconstruction via non-linear back project and diffusion backbone.

[[Paper](https://arxiv.org/abs/2602.06042)] &nbsp;|&nbsp; [[Pretrained Model (HuggingFace)](https://huggingface.co/yamitehr/SPNN)]

---

## Overview

This repository contains the official implementation of **SPNN**, a surjective classifier that satisfies the Moore-Penrose pseudo-inverse identities by construction. The classifier `g` maps images to attribute logits, and its pseudo-inverse `g'` maps logits back to images - without any additional training.

We apply SPNN to **attribute-guided face reconstruction** on CelebA-HQ: given a target set of facial attributes, the diffusion process is steered via **Nonlinear Back-Projection (NLBP)** to reconstruct a face that satisfies the target attributes.

---

## Installation

```bash
git clone https://github.com/yamitehr/SPNN
cd SPNN
pip install -r requirements.txt
```

> **Note:** Install PyTorch with the CUDA version matching your system from [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.

---

## Part 1 - Attribute-Guided Reconstruction (Diffusion + NLBP)

This section reproduces the main reconstruction task: using the SPNN pseudo-inverse inside a diffusion sampling loop to generate faces with target attributes.

### Example Result

<img src="assets/reconstruction_example.jpg" width="600"/>

*Reconstructed faces satisfying the target attributes.*

### Step 1 - Download the Diffusion Model

We use the CelebA-HQ 256×256 diffusion model from [SDEdit](https://github.com/ermongroup/SDEdit).

1. Download the checkpoint from the following link:
   **[celeba_hq.ckpt (Google Drive)](https://drive.google.com/file/d/1wSoA5fm_d6JBZk4RZ1SzWLMgev4WqH21/view?usp=share_link)**

2. Place it at the following path:
   ```
   DDNM/exp/logs/celeba/
   ```
   The final structure should look like:
   ```
   DDNM/
   └── exp/
       └── logs/
           └── celeba/
               └── celeba_hq.ckpt
   ```

### Step 2 - Download the Pretrained SPNN Classifier

The pretrained SPNN model (trained on CelebA-HQ, 40 attributes, 256×256) is available on HuggingFace and is downloaded automatically at runtime:

```
https://huggingface.co/yamitehr/SPNN
```

No manual download is required.

### Step 3 - Run Reconstruction

9 example CelebA-HQ images are already provided - you can use them directly to test the pipeline.

From the repository root:

```bash
cd DDNM

python main.py \
  --ni \
  --config celeba_hq.yml \
  --path_y celeba_hq \
  --image_folder results
```

**Arguments:**

| Argument | Description | Default |
|---|---|---|
| `--config` | Config file under `configs/` | required |
| `--path_y` | Directory of input images | required |
| `--image_folder` | Output folder name under `exp/image_samples/` | `images` |
| `--ni` | Non-interactive mode (no prompts, safe for scripts/Slurm) | - |

Results are saved to `exp/image_samples/results/`.

**Additional NLBP tuning arguments** (no need to change for standard use - defaults are tuned for best results on CelebA-HQ):

- `--min_bp_step` - step threshold; `lambda2` is used for steps at or above this (earlier, noisier stages) and `lambda1` for steps below it (later, cleaner stages) (default: `600`)
- `--lambda1` - back-projection strength for steps below `min_bp_step` (default: `1.0`)
- `--lambda2` - back-projection strength for steps at or above `min_bp_step` (default: `0.5`)
- `--nlbp_stop_cond` - skips back-projection when the attribute error drops below this threshold (default: `0.11`)

---

## Part 2 - Training the SPNN Classifier on CelebA-HQ

### Dataset

We use [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ), a dataset of 30,000 high-resolution (1024×1024) face images with 40 binary attribute annotations. We resize images to 256×256 during training. For our experiments, we used the version available on [Kaggle](https://www.kaggle.com/datasets/liusonghua/celebamaskhq).

**Expected directory structure:**
```
CelebAMask-HQ/
├── CelebA-HQ-img/          ← .jpg face images (30,000)
└── CelebAMask-HQ-attribute-anno.txt
```

**Option A - Provide your own path:**
```bash
python train_celeba.py --dataset_path /path/to/CelebAMask-HQ --is_forward_train --is_r_opt
```

**Option B - Auto-download via kagglehub** (requires [Kaggle API credentials](https://www.kaggle.com/docs/api)):
```bash
pip install kagglehub
python train_celeba.py --is_forward_train --is_r_opt
```

### Training

The script is split into two sequential stages, each controlled by a flag:

**Stage 1 - Forward training** (`--is_forward_train`): trains the SPNN classifier end-to-end with a combination of BCE classification loss, right-inverse loss, and image reconstruction loss.

**Stage 2 - r-network optimisation** (`--is_r_opt`): trains the auxiliary network `r` to predict the discarded null-space components from the compressed output, steering the pseudo-inverse toward the unique Natural Non-Linear Pseudo-Inverse - the minimal-norm solution on the pre-image manifold - rather than an arbitrary generalized inverse.

You can run both stages in one command:
```bash
python train_celeba.py \
  --dataset_path /path/to/CelebAMask-HQ \
  --is_forward_train \
  --is_r_opt \
  --img_size 256 \
  --epoch 10 \
  --batch_size 256 \
  --mix_type cayley
```

**Key training arguments:**

| Argument | Description | Default |
|---|---|---|
| `--dataset_path` | Path to CelebAMask-HQ root | auto-download |
| `--img_size` | Input resolution - must be `32`, `64`, or `256` | `256` |
| `--epoch` | Number of training epochs | `10` |
| `--batch_size` | Batch size | `256` |
| `--lr` | Learning rate | `2e-4` |
| `--mix_type` | Mixing layer type: `cayley` or `householder` | `cayley` |
| `--is_forward_train` | Run Stage 1 (forward training) | off |
| `--is_r_opt` | Run Stage 2 (r-network optimisation) | off |
| `--checkpoint_dir` | Where to save checkpoints | `check_points/` |
| `--lambda_bce` | Weight for classification loss | `1.0` |
| `--lambda_right_inverse` | Weight for right-inverse loss | `40.0` |
| `--lambda_img_rec` | Weight for image reconstruction loss | `40.0` |

Checkpoints are saved under `check_points/`: Stage 1 produces `best_model.pth` and Stage 2 produces `best_model_r_opt_real.pth`. After each stage, Penrose identity metrics are evaluated automatically on the test set.

---

## Part 3 - DIY Architecture (Custom Input Size)

For input sizes other than 32×32, 64×64, or 256×256, you can define your own block sequence using the `layer_channels` argument.

> **Recommended:** For best results, use our validated built-in architectures for `(3, 32, 32)`, `(3, 64, 64)`, and `(3, 256, 256)` (the default when `layer_channels` is not set). The DIY path is intended for non-standard inputs.

### Limitations

**1. Classification tasks only.** The current architecture is designed exclusively for classification. The network is built so that the output is always a flat vector: the final block must produce a spatial size of 1×1, giving a tensor of shape `[batch, num_classes, 1, 1]` that is squeezed into `[batch, num_classes]` logits. The user defines `num_classes` (the number of output classes), and the block design must be chosen so that the spatial dimensions reach exactly 1×1 at the end.

**2. Input H and W must be equal (square).** `PixelUnshuffleBlock` uses a single downscale factor `r` applied to both H and W equally. For the spatial dimensions to reach exactly 1×1 at the end, H and W must be equal and fully divisible by your chosen sequence of `r` values.

### Building Blocks

Two block types are available:

**`PixelUnshuffleBlock(r)`**
Rearranges spatial pixels into channels (no learned parameters). Reduces spatial size by factor `r` and multiplies channels by `r²`.
```
[C, H, W]  →  [C × r², H/r, W/r]
```
- `r` must divide both H and W exactly.
- Using `PixelUnshuffleBlock` is not mandatory, but we found it consistently improves results. By trading spatial resolution for channel depth, it enables the subsequent SPNN blocks to capture semantic features at multiple scales while progressively reducing dimensionality - as described in the paper's multi-scale architecture design. This is especially important when the input has few channels (e.g. RGB images with 3 channels), where starting directly with a `ConvPINNBlock` would give the network very little to work with.

**`ConvPINNBlock(in_ch, out_ch, hidden, scale_bound, feat_size)`**
A coupling layer that maps `in_ch` → `out_ch` channels (with `out_ch < in_ch`), discarding the remainder as invertible latents.
- `hidden`: internal width of the ConvMLP networks (`s`, `t`, `r`). A safe default is `64` for small inputs; use `128` for larger or more complex inputs (matches the built-in architectures).
- `scale_bound`: bounds the affine scale activations via `tanh(x) * scale_bound`. We use `2.0` across all our experiments - no need to change this.
- `feat_size`: the spatial size of the input feature map at this block. Controls the internal ConvMLP architecture:

  | `feat_size` | Architecture used |
  |---|---|
  | `> 1` (even) | U-Net style: stride-2 down → process → stride-2 up |
  | `1` | Pointwise 1×1 convolutions |
  | `None` | Safe default: 3×3 convolutions, no stride (works for any size) |

  Use `feat_size=None` when unsure - it always works correctly.

### Design Tips

1. **Alternate PixelUnshuffleBlock and ConvPINNBlock.** PixelUnshuffleBlock rearranges spatial pixels into the channel dimension. ConvPINNBlock is the surjective block that reduces channels (`out_ch < in_ch`). Interleaving them keeps the channel counts manageable: unshuffle first inflates channels from spatial rearrangement, then ConvPINNBlock compresses them down to the meaningful representation.
2. **Plan your `r` values so the spatial dims reach exactly 1×1**, and set `num_classes` to match the last block's `out_ch`. Since the final spatial size must be 1×1, `num_classes == out_ch` of the last ConvPINNBlock.
3. **`out_ch` must be strictly less than `in_ch`** in each ConvPINNBlock. The discarded `in_ch - out_ch` channels are stored as invertible latents for exact reconstruction. Violating this will cause an error.

### Example

Input: `(3, 16, 16)` → 7 output classes.

```python
from models import SPNN, ConvPINNBlock, PixelUnshuffleBlock

# Channel flow:
#   [3, 16, 16]
#   → PixelUnshuffleBlock(r=4)         → [48, 4, 4]
#   → ConvPINNBlock(48 → 12, feat=4)   → [12, 4, 4]
#   → PixelUnshuffleBlock(r=4)         → [192, 1, 1]
#   → ConvPINNBlock(192 → 7, feat=1)   → [7, 1, 1]

model = SPNN(
    img_ch=3,
    num_classes=7,      # must equal out_ch of the last ConvPINNBlock
    hidden=64,          # passed to SPNN but unused in DIY mode - ConvPINNBlock hidden is set per-block
    scale_bound=2.0,    # same - set per-block in layer_channels
    img_size=16,        # any value not in (32, 64, 256) triggers the DIY path
    layer_channels=[
        (PixelUnshuffleBlock, {"r": 4}),
        (ConvPINNBlock, {"in_ch": 48,  "out_ch": 12, "hidden": 64, "scale_bound": 2.0, "feat_size": 4}),
        (PixelUnshuffleBlock, {"r": 4}),
        (ConvPINNBlock, {"in_ch": 192, "out_ch": 7,  "hidden": 64, "scale_bound": 2.0, "feat_size": 1}),
    ],
)
```

---

## Project Structure

```
.
├── models.py              # SPNN, PINN, ConvPINNBlock, PixelUnshuffleBlock
├── train.py               # CelebATrainer (forward + r-opt training loops)
├── train_celeba.py        # Main training entry point
├── data_loader.py         # CelebAMask-HQ dataset loader
├── diagnostics.py         # Penrose identity checks and g' norm evaluation
├── logger.py              # Logging utilities
├── utils.py               # Argument validation helpers
├── requirements.txt
└── DDNM/                  # Diffusion + NLBP reconstruction pipeline
    ├── main.py
    ├── configs/
    │   └── celeba_hq.yml
    ├── guided_diffusion/  # Diffusion model and sampling logic
    │   ├── diffusion.py
    │   ├── unet.py
    │   └── ...
    └── exp/
        ├── logs/
        │   └── celeba/    # Place celeba_hq.ckpt here
        └── datasets/
            └── celeba_hq/
                └── face/  # 9 example input images provided
```

---

## Citation

```bibtex
@misc{spnn2026pseudo_invertible,
  title        = {Pseudo-Invertible Neural Networks},
  author       = {Yamit Ehrlich and Nimrod Berman and Assaf Shocher},
  year         = {2026},
  note         = {Preprint, under review},
  howpublished = {\url{https://github.com/yamitehr/SPNN}},
  url          = {https://github.com/yamitehr/SPNN}
}
```

---

## Acknowledgements

The diffusion backbone is based on [SDEdit](https://github.com/ermongroup/SDEdit). The NLBP sampling pipeline builds on [DDNM](https://github.com/wyhuai/DDNM).
