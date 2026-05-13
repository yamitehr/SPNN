import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models import SPNN, ConvPINNBlock, PixelUnshuffleBlock
from huggingface_hub import hf_hub_download

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import download

import torchvision.utils as tvu
import torch.nn.functional as F

from guided_diffusion.models import Model
from guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
import random


# bb-edit: pixel-space "inverse-inpainting" target.
# Same NLBP step as diffusion_copy.py, but only the bbox-region of the BP'd
# x0_t_hat is pasted back onto x0_t; xt_next is then formed from that
# (partially-BP'd) x0_t. Outside the box the diffusion runs free.
# Box: first GT box of x_orig (any class).


def smooth_paste(im1, im2, topleft, h, w, smoothness=1.5, alpha=1.0):
    H, W = im1.shape[-2:]
    top, left = topleft
    cy, cx = top + h / 2, left + w / 2
    sy, sx = smoothness * h, smoothness * w
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=im1.device),
        torch.arange(W, dtype=torch.float32, device=im1.device),
        indexing='ij',
    )
    weight = torch.exp(-0.5 * (((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2))
    weight = alpha * weight
    return weight * im1 + (1 - weight) * im2


# y-domain paste helpers --------------------------------------------------
# Channel layout of y for SPNN-CenterNet: [hmap (nc), regs (2), wh (2)] at
# stride 4. We localize the BP target by overwriting peak-cell values of
# y_cur, leaving everything else equal to y_cur (zero gradient there).

VOC_NAMES_NO_BG = (
    "aeroplane bicycle bird boat bottle bus car cat chair cow "
    "diningtable dog horse motorbike person pottedplant sheep sofa "
    "train tvmonitor"
).split()
VOC_LABEL = {n: i for i, n in enumerate(VOC_NAMES_NO_BG)}


def zeroize_y_outside_predicted_peaks(y, nc=20, score_thresh=0.1, top_k=100):
    """Operator B: keep hmap only at predicted-peak cells; keep regs+wh
    only at those cells; zero everywhere else. Used both during training
    of the SPNN (so r learns to invert this collapsed y) and during NLBP
    inference (apply B before every Ap)."""
    B, _, H, W = y.shape
    sig = torch.sigmoid(y[:, :nc])
    peak = (sig == F.max_pool2d(sig, 3, stride=1, padding=1)) & (sig >= score_thresh)
    flat = (sig * peak.float()).view(B, -1)
    vals, idx = flat.topk(top_k, dim=1)
    keep = torch.zeros_like(flat, dtype=torch.bool)
    keep.scatter_(1, idx, vals > 0)
    keep = keep.view(B, nc, H, W)
    cell = keep.any(dim=1, keepdim=True)
    out = torch.zeros_like(y)
    out[:, :nc] = y[:, :nc] * keep.float()
    out[:, nc:nc + 4] = y[:, nc:nc + 4] * cell.float()
    return out

VOC_TO_IMAGENET = {
    "aeroplane": 404,    # airliner
    "bicycle": 671,      # mountain bike
    "bird": 14,          # indigo bunting
    "boat": 814,         # speedboat
    "bottle": 898,       # water bottle
    "bus": 779,          # school bus
    "car": 817,          # sports car
    "cat": 281,          # tabby
    "chair": 765,        # rocking chair
    "cow": 345,          # ox
    "diningtable": 532,  # dining table
    "dog": 207,          # golden retriever
    "horse": 339,        # sorrel
    "motorbike": 670,    # motor scooter
    "person": 981,       # ballplayer
    "pottedplant": 738,  # pot
    "sheep": 348,        # ram
    "sofa": 831,         # studio couch
    "train": 466,        # bullet train
    "tvmonitor": 851,    # television
}


def _gaussian_radius(det_size, min_overlap=0.7):
    h, w = det_size
    a1 = 1; b1 = h + w; c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 - np.sqrt(b1 * b1 - 4 * a1 * c1)) / 2
    a2 = 4; b2 = 2 * (h + w); c2 = (1 - min_overlap) * w * h
    r2 = (b2 - np.sqrt(b2 * b2 - 4 * a2 * c2)) / 8
    a3 = 4 * min_overlap; b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    r3 = (b3 + np.sqrt(b3 * b3 - 4 * a3 * c3)) / 2
    return float(min(r1, r2, r3))


def _box_to_fmap(box_xyxy, stride=4):
    bx1, by1, bx2, by2 = box_xyxy
    cx = (bx1 + bx2) / 2.0
    cy = (by1 + by2) / 2.0
    return {
        "px": int(cx / stride), "py": int(cy / stride),
        "h_f": (by2 - by1) / stride, "w_f": (bx2 - bx1) / stride,
        "sub_x": cx / stride - int(cx / stride),
        "sub_y": cy / stride - int(cy / stride),
    }


def _gauss_kernel_2d(diameter, device, dtype):
    radius = (diameter - 1) // 2
    sigma = max(diameter / 6.0, 1e-6)
    yy, xx = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=device, dtype=dtype),
        torch.arange(-radius, radius + 1, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.exp(-(yy * yy + xx * xx) / (2 * sigma * sigma))


def _crop_window(px, py, radius, H, W):
    x0 = max(0, px - radius); x1 = min(W, px + radius + 1)
    y0 = max(0, py - radius); y1 = min(H, py + radius + 1)
    kx0 = x0 - (px - radius); kx1 = kx0 + (x1 - x0)
    ky0 = y0 - (py - radius); ky1 = ky0 + (y1 - y0)
    return (y0, y1, x0, x1), (ky0, ky1, kx0, kx1)


def y_paste_ref(y_cur, y_ref, gts, nc=20, stride=4):
    """Step 1: copy peak-cell region of y_ref onto y_cur for each GT box.
    hmap[label]: paste a Gaussian-radius region. regs/wh: peak cell only."""
    H, W = y_cur.shape[-2:]
    y_target = y_cur.clone()
    for cls_name, box in gts:
        if cls_name not in VOC_LABEL:
            continue
        label = VOC_LABEL[cls_name]
        f = _box_to_fmap(box, stride)
        if not (0 <= f["px"] < W and 0 <= f["py"] < H):
            continue
        radius = max(0, int(_gaussian_radius(
            (max(1, int(np.ceil(f["h_f"]))),
             max(1, int(np.ceil(f["w_f"]))))
        )))
        (y0, y1, x0, x1), _ = _crop_window(f["px"], f["py"], radius, H, W)
        y_target[:, label, y0:y1, x0:x1] = y_ref[:, label, y0:y1, x0:x1]
        y_target[:, nc:nc + 2, f["py"], f["px"]] = \
            y_ref[:, nc:nc + 2, f["py"], f["px"]]
        y_target[:, nc + 2:nc + 4, f["py"], f["px"]] = \
            y_ref[:, nc + 2:nc + 4, f["py"], f["px"]]
    return y_target


def finetune_r_single_input(detector, A, Ap, target_input, steps, lr):
    """Test-time, single-image fine-tune of all r networks (frozen
    everything else). Mirrors train._train_r_opt_classifier:

        loss = || G(g†(y)) - G(0) ||^2

    where G(x) for SPNN is the concat of per-block latents x_1. Returns
    (saved_r_state, losses) so the caller can restore r afterwards.
    """
    saved = {}
    r_blocks = []
    for name, m in detector.named_modules():
        if isinstance(m, ConvPINNBlock):
            saved[name] = {k: v.detach().clone()
                           for k, v in m.r.state_dict().items()}
            r_blocks.append((name, m))

    for p in detector.parameters():
        p.requires_grad_(False)
    r_params = []
    for _name, m in r_blocks:
        for p in m.r.parameters():
            p.requires_grad_(True)
            r_params.append(p)
    if not r_params:
        return saved, []

    detector.train()
    opt = torch.optim.Adam(r_params, lr=lr)

    # Reference latents: forward of x=0, computed once (no_grad).
    with torch.no_grad():
        x_shape = (target_input.shape[0], 3, 256, 256)
        x_zero = torch.zeros(x_shape, device=target_input.device,
                             dtype=target_input.dtype)
        _, z_list_0 = A(x_zero, return_latents=True)
        parts_0 = [z.flatten(start_dim=1) for z in z_list_0 if z is not None]
        G_0 = torch.cat(parts_0, dim=1) if parts_0 else None

    losses = []
    for _ in range(steps):
        opt.zero_grad()
        x = Ap(target_input)
        _, z_list = A(x, return_latents=True)
        parts = [z.flatten(start_dim=1) for z in z_list if z is not None]
        if not parts or G_0 is None:
            break
        G_pinv = torch.cat(parts, dim=1)
        loss = (G_pinv - G_0).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(r_params, max_norm=1.0)
        opt.step()
        losses.append(loss.item())

    detector.eval()
    for p in detector.parameters():
        p.requires_grad_(False)
    return saved, losses


def restore_r_state(detector, saved):
    for name, m in detector.named_modules():
        if isinstance(m, ConvPINNBlock) and name in saved:
            m.r.load_state_dict(saved[name])


def finetune_r_reconstruction(detector, A, Ap, masked_y, x_orig,
                              steps, lr, w_rec=1.0, w_norm=0.1,
                              bb_mask_64=None):
    """Test-time r training combining image reconstruction loss
    ||Ap(masked_y) - x_orig||^2 with outside-BB natural-PInv norm loss.
    For single-image, ||Ap-x_orig|| is achievable (unlike at training where
    it averages over the dataset and is impossible). Returns
    (saved_r_state, losses_per_step)."""
    saved = {}
    r_blocks = []
    for name, m in detector.named_modules():
        if isinstance(m, ConvPINNBlock):
            saved[name] = {k: v.detach().clone()
                           for k, v in m.r.state_dict().items()}
            r_blocks.append((name, m))
    for p in detector.parameters():
        p.requires_grad_(False)
    r_params = []
    for _name, m in r_blocks:
        for p in m.r.parameters():
            p.requires_grad_(True)
            r_params.append(p)
    if not r_params:
        return saved, []

    z_list_0 = compute_G0(A, device=masked_y.device, dtype=masked_y.dtype)

    detector.train()
    opt = torch.optim.Adam(r_params, lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        x_pred = Ap(masked_y)
        rec_loss = (x_pred - x_orig).pow(2).mean()
        norm_loss = torch.zeros((), device=masked_y.device, dtype=masked_y.dtype)
        if w_norm > 0 and z_list_0 is not None:
            _, z_list = A(x_pred, return_latents=True)
            tot_sq = 0.0; tot_w = 0.0
            for z, z_0 in zip(z_list, z_list_0):
                if z is None or z_0 is None:
                    continue
                diff_sq = (z - z_0).pow(2)
                if bb_mask_64 is not None:
                    m_in = _mask_at_block_resolution(z, bb_mask_64)
                    if m_in is not None:
                        m_out = (1.0 - m_in)
                        diff_sq = diff_sq * m_out
                        tot_sq = tot_sq + diff_sq.sum()
                        tot_w = tot_w + m_out.sum() * z.shape[1]
                        continue
                tot_sq = tot_sq + diff_sq.sum()
                tot_w = tot_w + float(z.numel())
            if isinstance(tot_w, float):
                if tot_w > 0:
                    norm_loss = tot_sq / tot_w
            else:
                norm_loss = tot_sq / tot_w.clamp(min=1.0)
        loss = w_rec * rec_loss + w_norm * norm_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(r_params, max_norm=1.0)
        opt.step()
        losses.append((rec_loss.item(),
                       norm_loss.item() if torch.is_tensor(norm_loss) else norm_loss))
    detector.eval()
    for p in detector.parameters():
        p.requires_grad_(False)
    return saved, losses


def learn_bg_y_and_r(detector, A, Ap, y_target, bb_mask, init_y_bg,
                     steps, lr_r, lr_bg):
    """Per-image joint optimization of (r, y_BG) on the natural-PInv loss
    for the input y_input = M*y_target + (1-M)*y_BG. Treats outside-BB y as
    a free degree of freedom and finds its most-natural completion of the
    BB constraint. Returns (saved_r_state, learned_y_bg, losses)."""
    saved = {}
    r_blocks = []
    for name, m in detector.named_modules():
        if isinstance(m, ConvPINNBlock):
            saved[name] = {k: v.detach().clone()
                           for k, v in m.r.state_dict().items()}
            r_blocks.append((name, m))
    for p in detector.parameters():
        p.requires_grad_(False)
    r_params = []
    for _name, m in r_blocks:
        for p in m.r.parameters():
            p.requires_grad_(True)
            r_params.append(p)

    y_bg = init_y_bg.detach().clone().requires_grad_(True)

    # Reference latents (forward of x=0).
    with torch.no_grad():
        x_zero = torch.zeros((y_target.shape[0], 3, 256, 256),
                             device=y_target.device, dtype=y_target.dtype)
        _, z_list_0 = A(x_zero, return_latents=True)
        parts_0 = [z.flatten(start_dim=1) for z in z_list_0 if z is not None]
        G_0 = torch.cat(parts_0, dim=1) if parts_0 else None
    if G_0 is None or not r_params:
        return saved, y_bg.detach(), []

    detector.train()
    opt = torch.optim.Adam([
        {"params": r_params, "lr": lr_r},
        {"params": [y_bg], "lr": lr_bg},
    ])

    losses = []
    for _ in range(steps):
        opt.zero_grad()
        y_input = bb_mask * y_target + (1 - bb_mask) * y_bg
        x = Ap(y_input)
        _, z_list = A(x, return_latents=True)
        parts = [z.flatten(start_dim=1) for z in z_list if z is not None]
        if not parts:
            break
        G_pinv = torch.cat(parts, dim=1)
        loss = (G_pinv - G_0).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(r_params + [y_bg], max_norm=1.0)
        opt.step()
        losses.append(loss.item())
    detector.eval()

    for p in detector.parameters():
        p.requires_grad_(False)
    return saved, y_bg.detach(), losses


def setup_online_r_training(detector, lr):
    """Make r networks trainable, freeze the rest, return (optimizer, r_params).
    Caller is expected to set detector.train()/eval() around forward calls
    and to restore r_state afterwards if state should not persist."""
    saved = {}
    r_params = []
    for name, m in detector.named_modules():
        if isinstance(m, ConvPINNBlock):
            saved[name] = {k: v.detach().clone()
                           for k, v in m.r.state_dict().items()}
    for p in detector.parameters():
        p.requires_grad_(False)
    for m in detector.modules():
        if isinstance(m, ConvPINNBlock):
            for p in m.r.parameters():
                p.requires_grad_(True)
                r_params.append(p)
    opt = torch.optim.Adam(r_params, lr=lr) if r_params else None
    return saved, opt, r_params


def compute_G0(A, device, dtype):
    """Latents of forward(x=0) — the natural-PInv reference. Returns the list
    of per-block latents (some entries are None for non-coupling blocks)."""
    with torch.no_grad():
        x_zero = torch.zeros((1, 3, 256, 256), device=device, dtype=dtype)
        _, z_list_0 = A(x_zero, return_latents=True)
    return [z.detach() if z is not None else None for z in z_list_0]


def _mask_at_block_resolution(z, bb_mask_64):
    """Avg-pool the 64x64 bb_mask down to z's spatial size. Returns
    [1,1,H,W] mask in [0,1] (1 inside BB, fractional at coarser scales)."""
    H_z = z.shape[-1]
    H_m = bb_mask_64.shape[-1]
    if H_z == H_m:
        return bb_mask_64
    if H_z > H_m or H_m % H_z != 0:
        return None  # bail out for unsupported shapes
    scale = H_m // H_z
    return F.avg_pool2d(bb_mask_64, scale)


def online_r_step(detector, A, Ap, masked_inputs, opt, r_params, z_list_0,
                  max_steps, loss_threshold, bb_mask_64=None):
    """Gentle r-only gradient updates on the natural-PInv loss
    ||z - z_0||² across blocks, averaged over `masked_inputs`. If
    `bb_mask_64` is provided, the loss at each block is restricted to
    OUTSIDE-BB spatial cells — so r is only nudged where we actually
    want a min-norm completion; inside-BB latents stay free (the BB
    constraint is what we care about there). Stops when loss<threshold."""
    if opt is None or z_list_0 is None:
        return None, 0
    if torch.is_tensor(masked_inputs):
        masked_inputs = [masked_inputs]
    if not masked_inputs:
        return None, 0
    detector.train()
    final_loss = None
    n_taken = 0
    for k in range(max_steps):
        opt.zero_grad()
        per_input_losses = []
        for inp in masked_inputs:
            x = Ap(inp)
            _, z_list = A(x, return_latents=True)
            tot_sq = 0.0
            tot_w = 0.0
            for z, z_0 in zip(z_list, z_list_0):
                if z is None or z_0 is None:
                    continue
                diff_sq = (z - z_0).pow(2)
                if bb_mask_64 is not None:
                    m_in = _mask_at_block_resolution(z, bb_mask_64)
                    if m_in is not None:
                        m_out = (1.0 - m_in)
                        diff_sq = diff_sq * m_out  # broadcast over channels
                        tot_sq = tot_sq + diff_sq.sum()
                        tot_w = tot_w + m_out.sum() * z.shape[1]
                        continue
                tot_sq = tot_sq + diff_sq.sum()
                tot_w = tot_w + float(z.numel())
            if isinstance(tot_w, float):
                if tot_w <= 0:
                    continue
                per_input_losses.append(tot_sq / tot_w)
            else:
                per_input_losses.append(tot_sq / tot_w.clamp(min=1.0))
        if not per_input_losses:
            break
        loss = sum(per_input_losses) / len(per_input_losses)
        final_loss = loss.item()
        n_taken = k + 1
        if final_loss < loss_threshold:
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(r_params, max_norm=1.0)
        opt.step()
    detector.eval()
    return final_loss, n_taken


def make_y_bg_const(nc, fmap, device, dtype,
                    hmap_bg=-6.0, regs_bg=0.5, wh_bg=18.0):
    """Constant 'natural BG' y values (matched to empirical y-tensor stats)
    used to fill outside-BB cells before Ap. Shape [1, nc+4, fmap, fmap]."""
    y = torch.empty(1, nc + 4, fmap, fmap, device=device, dtype=dtype)
    y[:, :nc] = hmap_bg
    y[:, nc:nc + 2] = regs_bg
    y[:, nc + 2:nc + 4] = wh_bg
    return y


def apply_crop_fill(y, mask_op, fill_const):
    """If fill_const is None: mask_op * y (zero-fill). Else: blend
    mask_op * y + (1 - mask_op) * fill_const (BG-fill). Reduces to y
    when mask_op == 1.0 in either case."""
    if fill_const is None:
        return mask_op * y
    return mask_op * y + (1 - mask_op) * fill_const


def bb_mask_from_gts(gts, fmap, stride, device, dtype, falloff=0.0):
    """Union of BB regions across all GT boxes. With falloff=0 returns a
    hard 0/1 mask; with falloff>0 returns 1 inside each BB and a Gaussian
    falloff outside with sigma=`falloff` cells (taking the union via max
    across boxes). Returns shape [1, 1, fmap, fmap]."""
    if falloff <= 0:
        M = torch.zeros(1, 1, fmap, fmap, device=device, dtype=dtype)
        for _cls, box in gts:
            bx1, by1, bx2, by2 = box
            px1 = max(0, int(bx1 // stride))
            px2 = min(fmap, int(np.ceil(bx2 / stride)))
            py1 = max(0, int(by1 // stride))
            py2 = min(fmap, int(np.ceil(by2 / stride)))
            if px2 > px1 and py2 > py1:
                M[..., py1:py2, px1:px2] = 1.0
        return M
    yy, xx = torch.meshgrid(
        torch.arange(fmap, device=device, dtype=dtype),
        torch.arange(fmap, device=device, dtype=dtype), indexing="ij")
    M = torch.zeros(fmap, fmap, device=device, dtype=dtype)
    for _cls, box in gts:
        bx1, by1, bx2, by2 = box
        cx1, cx2 = bx1 / stride, bx2 / stride
        cy1, cy2 = by1 / stride, by2 / stride
        # Distance from each cell to the BB rectangle (0 inside).
        dx = (cx1 - xx).clamp(min=0) + (xx - cx2).clamp(min=0)
        dy = (cy1 - yy).clamp(min=0) + (yy - cy2).clamp(min=0)
        d2 = dx * dx + dy * dy
        m = torch.exp(-d2 / (2.0 * falloff * falloff))
        M = torch.maximum(M, m)
    return M.unsqueeze(0).unsqueeze(0)


def y_paste_class_model(y_cur, gts, class_model, std_scale=0.0,
                        nc=20, stride=4, override_box_dims=True):
    """Restricted-override version: replaces only the cells that the
    detector loss actually constrains —
      * hmap (all classes) inside a circular splat of radius
        gaussian_radius(h_f, w_f) — the same radius used by the training
        target's draw_umich_gaussian. Outside this radius, the loss's
        (1-target)^4 negative weight is ~1, so the network was strongly
        supervised; inside the splat the negative weight is small, so
        the network had freedom and we should mimic empirical structure.
      * wh, regs at the EXACT peak cell only — `_reg_loss` masks by
        `ind_masks` so only the peak cell is supervised.
    Cells outside these regions stay equal to y_cur, so BP error there
    is zero and the diffusion runs free.

    `class_model` is a dict from build_class_y_model.py with keys:
      - mean[24, win_sz, win_sz], std[24, win_sz, win_sz]
      - window_radius (= (win_sz - 1) // 2), label, n_instances
    """
    mean = class_model["mean"]
    std = class_model["std"]
    win_r = class_model["window_radius"]
    H, W = y_cur.shape[-2:]
    y_target = y_cur.clone()
    for cls_name, box in gts:
        bx1, by1, bx2, by2 = box
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0
        px = int(cx / stride)
        py = int(cy / stride)
        if not (0 <= px < W and 0 <= py < H):
            continue

        # Sample (or use mean of) the window.
        if std_scale > 0:
            window = mean + std_scale * std * torch.randn_like(mean)
        else:
            window = mean
        window = window.to(y_cur.device, dtype=y_cur.dtype)

        # Splat radius — same formula as training (gaussian_radius).
        w_f = (bx2 - bx1) / stride
        h_f = (by2 - by1) / stride
        r_splat = max(0, int(_gaussian_radius(
            (max(1, int(np.ceil(h_f))),
             max(1, int(np.ceil(w_f)))))))
        r_splat = min(r_splat, win_r)  # don't reach beyond the model window

        # 1) Override hmap (all 20 classes) inside a circular splat of
        #    radius r_splat around the peak cell. Cells outside the
        #    splat stay at y_cur (zero BP error there).
        if r_splat > 0:
            y0 = max(0, py - r_splat); y1 = min(H, py + r_splat + 1)
            x0 = max(0, px - r_splat); x1 = min(W, px + r_splat + 1)
            wy0 = (win_r - r_splat) + (y0 - (py - r_splat))
            wx0 = (win_r - r_splat) + (x0 - (px - r_splat))
            wy1 = wy0 + (y1 - y0)
            wx1 = wx0 + (x1 - x0)
            # Build circular mask local to the placed region.
            yy_local, xx_local = torch.meshgrid(
                torch.arange(y0 - py, y1 - py, device=y_cur.device,
                             dtype=y_cur.dtype),
                torch.arange(x0 - px, x1 - px, device=y_cur.device,
                             dtype=y_cur.dtype),
                indexing="ij")
            circ = (yy_local * yy_local + xx_local * xx_local
                    <= r_splat * r_splat)  # [hh, ww]
            circ = circ.unsqueeze(0)  # [1, hh, ww] for broadcasting over channels
            # Override only the hmap channels (first nc).
            patch_hmap = window[:nc, wy0:wy1, wx0:wx1]
            cur_hmap = y_target[:, :nc, y0:y1, x0:x1]
            y_target[:, :nc, y0:y1, x0:x1] = torch.where(
                circ, patch_hmap.unsqueeze(0).expand_as(cur_hmap), cur_hmap)

        # 2) Override wh, regs at the peak cell only.
        if override_box_dims:
            sub_x = cx / stride - px
            sub_y = cy / stride - py
            y_target[:, nc, py, px] = sub_x
            y_target[:, nc + 1, py, px] = sub_y
            y_target[:, nc + 2, py, px] = w_f
            y_target[:, nc + 3, py, px] = h_f
        else:
            y_target[:, nc:nc + 4, py, px] = window[nc:nc + 4, win_r, win_r]
    return y_target


def y_paste_synth(y_cur, gts, nc=20, stride=4,
                  peak_logit=0.0, bg_logit=-6.0, other_peak_logit=-5.0,
                  smooth_hmap=False, smoothness=1.0):
    """Synth y_target with in-distribution per-class amplitudes.
    For target class: Gaussian splat from bg_logit at splat edge to
    peak_logit at center. For OTHER classes: Gaussian splat from
    bg_logit to other_peak_logit at center (default ~-5, matches the
    natural 'peak cell, other class' mean — the slight elevation of
    other classes at a peak). regs/wh: peak cell only.
    If smooth_hmap, blend the target-class hmap into y_cur with a
    Gaussian weight whose std is `smoothness * radius`."""
    H, W = y_cur.shape[-2:]
    y_target = y_cur.clone()
    device = y_cur.device
    dtype = y_cur.dtype
    for cls_name, box in gts:
        if cls_name not in VOC_LABEL:
            continue
        label = VOC_LABEL[cls_name]
        f = _box_to_fmap(box, stride)
        if not (0 <= f["px"] < W and 0 <= f["py"] < H):
            continue
        radius = max(0, int(_gaussian_radius(
            (max(1, int(np.ceil(f["h_f"]))),
             max(1, int(np.ceil(f["w_f"]))))
        )))
        diameter = 2 * radius + 1
        (y0, y1, x0, x1), (ky0, ky1, kx0, kx1) = _crop_window(
            f["px"], f["py"], radius, H, W)
        kernel = _gauss_kernel_2d(diameter, device, dtype)
        patch = kernel[ky0:ky1, kx0:kx1]
        synth_logit = bg_logit + patch * (peak_logit - bg_logit)
        synth_other = bg_logit + patch * (other_peak_logit - bg_logit)
        if smooth_hmap:
            sigma = max(smoothness * max(radius, 1), 1.0)
            yy, xx = torch.meshgrid(
                torch.arange(H, device=device, dtype=dtype),
                torch.arange(W, device=device, dtype=dtype), indexing="ij",
            )
            wfull = torch.exp(-((yy - f["py"]) ** 2 + (xx - f["px"]) ** 2)
                              / (2 * sigma * sigma))
            synth_full = torch.full_like(y_cur[0, label], bg_logit)
            synth_full[y0:y1, x0:x1] = synth_logit
            y_target[:, label] = (
                wfull * synth_full + (1 - wfull) * y_cur[:, label]
            )
        else:
            y_target[:, label, y0:y1, x0:x1] = synth_logit
        # Suppress other classes inside splat to natural other-peak value.
        # In real detections, at a peak cell other classes are at ~-5
        # (slight elevation above BG -6); we mirror that with the same
        # Gaussian falloff so the per-cell distribution stays natural.
        if other_peak_logit != bg_logit:
            for c in range(nc):
                if c == label:
                    continue
                y_target[:, c, y0:y1, x0:x1] = torch.minimum(
                    y_target[:, c, y0:y1, x0:x1], synth_other)
        # regs[0]=sub_x, regs[1]=sub_y; wh[0]=w, wh[1]=h
        y_target[:, nc, f["py"], f["px"]] = f["sub_x"]
        y_target[:, nc + 1, f["py"], f["px"]] = f["sub_y"]
        y_target[:, nc + 2, f["py"], f["px"]] = f["w_f"]
        y_target[:, nc + 3, f["py"], f["px"]] = f["h_f"]
    return y_target


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device


        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self):
        cls_fn = None

        if self.config.model.type == 'simple':
            model = Model(self.config)
            if self.config.data.dataset == 'CelebA_HQ':
                ckpt = os.path.join(self.args.exp, "logs/celeba/celeba_hq.ckpt")
                if not os.path.exists(ckpt):
                    download('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt', ckpt)
            else:
                raise ValueError(f"Unsupported dataset for 'simple' model type: {self.config.data.dataset}")
            model.load_state_dict(torch.load(ckpt, map_location=self.device, weights_only=False))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        elif self.config.model.type == 'openai':
            config_dict = vars(self.config.model)
            model = create_model(**config_dict)
            if self.config.model.use_fp16:
                model.convert_to_fp16()
            if getattr(self.config.model, 'class_cond', False):
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_diffusion.pt' % (
                    self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    # Fix: class-cond ckpt is xx_diffusion.pt, NOT _uncond.
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion.pt' % (
                            self.config.data.image_size, self.config.data.image_size), ckpt)
            else:
                ckpt = os.path.join(self.args.exp, "logs/imagenet/256x256_diffusion_uncond.pt")
                if not os.path.exists(ckpt):
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt',
                        ckpt)

            model.load_state_dict(torch.load(ckpt, map_location=self.device, weights_only=False))
            model.to(self.device)
            model.eval()
            model = torch.nn.DataParallel(model)

        else:
            raise ValueError(f"Unknown model type: {self.config.model.type}")

        print('Run Simplified DDNM.',
              f'{self.config.time_travel.T_sampling} sampling steps.',
              f'travel_length = {self.config.time_travel.travel_length},',
              f'travel_repeat = {self.config.time_travel.travel_repeat}.'
             )
        self.simplified_ddnm_plus(model)


    # ===================================================================
    # A / A† builders for the three supported tasks
    # ===================================================================

    def _build_imagenet_classification_A_Ap(self, args):
        """Original SPNN-classifier path: A: image → class logits."""
        spnn_ckpt = getattr(args, 'spnn_ckpt', None)
        num_classes = getattr(args, 'spnn_num_classes', 10)
        mix_type = getattr(args, 'spnn_mix_type', 'householder')
        scale_bound = getattr(args, 'spnn_scale_bound', 1.0)
        hidden = 256

        layer_channels = [
            (PixelUnshuffleBlock, {"r": 4}),
            (ConvPINNBlock, {"in_ch": 48, "out_ch": 24, "hidden": hidden,
                             "scale_bound": scale_bound, "feat_size": 64, "mix_type": mix_type}),
            (ConvPINNBlock, {"in_ch": 24, "out_ch": 12, "hidden": hidden,
                             "scale_bound": scale_bound, "feat_size": 64, "mix_type": mix_type}),
            (PixelUnshuffleBlock, {"r": 4}),
            (ConvPINNBlock, {"in_ch": 192, "out_ch": 96, "hidden": hidden,
                             "scale_bound": scale_bound, "feat_size": 16, "mix_type": mix_type}),
            (ConvPINNBlock, {"in_ch": 96, "out_ch": 48, "hidden": hidden,
                             "scale_bound": scale_bound, "feat_size": 16, "mix_type": mix_type}),
            (PixelUnshuffleBlock, {"r": 4}),
            (ConvPINNBlock, {"in_ch": 768, "out_ch": 192, "hidden": hidden,
                             "scale_bound": scale_bound, "feat_size": 4, "mix_type": mix_type}),
            (PixelUnshuffleBlock, {"r": 4}),
            (ConvPINNBlock, {"in_ch": 3072, "out_ch": 1024, "hidden": hidden,
                             "scale_bound": scale_bound, "feat_size": 1, "mix_type": mix_type}),
            (ConvPINNBlock, {"in_ch": 1024, "out_ch": num_classes, "hidden": hidden,
                             "scale_bound": scale_bound, "feat_size": 1, "mix_type": mix_type}),
        ]
        classifier = SPNN(img_ch=3, num_classes=num_classes, img_size=256,
                          layer_channels=layer_channels).to(self.device)

        assert spnn_ckpt is not None, "Must provide --spnn_ckpt for ImageNet DDNM"
        raw = torch.load(spnn_ckpt, map_location=self.device, weights_only=False)
        state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) and "state_dict" in raw else raw
        classifier.load_state_dict(state_dict)
        print(f"Loaded SPNN classifier ({num_classes} classes, {mix_type}) from {spnn_ckpt}")

        img_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        img_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        def diffusion_to_spnn(x):
            x01 = (x + 1.0) / 2.0
            return (x01 - img_mean) / img_std

        def spnn_to_diffusion(x):
            x01 = x * img_std + img_mean
            return x01 * 2.0 - 1.0

        A = lambda z, **kw: classifier(diffusion_to_spnn(z), **kw)
        Ap = lambda logits, **kw: spnn_to_diffusion(classifier.pinv(logits, **kw))
        return classifier, A, Ap

    def _build_celeba_classification_A_Ap(self):
        """CelebA SPNN classifier path."""
        classifier = SPNN(img_ch=3, num_classes=40, hidden=128,
                          scale_bound=2.0, img_size=256).to(self.device)
        ckpt_path = hf_hub_download(repo_id="yamitehr/SPNN",
                                    filename="spnn_celebahq_256.pth")
        print(f"Loading classifier from {ckpt_path}")
        classifier.load_state_dict(torch.load(ckpt_path, map_location=self.device,
                                              weights_only=False))
        A = lambda z, **kw: classifier(z, **kw)
        Ap = lambda logits, **kw: classifier.pinv(logits, **kw)
        return classifier, A, Ap

    def _build_detection_A_Ap(self, args):
        """SPNN-CenterNet path: A: image → full detector output (post-everything),
        a single tensor [B, num_classes+4, 64, 64].

        Bijective chain through ALL layers:
          diffusion (RGB, [-1,1])
            -> diffusion_to_detector: BGR + ImageNet-normalized
            -> detector.spnn  -> raw [B, 24, 64, 64]
            -> hmap channels: * scale, then orthogonal mix (if used), then + bias
            -> regs / w_h_ pass through unchanged
            -> y = cat([hmap, regs, w_h_], dim=1)

        A† reverses every step. The wrapper's affine + (optional) orthogonal
        mixer are both bijective, so they don't introduce extra latents — the
        latent list comes entirely from `detector.spnn`.
        """
        # Lazy import so the classification path doesn't pay the import cost
        # if centernet_ref isn't on sys.path.
        import sys as _sys
        _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        _centernet_ref = os.path.join(_project_root, "centernet_ref")
        for p in (_project_root, _centernet_ref):
            if p not in _sys.path:
                _sys.path.insert(0, p)
        from nets.spnn_centernet_copy import get_spnn_centernet  # noqa: E402

        assert args.detector_ckpt is not None, \
            "Must provide --detector_ckpt for detection DDNM"

        head_mix_reflections = (args.detector_head_mix_reflections
                                if args.detector_head_mix_reflections > 0 else None)
        detector = get_spnn_centernet(
            num_classes=args.detector_num_classes,
            pretrained_backbone=None,  # we're loading the full ckpt below
            hmap_init_scale=args.detector_hmap_init_scale,
            hmap_init_bias=args.detector_hmap_init_bias,
            head_mode=args.detector_head_mode,
            head_mix_type=args.detector_head_mix_type,
            head_mix_reflections=head_mix_reflections,
            deep_det_head=args.detector_deep_det_head,
            deep_head_hidden=args.detector_deep_head_hidden,
            two_block_head=args.detector_two_block_head,
            freeze_backbone=False,
            no_hmap_scale=getattr(args, 'no_hmap_scale', False),
            no_hmap_bias=getattr(args, 'no_hmap_bias', False),
            internal_head_affine=getattr(args, 'internal_head_affine', False),
        ).to(self.device)

        raw = torch.load(args.detector_ckpt, map_location=self.device,
                         weights_only=False)
        if isinstance(raw, dict) and 'state_dict' in raw:
            state = raw['state_dict']
        elif isinstance(raw, dict) and 'model' in raw:
            state = raw['model']
        else:
            state = raw
        # Strip 'module.' prefix from DataParallel / DDP saves
        state = {k[7:] if k.startswith('module.') else k: v
                 for k, v in state.items()}
        missing, unexpected = detector.load_state_dict(state, strict=False)
        print(f"[detector] Loaded {args.detector_ckpt} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
        if missing:
            print(f"[detector] first missing keys: {missing[:6]}")
        if unexpected:
            print(f"[detector] first unexpected keys: {unexpected[:6]}")
        detector.eval()
        for p in detector.parameters():
            p.requires_grad_(False)

        nc = args.detector_num_classes  # 20 for VOC
        img_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        img_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        def diffusion_to_detector(x):
            """[-1,1] RGB  →  BGR + ImageNet-normalized.

            VOC pascal.py loaded images via cv2.imread (BGR) and never swapped
            to RGB before applying RGB-named ImageNet stats — that's the
            CenterNet convention. To match the detector's training distribution
            we replicate it: RGB→BGR swap, then (x/255 − μ)/σ-equivalent step.
            (We start in [0,1] post-denorm, so it's just (x − μ)/σ.)
            """
            x01 = (x + 1.0) / 2.0                # [-1,1] RGB → [0,1] RGB
            x_bgr = x01[:, [2, 1, 0]]            # RGB → BGR
            return (x_bgr - img_mean) / img_std

        def detector_to_diffusion(x):
            """BGR + ImageNet-normalized  →  [-1,1] RGB."""
            x01_bgr = x * img_std + img_mean     # → [0,1] BGR
            x01_rgb = x01_bgr[:, [2, 1, 0]]      # BGR → RGB
            return x01_rgb * 2.0 - 1.0           # → [-1,1] RGB

        def _wrapper_post_spnn(raw_24ch):
            """Manually replay the wrapper's post-SPNN ops on raw [B,24,H,W].
            Used only by A's return_latents path (where we need access to
            SPNN's latents, which the wrapper's forward doesn't expose).

            Mirrors SPNNCenterNet.forward exactly: scale → mix → bias, with
            scale/bias skipped when use_hmap_scale / use_hmap_bias are off.
            When the head is configured with --no_hmap_scale --no_hmap_bias
            --internal_head_affine, the affine effect lives entirely inside
            spnn (in the last head block's s/t), so this function is just an
            (optional) orthogonal mix."""
            hmap = raw_24ch[:, :nc]
            if getattr(detector, 'use_hmap_scale', True):
                hmap = hmap * detector.hmap_scale
            if detector.head_mode == 'orthogonal_mix':
                hmap = detector.hmap_mix(hmap)
            if getattr(detector, 'use_hmap_bias', True):
                hmap = hmap + detector.hmap_bias
            regs = raw_24ch[:, nc:nc + 2]
            w_h_ = raw_24ch[:, nc + 2:]
            return torch.cat([hmap, regs, w_h_], dim=1)

        def A(z, return_latents=False):
            x_det = diffusion_to_detector(z)
            if return_latents:
                raw, latents = detector.spnn(x_det, return_latents=True)
                y = _wrapper_post_spnn(raw)
                return y, latents
            out = detector(x_det)                # SPNNCenterNet.forward
            hmap, regs, w_h_ = out[0]
            return torch.cat([hmap, regs, w_h_], dim=1)

        def Ap(y, latents=None):
            hmap = y[:, :nc]
            regs = y[:, nc:nc + 2]
            w_h_ = y[:, nc + 2:]
            hmap_raw = detector.hmap_to_raw(hmap)        # invert affine + mix
            raw = torch.cat([hmap_raw, regs, w_h_], dim=1)
            x_det = detector.spnn.pinv(raw, latents=latents)
            return detector_to_diffusion(x_det)

        return detector, A, Ap

    # 0..19 → VOC class name. The pascal_test2007.json categories list is
    # already alphabetical (id 1=aeroplane, ..., 20=tvmonitor), and SPNN's
    # output channels follow the same 0..19 ordering — so this list aligns
    # with the class index returned by ctdet_decode().
    _VOC_NAMES = [
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
        "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor",
    ]

    def _save_detection_grid(self, idx, results_dir, voc_dataset, classes,
                             orig_chw_01, y_orig, *, nc=20,
                             score_thresh=0.05, max_boxes=15, K=100):
        """2-panel figure: orig+GT (red) | orig+SPNN-pred (cyan, class:score).

        orig_chw_01: torch tensor [3, H, W], values in [0, 1] (CPU).
        y_orig:      detector output [1, nc+4, H/4, W/4] (any device).
        """
        # Lazy imports — only when running the detection task.
        import sys as _sys
        _project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", ".."))
        _centernet_ref = os.path.join(_project_root, "centernet_ref")
        for _p in (_project_root, _centernet_ref):
            if _p not in _sys.path:
                _sys.path.insert(0, _p)
        from utils.post_process import ctdet_decode  # noqa: E402
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        stride = 4  # CenterNet stride at 256→64
        H = orig_chw_01.shape[-1]

        voc_id = classes[0].item() if classes.dim() > 0 else classes.item()
        gt = voc_dataset.get_gt_in_image_coords(voc_id)

        with torch.no_grad():
            hmap = y_orig[:, :nc]
            regs = y_orig[:, nc:nc + 2]
            w_h_ = y_orig[:, nc + 2:nc + 4]
            dets = ctdet_decode(hmap, regs, w_h_, K=K)[0].cpu().numpy()
        # sort by score descending, drop low-confidence
        dets = dets[np.argsort(-dets[:, 4])]

        rgb = orig_chw_01.clamp(0, 1).permute(1, 2, 0).numpy()

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # --- Left: GT ---
        axes[0].imshow(rgb)
        axes[0].set_title(f"img {idx} (voc_id={voc_id})  GT  [{len(gt)} objs]",
                          fontsize=9)
        for name, (x1, y1, x2, y2) in gt:
            axes[0].add_patch(plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor="red", lw=1.5))
            axes[0].text(x1 + 1, y1 + 8, name,
                         color="white", fontsize=7,
                         bbox=dict(facecolor="red", alpha=0.6, pad=0.5))
        axes[0].set_xlim(0, H); axes[0].set_ylim(H, 0); axes[0].axis("off")

        # --- Right: SPNN predictions ---
        axes[1].imshow(rgb)
        shown = 0
        for d in dets:
            x1, y1, x2, y2, sc, cls = d
            if sc < score_thresh:
                continue
            x1p, y1p, x2p, y2p = (x1 * stride, y1 * stride,
                                  x2 * stride, y2 * stride)
            cls = int(cls)
            cname = self._VOC_NAMES[cls] if 0 <= cls < len(self._VOC_NAMES) else str(cls)
            axes[1].add_patch(plt.Rectangle(
                (x1p, y1p), x2p - x1p, y2p - y1p,
                fill=False, edgecolor="cyan", lw=1.5))
            axes[1].text(x1p + 1, y1p + 8, f"{cname}:{sc:.2f}",
                         color="white", fontsize=7,
                         bbox=dict(facecolor="darkblue", alpha=0.65, pad=0.5))
            shown += 1
            if shown >= max_boxes:
                break
        axes[1].set_title(
            f"SPNN top-K (score≥{score_thresh}, n={shown})", fontsize=9)
        axes[1].set_xlim(0, H); axes[1].set_ylim(H, 0); axes[1].axis("off")

        fig.tight_layout()
        out_path = os.path.join(results_dir, f"grid_dets_{idx}.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    def simplified_ddnm_plus(self, model):
        args, config = self.args, self.config

        # ---- dataset selection ---------------------------------------------
        # For the detection task we use VOC2007 test split. The default
        # get_dataset(...) doesn't know about VOC; bypass it.
        voc_dataset = None  # unwrapped handle, used for GT-on-grid overlay
        if getattr(args, "task", "classification") == "detection":
            from datasets.voc import VOCValForDDNM
            test_dataset = VOCValForDDNM(args.voc_data_dir,
                                         image_size=config.data.image_size)
            voc_dataset = test_dataset
        else:
            _, test_dataset = get_dataset(args, config)

        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')

        def seed_worker(worker_id):
            worker_seed = args.seed % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # ---- A / A† selection ---------------------------------------------
        if getattr(args, "task", "classification") == "detection":
            classifier, A, Ap = self._build_detection_A_Ap(args)
        elif config.data.dataset == 'ImageNet':
            classifier, A, Ap = self._build_imagenet_classification_A_Ap(args)
        else:
            classifier, A, Ap = self._build_celeba_classification_A_Ap()

        classifier.eval()

        # Optional: load empirical class-y model.
        _class_model = None
        if getattr(args, "y_class_model_path", None):
            _class_model = torch.load(args.y_class_model_path,
                                       map_location=self.device,
                                       weights_only=False)
            print(f"[class-y-model] Loaded "
                  f"{args.y_class_model_path} "
                  f"(class={_class_model.get('cls')}, "
                  f"n_instances={_class_model.get('n_instances')})")

        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        pbar = tqdm.tqdm(val_loader)
        for x_orig, classes in pbar:
            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)

            y = A(x_orig)

            if config.sampling.batch_size != 1:
                raise ValueError("please change the config file to set batch size as 1")

            # bb-edit: take the first GT box of x_orig (any class) as the
            # pixel-paste region, in 256-px coords. If this image has no
            # GT boxes that survive the center-crop, target_box stays None
            # and no paste happens.
            target_box = None
            target_cls = None
            gts_y_paste = []  # list of (cls_name, box_xyxy) for y-paste
            if voc_dataset is not None:
                gts = voc_dataset.get_gt_in_image_coords(int(classes.item()))
                if gts:
                    target_cls, target_box = gts[0]
                    if args.y_paste_mode == "synth_multi":
                        gts_y_paste = list(gts)
                    elif args.y_paste_mode in (
                            "ref", "synth_single", "synth_smooth_hmap",
                            "crop_pinv_ref", "crop_pinv_synth"):
                        gts_y_paste = [gts[0]]
                    # Optional class override for synth modes — keep the GT
                    # box but insert a different class at that location.
                    _ovr = getattr(args, "synth_override_class", None)
                    if _ovr and args.y_paste_mode in (
                            "synth_single", "synth_multi", "synth_smooth_hmap",
                            "crop_pinv_synth"):
                        gts_y_paste = [(_ovr, box) for (_cls, box) in gts_y_paste]
                        target_cls = _ovr

            # Truly no-ref: explicit box(es) override any GT-derived box.
            _box_str = getattr(args, "synth_box", None)
            _boxes_str = getattr(args, "synth_boxes", None)
            if _boxes_str and args.y_paste_mode in (
                    "synth_single", "synth_multi", "synth_smooth_hmap",
                    "crop_pinv_synth"):
                _box_list = []
                for _b in _boxes_str.split(";"):
                    _b = _b.strip()
                    if not _b:
                        continue
                    _c = tuple(float(v.strip()) for v in _b.split(","))
                    assert len(_c) == 4, f"box needs 4 vals, got {_c}"
                    _box_list.append(_c)
                _cls = (getattr(args, "synth_override_class", None)
                        or target_cls or "person")
                gts_y_paste = [(_cls, b) for b in _box_list]
                target_box = _box_list[0] if _box_list else target_box
                target_cls = _cls
            elif _box_str and args.y_paste_mode in (
                    "synth_single", "synth_multi", "synth_smooth_hmap",
                    "crop_pinv_synth"):
                _coords = tuple(float(v.strip()) for v in _box_str.split(","))
                assert len(_coords) == 4, f"--synth_box needs 4 vals, got {_coords}"
                _cls = (getattr(args, "synth_override_class", None)
                        or target_cls or "person")
                gts_y_paste = [(_cls, _coords)]
                target_box = _coords
                target_cls = _cls

            imagenet_class_for_image = int(getattr(args, "imagenet_class", 0))
            if getattr(args, "imagenet_class_from_gt", False) and target_cls in VOC_TO_IMAGENET:
                imagenet_class_for_image = VOC_TO_IMAGENET[target_cls]
                print(f"[class-cond] image cls={target_cls} -> imagenet={imagenet_class_for_image}")

            # Test-time r fine-tune for crop_pinv modes — mirrors
            # train.py:_train_r_opt_classifier but on a single zero-padded
            # input (M ⊙ y_orig). Restored at end of image so the next
            # image starts from the original ckpt r.
            saved_r_state = None
            online_r_opt = None
            online_r_params = []
            G_0_for_online = None
            crop_pinv_mode_local = args.y_paste_mode in (
                "crop_pinv_ref", "crop_pinv_synth")
            bb_mask_image = None
            y_bg_const_image = None
            if crop_pinv_mode_local and gts_y_paste:
                bb_mask_image = bb_mask_from_gts(
                    gts_y_paste, fmap=y.shape[-1], stride=4,
                    device=y.device, dtype=y.dtype,
                    falloff=getattr(args, "bb_mask_falloff", 0.0))
                if args.crop_pinv_fill == "bg":
                    y_bg_const_image = make_y_bg_const(
                        nc=args.detector_num_classes, fmap=y.shape[-1],
                        device=y.device, dtype=y.dtype,
                        hmap_bg=args.crop_pinv_bg_hmap,
                        regs_bg=args.crop_pinv_bg_regs,
                        wh_bg=args.crop_pinv_bg_wh,
                    )

            if args.r_finetune_steps > 0 and bb_mask_image is not None:
                _target_input = apply_crop_fill(y, bb_mask_image,
                                                y_bg_const_image)
                saved_r_state, _ft_losses = finetune_r_single_input(
                    classifier, A, Ap, _target_input,
                    steps=args.r_finetune_steps, lr=args.r_finetune_lr,
                )
                if _ft_losses:
                    print(f"  r-finetune ({len(_ft_losses)} steps): "
                          f"loss {_ft_losses[0]:.4e} -> {_ft_losses[-1]:.4e}")

            # Per-image: r training with image-reconstruction loss
            # ||Ap(M*y_orig) - x_orig|| + outside-BB natural-PInv loss.
            # Single-image, so the rec loss is achievable.
            if (args.r_rec_steps > 0 and bb_mask_image is not None
                    and crop_pinv_mode_local):
                _masked_y = bb_mask_image * y
                _saved_r3, _rec_losses = finetune_r_reconstruction(
                    classifier, A, Ap, _masked_y, x_orig,
                    steps=args.r_rec_steps, lr=args.r_rec_lr,
                    w_rec=args.r_rec_w_rec, w_norm=args.r_rec_w_norm,
                    bb_mask_64=bb_mask_image,
                )
                if saved_r_state is None:
                    saved_r_state = _saved_r3
                if _rec_losses:
                    print(f"  r-rec ({len(_rec_losses)} steps): "
                          f"rec {_rec_losses[0][0]:.3e} -> {_rec_losses[-1][0]:.3e}, "
                          f"norm {_rec_losses[0][1]:.3e} -> {_rec_losses[-1][1]:.3e}")

            # Per-image: jointly learn r AND y_BG outside BB.
            # Builds a learned y_BG that completes y_target most-naturally.
            learned_y_bg = None
            if (args.learn_bg_y_steps > 0 and bb_mask_image is not None
                    and crop_pinv_mode_local):
                # Determine y_target for the joint training (matches what BP
                # will use). For ref: y_target=y. For synth: synth-built.
                if args.y_paste_mode == "crop_pinv_ref":
                    _y_tgt_for_learn = y
                else:
                    # Build a one-shot synth y_target using y as the y_cur
                    # baseline (will only be used at non-peak cells; peaks
                    # are overwritten with synth values).
                    _y_tgt_for_learn = y_paste_synth(
                        y, gts_y_paste, nc=args.detector_num_classes,
                        stride=4,
                        peak_logit=args.y_paste_peak_logit,
                        bg_logit=args.y_paste_bg_logit,
                    )
                # Init y_BG: natural BG constants (matches network's expected
                # range; later optimization adjusts).
                _init_y_bg = make_y_bg_const(
                    nc=args.detector_num_classes, fmap=y.shape[-1],
                    device=y.device, dtype=y.dtype,
                    hmap_bg=args.crop_pinv_bg_hmap,
                    regs_bg=args.crop_pinv_bg_regs,
                    wh_bg=args.crop_pinv_bg_wh)
                _saved_r2, learned_y_bg, _ybg_losses = learn_bg_y_and_r(
                    classifier, A, Ap, _y_tgt_for_learn, bb_mask_image,
                    _init_y_bg,
                    steps=args.learn_bg_y_steps,
                    lr_r=args.r_finetune_lr, lr_bg=args.learn_bg_y_lr,
                )
                if saved_r_state is None:
                    saved_r_state = _saved_r2
                if _ybg_losses:
                    print(f"  learn_bg_y ({len(_ybg_losses)} steps): "
                          f"loss {_ybg_losses[0]:.4e} -> {_ybg_losses[-1]:.4e}")

            if args.r_online_steps > 0 and bb_mask_image is not None:
                if saved_r_state is None:
                    saved_r_state = {
                        n: {k: v.detach().clone() for k, v in m.r.state_dict().items()}
                        for n, m in classifier.named_modules()
                        if isinstance(m, ConvPINNBlock)
                    }
                _, online_r_opt, online_r_params = setup_online_r_training(
                    classifier, lr=args.r_online_lr)
                G_0_for_online = compute_G0(A, device=y.device, dtype=y.dtype)

            # init x_T — bb-edit: per-image seed so each x_orig sees a
            # different noise initialization (avoids every val image
            # starting from the same random latent).
            torch.manual_seed(args.seed + idx_so_far)
            x = torch.randn(
                y.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            with torch.no_grad():
                skip = config.diffusion.num_diffusion_timesteps // config.time_travel.T_sampling
                n = x.size(0)
                x0_preds = []
                xs = [x]

                times = get_schedule_jump(config.time_travel.T_sampling,
                                          config.time_travel.travel_length,
                                          config.time_travel.travel_repeat,
                                          )
                time_pairs = list(zip(times[:-1], times[1:]))
                bp_visited_t = set()
                descent_counter = 0

                for step_idx, (i, j) in enumerate(tqdm.tqdm(time_pairs)):
                    i, j = i * skip, j * skip
                    if j < 0: j = -1

                    if j < i:  # normal sampling
                        t = (torch.ones(n) * i).to(x.device)
                        next_t = (torch.ones(n) * j).to(x.device)
                        at = compute_alpha(self.betas, t.long())
                        at_next = compute_alpha(self.betas, next_t.long())
                        sigma_t = (1 - at / at_next).sqrt()
                        xt = xs[-1].to('cuda')

                        if getattr(config.model, 'class_cond', False):
                            _y_lbl = torch.full(
                                (xt.shape[0],),
                                int(imagenet_class_for_image),
                                dtype=torch.long, device=xt.device)
                            et = model(xt, t, y=_y_lbl)
                        else:
                            et = model(xt, t)

                        if et.size(1) == 6:
                            et = et[:, :3]

                        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                        x0_t_pre_bp = x0_t.clone()  # snapshot for debug viz
                        x0_t_hat = x0_t
                        min_bp_step = args.min_bp_step

                        lambda1 = args.lambda1
                        lambda2 = args.lambda2

                        if i < min_bp_step:
                            lambda_t = lambda1
                        else: # i > min
                            lambda_t = lambda2
                            # Optional ramp: at very high t (early sampling
                            # = mostly noise), don't have strong opinion.
                            # Ramp from 0 at ramp_top_t down to lambda2 at
                            # ramp_full_t (decreasing t). Defaults to no ramp.
                            _top_t = getattr(args, "ramp_top_t", 1000)
                            _full_t = getattr(args, "ramp_full_t", 1000)
                            if i >= _top_t:
                                lambda_t = 0.0
                            elif i > _full_t:
                                _w = (_top_t - i) / max(1, _top_t - _full_t)
                                lambda_t = lambda2 * _w

                        # None-Linear Back Projection
                        y_cur, z_cur = A(x0_t_hat, return_latents=True)

                        # Build BP target.
                        nc = args.detector_num_classes
                        crop_pinv = args.y_paste_mode in (
                            "crop_pinv_ref", "crop_pinv_synth")
                        bb_mask = bb_mask_image
                        if args.y_paste_mode == "ref" and gts_y_paste:
                            y_target = y_paste_ref(y_cur, y, gts_y_paste,
                                                   nc=nc, stride=4)
                        elif args.y_paste_mode in (
                                "synth_single", "synth_multi") and gts_y_paste:
                            if _class_model is not None:
                                y_target = y_paste_class_model(
                                    y_cur, gts_y_paste, _class_model,
                                    std_scale=args.y_class_model_std_scale,
                                    nc=nc, stride=4)
                            else:
                                y_target = y_paste_synth(
                                    y_cur, gts_y_paste, nc=nc, stride=4,
                                    peak_logit=args.y_paste_peak_logit,
                                    bg_logit=args.y_paste_bg_logit,
                                    other_peak_logit=getattr(
                                        args, "y_paste_other_peak_logit", -5.0))
                        elif args.y_paste_mode == "synth_smooth_hmap" \
                                and gts_y_paste:
                            y_target = y_paste_synth(
                                y_cur, gts_y_paste, nc=nc, stride=4,
                                peak_logit=args.y_paste_peak_logit,
                                bg_logit=args.y_paste_bg_logit,
                                other_peak_logit=getattr(
                                    args, "y_paste_other_peak_logit", -5.0),
                                smooth_hmap=True,
                                smoothness=args.y_paste_smoothness)
                        elif args.y_paste_mode == "crop_pinv_ref":
                            y_target = y
                        elif args.y_paste_mode == "crop_pinv_synth" and gts_y_paste:
                            y_target = y_paste_synth(
                                y_cur, gts_y_paste, nc=nc, stride=4,
                                peak_logit=args.y_paste_peak_logit,
                                bg_logit=args.y_paste_bg_logit,
                                other_peak_logit=getattr(
                                    args, "y_paste_other_peak_logit", -5.0))
                        else:
                            y_target = y

                        if bb_mask is not None:
                            diff_sig = (y_cur.sigmoid() - y_target.sigmoid()) * bb_mask
                            denom = bb_mask.expand_as(diff_sig).sum().clamp(min=1.0)
                            nlbp_error = diff_sig.abs().sum() / denom
                        else:
                            nlbp_error = (y_cur.sigmoid() - y_target.sigmoid()).abs().mean()

                        # Online r adaptation — train r on the actual input
                        # Ap will see at this step. For crop_pinv that's the
                        # masked diff M·(y_target − y_cur). Gentle threshold
                        # keeps r close to the original ckpt.
                        # Build the actual Ap inputs for this BP step. With
                        # `fill=y_cur` and smooth mask, outside BB blends
                        # smoothly to natural y_cur — natural completion.
                        if crop_pinv and bb_mask is not None:
                            if learned_y_bg is not None:
                                _fill = learned_y_bg
                                ap_in_target = bb_mask * y_target + (1 - bb_mask) * _fill
                                ap_in_cur = bb_mask * y_cur + (1 - bb_mask) * _fill
                            elif args.crop_pinv_fill == "y_cur":
                                _fill = y_cur
                                ap_in_target = bb_mask * y_target + (1 - bb_mask) * _fill
                                ap_in_cur = y_cur
                            elif args.crop_pinv_fill == "bg" and y_bg_const_image is not None:
                                _fill = y_bg_const_image
                                ap_in_target = bb_mask * y_target + (1 - bb_mask) * _fill
                                ap_in_cur = bb_mask * y_cur + (1 - bb_mask) * _fill
                            else:  # zero (literal cropping pinv)
                                ap_in_target = bb_mask * y_target
                                ap_in_cur = bb_mask * y_cur
                        else:
                            ap_in_target = y_target
                            ap_in_cur = y_cur

                        if getattr(args, "zeroize_peaks", False):
                            _zk = int(getattr(args, "zeroize_top_k", 100))
                            _zth = float(getattr(args, "zeroize_score_thresh", 0.1))
                            ap_in_target = zeroize_y_outside_predicted_peaks(
                                ap_in_target, nc=nc, score_thresh=_zth, top_k=_zk)
                            ap_in_cur = zeroize_y_outside_predicted_peaks(
                                ap_in_cur, nc=nc, score_thresh=_zth, top_k=_zk)

                        if (online_r_opt is not None
                                and bb_mask is not None):
                            _ap_target = ap_in_target.detach()
                            _ap_cur = ap_in_cur.detach()
                            with torch.enable_grad():
                                _online_loss, _online_n = online_r_step(
                                    classifier, A, Ap,
                                    [_ap_target, _ap_cur],
                                    online_r_opt, online_r_params,
                                    G_0_for_online,
                                    max_steps=args.r_online_steps,
                                    loss_threshold=args.r_online_loss_threshold,
                                    bb_mask_64=bb_mask_image,
                                )
                            if step_idx % 10 == 0 and _online_loss is not None:
                                print(f"  step {step_idx}: r-online "
                                      f"{_online_n} steps, loss={_online_loss:.4e}")

                        # bp_only_first_visit: BP locks in the constraint on
                        # the first descent through this t; subsequent
                        # time-travel revisits are pure prior denoising.
                        # bp_every: BP only every Nth descent step.
                        _bp_every = max(1, int(getattr(args, "bp_every", 1)))
                        do_bp = (lambda_t != 0.0
                                 and nlbp_error > args.nlbp_stop_cond
                                 and (not getattr(args, "bp_only_first_visit", False)
                                      or i not in bp_visited_t)
                                 and (descent_counter % _bp_every == 0))
                        descent_counter += 1
                        if do_bp:
                            bp_visited_t.add(i)
                            if getattr(args, "bp_pixel_space", False):
                                # Pixel-space BP: x0_t_hat = x_cur + lambda*
                                # (Ap(B(y_target)) - Ap(B(y_cur)))
                                _ap_t_out = Ap(ap_in_target)
                                _ap_c_out = Ap(ap_in_cur)
                                x0_t_hat = x0_t_hat + lambda_t * (_ap_t_out - _ap_c_out)
                            else:
                                # NLBP eq.: x' = G^{-1}(G(x_cur) + λ·[G(Ap(B(y))) -
                                # G(Ap(B(A(x_cur))))]). ap_in_target / ap_in_cur
                                # were built above based on crop_pinv_fill mode.
                                _ap_t_out = Ap(ap_in_target)
                                _ap_c_out = Ap(ap_in_cur)
                                if getattr(args, "clamp_ap_out", False):
                                    _ap_t_out = _ap_t_out.clamp(-1, 1)
                                    _ap_c_out = _ap_c_out.clamp(-1, 1)
                                y_tar, z_tar = A(_ap_t_out, return_latents=True)
                                y_proj, z_proj = A(_ap_c_out, return_latents=True)
                                z_final = []
                                for z0, z1, z2 in zip(z_cur, z_tar, z_proj):
                                    if z0 is not None:
                                        z_final.append(z0 + lambda_t * (z1 - z2))
                                    else:
                                        z_final.append(None)
                                y_final = y_cur + lambda_t * (y_tar - y_proj)
                                if getattr(args, "zeroize_peaks", False):
                                    _zk = int(getattr(args, "zeroize_top_k", 100))
                                    _zth = float(getattr(args, "zeroize_score_thresh", 0.1))
                                    y_final = zeroize_y_outside_predicted_peaks(
                                        y_final, nc=nc, score_thresh=_zth, top_k=_zk)
                                _lat = None if getattr(args, "no_nlbp_latents", False) else z_final
                                x0_t_hat = Ap(y_final, latents=_lat)
                            if getattr(args, "clamp_ap_out", False) or crop_pinv:
                                x0_t_hat = x0_t_hat.clamp(-1, 1)

                        # Pixel-space combination: x-paste blends post-BP
                        # x0_t_hat (BB constraint) over pre-BP x0_t (free
                        # diffusion BG). For multi-object: iterate over
                        # gts_y_paste, blending each box's region.
                        if getattr(args, "no_smooth_paste", False):
                            x0_t = x0_t_hat
                        elif (gts_y_paste and args.x_paste_smoothness > 0):
                            for _, _box_xyxy in gts_y_paste:
                                bx1, by1, bx2, by2 = _box_xyxy
                                top = int(by1); left = int(bx1)
                                h = max(1, int(np.ceil(by2)) - top)
                                w = max(1, int(np.ceil(bx2)) - left)
                                x0_t = smooth_paste(
                                    x0_t_hat, x0_t,
                                    topleft=(top, left), h=h, w=w,
                                    smoothness=args.x_paste_smoothness)
                        elif (target_box is not None
                                and args.x_paste_smoothness > 0):
                            bx1, by1, bx2, by2 = target_box
                            top = int(by1); left = int(bx1)
                            h = max(1, int(np.ceil(by2)) - top)
                            w = max(1, int(np.ceil(bx2)) - left)
                            x0_t = smooth_paste(
                                x0_t_hat, x0_t,
                                topleft=(top, left), h=h, w=w,
                                smoothness=args.x_paste_smoothness)
                        elif args.y_paste_mode != "none":
                            x0_t = x0_t_hat
                        elif target_box is not None:
                            _pn = getattr(args, "post_bp_noise", 0.025)
                            if _pn > 0:
                                x0_t_hat = x0_t_hat + _pn * torch.randn_like(x0_t_hat)
                            bx1, by1, bx2, by2 = target_box
                            top = int(by1)
                            left = int(bx1)
                            h = max(1, int(np.ceil(by2)) - top)
                            w = max(1, int(np.ceil(bx2)) - left)
                            x0_t = smooth_paste(
                                x0_t_hat, x0_t,
                                topleft=(top, left), h=h, w=w,
                                smoothness=getattr(args, "ref_paste_smoothness", 0.5),
                                alpha=getattr(args, "ref_paste_alpha", 1.0))


                        if step_idx % 10 == 0 or step_idx < 5:
                            print(f"  step {step_idx}: t={i} | x0_t range=[{x0_t.min():.3f}, {x0_t.max():.3f}] mean={x0_t.mean():.3f} | "
                                  f"x0_t_hat range=[{x0_t_hat.min():.3f}, {x0_t_hat.max():.3f}] mean={x0_t_hat.mean():.3f} | "
                                  f"nlbp_error={nlbp_error:.4f} lambda_t={lambda_t:.2f}")

                        # Per-step debug dump of x0_t and x0_t_hat for this
                        # image. inverse_data_transform maps [-1,1] → [0,1]
                        # for tvu.save_image. Lives in a `debug_x0` subdir
                        # of image_folder, with one subdir per image.
                        debug_dir = os.path.join(
                            self.args.image_folder, "debug_x0",
                            f"img_{idx_so_far}")
                        os.makedirs(debug_dir, exist_ok=True)
                        tvu.save_image(
                            inverse_data_transform(config, x0_t_pre_bp[0].cpu()),
                            os.path.join(debug_dir,
                                         f"step{step_idx:03d}_x0_t.png"))
                        tvu.save_image(
                            inverse_data_transform(config, x0_t_hat[0].cpu()),
                            os.path.join(debug_dir,
                                         f"step{step_idx:03d}_x0_t_hat.png"))

                        c2 = (1 - at_next - sigma_t ** 2).clamp(min=0).sqrt()
                        # bb-edit: roll the (partially-BP'd) x0_t forward,
                        # NOT x0_t_hat — that's what makes BP affect only
                        # the bbox region we just pasted.
                        _eta = getattr(args, "eta", 1.0)
                        xt_next = at_next.sqrt() * x0_t + c2 * et + _eta * sigma_t * torch.randn_like(x0_t)

                        x0_preds.append(x0_t.to('cpu'))
                        xs.append(xt_next.to('cpu'))
                    else: # time-travel back
                        next_t = (torch.ones(n) * j).to(x.device)
                        at_next = compute_alpha(self.betas, next_t.long())
                        x0_t = x0_preds[-1].to('cuda')

                        xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                        xs.append(xt_next.to('cpu'))

            final_x0 = inverse_data_transform(config, x0_preds[-1])

            tvu.save_image(
                final_x0[0], os.path.join(self.args.image_folder, f"{idx_so_far}_{0}.png")
            )
            orig = inverse_data_transform(config, x_orig[0])
            # Save result grid
            results_dir = self.args.image_folder
            os.makedirs(results_dir, exist_ok=True)
            # For the detection task, draw ONLY the GT box of the object we
            # actually pasted (the first GT — see target_box/target_cls
            # selected at the top of this image's loop body). Other GT
            # boxes are intentionally not drawn so the left/right grid is
            # a fair "what we conditioned on" vs "what came out".
            if voc_dataset is not None:
                from PIL import Image, ImageDraw  # lazy import
                arr = (orig.cpu().clamp(0, 1)
                       .permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                pil = Image.fromarray(arr)
                draw = ImageDraw.Draw(pil)
                if target_box is not None:
                    x1, y1, x2, y2 = target_box
                    draw.rectangle([x1, y1, x2, y2],
                                   outline=(255, 0, 0), width=2)
                    if target_cls is not None:
                        draw.text((x1 + 2, y1 + 2), target_cls,
                                  fill=(255, 255, 0))
                orig_for_grid = (torch.from_numpy(np.array(pil))
                                 .permute(2, 0, 1).float() / 255.0)
            else:
                orig_for_grid = orig.cpu()
            res_grid = torch.cat([orig_for_grid, final_x0[0].cpu()], dim=-1)
            grid_path = os.path.join(results_dir, f"grid_{idx_so_far}.png")
            tvu.save_image(res_grid, grid_path)

            # Per-image diagnostic: how well does A(generated) match A(original)?
            with torch.no_grad():
                y_orig = A(x_orig)
                y_gen = A(data_transform(config, final_x0.to(self.device)))

                # Detection task: also save grid_dets_<i>.png — a 2-panel
                # matplotlib figure with (left) original + GT boxes (red),
                # (right) original + SPNN top-K predictions (cyan, with
                # class:score). Same image both panels; lets you visually
                # compare GT to what the detector actually outputs.
                if voc_dataset is not None:
                    self._save_detection_grid(
                        idx_so_far, results_dir, voc_dataset, classes,
                        orig.cpu(), y_orig,
                        nc=args.detector_num_classes,
                        score_thresh=0.2, max_boxes=15,
                    )
                if getattr(args, "task", "classification") == "detection":
                    # Per-component error in the same space as the NLBP stopping
                    # condition above, so the numbers are comparable across runs.
                    nc = args.detector_num_classes
                    e_h = (y_orig[:, :nc].sigmoid()
                           - y_gen[:, :nc].sigmoid()).abs().mean().item()
                    e_r = (y_orig[:, nc:nc + 2]
                           - y_gen[:, nc:nc + 2]).abs().mean().item()
                    e_w = (y_orig[:, nc + 2:nc + 4]
                           - y_gen[:, nc + 2:nc + 4]).abs().mean().item()
                    voc_id = classes[0].item() if classes.dim() > 0 else classes.item()
                    print(f"  [img {idx_so_far}] voc_id={voc_id} | "
                          f"hmap_err(prob)={e_h:.4f} regs_err={e_r:.4f} wh_err={e_w:.4f} | "
                          f"weighted={(e_h + e_r + 0.1*e_w):.4f}")
                else:
                    pred_orig = y_orig.argmax(dim=1).item()
                    pred_gen = y_gen.argmax(dim=1).item()
                    true_cls = classes[0].item() if classes.dim() > 0 else classes.item()
                    print(f"  [img {idx_so_far}] true_class={true_cls} | "
                          f"spnn_on_orig={pred_orig} | spnn_on_generated={pred_gen}")

            mse = torch.mean((final_x0[0].to(self.device) - orig) ** 2)
            psnr = 10 * torch.log10(1 / mse)
            avg_psnr += psnr

            idx_so_far += y.shape[0]

            # Restore r to the original ckpt state so the next image starts
            # fresh (test-time fine-tune is per-image, not cumulative).
            if saved_r_state is not None:
                restore_r_state(classifier, saved_r_state)

            pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Number of samples: %d" % (idx_so_far - idx_init))
        print(f"Results saved to: {os.path.abspath(self.args.image_folder)}")



# Code form RePaint
def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)
    return ts

def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a
