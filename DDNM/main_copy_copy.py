"""Driver for the bb-edit / inverse-inpainting variant of the
reconstruction script. Same CLI surface as main_copy.py — only the
imported Diffusion class differs."""
import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
from guided_diffusion.diffusion_copy_copy import Diffusion

torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--exp", type=str, default="exp")
    parser.add_argument("--path_y", type=str, required=True)
    parser.add_argument("-i", "--image_folder", type=str, default="images")
    parser.add_argument("--verbose", type=str, default="info")
    parser.add_argument("--ni", action="store_true")
    parser.add_argument("--subset_start", type=int, default=-1)
    parser.add_argument("--subset_end", type=int, default=-1)
    parser.add_argument("--min_bp_step", type=int, default=300)
    parser.add_argument("--lambda1", type=float, default=0.0)
    parser.add_argument("--lambda2", type=float, default=1.0)
    parser.add_argument("--nlbp_stop_cond", type=float, default=-1)
    parser.add_argument("--spnn_ckpt", type=str, default=None)
    parser.add_argument("--spnn_num_classes", type=int, default=10)
    parser.add_argument("--spnn_mix_type", type=str, default="householder",
                        choices=["cayley", "householder"])
    parser.add_argument("--spnn_scale_bound", type=float, default=1.0)
    parser.add_argument("--task", choices=["classification", "detection"],
                        default="classification")
    parser.add_argument("--detector_ckpt", type=str, default=None)
    parser.add_argument("--voc_data_dir", type=str, default="./data")
    parser.add_argument("--detector_num_classes", type=int, default=20)
    parser.add_argument("--detector_deep_det_head", action="store_true")
    parser.add_argument("--detector_deep_head_hidden", type=int, default=128)
    parser.add_argument("--detector_two_block_head", action="store_true")
    parser.add_argument("--detector_head_mode", type=str, default="affine",
                        choices=["affine", "orthogonal_mix"])
    parser.add_argument("--detector_head_mix_type", type=str,
                        default="householder",
                        choices=["cayley", "householder"])
    parser.add_argument("--detector_head_mix_reflections", type=int, default=0)
    parser.add_argument("--detector_hmap_init_scale", type=float, default=0.01)
    parser.add_argument("--detector_hmap_init_bias", type=float, default=-2.19)
    parser.add_argument("--no_hmap_scale", action="store_true")
    parser.add_argument("--no_hmap_bias", action="store_true")
    parser.add_argument("--internal_head_affine", action="store_true")
    parser.add_argument("--detector_scale_bound", type=float, default=1.0)
    parser.add_argument("--detector_mix_type", type=str, default="householder",
                        choices=["cayley", "householder"])
    parser.add_argument("--detector_hidden", type=int, default=256)
    parser.add_argument("--y_paste_mode", type=str, default="none",
                        choices=["none", "ref", "synth_single",
                                 "synth_multi", "synth_smooth_hmap",
                                 "crop_pinv_ref", "crop_pinv_synth"])
    parser.add_argument("--y_paste_smoothness", type=float, default=1.0)
    parser.add_argument("--y_paste_peak_logit", type=float, default=0.0,
                        help="Peak logit value for synthesized hmap (matches "
                             "natural q90 of real peak cells from y-stats).")
    parser.add_argument("--y_paste_bg_logit", type=float, default=-6.0,
                        help="BG logit value (matches natural mean from "
                             "y-stats analysis).")
    parser.add_argument("--y_paste_other_peak_logit", type=float, default=-5.0,
                        help="Logit for OTHER classes at the splat peak "
                             "cell (matches natural 'peak-cell, other "
                             "class' mean = -5.10). Other classes follow "
                             "the same Gaussian falloff. Set equal to "
                             "bg_logit (-6) to skip suppression.")
    parser.add_argument("--y_class_model_path", type=str, default=None,
                        help="Path to .pt with per-class y-window stats "
                             "(built by build_class_y_model.py). If set "
                             "and class matches, replace analytical synth "
                             "with a sample (or mean) from this empirical "
                             "model — captures real class-confusion + "
                             "spatial structure.")
    parser.add_argument("--y_class_model_std_scale", type=float, default=0.0,
                        help="Multiplier on the per-cell std when sampling "
                             "from the class-y model. 0 = use mean only "
                             "(deterministic); 1 = full random sample.")
    parser.add_argument("--synth_override_class", type=str, default=None,
                        help="VOC class name to insert at the GT box "
                             "(synth_* modes). E.g., 'horse' to put a horse "
                             "where the GT object is.")
    parser.add_argument("--synth_box", type=str, default=None,
                        help="Explicit box 'x1,y1,x2,y2' in 256-px coords. "
                             "Overrides any GT/source-image box. Combined "
                             "with --synth_override_class gives truly no-ref "
                             "synth: no GT influence, just (class, box) "
                             "specification.")
    parser.add_argument("--synth_boxes", type=str, default=None,
                        help="Multiple boxes for multi-object synth, "
                             "semicolon-separated: "
                             "'x1,y1,x2,y2;x1,y1,x2,y2;...'. Each box uses "
                             "the same --synth_override_class. Overrides "
                             "--synth_box if both are set.")
    parser.add_argument("--ramp_top_t", type=int, default=1000,
                        help="t-value where BP ramp begins (decreasing t "
                             "from 1000). Above this t, BP is off (no "
                             "opinion at high noise). 1000 = ramp from "
                             "the very start (current behavior).")
    parser.add_argument("--ramp_full_t", type=int, default=1000,
                        help="t-value where BP ramp reaches full lambda2. "
                             "Between ramp_top_t and ramp_full_t, BP "
                             "linearly grows from 0 to lambda2. Default "
                             "1000 = no ramp (instant full).")
    parser.add_argument("--eta", type=float, default=1.0,
                        help="DDIM-like noise scale on the fresh-noise term "
                             "in xt_next. 1.0 = full DDPM-like (current); "
                             "0.85 = DDNM default; 0.0 = deterministic.")
    parser.add_argument("--post_bp_noise", type=float, default=0.025,
                        help="Std of Gaussian noise added to x0_t_hat after "
                             "BP, before pixel-space smooth_paste. Helps "
                             "the BB content blend more naturally with the "
                             "free BG. 0 = no noise.")
    parser.add_argument("--imagenet_class", type=int, default=339,
                        help="ImageNet class index for class-conditional "
                             "ADM sampling. Only used if config.model."
                             "class_cond is true. Common picks: 339=sorrel "
                             "(horse), 779=school_bus, 817=sports_car, "
                             "207=golden_retriever, 281=tabby_cat, "
                             "831=studio_couch (sofa). Full list at "
                             "https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a.")
    parser.add_argument("--imagenet_class_from_gt", action="store_true",
                        help="Per image, pick the ImageNet class from the "
                             "first GT object's VOC class via a built-in "
                             "VOC→ImageNet mapping (e.g. dog→golden_retriever, "
                             "train→bullet_train). Overrides --imagenet_class "
                             "for any image whose GT class is in the map.")
    parser.add_argument("--r_finetune_steps", type=int, default=0,
                        help="Test-time fine-tune r networks per image, "
                             "minimizing the natural-PInv objective on the "
                             "zero-padded BB-cropped y. 0 = off.")
    parser.add_argument("--r_finetune_lr", type=float, default=1e-4)
    parser.add_argument("--r_online_steps", type=int, default=0,
                        help="At each BP step, take up to N gradient steps on "
                             "r (natural-PInv loss on current M*y_cur) until "
                             "loss < threshold. 0 = off.")
    parser.add_argument("--r_online_lr", type=float, default=1e-4)
    parser.add_argument("--r_online_loss_threshold", type=float, default=1e-1,
                        help="Stop r updates when loss < this. Higher = gentler "
                             "(avoids over-fitting r into pathological weights).")
    parser.add_argument("--crop_pinv_fill", type=str, default="zero",
                        choices=["zero", "bg", "y_cur"],
                        help="Outside-BB values for Ap inputs in the "
                             "round-trip (y_tar/y_proj) computations: "
                             "'zero' (literal pinv of cropping; OOD for r), "
                             "'bg' (constant natural BG), or "
                             "'y_cur' (current detector output; "
                             "in-distribution, equivalent to y-paste "
                             "with full BB).")
    parser.add_argument("--crop_pinv_bg_hmap", type=float, default=-6.0)
    parser.add_argument("--crop_pinv_bg_regs", type=float, default=0.5)
    parser.add_argument("--crop_pinv_bg_wh", type=float, default=18.0)
    parser.add_argument("--bb_mask_falloff", type=float, default=0.0,
                        help="Soft-mask sigma (in fmap cells) for the BB. "
                             "0 = hard 0/1 mask. >0 = 1 inside BB, Gaussian "
                             "falloff outside with this sigma.")
    parser.add_argument("--x_paste_smoothness", type=float, default=0.0,
                        help="If >0 and crop_pinv mode is on, blend post-BP "
                             "x0_t_hat over pre-BP x0_t in pixel space using "
                             "a Gaussian centered at the BB (sigma = this * "
                             "box_dim). 0 = no x-paste (use x0_t_hat directly).")
    parser.add_argument("--learn_bg_y_steps", type=int, default=0,
                        help="Per-image: jointly train r AND outside-BB y "
                             "values to minimize natural-PInv loss on "
                             "M*y_target + (1-M)*y_BG. Treats y_BG as the "
                             "free 'don't care' variable. 0 = off; if >0, "
                             "uses the learned y_BG as the round-trip Ap "
                             "fill (overrides --crop_pinv_fill).")
    parser.add_argument("--learn_bg_y_lr", type=float, default=1e-2,
                        help="Learning rate for y_BG params (separate from "
                             "--r_finetune_lr which controls r's lr).")
    parser.add_argument("--r_rec_steps", type=int, default=0,
                        help="Per-image: pre-loop r training with image "
                             "reconstruction loss ||Ap(M*y_orig)-x_orig||^2 "
                             "+ outside-BB natural-PInv loss. 0 = off.")
    parser.add_argument("--r_rec_lr", type=float, default=1e-4)
    parser.add_argument("--r_rec_w_rec", type=float, default=1.0,
                        help="Weight for the image-reconstruction loss term.")
    parser.add_argument("--r_rec_w_norm", type=float, default=0.1,
                        help="Weight for the outside-BB natural-PInv loss term.")
    parser.add_argument("--bp_only_first_visit", action="store_true",
                        help="Apply BP only the first time the time-travel "
                             "schedule descends through each t. Re-passes "
                             "(due to travel_repeat) are pure prior, so the "
                             "extra steps refine without BP averaging.")
    parser.add_argument("--ref_paste_smoothness", type=float, default=0.5,
                        help="Gaussian sigma multiplier for the ref-x-paste "
                             "branch (was hardcoded 0.5). Smaller = sharper "
                             "boundary, more BG bleed near the BB edges.")
    parser.add_argument("--ref_paste_alpha", type=float, default=1.0,
                        help="Multiplier on the Gaussian weight in ref-x-paste. "
                             "1.0 = full BP at BB center; <1.0 = let some BG "
                             "(pre-BP x0_t) into the BB center too.")
    parser.add_argument("--bp_every", type=int, default=1,
                        help="Apply BP only every Nth descent step (1=every "
                             "step, 2=every other, 3=every third...). "
                             "Combines with --bp_only_first_visit. Higher "
                             "values let the diffusion prior dominate more.")
    parser.add_argument("--zeroize_peaks", action="store_true",
                        help="Apply B = zeroize_y_outside_predicted_peaks "
                             "before every Ap call (target and current). "
                             "Matches the way the SPNN was trained.")
    parser.add_argument("--zeroize_top_k", type=int, default=100,
                        help="Top-K peaks to keep in zeroize. 1 = single "
                             "strongest object only.")
    parser.add_argument("--zeroize_score_thresh", type=float, default=0.1,
                        help="Sigmoid score threshold for zeroize peak "
                             "selection.")
    parser.add_argument("--no_smooth_paste", action="store_true",
                        help="Skip the pixel-space smooth-paste entirely. "
                             "Use x0_t_hat directly as x0_t (clean NLBP).")
    parser.add_argument("--clamp_ap_out", action="store_true",
                        help="Clamp Ap output to [-1,1] inside the BP step. "
                             "Helps when Ap output is OOD (e.g. zeroize-trained "
                             "SPNN whose pinv has wider output range).")
    parser.add_argument("--no_nlbp_latents", action="store_true",
                        help="Skip the NLBP latent-residual trick. The final "
                             "Ap call uses latents=None (trained r-net "
                             "inverts B(y_final) directly). Recommended for "
                             "zeroize-trained SPNN where the r-net expects "
                             "B-zeroized inputs and external latents may "
                             "break it.")
    parser.add_argument("--bp_pixel_space", action="store_true",
                        help="Pixel-space BP: x0_t_hat = x_cur + lambda*"
                             "(Ap(B(y_target)) - Ap(B(y_cur))). Skips the "
                             "G/G^-1 latent algebra; just uses the trained "
                             "pinv as a black box. Most stable with the "
                             "zeroize-trained SPNN.")

    args = parser.parse_args()
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))
    handler1 = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s")
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
    args.image_folder = os.path.join(
        args.exp, "image_samples", args.image_folder)
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input(
                f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)")
            if response.upper() == "Y":
                overwrite = True
        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    try:
        runner = Diffusion(args, config)
        runner.sample()
    except Exception:
        logging.error(traceback.format_exc())
    return 0


if __name__ == "__main__":
    sys.exit(main())
