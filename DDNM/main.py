import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
# from runners.diffusion import Diffusion
from guided_diffusion.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Set different seeds for diverse results")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--path_y",
        type=str,
        required=True,
        help="Path of the test dataset.",
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument(
        '--subset_start', type=int, default=-1
    )
    parser.add_argument(
        '--subset_end', type=int, default=-1
    )
    parser.add_argument(
        "--min_bp_step", type=int, default=600, help="Minimum step to start back-projection"
    )
    parser.add_argument(
        "--lambda1", type=float, default=0.0, help="Lambda for BP when step < min_bp_step"
    )
    parser.add_argument(
        "--lambda2", type=float, default=0.5, help="Lambda for BP when step >= min_bp_step"
    )
    parser.add_argument(
        "--nlbp_stop_cond", type=float, default=0.11, help="Skip back-projection when attribute error is below this threshold"
    )
    # SPNN args for ImageNet/ImageNette (classification task)
    parser.add_argument("--spnn_ckpt", type=str, default=None, help="Path to SPNN classifier checkpoint")
    parser.add_argument("--spnn_num_classes", type=int, default=10, help="Number of SPNN classes (10 for ImageNette, 1000 for ImageNet)")
    parser.add_argument("--spnn_mix_type", type=str, default="householder", choices=["cayley", "householder"])
    parser.add_argument("--spnn_scale_bound", type=float, default=1.0)

    # Detection task: SPNN CenterNet detector as the degradation
    parser.add_argument("--task", choices=["classification", "detection"],
                        default="classification",
                        help="Degradation type. 'classification' uses an SPNN classifier "
                             "and reads from --spnn_ckpt. 'detection' uses an SPNN CenterNet "
                             "object detector and reads from --detector_ckpt.")
    parser.add_argument("--detector_ckpt", type=str, default=None,
                        help="Path to SPNN CenterNet checkpoint (e.g. ckpt/.../checkpoint.t7).")
    parser.add_argument("--voc_data_dir", type=str, default="./data",
                        help="Root containing voc/ subtree (images/, annotations/).")
    parser.add_argument("--detector_num_classes", type=int, default=20)
    # Detector architecture flags — must match the trained checkpoint.
    parser.add_argument("--detector_deep_det_head", action="store_true")
    parser.add_argument("--detector_deep_head_hidden", type=int, default=128)
    parser.add_argument("--detector_two_block_head", action="store_true")
    parser.add_argument("--detector_head_mode", type=str, default="affine",
                        choices=["affine", "orthogonal_mix"])
    parser.add_argument("--detector_head_mix_type", type=str, default="householder",
                        choices=["cayley", "householder"])
    parser.add_argument("--detector_head_mix_reflections", type=int, default=0)
    parser.add_argument("--detector_hmap_init_scale", type=float, default=0.01)
    parser.add_argument("--detector_hmap_init_bias", type=float, default=-2.19)
    parser.add_argument("--detector_scale_bound", type=float, default=1.0)
    parser.add_argument("--detector_mix_type", type=str, default="householder",
                        choices=["cayley", "householder"])
    parser.add_argument("--detector_hidden", type=int, default=256,
                        help="Hidden width for backbone ConvPINN blocks.")

    args = parser.parse_args()

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
    args.image_folder = os.path.join(
        args.exp, "image_samples", args.image_folder
    )
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input(
                f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
            )
            if response.upper() == "Y":
                overwrite = True

        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
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
