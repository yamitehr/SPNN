import os
import argparse
import torch
import numpy as np
import random

from utils import check_args
from data_loader import get_local_celebahq_loaders
from models import SPNN
from logger import setup_logger
from train import CelebATrainer
from diagnostics import PenroseChecker, GinvNormCalculator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--model_type', type=str, default='SPNN')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--fix_epoch', type=float, default=0.4)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--adaptive_weights', type=bool, default=True)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--warmup_iters', type=int, default=200)
    parser.add_argument('--float16', type=bool, default=False)
    parser.add_argument('--lambda_bce', type=float, default=1.0)
    parser.add_argument('--lambda_right_inverse', type=float, default=40.0)
    parser.add_argument('--lambda_img_rec', type=float, default=40.0)
    parser.add_argument('--lambda_r_norm', type=float, default=0.1)
    parser.add_argument('--lambda_r_rec', type=float, default=40.0)
    parser.add_argument('--lambda_r_cycle', type=float, default=1.0)
    parser.add_argument('--r_opt_epochs', type=int, default=50)
    parser.add_argument('--r_opt_lr', type=float, default=1e-4)
    parser.add_argument('--r_opt_batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=556)
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Checkpoint directory. Defaults to an auto-generated name under check_points/.')
    parser.add_argument('--setting_file', type=str, default='setting.txt')
    parser.add_argument('--log_file', type=str, default='log.txt')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to the CelebAMask-HQ root directory. If not provided, the dataset will be downloaded via kagglehub (requires Kaggle API credentials).')
    parser.add_argument('--mix_type', type=str, default='cayley', choices=['cayley', 'householder'])
    parser.add_argument('--is_r_opt', action="store_true")
    parser.add_argument('--is_forward_train', action="store_true")
    args = parser.parse_args()

    if args.checkpoint_dir is None:
        args.checkpoint_dir = "check_points"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    logger = setup_logger(args.checkpoint_dir, logfile_name=args.log_file, logger_name='att_cls')
    n_gpu = torch.cuda.device_count()

    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    check_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Load dataset
    if args.dataset_path is not None:
        path = args.dataset_path
    else:
        print("No --dataset_path provided. Downloading via kagglehub (requires Kaggle API credentials)...")
        import kagglehub
        path = kagglehub.dataset_download("liusonghua/celebamaskhq")
        path = os.path.join(path, "CelebAMask-HQ")

    print(f"Loading CelebA-HQ from: {path}")
    train_loader, dev_loader, test_loader = get_local_celebahq_loaders(
        root=path,
        batch_size=args.batch_size,
        img_size=args.img_size
    )

    # create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print("cuda_available:", torch.cuda.is_available())
        print("device_count:", torch.cuda.device_count())
        print("gpu:", torch.cuda.get_device_name(0))

    model = SPNN(
        img_ch=3,
        num_classes=40,
        hidden=128,
        scale_bound=2.0,
        img_size=args.img_size,
        mix_type=args.mix_type,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model total_params: {total_params:,}, trainable_params: {trainable_params:,}")

    trainer = CelebATrainer(args, model, train_loader, dev_loader, test_loader, device, logger, n_gpu)
    if args.is_forward_train:
        trainer.train()
        trainer.test()

    penrose_checker = PenroseChecker(logger)
    ginv_calculator = GinvNormCalculator(logger)

    ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    if os.path.exists(ckpt_path):
        penrose_before = penrose_checker.run_penrose_batched(
            checkpoint_path=ckpt_path, test_loader=test_loader,
            device=torch.device(device), img_ch=3, num_classes=40, hidden=128, scale_bound=2.0, img_size=args.img_size, mix_type=args.mix_type)

        ginv_real_before = ginv_calculator.run(
            checkpoint_path=ckpt_path,
            loader=test_loader,
            device=torch.device(device),
            model_cls=SPNN,
            model_kwargs=dict(img_ch=3, num_classes=40, hidden=128, scale_bound=2.0, img_size=args.img_size, mix_type=args.mix_type),
        )

        print("[Before r-opt] Penrose metrics:")
        for k, v in penrose_before.items():
            print(f"  penrose_before_r_opt/{k}: {v}")
        print(f"  test/g'(y)_norm_before_r_opt: {float(ginv_real_before)}")

        if args.is_r_opt:
            trainer.train_r_opt_on_real_logits(
                checkpoint_path=ckpt_path,
                device=torch.device(device),
                loader=train_loader,
                num_classes=40, H=args.img_size, W=args.img_size,
                epochs=args.r_opt_epochs, lr=args.r_opt_lr, batch_size=args.r_opt_batch_size, img_size=args.img_size,
                out_checkpoint_path=os.path.join(args.checkpoint_dir, "best_model_r_opt_real.pth"),
            )

            penrose_after = penrose_checker.run_penrose_batched(
                checkpoint_path=os.path.join(args.checkpoint_dir, "best_model_r_opt_real.pth"), test_loader=test_loader,
                device=torch.device(device), img_ch=3, num_classes=40, hidden=128, scale_bound=2.0, img_size=args.img_size, mix_type=args.mix_type)

            ginv_real_after = ginv_calculator.run(
                checkpoint_path=os.path.join(args.checkpoint_dir, "best_model_r_opt_real.pth"),
                loader=test_loader,
                device=torch.device(device),
                model_cls=SPNN,
                model_kwargs=dict(img_ch=3, num_classes=40, hidden=128, scale_bound=2.0, img_size=args.img_size, mix_type=args.mix_type),
            )

            print("[After r-opt] Penrose metrics:")
            for k, v in penrose_after.items():
                print(f"  penrose_after_r_opt/{k}: {v}")
            print(f"  test/g'(y)_norm_after_r_opt: {float(ginv_real_after)}")
    else:
        print(f"No checkpoint found at {ckpt_path}, skipping diagnostics.")


if __name__ == '__main__':
    main()
