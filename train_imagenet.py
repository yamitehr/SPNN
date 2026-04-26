"""
ImageNet classification training for SPNN.

Based on pytorch/examples/imagenet/main.py with minimal changes:
  - Model swapped to SPNN classification architecture
  - SPNN cycle losses added (right-inverse + image reconstruction)
  - Image size changed from 224 to 256 (for PixelUnshuffle divisibility)
  - Added --num-classes, --wandb, --lambda-cycle, --lambda-rec args
  - Added wandb logging

Usage:
  # ImageNette (10 classes, quick validation)
  python train_imagenet.py imagenette2-320 --num-classes 10 --epochs 50

  # Full ImageNet (1000 classes)
  python train_imagenet.py /path/to/imagenet --num-classes 1000 --epochs 90

  # Multi-GPU
  python train_imagenet.py /path/to/imagenet --multiprocessing-distributed --world-size 1 --rank 0
"""

import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset

from models import SPNN, ConvPINNBlock, PixelUnshuffleBlock


def build_classification_spnn(num_classes=1000, hidden=256, mix_type="cayley"):
    """Build SPNN for classification with the shared backbone architecture."""
    layer_channels = [
        # Backbone (shared with detection)
        (PixelUnshuffleBlock, {"r": 4}),
        (ConvPINNBlock, {"in_ch": 48, "out_ch": 24, "hidden": hidden,
                         "scale_bound": 2.0, "feat_size": 64, "mix_type": mix_type}),
        (ConvPINNBlock, {"in_ch": 24, "out_ch": 12, "hidden": hidden,
                         "scale_bound": 2.0, "feat_size": 64, "mix_type": mix_type}),
        (PixelUnshuffleBlock, {"r": 4}),
        (ConvPINNBlock, {"in_ch": 192, "out_ch": 96, "hidden": hidden,
                         "scale_bound": 2.0, "feat_size": 16, "mix_type": mix_type}),
        (ConvPINNBlock, {"in_ch": 96, "out_ch": 48, "hidden": hidden,
                         "scale_bound": 2.0, "feat_size": 16, "mix_type": mix_type}),
        # Classification head
        (PixelUnshuffleBlock, {"r": 4}),
        (ConvPINNBlock, {"in_ch": 768, "out_ch": 192, "hidden": hidden,
                         "scale_bound": 2.0, "feat_size": 4, "mix_type": mix_type}),
        (PixelUnshuffleBlock, {"r": 4}),
        (ConvPINNBlock, {"in_ch": 3072, "out_ch": 1024, "hidden": hidden,
                         "scale_bound": 2.0, "feat_size": 1, "mix_type": mix_type}),
        (ConvPINNBlock, {"in_ch": 1024, "out_ch": num_classes, "hidden": hidden,
                         "scale_bound": 2.0, "feat_size": 1, "mix_type": mix_type}),
    ]
    return SPNN(
        img_ch=3, num_classes=num_classes, img_size=256,
        layer_channels=layer_channels,
    )


# Image size for SPNN (256 instead of 224 for PixelUnshuffle divisibility)
IMG_SIZE = 256

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--no-accel', action='store_true',
                    help='disables accelerator')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")

# SPNN-specific arguments
parser.add_argument('--num-classes', default=1000, type=int,
                    help='number of classes (default: 1000 for ImageNet, 10 for ImageNette)')
parser.add_argument('--lambda-cycle', default=1.0, type=float,
                    help='weight for SPNN right-inverse cycle loss (default: 1.0)')
parser.add_argument('--lambda-rec', default=1.0, type=float,
                    help='weight for SPNN image reconstruction loss (default: 1.0)')
parser.add_argument('--scheduler', default='step', type=str, choices=['step', 'cosine'],
                    help='LR scheduler type (default: step)')
parser.add_argument('--warmup-epochs', default=0, type=int,
                    help='number of warmup epochs (default: 0, used with cosine scheduler)')
parser.add_argument('--wandb', action='store_true', help='enable wandb logging')
parser.add_argument('--wandb-project', default='spnn-imagenet', type=str)
parser.add_argument('--wandb-run-name', default=None, type=str)
parser.add_argument('--checkpoint-dir', default='check_points_cls', type=str,
                    help='directory to save checkpoints')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    use_accel = not args.no_accel and torch.accelerator.is_available()

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    if device.type =='cuda':
        ngpus_per_node = torch.accelerator.device_count()
        if ngpus_per_node == 1 and args.dist_backend == "nccl":
            warnings.warn("nccl backend >=2.5 requires GPU count>1, see https://github.com/NVIDIA/nccl/issues/103 perhaps use 'gloo'")
    else:
        ngpus_per_node = 1

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    use_accel = not args.no_accel and torch.accelerator.is_available()

    if use_accel:
        if args.gpu is not None:
            torch.accelerator.set_device_index(args.gpu)
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create SPNN classification model
    print(f"=> creating SPNN classification model (num_classes={args.num_classes})")
    model = build_classification_spnn(num_classes=args.num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total params: {total_params:,}")

    # wandb init (main process only)
    is_main = not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0)
    if args.wandb and is_main:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                   config=vars(args))
        # Log gradient norms and parameter norms every 100 steps
        wandb.watch(model, log="gradients", log_freq=100, log_graph=False)
    args.is_main = is_main

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if not use_accel:
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if device.type == 'cuda':
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif device.type == 'cuda':
        # Single GPU — no DataParallel (incompatible with SPNN's torch.eye)
        model.cuda()
    else:
        model.to(device)


    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    # Scheduler is created after data loaders (cosine needs len(train_loader))
    scheduler = None  # placeholder, created below

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = f'{device.type}:{args.gpu}'
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if isinstance(best_acc1, torch.Tensor):
                best_acc1 = best_acc1.item()
            # Load state_dict into unwrapped model (checkpoint saves unwrapped weights)
            unwrapped = model.module if hasattr(model, 'module') else model
            unwrapped.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # Scheduler state loaded after scheduler creation below
            args._scheduler_state = checkpoint.get('scheduler', None)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # Data loading code
    if args.dummy:
        print("=> Dummy data is used!")
        train_dataset = datasets.FakeData(1281167, (3, IMG_SIZE, IMG_SIZE), args.num_classes, transforms.ToTensor())
        val_dataset = datasets.FakeData(50000, (3, IMG_SIZE, IMG_SIZE), args.num_classes, transforms.ToTensor())
    else:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(IMG_SIZE),
                transforms.ToTensor(),
                normalize,
            ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    # Create scheduler (needs len(train_loader))
    if args.scheduler == 'cosine':
        from pytorch_optimization import get_cosine_schedule_with_warmup
        num_training_steps = len(train_loader) * args.epochs
        num_warmup_steps = len(train_loader) * args.warmup_epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps)
    else:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # Load scheduler state if resuming
    if hasattr(args, '_scheduler_state') and args._scheduler_state is not None:
        try:
            scheduler.load_state_dict(args._scheduler_state)
        except Exception as e:
            print(f"Warning: Could not load scheduler state ({e}), starting fresh scheduler")

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device, args,
              scheduler if args.scheduler == 'cosine' else None)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        if args.scheduler == 'step':
            scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # Penrose identity check every 5 epochs (main process only)
        penrose_freq = 5
        if args.is_main and (epoch + 1) % penrose_freq == 0:
            spnn_model = model.module if hasattr(model, 'module') else model
            spnn_model.eval()
            p1_sum, p2_sum, p3_sum, n_batches = 0.0, 0.0, 0.0, 0
            max_penrose_batches = 160  # ~10K images at batch=64
            with torch.no_grad():
                for val_imgs, _ in val_loader:
                    if n_batches >= max_penrose_batches:
                        break
                    val_imgs = val_imgs.to(device, non_blocking=True)

                    # Use real model outputs y = g(x) for all identities
                    y_g = spnn_model(val_imgs)

                    # 1. g(g'(g(x))) == g(x)
                    y_ggg = spnn_model(spnn_model.pinv(y_g))
                    p1_sum += (y_ggg - y_g).pow(2).mean().item()

                    # 2. g'(g(g'(y))) == g'(y)  [using y = g(x)]
                    x_gp = spnn_model.pinv(y_g)
                    x_gpgp = spnn_model.pinv(spnn_model(x_gp))
                    p2_sum += (x_gpgp - x_gp).pow(2).mean().item()

                    # 3. g(g'(y)) == y  [using y = g(x)]
                    y_cycle = spnn_model(spnn_model.pinv(y_g))
                    p3_sum += (y_cycle - y_g).pow(2).mean().item()

                    n_batches += 1

            p1 = p1_sum / max(1, n_batches)
            p2 = p2_sum / max(1, n_batches)
            p3 = p3_sum / max(1, n_batches)

            print(f"\n--- Penrose Check (epoch {epoch+1}, {n_batches} batches) ---")
            print(f"  g(g'(g(x)))=g(x)  MSE: {p1:.2e}")
            print(f"  g'(g(g'(y)))=g'(y) MSE: {p2:.2e}")
            print(f"  g(g'(y))=y         MSE: {p3:.2e}")
            print("--------------------------------------")

            if args.wandb:
                import wandb
                wandb.log({
                    "penrose/ggg_mse": p1,
                    "penrose/gpgp_mse": p2,
                    "penrose/gy_mse": p3,
                    "epoch": epoch,
                })
            spnn_model.train()

        # wandb val logging
        if args.wandb and args.is_main:
            import wandb
            wandb.log({"val/top1": acc1, "val/best_top1": best_acc1, "epoch": epoch})

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            # Save unwrapped state_dict for clean checkpoint
            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best, checkpoint_dir=args.checkpoint_dir)


def train(train_loader, model, criterion, optimizer, epoch, device, args, scheduler=None):
    
    use_accel = not args.no_accel and torch.accelerator.is_available()

    batch_time = AverageMeter('Time', use_accel, ':6.3f', Summary.NONE)
    data_time = AverageMeter('Data', use_accel, ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', use_accel, ':.4e', Summary.NONE)
    ce_losses = AverageMeter('CE', use_accel, ':.4e', Summary.NONE)
    cycle_losses = AverageMeter('Cycle', use_accel, ':.4e', Summary.NONE)
    rec_losses = AverageMeter('Rec', use_accel, ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', use_accel, ':6.2f', Summary.NONE)
    top5 = AverageMeter('Acc@5', use_accel, ':6.2f', Summary.NONE)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, ce_losses, cycle_losses, rec_losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        ce_loss = criterion(output, target)

        # SPNN cycle losses
        spnn_model = model.module if hasattr(model, 'module') else model
        x_inv = spnn_model.pinv(output)
        y_cycle = model(x_inv)
        cycle_loss = (y_cycle - output).pow(2).mean()
        rec_loss = (x_inv - images).pow(2).mean()

        loss = ce_loss + args.lambda_cycle * cycle_loss + args.lambda_rec * rec_loss

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        ce_losses.update(ce_loss.item(), images.size(0))
        cycle_losses.update(cycle_loss.item(), images.size(0))
        rec_losses.update(rec_loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # wandb logging
        if args.wandb and i % args.print_freq == 0 and args.is_main:
            import wandb
            wandb.log({
                "train/loss": loss.item(),
                "train/ce_loss": ce_loss.item(),
                "train/cycle_loss": cycle_loss.item(),
                "train/rec_loss": rec_loss.item(),
                "train/top1": acc1[0].item(),
                "train/top5": acc5[0].item(),
            })

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)


def validate(val_loader, model, criterion, args):

    use_accel = not args.no_accel and torch.accelerator.is_available()

    def run_validate(loader, base_progress=0):

        if use_accel:
            device = torch.accelerator.current_accelerator()
        else:
            device = torch.device("cpu")

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if use_accel:
                    if args.gpu is not None and device.type=='cuda':
                        torch.accelerator.set_device_index(args.gpu)
                        images = images.cuda(args.gpu, non_blocking=True)
                        target = target.cuda(args.gpu, non_blocking=True)
                    else:
                        images = images.to(device)
                        target = target.to(device)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', use_accel, ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', use_accel, ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', use_accel, ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', use_accel, ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return top1.avg


def save_checkpoint(state, is_best, checkpoint_dir='check_points_cls'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = os.path.join(checkpoint_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_dir, 'best_model.pth'))

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, use_accel, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.use_accel = use_accel
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):    
        if self.use_accel:
            device = torch.accelerator.current_accelerator()
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
