import os
import sys
import time
import signal
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.distributed as dist

from centernet_datasets.coco import COCO, COCO_eval
from centernet_datasets.pascal import PascalVOC, PascalVOC_eval

from nets.spnn_centernet import get_spnn_centernet

from utils.utils import _tranpose_and_gather_feature, load_model
from utils.image import transform_preds
from utils.losses import _neg_loss, _neg_loss_soft, _reg_loss, _vfl_loss
from utils.summary import create_summary, create_logger, create_saver, DisablePrint
from utils.post_process import ctdet_decode

# Training settings
parser = argparse.ArgumentParser(description='simple_centernet45')

parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--dist', action='store_true')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_name', type=str, default='test')
parser.add_argument('--pretrain_name', type=str, default='pretrain')

parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'pascal'])
parser.add_argument('--arch', type=str, default='large_hourglass')
parser.add_argument('--spnn_backbone', type=str, default=None,
                    help='Path to pretrained SPNN classifier checkpoint for backbone transfer')
parser.add_argument('--hmap_init_scale', type=float, default=0.01,
                    help='Initial scale for heatmap affine adapter (default: 0.01)')
parser.add_argument('--hmap_init_bias', type=float, default=-2.19,
                    help='Initial bias for heatmap affine adapter. CenterNet default '
                         '-2.19 caps SPNN max sigmoid; try 0.0 to remove the ceiling.')
parser.add_argument('--head_mode', type=str, default='affine',
                    choices=['affine', 'orthogonal_mix'],
                    help='Heatmap head structure (both bijective). '
                         '"affine": per-class scale + bias only. '
                         '"orthogonal_mix": adds a learnable C×C orthogonal '
                         'channel mixer between scale and bias.')
parser.add_argument('--head_mix_type', type=str, default='householder',
                    choices=['cayley', 'householder'],
                    help='Parameterization for the orthogonal_mix head. '
                         '"householder": product of K reflections, bit-exact '
                         'orthogonal (best for DDNM round-trips). '
                         '"cayley": matrix_exp(A−Aᵀ); identity init when A=0.')
parser.add_argument('--head_mix_reflections', type=int, default=0,
                    help='Number of Householder reflections (0 → default = '
                         'num_classes, which covers all of O(C)). Ignored '
                         'when head_mix_type=cayley.')
parser.add_argument('--deep_det_head', action='store_true',
                    help='Use the deeper 3-level U-net t/s/r networks from '
                         'models_deeper.py for the detector head (block 4) '
                         'only. Backbone (blocks 0-3) keeps models.py shallow '
                         'U-net so pretrained backbone weights load cleanly.')
parser.add_argument('--deep_head_hidden', type=int, default=128,
                    help='Hidden width for deep_det_head ConvMLPs. With '
                         'feat_size=64 the deeper U-net is h, 2h, 4h. '
                         '128 → ~30M extra params; 256 → ~120M extra. '
                         'Ignored when --deep_det_head is not set.')
parser.add_argument('--mlp_tail_hidden', type=int, default=0,
                    help='If > 0, append a per-pixel MLP residual tail '
                         '(1x1 Conv → GN → GELU stack, zero-init last conv) '
                         'inside each of s/t/r in the deep det head. Adds '
                         'nonlinear channel-direction capacity without '
                         'breaking bijectivity (the inverse picks up the '
                         'same residual). 0 = current behavior (no tail). '
                         'Requires --deep_det_head.')
parser.add_argument('--two_block_head', action='store_true',
                    help='Split the detector head into 2 ConvPINN blocks '
                         '(28→26→24) instead of the default single block '
                         '(28→24). Each split block has x1=2 conditioning '
                         'channels (vs x1=4 for the single-block variant). '
                         'Backbone (blocks 0-3) is unchanged and still '
                         'transfers fully from the classifier.')
parser.add_argument('--freeze_backbone', action='store_true',
                    help='Freeze the SPNN backbone (blocks 0-3, transferred '
                         'from the classifier) by detaching the tensor '
                         'between backbone and head during forward. Only the '
                         'head ConvPINN block(s) and the affine adapter / '
                         'orthogonal mixer receive gradients.')
parser.add_argument('--warm_start_full', type=str, default=None,
                    help='Path to a full SPNN-CenterNet checkpoint.t7 to '
                         'warm-start the entire model from (loaded with '
                         'strict=False so newly-introduced params like '
                         's_tail / t_tail / r_tail keep their constructed '
                         'init while every existing weight is overwritten). '
                         'Use this to resume / fine-tune a deep-head run '
                         'with --mlp_tail_hidden enabled.')

parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--split_ratio', type=float, default=1.0)

parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_step', type=str, default='90,120')
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--num_epochs', type=int, default=140)

parser.add_argument('--test_topk', type=int, default=100)

parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--val_interval', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=2)

# Distillation from a frozen teacher (e.g., ResNet-18 CenterNet)
parser.add_argument('--teacher_arch', type=str, default=None,
                    help='Teacher architecture for distillation (e.g., resnet_18). '
                         'If None, no distillation.')
parser.add_argument('--teacher_checkpoint', type=str, default=None,
                    help='Path to teacher checkpoint (.t7 / .pth)')
parser.add_argument('--lambda_distill_hmap', type=float, default=1.0,
                    help='Weight for hmap distillation (MSE in probability space)')
parser.add_argument('--lambda_distill_regs', type=float, default=1.0,
                    help='Weight for regs distillation (L1 at GT positions)')
parser.add_argument('--lambda_distill_wh', type=float, default=0.1,
                    help='Weight for w_h_ distillation (L1 at GT positions)')

# Varifocal loss for the heatmap (quality-aware classification target)
parser.add_argument('--vfl', action='store_true',
                    help='Use varifocal-style heatmap loss: positive target '
                         'becomes IoU(pred_box, gt_box) instead of 1. '
                         'Negatives keep CenterNet focal-loss form.')
parser.add_argument('--vfl_warmup_epochs', type=int, default=5,
                    help='Number of epochs to keep plain focal loss before '
                         'switching to VFL. Avoids cold-start collapse '
                         '(zero-init wh -> IoU=0 -> all peaks killed).')

# wandb logging (rank 0 only)
parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
parser.add_argument('--wandb_project', type=str, default='spnn-centernet',
                    help='wandb project name')
parser.add_argument('--wandb_run_name', type=str, default=None,
                    help='wandb run name (defaults to --log_name)')

cfg = parser.parse_args()

# torchrun sets LOCAL_RANK / RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT
# environment variables; honor LOCAL_RANK so the same train.py runs under
# both the legacy `--local_rank N` style and modern torchrun.
cfg.local_rank = int(os.environ.get('LOCAL_RANK', cfg.local_rank))

os.chdir(cfg.root_dir)

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)
cfg.pretrain_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.pretrain_name, 'checkpoint.t7')

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

cfg.lr_step = [int(s) for s in cfg.lr_step.split(',')]


def main():
  saver = create_saver(cfg.local_rank, save_dir=cfg.ckpt_dir)
  logger = create_logger(cfg.local_rank, save_dir=cfg.log_dir)
  summary_writer = create_summary(cfg.local_rank, log_dir=cfg.log_dir)
  print = logger.info
  print(cfg)

  wandb_run = None
  if cfg.wandb and cfg.local_rank == 0:
    import wandb
    wandb_run = wandb.init(project=cfg.wandb_project,
                           name=cfg.wandb_run_name or cfg.log_name,
                           config=vars(cfg),
                           dir=cfg.log_dir)

  torch.manual_seed(317)
  torch.backends.cudnn.benchmark = True  # disable this if OOM at beginning of training

  if cfg.dist:
    # Trust env (torchrun) over device_count: with one process per GPU,
    # device_count may report all visible GPUs even when this rank should
    # only use one.  WORLD_SIZE is authoritative for the sampler / batch math.
    num_gpus = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))
    cfg.device = torch.device('cuda:%d' % cfg.local_rank)
    torch.cuda.set_device(cfg.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=num_gpus, rank=cfg.local_rank)
  else:
    num_gpus = torch.cuda.device_count()
    cfg.device = torch.device('cuda')

  print('Setting up data...')
  Dataset = COCO if cfg.dataset == 'coco' else PascalVOC
  train_dataset = Dataset(cfg.data_dir, 'train', split_ratio=cfg.split_ratio, img_size=cfg.img_size)
  train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                  num_replicas=num_gpus,
                                                                  rank=cfg.local_rank)
  train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=cfg.batch_size // num_gpus
                                             if cfg.dist else cfg.batch_size,
                                             shuffle=not cfg.dist,
                                             num_workers=cfg.num_workers,
                                             pin_memory=True,
                                             drop_last=True,
                                             sampler=train_sampler if cfg.dist else None,
                                             persistent_workers=cfg.num_workers > 0,
                                             prefetch_factor=4)

  Dataset_eval = COCO_eval if cfg.dataset == 'coco' else PascalVOC_eval
  val_dataset = Dataset_eval(cfg.data_dir, 'val', test_scales=[1.], test_flip=False)
  val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                           shuffle=False, num_workers=1, pin_memory=True,
                                           collate_fn=val_dataset.collate_fn)

  print('Creating model...')
  if 'hourglass' in cfg.arch:
    from nets.hourglass import get_hourglass
    model = get_hourglass[cfg.arch]
  elif 'resdcn' in cfg.arch:
    from nets.resdcn import get_pose_net
    model = get_pose_net(num_layers=int(cfg.arch.split('_')[-1]),
                         head_conv=64, num_classes=train_dataset.num_classes)
  elif 'resnet' in cfg.arch:
    # Plain ResNet (no DCN) — uses nets/resnet.py with ImageNet pretrained
    from nets.resnet import get_pose_net as get_resnet_pose_net
    model = get_resnet_pose_net(num_layers=int(cfg.arch.split('_')[-1]),
                                 head_conv=64, num_classes=train_dataset.num_classes)
  elif cfg.arch == 'spnn':
    model = get_spnn_centernet(num_classes=train_dataset.num_classes,
                               pretrained_backbone=cfg.spnn_backbone,
                               hmap_init_scale=cfg.hmap_init_scale,
                               hmap_init_bias=cfg.hmap_init_bias,
                               head_mode=cfg.head_mode,
                               head_mix_type=cfg.head_mix_type,
                               head_mix_reflections=(cfg.head_mix_reflections
                                                     if cfg.head_mix_reflections > 0
                                                     else None),
                               deep_det_head=cfg.deep_det_head,
                               deep_head_hidden=cfg.deep_head_hidden,
                               mlp_tail_hidden=cfg.mlp_tail_hidden,
                               two_block_head=cfg.two_block_head,
                               freeze_backbone=cfg.freeze_backbone)
  else:
    raise NotImplementedError

  # Full-model warm start: load every weight that's compatible from a prior
  # SPNN-CenterNet checkpoint (strict=False so newly-introduced params like
  # the s_tail / t_tail / r_tail keys keep their constructed init).
  # Done BEFORE DDP/DataParallel wrap so we work with the un-prefixed keys.
  if cfg.arch == 'spnn' and cfg.warm_start_full is not None:
    print('[warm-start-full] loading %s' % cfg.warm_start_full)
    raw_w = torch.load(cfg.warm_start_full, map_location='cpu', weights_only=False)
    if isinstance(raw_w, dict) and 'state_dict' in raw_w:
      w_state = raw_w['state_dict']
    elif isinstance(raw_w, dict) and 'model' in raw_w:
      w_state = raw_w['model']
    else:
      w_state = raw_w
    w_state = {k[7:] if k.startswith('module.') else k: v for k, v in w_state.items()}
    missing, unexpected = model.load_state_dict(w_state, strict=False)
    print('[warm-start-full] loaded: missing=%d, unexpected=%d' %
          (len(missing), len(unexpected)))
    if missing:
      print('[warm-start-full] first missing keys (kept at constructed init): %s'
            % missing[:8])
    if unexpected:
      print('[warm-start-full] unexpected keys (ignored): %s' % unexpected[:8])

  if cfg.dist:
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(cfg.device)
    # find_unused_parameters=True is required because:
    #  1) every coupling block has an `r` network used only by model.pinv()
    #     (the inverse direction for DDNM); during detection training those
    #     params never receive gradient.
    #  2) when --mlp_tail_hidden > 0 the tail's last 1x1 conv is zero-init,
    #     so at step 0 the tail's inner layers see d_loss/d_input = 0 and
    #     also miss gradient until W_last starts moving.
    # DataParallel tolerated both silently; DDP needs to be told explicitly.
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[cfg.local_rank, ],
                                                output_device=cfg.local_rank,
                                                find_unused_parameters=True)
  else:
    model = nn.DataParallel(model).to(cfg.device)

  if os.path.isfile(cfg.pretrain_dir):
    model = load_model(model, cfg.pretrain_dir)

  teacher_model = None
  if cfg.teacher_arch is not None and cfg.teacher_checkpoint is not None:
    print('Building teacher: %s' % cfg.teacher_arch)
    if 'resnet' in cfg.teacher_arch:
      from nets.resnet import get_pose_net as get_resnet_pose_net
      teacher_model = get_resnet_pose_net(num_layers=int(cfg.teacher_arch.split('_')[-1]),
                                          head_conv=64, num_classes=train_dataset.num_classes)
    elif 'resdcn' in cfg.teacher_arch:
      from nets.resdcn import get_pose_net as get_resdcn_pose_net
      teacher_model = get_resdcn_pose_net(num_layers=int(cfg.teacher_arch.split('_')[-1]),
                                          head_conv=64, num_classes=train_dataset.num_classes)
    else:
      raise NotImplementedError('Teacher arch %s not supported' % cfg.teacher_arch)

    raw_t = torch.load(cfg.teacher_checkpoint, map_location='cpu', weights_only=False)
    if isinstance(raw_t, dict) and 'state_dict' in raw_t:
      t_state = raw_t['state_dict']
    elif isinstance(raw_t, dict) and 'model' in raw_t:
      t_state = raw_t['model']
    else:
      t_state = raw_t
    t_state = {k[7:] if k.startswith('module.') else k: v for k, v in t_state.items()}
    missing, unexpected = teacher_model.load_state_dict(t_state, strict=False)
    print('[teacher] Loaded %s from %s (missing=%d, unexpected=%d)' %
          (cfg.teacher_arch, cfg.teacher_checkpoint, len(missing), len(unexpected)))
    teacher_model = teacher_model.to(cfg.device)
    teacher_model.eval()
    for p in teacher_model.parameters():
      p.requires_grad_(False)
    if not cfg.dist:
      teacher_model = nn.DataParallel(teacher_model)

  optimizer = torch.optim.Adam(model.parameters(), cfg.lr)
  lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.lr_step, gamma=0.1)

  current_epoch = [0]

  def _save_and_exit(signum, frame):
    try:
      ckpt_path = os.path.join(cfg.ckpt_dir, 'preempt_checkpoint.pth')
      tmp = ckpt_path + '.tmp'
      m = model.module if hasattr(model, 'module') else model
      torch.save({
        'model': m.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': current_epoch[0],
      }, tmp)
      os.replace(tmp, ckpt_path)
      print('[preempt] checkpoint saved to %s at epoch %d' % (ckpt_path, current_epoch[0]), flush=True)
    except Exception as e:
      print('[preempt] checkpoint save failed: %s' % e, flush=True)
    sys.exit(0)

  signal.signal(signal.SIGTERM, _save_and_exit)

  def train(epoch):
    print('\n Epoch: %d' % epoch)
    model.train()
    tic = time.perf_counter()
    for batch_idx, batch in enumerate(train_loader):
      for k in batch:
        if k != 'meta':
          batch[k] = batch[k].to(device=cfg.device, non_blocking=True)

      outputs = model(batch['image'])
      hmap, regs, w_h_ = zip(*outputs)
      regs = [_tranpose_and_gather_feature(r, batch['inds']) for r in regs]
      w_h_ = [_tranpose_and_gather_feature(r, batch['inds']) for r in w_h_]

      if cfg.vfl and epoch > cfg.vfl_warmup_epochs:
        hmap_loss = _vfl_loss(hmap, batch['hmap'],
                              regs[0], w_h_[0],
                              batch['regs'], batch['w_h_'],
                              batch['inds'], batch['ind_masks'])
      else:
        hmap_loss = _neg_loss(hmap, batch['hmap'])
      reg_loss = _reg_loss(regs, batch['regs'], batch['ind_masks'])
      w_h_loss = _reg_loss(w_h_, batch['w_h_'], batch['ind_masks'])
      loss = hmap_loss + 1 * reg_loss + 0.1 * w_h_loss

      d_hmap_val = d_regs_val = d_wh_val = 0.0
      if teacher_model is not None:
        with torch.no_grad():
          t_outputs = teacher_model(batch['image'])
        t_hmap, t_regs, t_w_h_ = zip(*t_outputs)
        t_regs_g = [_tranpose_and_gather_feature(r, batch['inds']).detach() for r in t_regs]
        t_w_h_g = [_tranpose_and_gather_feature(r, batch['inds']).detach() for r in t_w_h_]

        # Soft focal loss with teacher's sigmoid hmap as soft target.
        # Magnitude is comparable to supervised hmap_loss, so lambda~1.0 works.
        t_hmap_soft = [t.sigmoid().detach() for t in t_hmap]
        d_hmap = _neg_loss_soft(hmap, t_hmap_soft[0])
        d_regs = _reg_loss(regs, t_regs_g[0], batch['ind_masks'])
        d_wh = _reg_loss(w_h_, t_w_h_g[0], batch['ind_masks'])

        loss = (loss + cfg.lambda_distill_hmap * d_hmap
                + cfg.lambda_distill_regs * d_regs
                + cfg.lambda_distill_wh * d_wh)
        d_hmap_val = d_hmap.item()
        d_regs_val = d_regs.item()
        d_wh_val = d_wh.item()

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if batch_idx % cfg.log_interval == 0:
        duration = time.perf_counter() - tic
        tic = time.perf_counter()
        msg = ('[%d/%d-%d/%d] ' % (epoch, cfg.num_epochs, batch_idx, len(train_loader)) +
               ' hmap_loss= %.5f reg_loss= %.5f w_h_loss= %.5f' %
               (hmap_loss.item(), reg_loss.item(), w_h_loss.item()))
        if teacher_model is not None:
          msg += ' d_hmap= %.5f d_reg= %.5f d_wh= %.5f' % (d_hmap_val, d_regs_val, d_wh_val)
        msg += ' (%d samples/sec)' % (cfg.batch_size * cfg.log_interval / duration)
        print(msg)

        step = len(train_loader) * epoch + batch_idx
        summary_writer.add_scalar('hmap_loss', hmap_loss.item(), step)
        summary_writer.add_scalar('reg_loss', reg_loss.item(), step)
        summary_writer.add_scalar('w_h_loss', w_h_loss.item(), step)
        if teacher_model is not None:
          summary_writer.add_scalar('distill/hmap', d_hmap_val, step)
          summary_writer.add_scalar('distill/regs', d_regs_val, step)
          summary_writer.add_scalar('distill/wh', d_wh_val, step)

        if wandb_run is not None:
          wandb_log = {
            'train/hmap_loss': hmap_loss.item(),
            'train/reg_loss': reg_loss.item(),
            'train/w_h_loss': w_h_loss.item(),
            'train/total_loss': loss.item(),
            'train/lr': optimizer.param_groups[0]['lr'],
            'train/step': step,
            'epoch': epoch,
          }
          if teacher_model is not None:
            wandb_log['train/distill_hmap'] = d_hmap_val
            wandb_log['train/distill_regs'] = d_regs_val
            wandb_log['train/distill_wh'] = d_wh_val
          wandb_run.log(wandb_log)
    return

  def val_map(epoch):
    print('\n Val@Epoch: %d' % epoch)
    model.eval()
    torch.cuda.empty_cache()
    max_per_image = 100

    results = {}
    with torch.no_grad():
      for inputs in val_loader:
        img_id, inputs = inputs[0]

        detections = []
        for scale in inputs:
          inputs[scale]['image'] = inputs[scale]['image'].to(cfg.device)
          output = model(inputs[scale]['image'])[-1]

          dets = ctdet_decode(*output, K=cfg.test_topk)
          dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0]

          top_preds = {}
          dets[:, :2] = transform_preds(dets[:, 0:2],
                                        inputs[scale]['center'],
                                        inputs[scale]['scale'],
                                        (inputs[scale]['fmap_w'], inputs[scale]['fmap_h']))
          dets[:, 2:4] = transform_preds(dets[:, 2:4],
                                         inputs[scale]['center'],
                                         inputs[scale]['scale'],
                                         (inputs[scale]['fmap_w'], inputs[scale]['fmap_h']))
          clses = dets[:, -1]
          for j in range(val_dataset.num_classes):
            inds = (clses == j)
            top_preds[j + 1] = dets[inds, :5].astype(np.float32)
            top_preds[j + 1][:, :4] /= scale

          detections.append(top_preds)

        bbox_and_scores = {j: np.concatenate([d[j] for d in detections], axis=0)
                           for j in range(1, val_dataset.num_classes + 1)}
        scores = np.hstack([bbox_and_scores[j][:, 4] for j in range(1, val_dataset.num_classes + 1)])
        if len(scores) > max_per_image:
          kth = len(scores) - max_per_image
          thresh = np.partition(scores, kth)[kth]
          for j in range(1, val_dataset.num_classes + 1):
            keep_inds = (bbox_and_scores[j][:, 4] >= thresh)
            bbox_and_scores[j] = bbox_and_scores[j][keep_inds]

        results[img_id] = bbox_and_scores

    eval_results = val_dataset.run_eval(results, save_dir=cfg.ckpt_dir)
    print(eval_results)
    summary_writer.add_scalar('val_mAP/mAP', eval_results[0], epoch)
    return eval_results

  print('Starting training...')
  for epoch in range(1, cfg.num_epochs + 1):
    current_epoch[0] = epoch
    train_sampler.set_epoch(epoch)
    train(epoch)
    # Validation is rank-0-only: the val loader has no DistributedSampler, so
    # every rank would otherwise iterate the entire test set independently and
    # race when val_dataset.run_eval writes detection files into cfg.ckpt_dir.
    # DDP does not deadlock here because val_map performs no collective ops;
    # ranks > 0 proceed to next epoch's train and DDP allreduce blocks them
    # at the first backward until rank-0 catches up.
    if cfg.val_interval > 0 and epoch % cfg.val_interval == 0 and cfg.local_rank == 0:
      eval_results = val_map(epoch)
      if wandb_run is not None and eval_results is not None:
        wandb_log = {'val/mAP': float(eval_results[0]), 'epoch': epoch}
        if cfg.dataset == 'pascal' and len(eval_results) > 1:
          from centernet_datasets.pascal import VOC_NAMES
          for cls_name, ap in zip(VOC_NAMES[1:], eval_results[1]):
            wandb_log['val/AP_%s' % cls_name] = float(ap)
        wandb_run.log(wandb_log)
    # saver.save is already rank-aware (no-op for rank > 0)
    print(saver.save(model.module.state_dict(), 'checkpoint'))
    lr_scheduler.step(epoch)  # move to here after pytorch1.1.0

  summary_writer.close()
  if wandb_run is not None:
    wandb_run.finish()


if __name__ == '__main__':
  with DisablePrint(local_rank=cfg.local_rank):
    main()
