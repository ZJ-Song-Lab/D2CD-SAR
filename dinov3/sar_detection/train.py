# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with the
# terms of the DINOv3 License Agreement.

"""
Training script for D²CD-SAR cross-modal knowledge distillation.

Implements Algorithm 1 of the D²CD-SAR paper:
  1. student multi-scale features {S3, S4, S5} + AIFI -> F5;
  2. frozen DINOv3-ViT-Base teacher patch features F_tea^sp;
  3. DRCP alignment -> L_DRCP, detection losses -> L_task, and the
     A^2TD-LoRA regularizers L_ortho / L_sparsity;
  4. direction-aware variance-gate update on the AIFI LoRA space;
  5. minimize L_total and step the optimizer;
  6. at the end, reparameterize the LoRA branches into the frozen backbone
     (Eq. 15) and export a zero-overhead RT-DETR-R18 deployment checkpoint.

Defaults follow the paper's Implementation Details:
  AdamW, lr = 1e-4, weight_decay = 1e-4, 120 epochs, batch size 16.
"""

import os
import sys
import argparse
import random
import numpy as np
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Make the package importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dinov3.sar_detection.distillation import build_distiller
from dinov3.data.SSDD.ssdd_dataset import SSDDDataset, collate_fn
from dinov3.data.SSDD.transforms import (
    make_train_transforms,
    make_val_transforms,
    DATASET_STATS,
)
from dinov3.data.HRSID.hrsid_dataset import HRSIDDataset


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    """Seed Python, NumPy and PyTorch (CPU + CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------
def setup_distributed():
    """Initialize DDP if launched with torchrun, otherwise run single-process."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
    else:
        rank, world_size, gpu = 0, 1, 0

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://",
                                 world_size=world_size, rank=rank)
        dist.barrier()
    return rank, world_size, gpu


def unwrap(model):
    return model.module if isinstance(model, (DDP, nn.parallel.DataParallel)) else model


def set_train_mode(distiller):
    """Put the distiller in train mode, but keep the frozen backbone and the
    frozen teacher in eval mode so their BatchNorm / running stats are frozen."""
    distiller.train()
    distiller.student.backbone.eval()
    distiller.teacher.eval()
    return distiller


def collate_batch(images, device):
    """The collate_fn returns a list of [3, H, W] tensors; stack to [B,3,H,W]."""
    return torch.stack(images, dim=0).to(device, non_blocking=True)


# ---------------------------------------------------------------------------
# mAP evaluation (COCO-style, IoU 0.50:0.95)
# ---------------------------------------------------------------------------
def evaluate(distiller, data_loader, device, num_classes, img_size, max_dets=300):
    """COCO-style mAP@[.5:.95] and AP50 via the standard COCO evaluator."""
    from dinov3.sar_detection.rtdetr import PostProcess
    from dinov3.sar_detection.evaluate import compute_map

    core = unwrap(distiller)
    core.eval()
    post = PostProcess()

    gt_by_img = {}
    det_list = []

    for images, targets in data_loader:
        images = collate_batch(images, device)
        bsz = images.shape[0]
        outputs = core.student(images)
        target_sizes = torch.full((bsz, 2), float(img_size), device=device)
        results = post(outputs, target_sizes)
        t_dev = [{k: v.to(device) for k, v in t.items()} for t in targets]

        for b in range(bsz):
            iid = int(t_dev[b]["image_id"].item())
            gt_by_img[iid] = {
                "boxes": t_dev[b]["boxes"],
                "labels": t_dev[b]["labels"],
            }
            r = results[b]
            scores, labels, boxes = r["scores"], r["labels"], r["boxes"]
            for k in range(len(scores)):
                det_list.append((iid, float(scores[k]), boxes[k].clone(), int(labels[k])))

    return compute_map(gt_by_img, det_list, num_classes, device, max_dets=max_dets)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one_epoch(distiller, data_loader, optimizer, device, epoch, rank,
                    grad_clip=0.0, print_freq=50):
    set_train_mode(distiller)
    core = unwrap(distiller)
    meters = defaultdict(float)
    n = 0

    for i, (images, targets) in enumerate(data_loader):
        images = collate_batch(images, device)
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        out = distiller(images, targets)                 # DDP forward
        loss = out["loss_total"]

        # Step 3 of Algorithm 1: refresh the direction-aware variance gate on
        # the AIFI LoRA space (retain_graph=True keeps the graph for backward).
        core.update_gate(out["loss_drcp"], out["loss_task"], out["z_distill"], out["z_task"])

        # Step 4: optimize.
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in distiller.parameters() if p.requires_grad], grad_clip)
        optimizer.step()

        n += 1
        meters["loss_total"] += loss.item()
        for k in ("loss_cls", "loss_box", "loss_drcp", "loss_ortho", "loss_sparsity"):
            meters[k] += out[k].item()
        meters["gate"] += out["gate_value"]

        if rank == 0 and i % print_freq == 0 and i > 0:
            avg = meters["loss_total"] / print_freq
            print(f"[Epoch {epoch}] [{i}/{len(data_loader)}] "
                  f"loss={avg:.4f} (cls {meters['loss_cls']/print_freq:.4f}, "
                  f"box {meters['loss_box']/print_freq:.4f}, "
                  f"drcp {meters['loss_drcp']/print_freq:.4f}, "
                  f"ortho {meters['loss_ortho']/print_freq:.4f}, "
                  f"sparse {meters['loss_sparsity']/print_freq:.4f}, "
                  f"gate {meters['gate']/print_freq:.3f})", flush=True)
            meters.clear()

    return {k: v / max(n, 1) for k, v in meters.items()}


def build_optimizer(distiller, lr, weight_decay):
    params = [p for p in distiller.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def main(args):
    if args.data_root is None:
        args.data_root = f"./dinov3/data/{args.dataset.upper()}"
    set_seed(args.seed)
    rank, world_size, gpu = setup_distributed()
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print("=== D²CD-SAR cross-modal distillation ===")
        print(f"Dataset: {args.dataset}, Data root: {args.data_root}")
        print(f"World size: {world_size}, Rank: {rank}, Device: {device}")
        print(f"Seed: {args.seed}")
        print(f"Arguments: {args}")

    norm_mean, norm_std = DATASET_STATS[args.dataset]
    distiller = build_distiller(
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        r_lora=args.r_lora,
        lambda_distill=args.lambda_distill,
        lambda_ortho=args.lambda_ortho,
        lambda_sparsity=args.lambda_sparsity,
        teacher_pretrained=args.teacher_pretrained,
        backbone_pretrained=args.backbone_pretrained,
        freeze_backbone=True,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )
    distiller = distiller.to(device)

    if world_size > 1:
        distiller = DDP(distiller, device_ids=[gpu] if torch.cuda.is_available() else None,
                        find_unused_parameters=True)

    # Data
    DatasetClass = SSDDDataset if args.dataset == "ssdd" else HRSIDDataset
    train_tf = make_train_transforms(img_size=args.img_size, mean=norm_mean, std=norm_std)
    val_tf = make_val_transforms(img_size=args.img_size, mean=norm_mean, std=norm_std)
    train_set = DatasetClass(root_dir=args.data_root, split="train", transform=train_tf)
    val_set = DatasetClass(root_dir=args.data_root, split="val", transform=val_tf)
    train_sampler = DistributedSampler(train_set) if world_size > 1 else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler,
                              shuffle=(train_sampler is None), num_workers=args.num_workers,
                              collate_fn=collate_fn, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.eval_batch_size, sampler=val_sampler,
                            shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn,
                            pin_memory=True)

    if rank == 0:
        print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")

    optimizer = build_optimizer(distiller, args.lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_map = 0.0

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            distiller, train_loader, optimizer, device, epoch, rank,
            grad_clip=args.grad_clip, print_freq=args.print_freq)
        scheduler.step()

        if rank == 0:
            metrics = evaluate(distiller, val_loader, device, args.num_classes, args.img_size)
            print(f"Epoch {epoch}: train_loss={train_stats.get('loss_total', 0):.4f} | "
                  f"mAP={metrics['mAP']:.4f} AP50={metrics['AP50']:.4f}", flush=True)

            if metrics["mAP"] > best_map:
                best_map = metrics["mAP"]
                torch.save({"epoch": epoch, "map": best_map, "args": vars(args),
                            "model": unwrap(distiller).state_dict()},
                           output_dir / "best.pth")
                print(f"  -> new best mAP={best_map:.4f}, saved best.pth", flush=True)

            if (epoch + 1) % args.save_freq == 0:
                torch.save({"epoch": epoch, "model": unwrap(distiller).state_dict()},
                           output_dir / f"checkpoint_epoch_{epoch}.pth")

        if world_size > 1:
            dist.barrier()

    # Step 5-6: reparameterize the LoRA branches into the frozen backbone (Eq. 15)
    # and export a zero-overhead RT-DETR-R18 deployment checkpoint.
    if rank == 0:
        core = unwrap(distiller)
        deploy_student = core.reparameterize()
        torch.save({"student": deploy_student.state_dict(),
                    "num_classes": args.num_classes,
                    "num_queries": args.num_queries,
                    "gate_value": core.gate.value(),
                    "seed": args.seed,
                    "epoch": args.epochs},
                   output_dir / "deploy_student.pth")
        print(f"Deployment student saved to {output_dir / 'deploy_student.pth'} "
              f"(LoRA merged, final gate w={core.gate.value():.4f}).", flush=True)
        print("Training completed!")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("D²CD-SAR distillation training")
    # Data
    parser.add_argument("--dataset", default="ssdd", choices=["ssdd", "hrsid"], type=str,
                        help="Dataset to train on")
    parser.add_argument("--data-root", default=None, type=str,
                        help="Root dir of dataset (default: ./dinov3/data/<DATASET>)")
    parser.add_argument("--output-dir", default="./outputs/d2cd_sar", type=str)
    parser.add_argument("--img-size", default=896, type=int)
    # Model (paper Implementation Details)
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--num-queries", default=300, type=int)
    parser.add_argument("--r-lora", default=16, type=int)
    parser.add_argument("--lambda-distill", default=1.0, type=float)
    parser.add_argument("--lambda-ortho", default=0.1, type=float)
    parser.add_argument("--lambda-sparsity", default=0.01, type=float)
    parser.add_argument("--teacher-pretrained", action="store_true", default=True)
    parser.add_argument("--no-teacher-pretrained", dest="teacher_pretrained", action="store_false")
    parser.add_argument("--no-backbone-pretrained", dest="backbone_pretrained", action="store_false")
    parser.set_defaults(backbone_pretrained=True)
    # Training (paper Implementation Details)
    parser.add_argument("--epochs", default=120, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--eval-batch-size", default=8, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--grad-clip", default=0.0, type=float)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--print-freq", default=50, type=int)
    parser.add_argument("--save-freq", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int,
                        help="Random seed for reproducible training (paper reports mean ± std over multiple seeds)")
    args = parser.parse_args()
    main(args)
