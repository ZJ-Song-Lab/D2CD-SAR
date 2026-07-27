# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
Training script for SAR Ship Detection using DINOv3 ViT-7B/16
Optimized for multi-GPU training (6x RTX 5880)
ViT-7B/16: 6.7B parameters, pretrained on SAT-493M satellite dataset
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dinov3.hub.backbones import dinov3_vit7b16, Weights
from dinov3.eval.detection.models.detr import build_model, PostProcess
from dinov3.eval.detection.config import DetectionHeadConfig
from dinov3.eval.detection.models.position_encoding import PositionEncoding
from dinov3.data.SSDD.ssdd_dataset import SSDDDataset, collate_fn
from dinov3.data.SSDD.transforms import make_train_transforms, make_val_transforms


class SARDetector(nn.Module):
    """SAR Ship Detector based on DINOv3 ViT-7B/16"""

    def __init__(self, backbone, config):
        super().__init__()
        self.detector = build_model(backbone, config)

    def forward(self, samples):
        return self.detector(samples)


def setup_distributed():
    """Initialize distributed training"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        gpu = 0

    torch.cuda.set_device(gpu)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    dist.barrier()
    return rank, world_size, gpu


def build_sar_detector_vit7b(num_classes=1, pretrained=True):
    """Build SAR detector with DINOv3 ViT-7B/16 backbone"""

    # Detection config optimized for ViT-7B/16 and SAR ship detection
    # ViT-7B/16 config: embed_dim=4096, depth=40, num_heads=32
    detection_kwargs = dict(
        with_box_refine=True,
        two_stage=True,
        mixed_selection=True,
        look_forward_twice=True,
        k_one2many=6,
        lambda_one2many=1.0,
        num_queries_one2one=1500,  # Increased for 7B model
        num_queries_one2many=1500,
        reparam=True,
        position_embedding=PositionEncoding.SINE,
        num_feature_levels=1,
        dec_layers=6,
        dim_feedforward=2048,
        dropout=0.0,
        norm_type="pre_norm",
        proposal_feature_levels=4,
        proposal_min_size=20,  # Smaller for SAR ships
        decoder_type="global_rpe_decomp",
        decoder_use_checkpoint=False,
        decoder_rpe_hidden_dim=512,
        decoder_rpe_type="linear",
        layers_to_use=None,
        blocks_to_train=None,
        add_transformer_encoder=True,
        num_encoder_layers=6,
        backbone_use_layernorm=False,
        num_classes=num_classes + 1,  # +1 for background
        aux_loss=True,
        topk=1500,
        hidden_dim=768,  # Keep detection head hidden dim at 768
        nheads=8,
    )

    config = DetectionHeadConfig(**detection_kwargs)

    # Load DINOv3 ViT-7B/16 backbone (satellite pretrained)
    print("Loading DINOv3 ViT-7B/16 backbone (satellite pretrained SAT-493M)...")
    print("Model specs: 6.7B parameters, embed_dim=4096, depth=40, num_heads=32")

    backbone = dinov3_vit7b16(
        pretrained=pretrained,
        weights=Weights.SAT493M,  # Use satellite pretrained weights
        check_hash=False,
    )
    backbone.eval()

    # Freeze backbone layers (optional: can be unfrozen for fine-tuning)
    # Note: ViT-7B is very large, consider using DeepSpeed or FSDP for full fine-tuning
    for param in backbone.parameters():
        param.requires_grad = False

    # Configure for SAR detection
    # ViT-7B uses 3x3 windows (n_windows_sqrt=3)
    config.n_windows_sqrt = 3
    config.proposal_in_stride = backbone.patch_size
    config.proposal_tgt_strides = [int(m * backbone.patch_size) for m in (0.5, 1, 2, 4)]

    if config.layers_to_use is None:
        # For depth=40, use layers [9, 19, 29, 39]
        config.layers_to_use = [m * backbone.n_blocks // 4 - 1 for m in range(1, 5)]
        print(f"Using backbone layers: {config.layers_to_use}")

    detector = SARDetector(backbone, config)

    return detector, config


def train_one_epoch(
    model,
    criterion,
    data_loader,
    optimizer,
    device,
    epoch,
    rank,
    print_freq=50,
):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_loss_dict = {}

    for i, (images, targets) in enumerate(data_loader):
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Track metrics
        running_loss += losses.item()
        for k, v in loss_dict.items():
            if k not in running_loss_dict:
                running_loss_dict[k] = 0.0
            running_loss_dict[k] += v.item()

        if rank == 0 and i % print_freq == 0 and i > 0:
            avg_loss = running_loss / print_freq
            print(f"Epoch [{epoch}] Iter [{i}/{len(data_loader)}] Loss: {avg_loss:.4f}")
            running_loss = 0.0
            running_loss_dict = {}

    return running_loss / len(data_loader)


@torch.no_grad()
def evaluate(model, data_loader, device, rank):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)
        # Compute validation metrics here if needed
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main(args):
    """Main training function"""

    # Setup distributed training
    rank, world_size, gpu = setup_distributed()
    device = torch.device(f"cuda:{gpu}")

    if rank == 0:
        print(f"=" * 60)
        print(f"SAR Ship Detection Training with DINOv3 ViT-7B/16")
        print(f"=" * 60)
        print(f"World size: {world_size}, Rank: {rank}, GPU: {gpu}")
        print(f"Backbone: ViT-7B/16 (6.7B parameters, SAT-493M pretrained)")
        print(f"Arguments: {args}")
        print(f"=" * 60)

    # Build model
    model, config = build_sar_detector_vit7b(num_classes=1, pretrained=True)
    model = model.to(device)

    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[gpu], find_unused_parameters=True)

    # Create datasets
    train_transform = make_train_transforms(img_size=args.img_size)
    val_transform = make_val_transforms(img_size=args.img_size)

    train_dataset = SSDDDataset(
        root_dir=args.data_root,
        split="train",
        transform=train_transform,
    )

    val_dataset = SSDDDataset(
        root_dir=args.data_root,
        split="val",
        transform=val_transform,
    )

    # Create samplers for distributed training
    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    if rank == 0:
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Build criterion (DETR loss)
    from dinov3.eval.detection.models.detr import build_criterion
    criterion = build_criterion(config)
    criterion.to(device)

    # Optimizer - only train detection head
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "detector" in n and p.requires_grad],
            "lr": args.lr,
        },
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=0.1)

    # Training loop
    best_loss = float("inf")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train
        train_loss = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch, rank
        )

        # Validate
        if rank == 0:
            val_loss = evaluate(model, val_loader, device, rank)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

            # Save checkpoint
            if val_loss < best_loss:
                best_loss = val_loss
                checkpoint = {
                    "epoch": epoch,
                    "model": model.module.state_dict() if world_size > 1 else model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "config": config,
                    "args": args,
                }
                torch.save(checkpoint, output_dir / "best_checkpoint.pth")
                print(f"Saved best checkpoint with loss {best_loss:.4f}")

            # Save regular checkpoint
            if (epoch + 1) % args.save_freq == 0:
                checkpoint = {
                    "epoch": epoch,
                    "model": model.module.state_dict() if world_size > 1 else model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "config": config,
                    "args": args,
                }
                torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch}.pth")

        lr_scheduler.step()
        dist.barrier()

    if rank == 0:
        print("Training completed!")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SAR Ship Detection Training with ViT-7B/16")

    # Data arguments
    parser.add_argument("--data-root", default="./dinov3/data/SSDD", type=str)
    parser.add_argument("--output-dir", default="./outputs/sar_detection_vit7b", type=str)

    # Model arguments
    parser.add_argument("--img-size", default=896, type=int)
    parser.add_argument("--backbone", default="dinov3_vit7b16", type=str, help="ViT-7B/16 with SAT-493M weights")

    # Training arguments
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch-size", default=2, type=int, help="Batch size per GPU (reduce for 7B model)")
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr-drop", default=40, type=int)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--save-freq", default=10, type=int)

    # Distributed arguments
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--dist-url", default="env://", type=str)

    args = parser.parse_args()
    main(args)
