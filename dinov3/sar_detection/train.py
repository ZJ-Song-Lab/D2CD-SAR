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
# Training (step-based schedule, Algorithm 1)
# ---------------------------------------------------------------------------
def _is_finite_tensor(x) -> bool:
    return isinstance(x, torch.Tensor) and torch.isfinite(x).all().item()


def _is_finite_loss(out) -> bool:
    """Check all data-loss, regularizer, and gate-statistic components are finite."""
    for k in ("loss_total", "loss_drcp", "loss_task",
              "loss_cls", "loss_box", "loss_ortho", "loss_sparsity", "gate_value"):
        v = out.get(k)
        if v is None:
            continue
        if isinstance(v, torch.Tensor) and not torch.isfinite(v).all().item():
            return False
    return True


def train_step_based(distiller, data_loader, optimizer, scheduler, scaler, device,
                     rank, max_steps, warmup_steps, grad_clip, print_freq,
                     amp_dtype):
    """Step-based training loop matching Algorithm 1 of the D^2CD-SAR paper.

    Schedule: 750-step linear warm-up then cosine decay to 1e-6 over
    N_sched=18,002 scheduled steps. Invalid steps (non-finite loss / probe /
    gradient / gate-statistic) are skipped without extending the schedule or
    mutating the gate buffer (Algorithm 1 lines 11-14, 16-18).
    """
    set_train_mode(distiller)
    core = unwrap(distiller)
    meters = defaultdict(float)
    skipped_total = 0
    n_print = 0

    data_iter = iter(data_loader)
    global_step = 0

    while global_step < max_steps:
        # ---- fetch a new batch (cycle the loader if epoch boundary reached) --
        try:
            batch = next(data_iter)
        except StopIteration:
            if dist.is_available() and dist.is_initialized():
                try:
                    data_loader.sampler.set_epoch(
                        getattr(data_loader.sampler, "epoch", 0) + 1
                    )
                except Exception:
                    pass
            data_iter = iter(data_loader)
            batch = next(data_iter)
        images, targets = batch
        images = collate_batch(images, device)
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        # ---- AMP autocast forward + gate candidate -------------------------
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            out = distiller(images, targets)
            loss = out["loss_total"]

        # Algorithm 1 Line 6: probe gradients → compute gate candidate (DO NOT commit yet).
        gate_candidate = None
        invalid_source = None
        try:
            gate_candidate = core.compute_gate_candidate(
                out["loss_drcp"], out["loss_task"],
                out["z_distill"], out["z_task"])
        except Exception:
            invalid_source = "gate_update"

        if not _is_finite_loss(out):
            invalid_source = invalid_source or "non_finite_loss"

        # ---- backward + grad clip + NaN check ------------------------------
        if invalid_source is None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in distiller.parameters() if p.requires_grad], grad_clip
                )
                if isinstance(grad_norm, torch.Tensor) and not torch.isfinite(grad_norm).item():
                    invalid_source = "non_finite_grad_norm"
            # Final validity check on the trainable parameter gradients.
            if invalid_source is None:
                for p in distiller.parameters():
                    if p.requires_grad and p.grad is not None:
                        if not torch.isfinite(p.grad).all().item():
                            invalid_source = "non_finite_param_grad"
                            break

        # ---- commit step only when fully valid (Algorithm 1 Line 19) ------
        if invalid_source is None:
            scaler.step(optimizer)
            scaler.update()
            step_ok = True
            # Algorithm 1 Line 19: commit gate state and buffer only after
            # all loss/probe/gradient checks pass.
            if gate_candidate is not None:
                core.commit_gate(gate_candidate)
        else:
            # Skip without changing gate buffer or stepping optimizer.
            optimizer.zero_grad(set_to_none=True)
            skipped_total += 1
            step_ok = False
            if rank == 0 and n_print == 0:
                print(f"  [!] skipped invalid step (source={invalid_source})",
                      flush=True)

        # ---- warm-up + cosine LR schedule (every scheduled step, even skip) -
        # The schedule counts N_sched attempted minibatches.
        # Save base LR BEFORE any adjustment to avoid clobbering with lr=0
        # at step 0 (warmup multiplier is 0 at the first step).
        if global_step == 0:
            for pg in optimizer.param_groups:
                pg.setdefault("_base_lr", pg["lr"])
        if global_step < warmup_steps:
            # Linear warm-up.
            lr_mult = max(global_step / max(warmup_steps, 1), 0.0)
            for pg in optimizer.param_groups:
                pg["lr"] = pg["_base_lr"] * lr_mult
        else:
            scheduler.step()

        global_step += 1

        # ---- logging --------------------------------------------------------
        if step_ok:
            n_print += 1
            meters["loss_total"] += float(loss.item())
            for k in ("loss_cls", "loss_box", "loss_drcp", "loss_ortho", "loss_sparsity"):
                meters[k] += float(out[k].item())
            meters["gate"] += float(out["gate_value"])

        if rank == 0 and global_step % print_freq == 0:
            denom = max(n_print, 1)
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"[Step {global_step}/{max_steps}] "
                  f"loss={meters['loss_total']/denom:.4f} "
                  f"(cls {meters['loss_cls']/denom:.4f}, "
                  f"box {meters['loss_box']/denom:.4f}, "
                  f"drcp {meters['loss_drcp']/denom:.4f}, "
                  f"ortho {meters['loss_ortho']/denom:.4f}, "
                  f"sparse {meters['loss_sparsity']/denom:.4f}, "
                  f"gate {meters['gate']/denom:.3f}) "
                  f"lr={lr_now:.2e} skip={skipped_total}", flush=True)
            meters.clear()
            n_print = 0

    avg = {k: v / max(n_print, 1) for k, v in meters.items()}
    avg["skipped"] = skipped_total
    return avg, global_step


def build_optimizer(distiller, lr, weight_decay):
    params = [p for p in distiller.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def main(args):
    if args.data_root is None:
        args.data_root = f"./dinov3/data/{args.dataset.upper()}"
    set_seed(args.seed)
    rank, world_size, gpu = setup_distributed()
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    # AMP dtype
    if args.amp_dtype == "bfloat16":
        amp_dtype = torch.bfloat16
    elif args.amp_dtype == "float16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32
    use_amp = amp_dtype != torch.float32 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    if rank == 0:
        print("=== D²CD-SAR cross-modal distillation ===")
        print(f"Dataset: {args.dataset}, Data root: {args.data_root}")
        print(f"World size: {world_size}, Rank: {rank}, Device: {device}")
        print(f"Seed: {args.seed}  AMP: {args.amp_dtype}  use_grad_scaler={amp_dtype==torch.float16}")
        print(f"Schedule: N_sched={args.max_steps} steps, warmup={args.warmup_steps}, "
              f"lr_min={args.lr_min:.1e}, grad_clip={args.grad_clip}")
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

    # Bind the train-set reference into Mosaic and MixUp (if present) so these
    # transforms can fetch companion samples for 4-image mosaic and 2-image mixup
    # as specified in the paper Table 3 frozen augmentation config.
    try:
        for t in train_tf.transforms:
            if hasattr(t, "bind_dataset"):
                t.bind_dataset(train_set)
    except Exception:
        pass
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

    # Optimizer + post-warmup cosine scheduler (starts acting after warmup_steps).
    optimizer = build_optimizer(distiller, args.lr, args.weight_decay)
    post_warmup_steps = max(args.max_steps - args.warmup_steps, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=post_warmup_steps, eta_min=args.lr_min
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_map = 0.0

    # --- Step-based training loop (paper Algorithm 1) ------------------------
    train_stats, total_steps = train_step_based(
        distiller=distiller,
        data_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        rank=rank,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        print_freq=args.print_freq,
        amp_dtype=amp_dtype if device.type == "cuda" else torch.float32,
    )

    # --- Periodic validation + checkpointing (executed once post-training here
    # for the registry-compliant single validation-selection rule described in
    # Sec. 4.3). In longer runs, uncomment the in-loop validator below.
    if rank == 0:
        metrics = evaluate(distiller, val_loader, device, args.num_classes, args.img_size)
        skip_rate = 100.0 * train_stats.get("skipped", 0) / max(args.max_steps, 1)
        print(f"[Final step {total_steps}] train_loss={train_stats.get('loss_total', 0):.4f} | "
              f"mAP={metrics['mAP']:.4f} AP50={metrics['AP50']:.4f} "
              f"skip_rate={skip_rate:.3f}%", flush=True)
        best_map = metrics["mAP"]
        torch.save({"step": total_steps, "map": best_map, "args": vars(args),
                    "model": unwrap(distiller).state_dict()},
                   output_dir / "best.pth")
        print(f"  -> saved best.pth, mAP={best_map:.4f}", flush=True)

        if args.save_freq > 0:
            torch.save({"step": total_steps, "model": unwrap(distiller).state_dict()},
                       output_dir / f"checkpoint_step_{total_steps}.pth")

    if world_size > 1:
        dist.barrier()

    # Reparameterize LoRA branches (Eq. 15) and export deployment student.
    if rank == 0:
        core = unwrap(distiller)
        deploy_student = core.reparameterize()
        torch.save({"student": deploy_student.state_dict(),
                    "num_classes": args.num_classes,
                    "num_queries": args.num_queries,
                    "gate_value": core.gate.value(),
                    "seed": args.seed,
                    "total_steps": total_steps,
                    "skip_rate_pct": 100.0 * train_stats.get("skipped", 0) / max(args.max_steps, 1)},
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
    parser.add_argument("--img-size", default=640, type=int,
                        help="Input resolution (paper: 640x640)")
    # Model (paper Implementation Details / Table 3 frozen config)
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--num-queries", default=300, type=int)
    parser.add_argument("--r-lora", default=100, type=int,
                        help="LoRA rank per branch (paper r_max=100, combined R=200)")
    parser.add_argument("--lambda-distill", default=1.0, type=float)
    parser.add_argument("--lambda-ortho", default=0.1, type=float)
    parser.add_argument("--lambda-sparsity", default=0.01, type=float)
    parser.add_argument("--teacher-pretrained", action="store_true", default=True)
    parser.add_argument("--no-teacher-pretrained", dest="teacher_pretrained", action="store_false")
    parser.add_argument("--no-backbone-pretrained", dest="backbone_pretrained", action="store_false")
    parser.set_defaults(backbone_pretrained=True)
    # Training (paper Implementation Details / Table 3 frozen config)
    parser.add_argument("--max-steps", default=18002, type=int,
                        help="Total scheduled optimization steps (paper N_sched=18,002)")
    parser.add_argument("--warmup-steps", default=750, type=int,
                        help="Linear LR warm-up steps (paper 750)")
    parser.add_argument("--lr-min", default=1e-6, type=float,
                        help="Cosine decay minimum LR (paper 1e-6)")
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--eval-batch-size", default=8, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--grad-clip", default=1.0, type=float,
                        help="Gradient clipping max norm (paper 1.0)")
    parser.add_argument("--amp-dtype", default="bfloat16", type=str,
                        choices=["float32", "bfloat16", "float16"],
                        help="Mixed-precision dtype (paper uses bfloat16)")
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--print-freq", default=50, type=int)
    parser.add_argument("--save-freq", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int,
                        help="Random seed for reproducible training (paper reports mean ± std over multiple seeds)")
    args = parser.parse_args()
    main(args)
