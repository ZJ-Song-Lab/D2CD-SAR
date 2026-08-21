# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with the
# terms of the DINOv3 License Agreement.

"""Standalone evaluation script for the SAR-RTDETR deployment model.

Produces the reproducibility artifacts referenced in the paper's Experiment
section:

  * ``predictions.json``  — COCO-style detection results
    ``[{image_id, category_id, bbox:[x,y,w,h], score}, ...]``.
  * ``latency.json``       — per-image inference latency (ms) and FPS,
    measured after a fixed warm-up on the *deployment* student (LoRA merged,
    zero teacher / DRCP overhead).
  * ``metrics_<seed>.json`` — mAP@[.5:.95] and AP50 for one seed.
  * ``seed_summary.json``  — mean ± std of mAP / AP50 / latency across all
    seeds (only written when more than one seed is evaluated).

Usage
-----
Single checkpoint::

    python -m dinov3.sar_detection.evaluate \
        --checkpoint outputs/sar_rtdetr/deploy_student.pth \
        --dataset ssdd --data-root ./dinov3/data/SSDD \
        --split val --output-dir outputs/sar_rtdetr/eval

Multiple seeds (aggregates mean ± std)::

    python -m dinov3.sar_detection.evaluate \
        --checkpoint outputs/sar_rtdetr/deploy_student.pth \
        --dataset ssdd --data-root ./dinov3/data/SSDD \
        --split val --seeds 42,123,2024 --output-dir outputs/sar_rtdetr/eval
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dinov3.data.SSDD.ssdd_dataset import SSDDDataset, collate_fn
from dinov3.data.SSDD.transforms import make_val_transforms, DATASET_STATS
from dinov3.data.HRSID.hrsid_dataset import HRSIDDataset
from dinov3.sar_detection.rtdetr import RTDETRStudent, PostProcess, box_cxcywh_to_xyxy


# ---------------------------------------------------------------------------
# mAP computation (COCO-style, IoU 0.50:0.95) — same logic as train.py
# ---------------------------------------------------------------------------
def _iou_xyxy(b1, b2):
    a1 = (b1[:, 2] - b1[:, 0]).clamp(min=0) * (b1[:, 3] - b1[:, 1]).clamp(min=0)
    a2 = (b2[:, 2] - b2[:, 0]).clamp(min=0) * (b2[:, 3] - b2[:, 1]).clamp(min=0)
    lt = torch.max(b1[:, None, :2], b2[None, :, :2])
    rb = torch.min(b1[:, None, 2:], b2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (a1[:, None] + a2[None, :] - inter + 1e-6)


def _compute_map_coco(gt_by_img, det_list, num_classes, device, max_dets=300):
    """Standard pycocotools COCO evaluator (AP@.5:.95, AP50)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import itertools

    images = []
    annotations = []
    ann_id = 0
    for iid, gt in gt_by_img.items():
        boxes = gt["boxes"].cpu()
        labels = gt["labels"].cpu()
        H = W = 896  # default; area-based AP not used for primary endpoints
        images.append({"id": int(iid), "width": W, "height": H})
        for b in range(boxes.shape[0]):
            cx, cy, w, h = boxes[b].tolist()
            cat = int(labels[b].item())
            annotations.append({
                "id": ann_id, "image_id": int(iid),
                "category_id": cat + 1,
                "bbox": [cx - w / 2, cy - h / 2, w, h],
                "area": float(w * h), "iscrowd": 0,
            })
            ann_id += 1

    categories = [{"id": c + 1, "name": str(c)} for c in range(num_classes)]
    coco_gt = COCO()
    coco_gt.dataset = {"images": images, "annotations": annotations, "categories": categories}
    coco_gt.createIndex()

    coco_dt = []
    for iid, score, box, label in det_list:
        x1, y1, x2, y2 = box.tolist()
        coco_dt.append({
            "image_id": int(iid),
            "category_id": int(label) + 1,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(score),
        })

    if not coco_dt:
        return {"mAP": 0.0, "AP50": 0.0}

    coco_eval = COCOeval(coco_gt, coco_gt.loadRes(coco_dt), "bbox")
    coco_eval.params.maxDets = [max_dets]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {"mAP": float(coco_eval.stats[0]), "AP50": float(coco_eval.stats[1])}


def _compute_map_pr_area(gt_by_img, det_list, num_classes, device, max_dets=300):
    """Area-under-PR fallback when pycocotools is unavailable."""
    iou_ts = torch.linspace(0.5, 0.95, 10, device=device)
    gt_by_class = {c: {} for c in range(num_classes)}
    det_by_class = {c: [] for c in range(num_classes)}
    gt_count = {c: 0 for c in range(num_classes)}

    for iid, gt in gt_by_img.items():
        gboxes = gt["boxes"]
        glabels = gt["labels"]
        for c in range(num_classes):
            sel = (glabels == c).nonzero(as_tuple=True)[0]
            gt_by_class[c].setdefault(iid, torch.empty((0, 4), device=device))
            if sel.numel():
                gt_by_class[c][iid] = torch.cat(
                    [gt_by_class[c][iid], gboxes[sel]], 0
                )
            gt_count[c] += sel.numel()

    for iid, score, box, label in det_list:
        det_by_class[label].append((iid, float(score), box))

    ap_per_class = []
    for c in range(num_classes):
        n_gt = gt_count[c]
        if n_gt == 0:
            continue
        dets = sorted(det_by_class[c], key=lambda x: -x[1])
        gt_xy = {
            iid: box_cxcywh_to_xyxy(b.to(device))
            for iid, b in gt_by_class[c].items()
        }
        aps_iou = []
        for t in iou_ts:
            used = {
                iid: torch.zeros(b.shape[0], dtype=torch.bool, device=device)
                for iid, b in gt_xy.items()
            }
            tp = torch.zeros(len(dets), device=device)
            fp = torch.zeros(len(dets), device=device)
            for i, (iid, _score, box) in enumerate(dets):
                gts = gt_xy.get(iid)
                if gts is None or gts.shape[0] == 0:
                    fp[i] = 1
                    continue
                iou = _iou_xyxy(box.unsqueeze(0).to(device), gts)[0]
                best_iou, best_j = torch.max(iou, 0)
                if best_iou >= t and not used[iid][best_j]:
                    used[iid][best_j] = True
                    tp[i] = 1
                else:
                    fp[i] = 1
            tp_cum = torch.cumsum(tp, 0)
            fp_cum = torch.cumsum(fp, 0)
            rc = torch.cat(
                [torch.zeros(1, device=device), tp_cum / (n_gt + 1e-6),
                 torch.ones(1, device=device)]
            )
            pr = torch.cat(
                [torch.zeros(1, device=device),
                 tp_cum / (tp_cum + fp_cum + 1e-6),
                 torch.zeros(1, device=device)]
            )
            for i in range(len(pr) - 1, 0, -1):
                pr[i - 1] = torch.max(pr[i - 1], pr[i])
            idx = torch.nonzero(rc[1:] != rc[:-1], as_tuple=True)[0]
            ap_t = (
                torch.sum((rc[1:][idx] - rc[:-1][idx]) * pr[1:][idx])
                if idx.numel()
                else torch.zeros((), device=device)
            )
            aps_iou.append(float(ap_t))
        ap_per_class.append(aps_iou)

    if not ap_per_class:
        return {"mAP": 0.0, "AP50": 0.0}
    mAP = sum(sum(r) for r in ap_per_class) / (len(ap_per_class) * 10)
    ap50 = sum(r[0] for r in ap_per_class) / len(ap_per_class)
    return {"mAP": mAP, "AP50": ap50}


def compute_map(gt_by_img, det_list, num_classes, device, max_dets=300):
    """COCO-style mAP@[.5:.95] and AP50.

    Uses the standard pycocotools COCO evaluator when available; falls back
    to the area-under-PR approximation otherwise.
    """
    try:
        return _compute_map_coco(gt_by_img, det_list, num_classes, device, max_dets)
    except Exception:
        return _compute_map_pr_area(gt_by_img, det_list, num_classes, device, max_dets)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def load_student(checkpoint_path, device, num_classes=1, num_queries=300,
                 r_lora=16):
    """Load a deployment or training checkpoint into an RTDETRStudent.

    ``deploy_student.pth`` stores only the merged student weights (the LoRA
    branches have been folded into the frozen AIFI linears by
    ``Distiller.reparameterize``).  ``best.pth`` / ``checkpoint_epoch_N.pth``
    store the full distiller state dict under the ``model`` key; the student
    sub-tree is extracted with a prefix strip.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    student = RTDETRStudent(
        num_classes=num_classes,
        num_queries=num_queries,
        r_lora=r_lora,
        backbone_pretrained=False,
        freeze_backbone=False,
    ).to(device)
    student.eval()

    if "student" in ckpt:
        state = ckpt["student"]
    elif "model" in ckpt:
        state = ckpt["model"]
        if all(k.startswith("student.") for k in state):
            state = {k[len("student."):]: v for k, v in state.items()}
    else:
        state = ckpt

    student.load_state_dict(state, strict=False)
    return student


# ---------------------------------------------------------------------------
# Evaluation + prediction JSON + latency
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_split(student, data_loader, device, num_classes, img_size,
                   postprocess, output_dir, seed, conf_threshold=0.0):
    """Run one full pass over *data_loader*.

    Returns a dict with mAP, AP50, latency stats, and writes:
      ``predictions_<seed>.json``  — COCO-style detections
      ``latency_<seed>.json``       — per-image timing
    """
    coco_preds = []
    gt_by_img = {}
    det_list = []
    latencies_ms = []

    for images, targets in data_loader:
        images = torch.stack(images, dim=0).to(device, non_blocking=True)
        bsz = images.shape[0]
        target_sizes = torch.full((bsz, 2), float(img_size), device=device)
        t_dev = [{k: v.to(device) for k, v in t.items()} for t in targets]

        start = time.perf_counter()
        outputs = student(images)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms / bsz)

        results = postprocess(outputs, target_sizes)

        for b in range(bsz):
            iid = int(t_dev[b]["image_id"].item())
            gboxes = t_dev[b]["boxes"]
            glabels = t_dev[b]["labels"]
            gt_by_img[iid] = {"boxes": gboxes, "labels": glabels}

            r = results[b]
            scores = r["scores"]
            labels = r["labels"]
            boxes_xyxy = r["boxes"]
            for k in range(len(scores)):
                sc = float(scores[k])
                if sc < conf_threshold:
                    continue
                lbl = int(labels[k])
                x1, y1, x2, y2 = boxes_xyxy[k].tolist()
                coco_preds.append({
                    "image_id": iid,
                    "category_id": lbl + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": sc,
                })
                det_list.append((iid, sc, boxes_xyxy[k].clone(), lbl))

    metrics = compute_map(gt_by_img, det_list, num_classes, device)

    latencies_tensor = torch.tensor(latencies_ms)
    latency_stats = {
        "mean_ms": float(latencies_tensor.mean()),
        "std_ms": float(latencies_tensor.std()) if len(latencies_ms) > 1 else 0.0,
        "median_ms": float(latencies_tensor.median()),
        "p95_ms": float(latencies_tensor.quantile(0.95)),
        "fps": float(1000.0 / latencies_tensor.mean()),
        "num_images": len(latencies_ms) * data_loader.batch_size,
        "img_size": img_size,
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"predictions_{seed}.json", "w") as f:
        json.dump(coco_preds, f, indent=2)
    with open(out / f"latency_{seed}.json", "w") as f:
        json.dump(latency_stats, f, indent=2)

    metrics.update(latency_stats)
    with open(out / f"metrics_{seed}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ---------------------------------------------------------------------------
# Seed aggregation
# ---------------------------------------------------------------------------
def aggregate_seeds(per_seed_metrics, output_dir):
    if len(per_seed_metrics) <= 1:
        return
    keys = ["mAP", "AP50", "mean_ms", "fps", "median_ms", "p95_ms"]
    summary = {}
    for k in keys:
        vals = [m[k] for m in per_seed_metrics if k in m]
        if not vals:
            continue
        summary[k] = {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "values": vals,
        }
    with open(Path(output_dir) / "seed_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== SAR-RTDETR evaluation ===")
    print(f"Device: {device}, Dataset: {args.dataset}, Split: {args.split}")

    norm_mean, norm_std = DATASET_STATS[args.dataset]
    val_tf = make_val_transforms(img_size=args.img_size, mean=norm_mean, std=norm_std)
    DatasetClass = SSDDDataset if args.dataset == "ssdd" else HRSIDDataset
    dataset = DatasetClass(
        root_dir=args.data_root, split=args.split, transform=val_tf
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    print(f"Eval samples: {len(dataset)}")

    postprocess = PostProcess()
    seeds = [int(s) for s in args.seeds.split(",")]
    per_seed = []

    for seed in seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"\n--- Seed {seed} ---")

        student = load_student(
            args.checkpoint, device,
            num_classes=args.num_classes,
            num_queries=args.num_queries,
            r_lora=args.r_lora,
        )

        if args.warmup > 0:
            print(f"Warming up ({args.warmup} batches)...")
            wi = 0
            for images, _ in loader:
                images = torch.stack(images, dim=0).to(device)
                _ = student(images)
                wi += 1
                if wi >= args.warmup:
                    break

        metrics = evaluate_split(
            student, loader, device, args.num_classes, args.img_size,
            postprocess, args.output_dir, seed, args.conf_threshold,
        )
        per_seed.append(metrics)
        print(f"  mAP={metrics['mAP']:.4f}  AP50={metrics['AP50']:.4f}  "
              f"latency={metrics['mean_ms']:.2f}ms  FPS={metrics['fps']:.1f}")

    if len(per_seed) > 1:
        summary = aggregate_seeds(per_seed, args.output_dir)
        print(f"\n=== Seed summary (mean ± std, n={len(per_seed)}) ===")
        for k in ("mAP", "AP50", "mean_ms", "fps"):
            if k in summary:
                s = summary[k]
                print(f"  {k}: {s['mean']:.4f} ± {s['std']:.4f}")

    print(f"\nArtifacts written to {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SAR-RTDETR evaluation")
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to deploy_student.pth or best.pth")
    parser.add_argument("--dataset", default="ssdd", choices=["ssdd", "hrsid"])
    parser.add_argument("--data-root", default=None, type=str)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default="./outputs/sar_rtdetr/eval", type=str)
    parser.add_argument("--img-size", default=896, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--num-queries", default=300, type=int)
    parser.add_argument("--r-lora", default=16, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--seeds", default="42", type=str,
                        help="Comma-separated seeds (e.g. 42,123,2024)")
    parser.add_argument("--warmup", default=3, type=int,
                        help="Warm-up batches before timing latency")
    parser.add_argument("--conf-threshold", default=0.0, type=float,
                        help="Minimum score to include in predictions.json")
    args = parser.parse_args()
    if args.data_root is None:
        args.data_root = f"./dinov3/data/{args.dataset.upper()}"
    main(args)
