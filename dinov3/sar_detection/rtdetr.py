# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with the
# terms of the DINOv3 License Agreement.

"""RT-DETR-R18 student detector for the SAR-RTDETR distillation framework.

Contains:
  * ResNet-18 backbone producing {S3, S4, S5} (strides 8 / 16 / 32).
  * Hybrid encoder: AIFI (single-scale transformer on S5, attention output
    projection is an A^2TD-LoRA module) + CCFF (top-down multi-scale fusion).
  * DETR-style decoder producing {pred_logits, pred_boxes}.
  * HungarianMatcher + SetCriterion (focal classification + L1 + GIoU),
    providing L_cls and L_box used by the paper's total loss.

The backbone is frozen by default (only AIFI-LoRA + CCFF + decoder heads are
trained), matching the paper's "frozen backbone + trainable LoRA / heads".
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm

from dinov3.sar_detection.atd_lora import ATDLoRALinear, VarianceGate


# ---------------------------------------------------------------------------
# Box utilities (cxcywh <-> xyxy, IoU, GIoU)
# ---------------------------------------------------------------------------
def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def box_iou(box1, box2):
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    lt = torch.max(box1[..., :2], box2[..., :2])
    rb = torch.min(box1[..., 2:], box2[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1 + area2 - inter
    return inter / (union + 1e-6), union


def generalized_box_iou(boxes1, boxes2):
    assert (boxes1[:, -2:] >= boxes1[:, :2]).all(), "xyxy boxes must satisfy x2>=x1, y2>=y1"
    assert (boxes2[:, -2:] >= boxes2[:, :2]).all(), "xyxy boxes must satisfy x2>=x1, y2>=y1"
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[..., 0] * wh[..., 1]
    return iou - (area - union) / (area + 1e-6)


# ---------------------------------------------------------------------------
# ResNet-18 backbone -> {S3, S4, S5}
# ---------------------------------------------------------------------------
class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()
        try:
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = tvm.resnet18(weights=weights)
        except Exception:
            net = tvm.resnet18(pretrained=pretrained)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1  # 64,  stride 4
        self.layer2 = net.layer2  # 128, stride 8  -> S3
        self.layer3 = net.layer3  # 256, stride 16 -> S4
        self.layer4 = net.layer4  # 512, stride 32 -> S5
        self.dims = (128, 256, 512)
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)
            self.eval()

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        s3 = self.layer2(x)
        s4 = self.layer3(s3)
        s5 = self.layer4(s4)
        return s3, s4, s5


# ---------------------------------------------------------------------------
# AIFI encoder (attention output projection uses A^2TD-LoRA)
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    """Multi-head self-attention with separable output projection."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.in_proj = nn.Linear(d_model, 3 * d_model)
        self.dropout = dropout

    def forward(self, q, k, v):
        B, N, C = q.shape
        qkv = self.in_proj(q)  # project then chunk (standard MHA)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(B, N, self.nhead, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.nhead, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.nhead, self.head_dim).transpose(1, 2)
        scale = self.head_dim ** -0.5
        attn = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return out


class AIFIIEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, gate: VarianceGate, r: int, dropout=0.0):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, dropout)
        self.attn_out_proj = ATDLoRALinear(nn.Linear(d_model, d_model), r, r, gate)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src, pos, detach_distill=False, detach_det=False):
        q = k = v = src + pos
        sa = self.self_attn(q, k, v)
        sa = self.attn_out_proj(sa, detach_distill=detach_distill, detach_det=detach_det)
        src = self.norm1(src + self.dropout(sa))
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout(ff))
        return src


class AIFI(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, gate: VarianceGate, r: int, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [AIFIIEncoderLayer(d_model, nhead, dim_feedforward, gate, r, dropout) for _ in range(num_layers)]
        )

    def forward(self, src, pos, detach_distill=False, detach_det=False):
        for layer in self.layers:
            src = layer(src, pos, detach_distill=detach_distill, detach_det=detach_det)
        return src


# ---------------------------------------------------------------------------
# CCFF: top-down multi-scale fusion with RepConv blocks
# ---------------------------------------------------------------------------
class RepConv(nn.Module):
    def __init__(self, dim, dilation=2**2 - 1):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv1(x) + self.conv2(x)))


class CCFF(nn.Module):
    def __init__(self, dims, d_model):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(d, d_model, 1, bias=False) for d in dims])
        self.output = nn.ModuleList([RepConv(d_model) for _ in dims])

    def forward(self, feats):
        laterals = [l(f) for l, f in zip(self.lateral, feats)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="bilinear", align_corners=False
            )
        return [m(l) for m, l in zip(self.output, laterals)]


# ---------------------------------------------------------------------------
# 2D sine positional encoding + level embedding
# ---------------------------------------------------------------------------
def get_2d_sine_pos(h, w, dim, device):
    ys = torch.arange(h, device=device, dtype=torch.float32).unsqueeze(1)
    xs = torch.arange(w, device=device, dtype=torch.float32).unsqueeze(1)
    half = dim // 4
    div = torch.exp(torch.arange(0, half, device=device, dtype=torch.float32) * (-math.log(10000) / max(half - 1, 1)))
    # Interleave sin and cos (standard 2D sincos PE) instead of duplicating sin.
    pe_y = torch.stack([torch.sin(ys * div), torch.cos(ys * div)], dim=-1).reshape(h, 2 * half)
    pe_x = torch.stack([torch.sin(xs * div), torch.cos(xs * div)], dim=-1).reshape(w, 2 * half)
    pos = torch.cat([
        pe_y.unsqueeze(1).expand(h, w, 2 * half),
        pe_x.unsqueeze(0).expand(h, w, 2 * half),
    ], dim=-1)
    return pos.reshape(h * w, dim)


# ---------------------------------------------------------------------------
# DETR-style decoder
# ---------------------------------------------------------------------------
class DETRDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, tgt, query_pos, memory, pos, q_mask=None, k_mask=None):
        q = k = tgt + query_pos
        sa, _ = self.self_attn(q, k, value=tgt, key_padding_mask=q_mask, need_weights=False)
        tgt = self.norm1(tgt + self.dropout(sa))
        ca, _ = self.cross_attn(tgt + query_pos, memory + pos, value=memory, key_padding_mask=k_mask, need_weights=False)
        tgt = self.norm2(tgt + self.dropout(ca))
        ff = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout(ff))
        return tgt


class RTDETRHead(nn.Module):
    def __init__(self, num_queries, num_classes, d_model, nhead, dim_feedforward, num_layers, dropout=0.0):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model
        self.num_classes = num_classes
        # RT-DETR query selection: top-k encoder-memory positions scored by
        # a class-aware linear, with learned positional embedding.
        self.query_selection = nn.Linear(d_model, num_classes)
        self.query_pos_embed = nn.Embedding(num_queries, d_model)
        self.decoder = nn.ModuleList(
            [DETRDecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)

    def forward(self, memory, pos, mem_mask=None):
        bs = memory.shape[0]
        # Query selection: score memory, pick top-k as initial content queries.
        cls_score = self.query_selection(memory)  # [B, N, num_classes]
        k = min(self.num_queries, cls_score.shape[1])
        topk_score, topk_idx = torch.topk(cls_score.flatten(1), k, dim=1)
        mem_idx = topk_idx // self.num_classes  # [B, k]
        gather_idx = mem_idx.unsqueeze(-1).expand(-1, -1, self.d_model)
        tgt = torch.gather(memory, 1, gather_idx)  # [B, k, C]
        query_pos = self.query_pos_embed.weight[:k].unsqueeze(0).expand(bs, -1, -1)
        for layer in self.decoder:
            tgt = layer(tgt, query_pos, memory, pos, q_mask=None, k_mask=mem_mask)
        logits = self.class_embed(tgt)
        boxes = torch.sigmoid(self.bbox_embed(tgt))
        return {"pred_logits": logits, "pred_boxes": boxes}


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers):
        super().__init__()
        h = [hidden] * (num_layers - 1)
        self.layers = nn.Sequential(*[m for a, b in zip([in_dim] + h, h + [out_dim]) for m in [nn.Linear(a, b)]])
        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layers(x)


# ---------------------------------------------------------------------------
# Student
# ---------------------------------------------------------------------------
class RTDETRStudent(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        num_queries: int = 300,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        enc_layers: int = 1,
        dec_layers: int = 3,
        r_lora: int = 100,
        backbone_pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.backbone = ResNet18Backbone(pretrained=backbone_pretrained, freeze=freeze_backbone)
        self.input_proj = nn.ModuleList([nn.Conv2d(c, d_model, 1, bias=False) for c in self.backbone.dims])
        self.level_embed = nn.Parameter(torch.zeros(3, d_model))
        self.gate = VarianceGate()
        self.aifi = AIFI(d_model, nhead, dim_feedforward, enc_layers, self.gate, r_lora)
        self.ccff = CCFF((self.backbone.dims[0], self.backbone.dims[1], d_model), d_model)
        self.head = RTDETRHead(num_queries, num_classes, d_model, nhead, dim_feedforward, dec_layers)

    def atd_modules(self):
        mods = []
        for layer in self.aifi.layers:
            mods.append(layer.attn_out_proj)
        return mods

    def forward(self, samples, detach_distill=False, detach_det=False, backbone_features=None):
        if backbone_features is not None:
            s3, s4, s5 = backbone_features
        else:
            s3, s4, s5 = self.backbone(samples)
        # AIFI on the projected deepest level.
        p5 = self.input_proj[2](s5)
        b, _, h5, w5 = p5.shape
        # batch_first=True: need [B, HW, C]
        pos5 = get_2d_sine_pos(h5, w5, self.d_model, p5.device)  # [HW, C]
        pos5 = pos5.unsqueeze(0).expand(b, -1, -1)  # [B, HW, C]
        z = p5.flatten(2).permute(0, 2, 1)  # AIFI input activation z = Flatten(S5)
        f5_seq = self.aifi(z, pos5, detach_distill=detach_distill, detach_det=detach_det)
        f5 = f5_seq.permute(0, 2, 1).reshape(b, self.d_model, h5, w5)  # [B, C, H, W]
        # CCFF fuses {S3, S4, F5}.
        feats = self.ccff([s3, s4, f5])
        # Flatten multi-scale memory with pos + level embeddings.
        tokens, pos = [], []
        masks = []
        for lvl, f in enumerate(feats):
            _, _, h, w = f.shape
            t = f.flatten(2).permute(0, 2, 1)  # [B, HW, C]
            p = get_2d_sine_pos(h, w, self.d_model, f.device)  # [HW, C]
            p = p.unsqueeze(0).expand(b, -1, -1)
            t = t + self.level_embed[lvl]
            tokens.append(t)
            pos.append(p)
            masks.append(torch.zeros(b, h * w, device=f.device, dtype=torch.bool))
        memory = torch.cat(tokens, dim=1)
        pos = torch.cat(pos, dim=1)
        mem_mask = torch.cat(masks, dim=1)
        out = self.head(memory, pos, mem_mask)
        out["s3"], out["s4"], out["s5"], out["f5"] = s3, s4, s5, f5
        out["z"] = z  # AIFI input activation for gradient probes (Eq. activation_probes)
        return out


# ---------------------------------------------------------------------------
# Hungarian matcher + SetCriterion (L_cls + L_box)
# ---------------------------------------------------------------------------
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1.0, cost_bbox=1.0, cost_giou=1.0, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.alpha = focal_alpha
        self.gamma = focal_gamma

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, nq = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        tgt_cls = torch.cat([t["labels"] for t in targets])
        tgt_bbox = torch.cat([t["boxes"] for t in targets])
        # focal-style classification cost (negation of the target probability)
        neg_cost = self.alpha * out_prob ** self.gamma * -(out_prob + 1e-8).log()
        pos_cost = (1 - self.alpha) * (1 - out_prob) ** self.gamma * -(1 - out_prob + 1e-8).log()
        cost_clf = pos_cost[:, tgt_cls] - neg_cost[:, tgt_cls]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_clf + self.cost_giou * cost_giou
        C = C.view(bs, nq, -1).cpu()
        sizes = [t["boxes"].shape[0] for t in targets]
        indices = []
        for i, (c, s) in enumerate(zip(C.split(sizes, -1), sizes)):
            if s > 0:
                row, col = _linear_sum_assignment(c[i])
            else:
                row, col = (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
            indices.append((torch.as_tensor(row, dtype=torch.long), torch.as_tensor(col, dtype=torch.long)))
        return [(i.to(out_bbox.device), j.to(out_bbox.device)) for (i, j) in indices]


def _linear_sum_assignment(cost):
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost)
        return r, c
    except Exception:
        b, n = cost.shape  # b queries, n targets
        # greedy fallback
        used = set()
        rows, cols = [], []
        order = torch.argsort(cost.min(dim=1).values)
        for r in order.tolist():
            row_cost = cost[r]
            for c in torch.argsort(row_cost).tolist():
                if c not in used:
                    used.add(c); rows.append(r); cols.append(c); break
        return torch.as_tensor(rows), torch.as_tensor(cols)


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
    """Canonical DETR sigmoid focal loss (handles the no-object column implicitly)."""
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha=0.25, focal_gamma=2.0, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        self.eos_coef = eos_coef

    def loss_labels(self, outputs, targets, indices):
        """Sigmoid focal classification loss (L_cls)."""
        src_logits = outputs["pred_logits"]  # [B, NQ, num_classes+1]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        # One-hot of size num_classes+2 (so the no-object index is settable), then drop
        # the last column -> matches src_logits shape [B, NQ, num_classes+1].
        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, self.alpha, self.gamma) * src_logits.shape[1]
        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses = {"loss_bbox": loss_bbox.sum() / num_boxes}
        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(i, k) for k, (i, _) in enumerate(indices)])
        src_idx = torch.cat([i for i, _ in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        num_boxes = sum(t["boxes"].shape[0] for t in targets)
        num_boxes = max(num_boxes, 1)
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        return losses


class PostProcess(nn.Module):
    def forward(self, outputs, target_sizes):
        out_logits, out_boxes = outputs["pred_logits"], outputs["pred_boxes"]
        prob = out_logits.sigmoid()
        num_total = prob.shape[1] * prob.shape[2]
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), min(100, num_total), dim=1)
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = out_boxes.gather(1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        # Convert cxcywh -> xyxy *before* scaling to pixel space.
        boxes = box_cxcywh_to_xyxy(boxes)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
        boxes = boxes * scale_fct[:, None, :]
        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(topk_values, labels, boxes)]
        return results


def build_student(num_classes=1, num_queries=300, r_lora=100, backbone_pretrained=True, freeze_backbone=True):
    student = RTDETRStudent(
        num_classes=num_classes,
        num_queries=num_queries,
        r_lora=r_lora,
        backbone_pretrained=backbone_pretrained,
        freeze_backbone=freeze_backbone,
    )
    matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
    weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    criterion = SetCriterion(num_classes, matcher, weight_dict)
    postprocessors = {"bbox": PostProcess()}
    return student, criterion, postprocessors
