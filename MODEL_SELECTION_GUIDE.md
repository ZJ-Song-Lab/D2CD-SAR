# DINOv3 模型选择指南

本项目支持两种 DINOv3 卫星预训练模型用于 SAR 舰船检测：

## 模型对比

| 特性 | ViT-L/16 | ViT-7B/16 |
|------|----------|-----------|
| **参数量** | 300M | 6.7B |
| **架构** | embed_dim=1024, depth=24, heads=16 | embed_dim=4096, depth=40, heads=32 |
| **预训练数据** | SAT-493M (卫星图像) | SAT-493M (卫星图像) |
| **窗口大小** | 2×2 | 3×3 |
| **检测查询数** | 300 | 1500 |
| **显存需求** | ~16GB/GPU | ~48GB/GPU |
| **推荐Batch Size** | 4/GPU | 2/GPU |
| **训练速度** | 快 | 慢 |
| **检测精度** | 良好 | 更高 |

## 选择建议

### 选择 ViT-L/16 如果：
- GPU 显存有限（16GB以下）
- 需要快速训练和迭代
- 对精度要求不是极高
- 只有少量训练数据

### 选择 ViT-7B/16 如果：
- 有充足的 GPU 显存（48GB以上）
- 追求最高检测精度
- 有大量训练数据
- 可以接受更长的训练时间

## 使用方法

### ViT-L/16 (默认)

```bash
# 训练
bash scripts/train_sar.sh

# 推理
bash scripts/inference_sar.sh /path/to/image.jpg
```

### ViT-7B/16

```bash
# 训练
bash scripts/train_sar_vit7b.sh

# 推理
bash scripts/inference_sar_vit7b.sh /path/to/image.jpg
```

## 显存优化建议

### 对于 6×RTX 5880 (48GB) 配置：

**ViT-L/16:**
- Batch size: 4/GPU (总 batch size 24)
- 训练时间: ~2-3小时 (50 epochs)

**ViT-7B/16:**
- Batch size: 2/GPU (总 batch size 12)
- 训练时间: ~6-8小时 (50 epochs)
- 可能需要启用梯度检查点

### 内存不足解决方案：

1. **减小 Batch Size**
   ```bash
   # ViT-7B/16 使用 batch size 1
   --batch-size 1
   ```

2. **减小图像尺寸**
   ```bash
   --img-size 768
   ```

3. **启用梯度检查点** (需要修改代码)
   ```python
   # 在 train_vit7b.py 中添加
   torch.utils.checkpoint.checkpoint(...)
   ```

4. **使用 DeepSpeed ZeRO**
   ```bash
   deepspeed train_vit7b.py --deepspeed ds_config.json
   ```

## 性能预期

### ViT-L/16
- mAP@0.5: ~0.85-0.90
- 推理速度: ~5-10 FPS (单张 896×896)

### ViT-7B/16
- mAP@0.5: ~0.90-0.95
- 推理速度: ~2-5 FPS (单张 896×896)

## 预训练权重下载

两个模型都需要从 Meta AI 下载预训练权重：

1. 访问 [DINOv3 Downloads](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
2. 申请访问权限
3. 下载 SAT-493M 预训练权重：
   - `dinov3_vitl16_pretrain_sat493m.pth`
   - `dinov3_vit7b16_pretrain_sat493m.pth`

## 快速开始

### 推荐配置 (6×RTX 5880)

**方案1: 快速实验 (ViT-L/16)**
```bash
export DATA_ROOT="./dinov3/data/SSDD"
export EPOCHS=30
export BATCH_SIZE=4
bash scripts/train_sar.sh
```

**方案2: 最佳性能 (ViT-7B/16)**
```bash
export DATA_ROOT="./dinov3/data/SSDD"
export EPOCHS=50
export BATCH_SIZE=2
bash scripts/train_sar_vit7b.sh
```

## 故障排除

### Out of Memory 错误

**ViT-7B/16 常见解决方案：**

1. 检查显存是否被其他进程占用
   ```bash
   nvidia-smi
   ```

2. 减小 batch size
   ```bash
   BATCH_SIZE=1 bash scripts/train_sar_vit7b.sh
   ```

3. 使用混合精度训练 (需要修改代码)
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```

### 模型加载失败

确保预训练权重已正确下载并放置在正确位置，或者使用直接 URL：

```python
backbone = dinov3_vit7b16(
    pretrained=True,
    weights="https://path.to.weights.pth",
    check_hash=False,
)
```

## 联系与支持

如有问题，请参考：
- [DINOv3 官方文档](https://github.com/facebookresearch/dinov3)
- [SAR_DETECTION_README.md](./SAR_DETECTION_README.md)
