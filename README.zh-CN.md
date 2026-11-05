# EGMF-NET

EGMF-NET 使用 gated 融合进行文档图像增强与二值化，提供独立的训练和测试入口。

## 运行方法

1. 安装依赖：`pip install -r requirements.txt`。
2. 在根目录 `conf.yml` 填写图像、二值 GT 和 RGB 增强 GT 路径。
3. `mode: train` 表示训练，`mode: test` 表示独立测试。
4. 直接运行 `python main.py`，也可在 IDE 直接运行主函数。

相对数据路径、输出路径均相对项目根目录解析。

## 模型结构

RGB 输入 → 全局/局部 MSCAN 编码器 → 增强解码器与受增强特征引导的 EMCAD-Lite 二值解码器 → gated 融合。
默认两阶段级联共享参数；阶段一的增强结果作为阶段二输入。上采样固定为双线性插值加 1×1 卷积。

级联数和参数共享通过 `conf.yml` 配置：

```yaml
model:
  dual_cascade_stages: 2
  independent_cascade_params: false
```

`dual_cascade_stages` 是正整数，设为 1 表示单阶段；`false` 表示各阶段共享参数，`true` 表示各阶段独立参数。
独立参数会增加参数量和内存开销。两阶段损失权重为 0.3/0.7，其他阶段数均匀加权。
推理从检查点恢复级联设置；没有元信息的裸权重使用当前 YAML 设置。续训时保持检查点中的级联数和共享方式。

默认两阶段共享参数模型为 12,758,948 个可训练参数。

## 训练

手动将数据分为训练和测试目录，并在 `conf.yml` 中配置路径。

每张训练图与 GT、增强 GT 的文件名主干、图像尺寸必须一致。
`train_enhance_gt_dir: ""` 表示用输入 RGB 本身作为监督，训练 RGB 重建。

训练参数：

| 配置 | 含义 |
|---|---|
| `batch_size` | 每次更新的局部 patch 数 |
| `max_epochs` | 总训练轮数，默认 60 |
| `patch_size` / `full_image_size` | 局部/全局尺寸，默认 256/512 |
| `patch_stride` | 训练裁块步长，默认 256 |
| `lr` / `lr_end` | AdamW 余弦学习率起始/结束值 |
| `weight_decay` | AdamW 权重衰减 |
| `augment` | 是否增强，默认关闭 |
| `save_every` | 权重保存间隔，默认每 10 轮一次，末轮也保存 |
| `resume_path` | 发布版完整断点路径，空值表示从头训练 |

输出目录包含 `model_last.pth`、`model_epochNNN.pth`、训练损失 CSV/曲线、实际配置和训练文件清单。
两种权重文件均按 `save_every` 间隔及末轮保存，`model_last.pth` 指向最近一次保存的断点。

## 独立测试

修改 `mode: test`，配置 `test.checkpoint_path`、`data.test_img_dir` 和 `test.save_dir`。
测试原尺寸文档，使用全局缩放、局部重叠滑窗与检查点记录的级联 gated 融合。

有测试 GT 时单独输出 FM/pFM/PSNR/DRD；`data.test_gt_dir: ""` 时只预测，不计算指标。
预测结果分为 `binary/`、`probability/`、`enhanced/`。

原最终 gated/shared/RGB/interpolate 权重可以严格加载推理，仅移除原来未参与 gated 路径的两组 conv 融合层权重。
由于移除了未使用参数，旧版完整训练断点不能直接用于发布版续训；发布版断点可正常续训。

## 验证

运行 `python -B -m unittest discover -s tests -v`。
测试覆盖参数梯度、训练数据加载、CPU 零 worker 续训与连续训练逐值一致，以及训练和独立测试流程。
对照原始代码，同权重下两阶段输出、损失及原尺寸滑窗推理结果一致。

完整结构和损失权重见 [英文 README](README.md)。
