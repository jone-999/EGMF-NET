# EGMF-NET 最终 gated 版本

该目录是独立发布副本，原 EGMF-NET 目录保持不变。

## 运行方法

1. 安装依赖：`pip install -r requirements.txt`。
2. 在根目录 `conf.yml` 填写图像、二值 GT 和 RGB 增强 GT 路径。
3. `mode: train` 表示训练，`mode: test` 表示独立测试。
4. 直接运行 `python main.py`，也可在 IDE 直接运行主函数。

无需拼接额外参数。相对数据路径、输出路径均相对项目根目录解析。

## 最终保留的结构

RGB 输入 → 全局/局部 MSCAN 编码器 → 增强解码器与受增强特征引导的 EMCAD-Lite 二值解码器 → gated 融合。
两阶段级联共享参数；阶段一的增强结果作为阶段二输入。上采样固定为双线性插值加 1×1 卷积。

已经删除 conv、union、spatial_gate、residual_conv、token/axial bottleneck、独立级联参数、灰度增强和转置卷积等未采用选项。
固定结构不再作为 conf 中的开关出现。模型为 12,758,948 个可训练参数。

## 仅训练

训练读取 `data.train_img_dir` 下的全部文档，没有随机验证划分、留一验证、验证指标计算或验证选优。
请提前分开训练集与测试集；例如测试 DIBCO2019 时，训练目录中应只放其他年份数据。

每张训练图与 GT、增强 GT 的文件名主干、图像尺寸必须一致。
`train_enhance_gt_dir: ""` 表示用输入 RGB 本身作为增强监督，这是重建任务，不等同于干净增强 GT。

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
| `save_every` | 每隔多少轮保存一次，末轮总会保存 |
| `resume_path` | 发布版完整断点路径，空值表示从头训练 |

输出目录包含 `model_last.pth`、`model_epochNNN.pth`、训练损失 CSV/曲线、实际配置和训练文件清单。
没有 `best_fm` 等依赖验证集的权重。

## 独立测试

修改 `mode: test`，配置 `test.checkpoint_path`、`data.test_img_dir` 和 `test.save_dir`。
测试原尺寸文档，使用全局缩放、局部重叠滑窗与两阶段 gated 融合。

有测试 GT 时单独输出 FM/pFM/PSNR/DRD；`data.test_gt_dir: ""` 时只预测，不计算指标。
预测结果分为 `binary/`、`probability/`、`enhanced/`。

原最终 gated/shared/RGB/interpolate 权重可以严格加载推理，仅移除原来未参与 gated 路径的两组 conv 融合层权重。
由于移除了未使用参数，旧版完整训练断点不能直接用于发布版续训；发布版断点可正常续训。

## 验证

运行 `python -B -m unittest discover -s tests -v`。
已经验证所有可训练参数有梯度、训练数据不被划分、CPU 零 worker 续训与连续训练逐值一致，以及训练和独立测试流程。
对照原始代码，同权重下两阶段输出、损失及原尺寸滑窗推理结果一致。

完整结构和损失权重见 [英文 README](README.md)。
