# EGMF-NET

The final gated configuration for document image enhancement and binarization.
This release contains one architecture and separate training/testing programs.

[中文说明](README.zh-CN.md)

## Architecture

- RGB global page branch and RGB local patch branch.
- MSCAN encoders, GroupNorm, and EMCAD-Lite decoders.
- Enhancement decoder features guide the binary decoders through addition.
- Feature gates fuse global/local binary logits and RGB enhancements.
- Configurable cascade stages with shared or independent parameters. The enhanced result feeds the next stage.
- Bilinear interpolation with a 1×1 projection performs decoder upsampling.

The default two-stage shared network has **12,758,948 trainable parameters**.

Cascade settings remain available in `conf.yml`:

```yaml
model:
  dual_cascade_stages: 2
  independent_cascade_params: false
```

`false` shares parameters across stages; `true` gives each stage its own parameters.
The stage count must be a positive integer. Independent stages increase model size and memory use.

## Installation

Use Python 3.10 or later and install a PyTorch/torchvision build appropriate for
your CPU or CUDA environment, followed by the remaining requirements:

```bash
pip install -r requirements.txt
```

Development checks were run with Python 3.12, PyTorch 2.5.1 and CPU synthetic inputs.

## Data preparation

```text
data/
  train/
    img/            # original RGB pages
    gt/             # binary GT: black foreground, white background
    enhancement/    # RGB enhancement targets
  test/
    img/
    gt/             # optional for prediction-only testing
```

An image and its target files must have identical filename stems and dimensions.
PNG, JPEG, BMP and TIFF are supported. Duplicate stems or missing targets raise
an error rather than silently removing documents.

The loader reads **every document in the configured training directory**. It
does not reserve a validation subset or automatically exclude a dataset year.
Prepare training and test directories separately; for a held-out DIBCO year,
put that year's pages only in the test directories.

Set `data.train_enhance_gt_dir: ""` to use the input RGB as the enhancement target.
This is a reconstruction objective, not a replacement for clean enhancement GT.
Set `data.test_gt_dir: ""` to predict without computing test metrics.

## Training

Edit the paths in `conf.yml`, retain `mode: train`, then run:

```bash
python main.py
```

The equivalent standalone entry point is `python train.py`. Relative paths are
resolved against the project directory, independently of the working directory.
An alternate configuration can optionally be selected with `--config` on `main.py`.

Training uses all supplied documents, a 512×512 global image and 256×256 local
patches by default. There is **no validation loader, validation metric or best
validation checkpoint**.

The final optimizer is AdamW (betas 0.9/0.999), with cosine learning rate from
`train.lr` toward `train.lr_end`. The default run lasts 60 epochs with batch
size 4. `train.augment` defaults to false, matching the selected configuration.
When enabled, the same page transformation is used for global and local views.

Per-stage loss is:

```text
0.5 × binary(fused) + 0.3 × binary(global) + 0.3 × binary(local)
+ 0.3 × L1(enhanced_fused) + 0.1 × L1(enhanced_global) + 0.1 × L1(enhanced_local)
```

`binary = BCEWithLogits + Dice`; two cascade stages are weighted 0.3/0.7.
Other stage counts use equal stage weights. Loss coefficients are fixed in the model code.

Outputs in `train.save_dir`:

| File | Contents |
|---|---|
| `model_last.pth` | latest complete training checkpoint |
| `model_epochNNN.pth` | periodic checkpoint and final-epoch checkpoint |
| `loss_history.csv`, `history.json` | training losses and learning rate |
| `loss_curve.png` | training loss curve |
| `config.json`, `training_documents.json` | effective settings and document list |
| `log.txt` | epoch summaries |

Resume by setting `train.resume_path` to a release checkpoint. `max_epochs` is
the total target epoch count, not the number of additional epochs. Optimizer
moments and random states are restored; the current configuration determines
the cosine schedule and weight decay. Changing the total epoch count changes
the future cosine schedule. Changing the cascade count or sharing during resume is rejected. Exact resumption was tested on CPU with zero workers.

## Testing

Set `mode: test`, `test.checkpoint_path` and the test data paths in `conf.yml`:

```bash
python main.py
# or:
python test.py
```

Testing processes native-size pages with a resized global view and overlapping
local tiles. The stage count and parameter sharing are restored from checkpoint metadata.
For a raw state dictionary without metadata, the current YAML cascade settings are used.
`test.binary_threshold` controls the final foreground decision.

Outputs are saved under `test.save_dir/binary`, `probability` and `enhanced`.
If GT is supplied, `metrics.csv` reports per-document FM, pFM, PSNR and DRD;
`metrics.json` contains their document means. Testing is independent of training
and never selects or updates training checkpoints.

Original final **gated/shared/RGB/interpolate** state dictionaries can be loaded
for inference. Four tensors from the original unused conv-fusion heads are
discarded; every other parameter must match strictly. Original full training
checkpoints cannot be resumed because the unused parameters have been removed.

## Project layout

```text
main.py                       # default conf.yml entry point
conf.yml                      # training/testing configuration
train.py                      # training only
test.py                       # independent native-page testing
model/egmf_net.py              # final gated network and losses
model/blocks.py                # MSCAN / EMCAD-Lite components
datasets/documentloader.py     # pairing, global resizing, local crops
evaluation/document_metrics.py # test metrics only
utils/config.py               # strict configuration and paths
utils/checkpoint.py           # random states and weight loading
tests/test_release.py          # regression tests
```

## Verification

```bash
python -B -m unittest discover -s tests -v
```

Tests cover complete training data usage, strict pairing, rejected obsolete
configuration, gradient coverage, training/resume equivalence and separate
testing. During release preparation, both cascade-stage outputs, loss values and
native-page tiled inference were compared with the original selected gated code.
They matched with the same input tensors and weights. These synthetic checks do
not constitute a new dataset accuracy evaluation.
