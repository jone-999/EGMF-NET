"""Run full-page inference with a trained EGMF-NET model."""
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

from datasets.documentloader import image_files, axis_starts, normalize_image
from model import EGMFNet
from utils.checkpoint import load_weights, cascade_config
from utils.config import load_config, validate_config


def blend_window(height, width, device):
    def axis(length):
        if length == 1:
            return torch.ones(1, device=device)
        values = torch.linspace(0, 1, length, device=device)
        return (1 - (2 * values - 1).abs()).clamp_min(0.05)
    return axis(height).view(1, 1, height, 1) * axis(width).view(1, 1, 1, width)


@torch.no_grad()
def predict_page(model, image, full_size=512, tile_size=256, overlap=128, use_amp=False):
    """Restore global features and blend local tiles before gated fusion."""
    model.eval()
    device = image.device
    height, width = image.shape[-2:]
    current = image
    result = None
    for stage in range(model.stages):
        stage_model = model.get_stage_model(stage)
        resized = F.interpolate(current, size=(full_size, full_size), mode="bilinear", align_corners=False)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            global_output = stage_model.full_branch(resized)
        global_output = {key: F.interpolate(value.float(), size=(height, width), mode="bilinear", align_corners=False)
                         for key, value in global_output.items()}
        channels = {"bin_logits": 1, "enhanced_pred": 3, "pred_residual": 3,
                    "decoder_feature": stage_model.bin_gate_head.in_channels // 2}
        local_output = {key: torch.zeros((1, count, height, width), device=device) for key, count in channels.items()}
        counts = torch.zeros((1, 1, height, width), device=device)
        for top in axis_starts(height, tile_size, tile_size - overlap):
            for left in axis_starts(width, tile_size, tile_size - overlap):
                patch = current[:, :, top:top + tile_size, left:left + tile_size]
                ph, pw = patch.shape[-2:]
                patch = F.pad(patch, (0, tile_size - pw, 0, tile_size - ph), mode="replicate")
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    output = stage_model.patch_branch(patch)
                window = blend_window(ph, pw, device)
                for key in local_output:
                    local_output[key][:, :, top:top + ph, left:left + pw] += output[key][:, :, :ph, :pw].float() * window
                counts[:, :, top:top + ph, left:left + pw] += window
        if torch.any(counts == 0):
            raise RuntimeError("Local tiling left uncovered pixels")
        local_output = {key: value / counts for key, value in local_output.items()}
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, enhanced = stage_model.fuse(global_output, local_output)
        result = {"probability": logits.float().sigmoid(), "enhanced": enhanced.float()}
        if stage + 1 < model.stages:
            current = enhanced.detach() * 2 - 1
    return result


def run_test(config):
    config = validate_config(config)
    cfg, runtime, data = (config[name] for name in ("test", "runtime", "data"))
    requested = runtime["device"]
    device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested)
    checkpoint = torch.load(cfg["checkpoint_path"], map_location="cpu", weights_only=False)
    architecture = cascade_config(checkpoint, config["model"])
    model = EGMFNet(stages=architecture["dual_cascade_stages"],
                    independent_cascade_params=architecture["independent_cascade_params"]).to(device)
    load_weights(model, checkpoint)
    model.eval()
    images = image_files(data["test_img_dir"])
    output = Path(cfg["save_dir"])
    if output.resolve() == Path(data["test_img_dir"]).resolve():
        raise ValueError("test.save_dir must differ from the input directory")
    for subdir in ("binary", "probability", "enhanced"):
        (output / subdir).mkdir(parents=True, exist_ok=True)
    stored = checkpoint.get("config", {})
    full_size = stored.get("train", {}).get("full_image_size", stored.get("full_image_size", config["train"]["full_image_size"]))
    for path in images:
        with Image.open(path) as handle:
            image = normalize_image(handle.convert("RGB")).unsqueeze(0).to(device)
        prediction = predict_page(model, image, full_size, cfg["tile_size"], cfg["tile_overlap"], runtime["use_amp"] and device.type == "cuda")
        probability = prediction["probability"][0, 0].cpu().numpy()
        mask = (probability >= cfg["binary_threshold"]).astype(np.uint8)
        Image.fromarray((1 - mask) * 255).save(output / "binary" / f"{path.stem}.png")
        Image.fromarray(((1 - probability) * 255).clip(0, 255).astype(np.uint8)).save(output / "probability" / f"{path.stem}.png")
        TF.to_pil_image(prediction["enhanced"][0].clamp(0, 1).cpu()).save(output / "enhanced" / f"{path.stem}.png")
        print(output / "binary" / f"{path.stem}.png")


if __name__ == "__main__":
    config = load_config()
    config["mode"] = "test"
    run_test(config)
