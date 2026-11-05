"""Train on every supplied document. No validation split or metric selection."""
import csv
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.documentloader import DocumentDataset, paired_documents
from model import EGMFNet, compute_loss
from utils.checkpoint import FORMAT, capture_rng, restore_rng
from utils.config import load_config, validate_config


def select_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def initialize_weights(model):
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def learning_rate(config, epoch):
    return config["lr_end"] + (config["lr"] - config["lr_end"]) * (1 + math.cos(math.pi * (epoch - 1) / config["max_epochs"])) / 2


def save_history(history, output):
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    with (output / "loss_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "lr", "train_loss", "train_loss_bin", "train_loss_enhance"])
        writer.writeheader()
        writer.writerows(history)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="Training loss")
    ax.set(xlabel="Epoch", ylabel="Loss", title="EGMF-NET training")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "loss_curve.png", dpi=150)
    plt.close(fig)


def train(config):
    config = validate_config(config)
    cfg, runtime, data = (config[name] for name in ("train", "runtime", "data"))
    seed = runtime["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = select_device(runtime["device"])
    use_amp = runtime["use_amp"] and device.type == "cuda"
    pairs = paired_documents(data["train_img_dir"], data["train_gt_dir"], data["train_enhance_gt_dir"] or None)
    dataset = DocumentDataset(pairs, cfg["patch_size"], cfg["patch_stride"], cfg["full_image_size"], cfg["augment"])
    manifest = [[str(value) if value else None for value in pair] for pair in pairs]
    output = Path(cfg["save_dir"])
    if not cfg["resume_path"] and output.exists() and any(output.iterdir()):
        raise FileExistsError("Training output is not empty; use train.resume_path or a new train.save_dir")
    output.mkdir(parents=True, exist_ok=True)
    model = EGMFNet().to(device)
    initialize_weights(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    generator = torch.Generator().manual_seed(seed)
    history, start_epoch, global_step = [], 1, 0
    if cfg["resume_path"]:
        checkpoint = torch.load(cfg["resume_path"], map_location="cpu", weights_only=False)
        if checkpoint.get("format") != FORMAT:
            raise ValueError("Resume requires a full checkpoint produced by this release")
        if checkpoint["documents"] != manifest:
            raise ValueError("Resume training documents differ from the checkpoint")
        old = checkpoint["config"]["train"]
        for key in ("patch_size", "patch_stride", "full_image_size"):
            if old[key] != cfg[key]:
                raise ValueError(f"Cannot change {key} during resume")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["weight_decay"] = cfg["weight_decay"]
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history = checkpoint["history"]
        start_epoch, global_step = checkpoint["epoch"] + 1, checkpoint["global_step"]
        generator.set_state(checkpoint["loader_rng"])
        restore_rng(checkpoint["rng"])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=runtime["num_workers"], pin_memory=device.type == "cuda", generator=generator)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "training_documents.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    message = f"Training documents={len(pairs)} patches={len(dataset)} device={device}; fixed gated model, no validation"
    print(message)
    log_path = output / "log.txt"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
    for epoch in range(start_epoch, cfg["max_epochs"] + 1):
        lr = learning_rate(cfg, epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        sums = {key: 0.0 for key in ("loss_total", "loss_bin", "loss_enhance")}
        examples = 0
        progress = tqdm(loader, desc=f"EGMF-NET {epoch}/{cfg['max_epochs']}")
        for batch in progress:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                result = model(batch["full_image"], batch["image"], batch["crop_meta"])
                losses = compute_loss(result, batch["target"], batch["enhance_target"], batch["full_target"], batch["full_enhance_target"])
            if not torch.isfinite(losses["loss_total"]):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(losses["loss_total"]).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            count = batch["image"].shape[0]
            examples += count
            for key in sums:
                sums[key] += float(losses[key].detach()) * count
            progress.set_postfix(loss=f"{float(losses['loss_total'].detach()):.5f}")
        row = {"epoch": epoch, "lr": lr, "train_loss": sums["loss_total"] / examples,
               "train_loss_bin": sums["loss_bin"] / examples, "train_loss_enhance": sums["loss_enhance"] / examples}
        history.append(row)
        save_history(history, output)
        checkpoint = {"format": FORMAT, "model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(),
                      "epoch": epoch, "global_step": global_step, "history": history, "config": config,
                      "documents": manifest, "rng": capture_rng(), "loader_rng": generator.get_state()}
        torch.save(checkpoint, output / "model_last.pth")
        if epoch % cfg["save_every"] == 0 or epoch == cfg["max_epochs"]:
            torch.save(checkpoint, output / f"model_epoch{epoch:03d}.pth")
        summary = f"epoch={epoch} lr={lr:.8g} loss={row['train_loss']:.6f}"
        print(summary)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(summary + "\n")
    return history


if __name__ == "__main__":
    config = load_config()
    config["mode"] = "train"
    train(config)
