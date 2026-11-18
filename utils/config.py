"""Load and check EGMF-NET configuration."""
from copy import deepcopy
from pathlib import Path
import math
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "conf.yml"
SCHEMA = {
    "model": {"dual_cascade_stages", "independent_cascade_params"},
    "data": {"train_img_dir", "train_gt_dir", "train_enhance_gt_dir", "test_img_dir"},
    "train": {"save_dir", "resume_path", "batch_size", "max_epochs", "patch_size", "full_image_size",
              "patch_stride", "lr", "lr_end", "weight_decay", "augment", "save_every"},
    "test": {"checkpoint_path", "save_dir", "tile_size", "tile_overlap", "binary_threshold"},
    "runtime": {"seed", "device", "num_workers", "use_amp"},
}


def validate_config(config):
    config = deepcopy(config)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    if set(config) != set(SCHEMA) | {"mode"}:
        raise ValueError("Required sections: mode, model, data, train, test, runtime")
    if isinstance(config["data"], dict):
        config["data"].pop("test_gt_dir", None)
    for section, keys in SCHEMA.items():
        if not isinstance(config[section], dict) or set(config[section]) != keys:
            raise ValueError(f"Invalid or missing {section} keys; expected: {sorted(keys)}")
    if config["mode"] not in {"train", "test"}:
        raise ValueError("mode must be train or test")
    train, test, runtime = (config[name] for name in ("train", "test", "runtime"))
    stages = config["model"]["dual_cascade_stages"]
    if type(stages) is not int or stages <= 0:
        raise ValueError("model.dual_cascade_stages must be a positive integer")
    for section, key in (("train", "batch_size"), ("train", "max_epochs"), ("train", "patch_size"),
                         ("train", "full_image_size"), ("train", "patch_stride"), ("train", "save_every"),
                         ("test", "tile_size")):
        value = config[section][key]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{section}.{key} must be a positive integer")
    for key in ("patch_size", "full_image_size"):
        if train[key] % 32:
            raise ValueError(f"train.{key} must be a multiple of 32")
    if test["tile_size"] % 32:
        raise ValueError("test.tile_size must be a multiple of 32")
    if train["patch_stride"] > train["patch_size"]:
        raise ValueError("patch_stride must not exceed patch_size")
    if type(test["tile_overlap"]) is not int or not 0 <= test["tile_overlap"] < test["tile_size"]:
        raise ValueError("Require 0 <= tile_overlap < tile_size")
    for key in ("lr", "lr_end", "weight_decay"):
        train[key] = float(train[key])
        if not math.isfinite(train[key]) or train[key] < 0:
            raise ValueError(f"train.{key} must be finite and nonnegative")
    if not 0 < train["lr"] or train["lr_end"] > train["lr"]:
        raise ValueError("Require 0 <= lr_end <= lr and lr > 0")
    test["binary_threshold"] = float(test["binary_threshold"])
    if not 0 <= test["binary_threshold"] <= 1:
        raise ValueError("binary_threshold must be in [0,1]")
    for section, key in (("train", "augment"), ("runtime", "use_amp"), ("model", "independent_cascade_params")):
        if type(config[section][key]) is not bool:
            raise ValueError(f"{section}.{key} must be a YAML boolean")
    if type(runtime["num_workers"]) is not int or runtime["num_workers"] < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if type(runtime["seed"]) is not int or not 0 <= runtime["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if runtime["device"] not in {"auto", "cpu", "cuda"} and not str(runtime["device"]).startswith("cuda:"):
        raise ValueError("device must be auto, cpu, cuda or cuda:<index>")
    required = [("train", "save_dir")] if config["mode"] == "train" else [("test", "checkpoint_path"), ("test", "save_dir")]
    required += [("data", "train_img_dir"), ("data", "train_gt_dir")] if config["mode"] == "train" else [("data", "test_img_dir")]
    for section, key in required:
        if not config[section][key]:
            raise ValueError(f"Set {section}.{key} in conf.yml")
    for section, keys in (("data", SCHEMA["data"]), ("train", {"save_dir", "resume_path"}),
                          ("test", {"checkpoint_path", "save_dir"})):
        for key in keys:
            value = config[section][key]
            if not isinstance(value, str):
                raise ValueError(f"{section}.{key} must be a path string")
            if value:
                path = Path(value).expanduser()
                config[section][key] = str(path if path.is_absolute() else (ROOT / path).resolve())
    return config


def load_config(path=DEFAULT_CONFIG):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with path.open(encoding="utf-8-sig") as handle:
        return validate_config(yaml.safe_load(handle))
