"""Strict document pairing and aligned global/page-patch training samples."""
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_files(directory):
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    images = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not images:
        raise ValueError(f"No images in {root}")
    if len({p.stem for p in images}) != len(images):
        raise ValueError(f"Duplicate image stems in {root}")
    return images


def paired_documents(image_dir, gt_dir, enhance_dir=None):
    images = image_files(image_dir)
    masks = {p.stem: p for p in image_files(gt_dir)}
    enhanced = {p.stem: p for p in image_files(enhance_dir)} if enhance_dir else {}
    pairs = []
    for image in images:
        if image.stem not in masks:
            raise FileNotFoundError(f"Missing binary GT for {image.name}")
        if enhance_dir and image.stem not in enhanced:
            raise FileNotFoundError(f"Missing RGB enhancement GT for {image.name}")
        pair = (image, masks[image.stem], enhanced.get(image.stem))
        with Image.open(image) as source:
            size = source.size
        for target in pair[1:]:
            if target is not None:
                with Image.open(target) as handle:
                    if handle.size != size:
                        raise ValueError(f"Image/target dimensions differ: {image.name}, {target.name}")
        pairs.append(pair)
    return pairs


def axis_starts(length, side, stride):
    if not 0 < stride <= side:
        raise ValueError("Require 0 < stride <= patch size")
    starts = list(range(0, max(length - side + 1, 1), stride))
    return sorted(set(starts + [max(length - side, 0)]))


def normalize_image(image):
    return TF.to_tensor(image) * 2 - 1


class DocumentDataset(Dataset):
    """All supplied documents are training data; there is no internal split."""
    def __init__(self, pairs, patch_size=256, patch_stride=256, full_image_size=512, augment=False):
        self.pairs = pairs
        self.patch_size = patch_size
        self.full_image_size = full_image_size
        self.augment = augment
        self.index = []
        for index, (image, _, _) in enumerate(pairs):
            with Image.open(image) as handle:
                width, height = handle.size
            for top in axis_starts(height, patch_size, patch_stride):
                for left in axis_starts(width, patch_size, patch_stride):
                    self.index.append((index, left, top))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        document, left, top = self.index[index]
        image_path, gt_path, enhance_path = self.pairs[document]
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        with Image.open(gt_path) as handle:
            mask = handle.convert("L")
        if enhance_path:
            with Image.open(enhance_path) as handle:
                enhanced = handle.convert("RGB")
        else:
            enhanced = image.copy()
        width, height = image.size
        images = [image, mask, enhanced]
        # Augment the page once before global resizing and local cropping,
        # so geometric correspondence is preserved across the two branches.
        if self.augment:
            if random.random() < 0.3:
                angle = random.uniform(-10, 10)
                images = [TF.rotate(item, angle, interpolation=InterpolationMode.NEAREST if item.mode == "L" else InterpolationMode.BILINEAR,
                                    fill=255 if item.mode == "L" else (255, 255, 255)) for item in images]
            if random.random() < 0.5:
                images[0] = TF.adjust_brightness(images[0], random.uniform(0.85, 1.2))
            if random.random() < 0.5:
                images[0] = TF.adjust_contrast(images[0], random.uniform(0.85, 1.2))
            if random.random() < 0.5:
                images[0] = TF.adjust_saturation(images[0], random.uniform(0.85, 1.15))
                images[0] = TF.adjust_hue(images[0], random.uniform(-0.08, 0.08))
        full = [TF.resize(item, [self.full_image_size] * 2,
                          interpolation=InterpolationMode.NEAREST if item.mode == "L" else InterpolationMode.BILINEAR) for item in images]
        padding = [0, 0, max(0, self.patch_size - width), max(0, self.patch_size - height)]
        local = [TF.crop(TF.pad(item, padding, fill=255 if item.mode == "L" else (255, 255, 255)), top, left,
                         self.patch_size, self.patch_size) for item in images]

        def binary(item):
            return torch.from_numpy((np.asarray(item) < 128).astype(np.float32)).unsqueeze(0)

        return {"image": normalize_image(local[0]), "target": binary(local[1]),
                "enhance_target": TF.to_tensor(local[2]), "full_image": normalize_image(full[0]),
                "full_target": binary(full[1]), "full_enhance_target": TF.to_tensor(full[2]),
                "crop_meta": torch.tensor([left, top, self.patch_size, self.patch_size, width, height], dtype=torch.float32)}
