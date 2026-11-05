import numpy as np


def foreground_to_binary_image(mask):
    return np.where(mask > 0, 0, 255).astype(np.uint8)


def _to_foreground_mask(mask):
    return (np.asarray(mask) > 0).astype(np.uint8)


def _zhang_suen_thinning(mask):
    skeleton = _to_foreground_mask(mask).copy()
    if skeleton.size == 0 or skeleton.sum() == 0:
        return skeleton

    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            padded = np.pad(skeleton, 1, mode="constant", constant_values=0)
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )

            if step == 0:
                step_condition = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                step_condition = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)

            removable = (
                (skeleton == 1)
                & (neighbors >= 2)
                & (neighbors <= 6)
                & (transitions == 1)
                & step_condition
            )
            if np.any(removable):
                skeleton[removable] = 0
                changed = True
    return skeleton


def _fmeasure(precision, recall):
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)


def _precision_recall(prediction, target):
    pred_flat = prediction.reshape(-1).astype(np.uint8)
    tgt_flat = target.reshape(-1).astype(np.uint8)
    tp = int(np.logical_and(pred_flat == 1, tgt_flat == 1).sum())
    fp = int(np.logical_and(pred_flat == 1, tgt_flat == 0).sum())
    fn = int(np.logical_and(pred_flat == 0, tgt_flat == 1).sum())
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    return precision, recall


def _pseudo_fmeasure(prediction, target, precision):
    skeleton = _zhang_suen_thinning(target)
    skeleton_pixels = int(skeleton.sum())
    if skeleton_pixels == 0:
        pseudo_recall = 1.0 if int(prediction.sum()) == 0 else 0.0
    else:
        pseudo_recall = float(np.logical_and(prediction == 1, skeleton == 1).sum() / skeleton_pixels)
    return _fmeasure(precision, pseudo_recall)


def _psnr(prediction, target):
    pred_image = foreground_to_binary_image(prediction).astype(np.float32)
    target_image = foreground_to_binary_image(target).astype(np.float32)
    mse = np.mean((pred_image - target_image) ** 2)
    return 99.0 if mse == 0 else float(10.0 * np.log10((255.0 ** 2) / mse))


def _non_uniform_block_count(target, block_size=8):
    height, width = target.shape
    count = 0
    for top in range(0, height - block_size + 1, block_size):
        for left in range(0, width - block_size + 1, block_size):
            block = target[top : top + block_size, left : left + block_size]
            if block.any() and not block.all():
                count += 1
    return max(count, 1)


def _drd(prediction, target):
    mismatch_y, mismatch_x = np.nonzero(prediction != target)
    if mismatch_y.size == 0:
        return 0.0

    offsets = []
    weights = []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dy == 0 and dx == 0:
                continue
            offsets.append((dy, dx))
            weights.append(1.0 / np.sqrt(float(dy * dy + dx * dx)))
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()

    target_int = target.astype(int)
    total_distortion = 0.0
    rows, cols = target.shape
    for y, x in zip(mismatch_y, mismatch_x):
        gk = int(prediction[y, x])
        local_distortion = 0.0
        for weight, (dy, dx) in zip(weights, offsets):
            ny, nx = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx < cols:
                local_distortion += abs(int(target_int[ny, nx]) - gk) * weight
            else:
                local_distortion += abs(gk - gk) * weight
        total_distortion += local_distortion
    return float(total_distortion / _non_uniform_block_count(target))


def calculate_document_metrics(prediction, target):
    prediction = _to_foreground_mask(prediction)
    target = _to_foreground_mask(target)

    precision, recall = _precision_recall(prediction, target)
    return {
        "fm": float(_fmeasure(precision, recall)),
        "pfm": float(_pseudo_fmeasure(prediction, target, precision)),
        "psnr": float(_psnr(prediction, target)),
        "drd": float(_drd(prediction, target)),
    }
