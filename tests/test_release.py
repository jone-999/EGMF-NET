"""Regression tests use tiny synthetic pages and temporary output directories."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from datasets.documentloader import DocumentDataset, paired_documents, axis_starts
from model import EGMFNet, compute_loss
from utils.config import load_config, validate_config
from utils.checkpoint import load_weights
from utils.checkpoint import cascade_config, FORMAT
from train import train
from test import run_test


def small_model(**kwargs):
    return EGMFNet(base_channels=4, channels=(8, 16, 32, 32), **kwargs)


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make_case(self, folder):
        root = Path(folder)
        for name in ("img", "gt", "enhancement"):
            (root / name).mkdir()
        for index in range(2):
            image = np.full((32, 32, 3), 255, dtype=np.uint8)
            image[4:28, 10 + index:14 + index] = 0
            Image.fromarray(image).save(root / "img" / f"page{index}.png")
            Image.fromarray(image[:, :, 0]).save(root / "gt" / f"page{index}.png")
            Image.fromarray(image).save(root / "enhancement" / f"page{index}.png")
        config = load_config()
        config["data"].update(train_img_dir=str(root / "img"), train_gt_dir=str(root / "gt"),
                              train_enhance_gt_dir=str(root / "enhancement"),
                              test_img_dir=str(root / "img"), test_gt_dir=str(root / "gt"))
        config["train"].update(save_dir=str(root / "out"), batch_size=1, max_epochs=2,
                               patch_size=32, full_image_size=64, patch_stride=32, save_every=1)
        config["test"].update(save_dir=str(root / "pred"), tile_size=32, tile_overlap=16)
        config["runtime"].update(device="cpu", num_workers=0)
        return root, config

    def test_rejects_old_options_and_tiling_gaps(self):
        config = load_config()
        for section, key, value in (("train", "val_ratio", 0.2), ("train", "optimizer", "sgd"),
                                    ("train", "augment", "false"), ("train", "patch_stride", 512),
                                    ("test", "tile_overlap", 256),
                                    ("model", "dual_cascade_stages", 0),
                                    ("model", "independent_cascade_params", "false")):
            changed = deepcopy(config)
            changed[section][key] = value
            with self.assertRaises(ValueError):
                validate_config(changed)
        with self.assertRaises(ValueError):
            axis_starts(800, 256, 400)

    def test_cascade_counts_sharing_and_gradients(self):
        for stages, independent in ((1, False), (3, False), (2, True)):
            with self.subTest(stages=stages, independent=independent):
                model = small_model(stages=stages, independent_cascade_params=independent)
                if stages > 1:
                    first = next(model.get_stage_model(0).parameters())
                    second = next(model.get_stage_model(1).parameters())
                    self.assertEqual(first is second, not independent)
                output = model(torch.randn(1, 3, 64, 64), torch.randn(1, 3, 32, 32),
                               torch.tensor([[0, 0, 32, 32, 32, 32]], dtype=torch.float32))
                self.assertEqual(len(output["stage_outputs"]), stages)
                losses = compute_loss(output, torch.zeros(1, 1, 32, 32), torch.rand(1, 3, 32, 32),
                                      torch.zeros(1, 1, 64, 64), torch.rand(1, 3, 64, 64))
                losses["loss_total"].backward()
                self.assertTrue(all(p.grad is not None for p in model.parameters()))
        self.assertEqual(cascade_config({"format": FORMAT, "config": {}}),
                         {"dual_cascade_stages": 2, "independent_cascade_params": False})

    def test_independent_cascade_training_and_checkpoint_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            root, config = self.make_case(folder)
            config["model"]["independent_cascade_params"] = True
            config["train"]["max_epochs"] = 1
            with patch("train.EGMFNet", side_effect=small_model):
                train(config)
            checkpoint = torch.load(root / "out" / "model_last.pth", map_location="cpu", weights_only=False)
            self.assertTrue(checkpoint["config"]["model"]["independent_cascade_params"])
            self.assertTrue(any(key.startswith("cascade_stage_models.1.") for key in checkpoint["model_state_dict"]))
            config["mode"] = "test"
            config["test"]["checkpoint_path"] = str(root / "out" / "model_last.pth")
            # Inference reconstructs the trained checkpoint, even if the
            # current YAML's sharing setting was subsequently changed.
            config["model"]["independent_cascade_params"] = False
            with patch("test.EGMFNet", side_effect=small_model) as constructor:
                run_test(config)
                self.assertTrue(constructor.call_args.kwargs["independent_cascade_params"])
            config["mode"] = "train"
            config["train"]["resume_path"] = str(root / "out" / "model_last.pth")
            with patch("train.EGMFNet", side_effect=small_model), self.assertRaisesRegex(ValueError, "sharing"):
                train(config)

    def test_pairing_is_strict_and_all_documents_are_used(self):
        with tempfile.TemporaryDirectory() as folder:
            root, config = self.make_case(folder)
            pairs = paired_documents(root / "img", root / "gt", root / "enhancement")
            dataset = DocumentDataset(pairs, 32, 32, 64)
            self.assertEqual(len(dataset.pairs), 2)
            self.assertEqual(len(dataset), 2)
            sample = dataset[0]
            self.assertEqual(tuple(sample["full_image"].shape), (3, 64, 64))
            self.assertEqual(tuple(sample["image"].shape), (3, 32, 32))
            # A missing GT must fail, not silently discard a training document.
            (root / "gt" / "page0.png").rename(root / "gt" / "unmatched.png")
            with self.assertRaises(FileNotFoundError):
                paired_documents(root / "img", root / "gt", root / "enhancement")

    def test_fixed_gated_model_has_no_unused_parameters(self):
        model = small_model()
        image = torch.randn(1, 3, 32, 32)
        full = torch.randn(1, 3, 64, 64)
        meta = torch.tensor([[0, 0, 32, 32, 32, 32]], dtype=torch.float32)
        output = model(full, image, meta)
        losses = compute_loss(output, torch.zeros(1, 1, 32, 32), torch.rand(1, 3, 32, 32),
                              torch.zeros(1, 1, 64, 64), torch.rand(1, 3, 64, 64))
        losses["loss_total"].backward()
        self.assertEqual(len(output["stage_outputs"]), 2)
        self.assertEqual([name for name, parameter in model.named_parameters() if parameter.grad is None], [])
        state = dict(model.state_dict())
        state["bin_fusion_head.weight"] = torch.zeros(1, 2, 1, 1)
        load_weights(small_model(), state)
        state["unexpected.weight"] = torch.zeros(1)
        with self.assertRaises(RuntimeError):
            load_weights(small_model(), state)

    def test_training_resume_matches_uninterrupted_and_separate_testing(self):
        with tempfile.TemporaryDirectory() as folder:
            root, config = self.make_case(folder)
            with patch("train.EGMFNet", side_effect=small_model):
                history = train(config)
            self.assertEqual(len(history), 2)
            self.assertTrue(all(not key.startswith("val") for row in history for key in row))
            out = root / "out"
            self.assertFalse(list(out.glob("*best*")))
            checkpoint = torch.load(out / "model_last.pth", map_location="cpu", weights_only=False)
            self.assertEqual(len(checkpoint["documents"]), 2)
            resumed = deepcopy(config)
            resumed["train"].update(save_dir=str(root / "resumed"), resume_path=str(out / "model_epoch001.pth"))
            with patch("train.EGMFNet", side_effect=small_model):
                train(resumed)
            restored = torch.load(root / "resumed" / "model_last.pth", map_location="cpu", weights_only=False)
            for key in checkpoint["model_state_dict"]:
                torch.testing.assert_close(checkpoint["model_state_dict"][key], restored["model_state_dict"][key], rtol=0, atol=0)
            test_config = deepcopy(config)
            test_config["mode"] = "test"
            test_config["test"]["checkpoint_path"] = str(out / "model_last.pth")
            with patch("test.EGMFNet", side_effect=small_model):
                run_test(test_config)
            self.assertEqual(len(list((root / "pred" / "binary").glob("*.png"))), 2)
            self.assertTrue((root / "pred" / "metrics.csv").exists())


if __name__ == "__main__":
    unittest.main()
