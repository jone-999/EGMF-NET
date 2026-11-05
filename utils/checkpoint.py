"""Checkpoint handling for the fixed gated network."""
import random
import numpy as np
import torch

FORMAT = "egmf_gated_shared_v1"


def capture_rng():
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def load_weights(model, checkpoint):
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    # The old gated implementation registered two conv-fusion heads that were
    # never used by its gated path. Only these exact obsolete keys are dropped.
    removed = {"bin_fusion_head.weight", "bin_fusion_head.bias", "enhance_fusion_head.weight", "enhance_fusion_head.bias"}
    state = {key: value for key, value in state.items() if key not in removed}
    model.load_state_dict(state, strict=True)
