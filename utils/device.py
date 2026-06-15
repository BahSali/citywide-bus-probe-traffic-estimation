from __future__ import annotations

import torch


def get_device(prefer_mps: bool = True) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")