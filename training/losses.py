from __future__ import annotations

import torch


def masked_mse_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if mask.sum() == 0:
        return y_hat.new_tensor(0.0)

    error = (y_hat - y) ** 2
    return error[mask].mean()


def temporal_delta_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if y_hat.shape[0] < 2:
        return y_hat.new_tensor(0.0)

    valid_pairs = mask[1:] & mask[:-1]

    if valid_pairs.sum() == 0:
        return y_hat.new_tensor(0.0)

    delta_hat = y_hat[1:] - y_hat[:-1]
    delta = y[1:] - y[:-1]

    error = (delta_hat - delta) ** 2
    return error[valid_pairs].mean()