from __future__ import annotations

import torch
import torch.nn as nn


class DualBranchSTGCNN(nn.Module):
    """Gated dual-branch spatiotemporal GCNN.

    The model maps bus-derived speed observations to reference traffic speed.
    A stack of spatiotemporal encoder blocks combines temporal convolution,
    temporal self-attention, spatial attention over the road graph, and a
    diffusion graph convolution. A gated dual-branch decoder then separates a
    coarse base estimate from a localized residual correction.

    Input:
        x: [batch, window, num_segments]

    Output:
        y_hat: [batch, num_segments]

    The estimate corresponds to the last timestamp of each input window.
    """

    def __init__(
        self,
        num_segments: int,
        input_window: int,
        adjacency: torch.Tensor,
        hidden_dim: int = 64,
        num_blocks: int = 2,
        temporal_kernel_size: int = 3,
        attention_dim: int | None = None,
        diffusion_hops: int = 2,
        base_window: int = 4,
        residual_topk: int = 2,
        enforce_nonnegative: bool = False,
    ):
        super().__init__()

        if base_window > input_window:
            raise ValueError("base_window cannot exceed the input window length.")

        if residual_topk > base_window:
            raise ValueError("residual_topk cannot exceed base_window.")

        self.num_segments = num_segments
        self.input_window = input_window
        self.base_window = base_window
        self.residual_topk = residual_topk
        self.enforce_nonnegative = enforce_nonnegative

        attention_dim = attention_dim or hidden_dim

        adjacency = _prepare_adjacency(adjacency)
        self.register_buffer("adjacency", adjacency)

        self.input_projection = nn.Linear(1, hidden_dim)

        self.blocks = nn.ModuleList(
            SpatioTemporalBlock(
                hidden_dim=hidden_dim,
                temporal_kernel_size=temporal_kernel_size,
                attention_dim=attention_dim,
                diffusion_hops=diffusion_hops,
            )
            for _ in range(num_blocks)
        )

        self.decoder = GatedDualBranchDecoder(
            hidden_dim=hidden_dim,
            base_window=base_window,
            residual_topk=residual_topk,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected input with shape [batch, window, num_segments], got {tuple(x.shape)}."
            )

        if x.shape[2] != self.num_segments:
            raise ValueError(
                f"Expected {self.num_segments} segments, got {x.shape[2]}."
            )

        h = self.input_projection(x.unsqueeze(-1))

        for block in self.blocks:
            h = block(h, self.adjacency)

        y_hat = self.decoder(h)

        if self.enforce_nonnegative:
            y_hat = torch.nn.functional.softplus(y_hat)

        return y_hat


class SpatioTemporalBlock(nn.Module):
    """One encoder block: temporal modeling followed by spatial aggregation."""

    def __init__(
        self,
        hidden_dim: int,
        temporal_kernel_size: int,
        attention_dim: int,
        diffusion_hops: int,
    ):
        super().__init__()

        self.temporal = TemporalSubmodule(
            hidden_dim=hidden_dim,
            kernel_size=temporal_kernel_size,
            attention_dim=attention_dim,
        )

        self.spatial = SpatialSubmodule(
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            diffusion_hops=diffusion_hops,
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = self.temporal(x)
        h = self.spatial(h, adjacency)

        return h


class TemporalSubmodule(nn.Module):
    """Causal temporal convolution followed by temporal self-attention.

    Operates independently on each road segment.

    Input:
        x: [B, W, N, H]

    Output:
        h: [B, W, N, H]
    """

    def __init__(self, hidden_dim: int, kernel_size: int, attention_dim: int):
        super().__init__()

        self.padding = kernel_size - 1

        self.conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
        )
        self.activation = nn.ReLU()

        self.query = nn.Linear(hidden_dim, attention_dim)
        self.key = nn.Linear(hidden_dim, attention_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.scale = attention_dim ** 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, window, num_segments, hidden_dim = x.shape

        flat = x.permute(0, 2, 3, 1).reshape(
            batch_size * num_segments, hidden_dim, window
        )

        conv = self.conv(flat)
        if self.padding > 0:
            conv = conv[..., :window]
        conv = self.activation(conv)

        conv = conv.permute(0, 2, 1)

        q = self.query(conv)
        k = self.key(conv)
        v = self.value(conv)

        scores = torch.matmul(q, k.transpose(1, 2)) / self.scale
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v)

        attended = attended.reshape(
            batch_size, num_segments, window, hidden_dim
        ).permute(0, 2, 1, 3)

        return attended


class SpatialSubmodule(nn.Module):
    """Spatial attention over the road graph and diffusion graph convolution.

    The binary adjacency restricts attention to connected segments. Attention
    weights modulate the adjacency at each timestamp, and a diffusion-style
    convolution aggregates information from neighbours within a fixed number of
    hops.

    Input:
        x: [B, W, N, H]
        adjacency: [N, N] binary connectivity with self-loops

    Output:
        h: [B, W, N, H]
    """

    def __init__(self, hidden_dim: int, attention_dim: int, diffusion_hops: int):
        super().__init__()

        self.diffusion_hops = diffusion_hops

        self.query = nn.Linear(hidden_dim, attention_dim)
        self.key = nn.Linear(hidden_dim, attention_dim)
        self.scale = attention_dim ** 0.5

        self.hop_projections = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(diffusion_hops + 1)
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        mask = adjacency > 0

        q = self.query(x)
        k = self.key(x)

        scores = torch.einsum("bwnh,bwmh->bwnm", q, k) / self.scale
        scores = scores.masked_fill(~mask, float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        attention = attention * adjacency

        degree = attention.sum(dim=-1, keepdim=True).clamp(min=1.0)
        transition = attention / degree

        out = self.hop_projections[0](x)
        propagated = x

        for hop in range(1, self.diffusion_hops + 1):
            propagated = torch.einsum("bwnm,bwmh->bwnh", transition, propagated)
            out = out + self.hop_projections[hop](propagated)

        return self.activation(out)


class GatedDualBranchDecoder(nn.Module):
    """Gated dual-branch decoder.

    The base branch aggregates the latent states within a causal temporal
    neighbourhood using a learnable pooling kernel whose weights are
    non-negative and sum to one. The residual branch models localized
    deviations by selecting the residual vectors with the largest L1 norm. A
    learnable gate regulates the residual contribution before fusion.

    Input:
        x: [B, W, N, H]

    Output:
        y_hat: [B, N]
    """

    def __init__(self, hidden_dim: int, base_window: int, residual_topk: int):
        super().__init__()

        self.base_window = base_window
        self.residual_topk = residual_topk

        self.pooling_logits = nn.Parameter(torch.zeros(base_window))

        self.base_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.gate_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        neighbourhood = x[:, -self.base_window:, :, :]

        pooling_weights = torch.softmax(self.pooling_logits, dim=0)
        base_context = torch.einsum(
            "m,bmnh->bnh", pooling_weights, neighbourhood
        )

        residuals = neighbourhood - base_context.unsqueeze(1)

        norms = residuals.abs().sum(dim=-1)
        _, top_indices = torch.topk(norms, k=self.residual_topk, dim=1)

        gathered = torch.gather(
            residuals,
            dim=1,
            index=top_indices.unsqueeze(-1).expand(-1, -1, -1, residuals.shape[-1]),
        )
        residual_context = gathered.mean(dim=1)

        base = self.base_head(base_context).squeeze(-1)
        residual = self.residual_head(residual_context).squeeze(-1)
        gate = self.gate_head(base_context).squeeze(-1)

        return base + gate * residual


def _prepare_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    if not isinstance(adjacency, torch.Tensor):
        adjacency = torch.as_tensor(adjacency, dtype=torch.float32)

    adjacency = adjacency.float()

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Adjacency matrix must be square.")

    adjacency = (adjacency > 0).float()
    adjacency = adjacency.clone()
    adjacency.fill_diagonal_(1.0)

    return adjacency
