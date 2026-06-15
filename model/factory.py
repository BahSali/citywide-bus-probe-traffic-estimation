from __future__ import annotations

from model.dual_branch_stgcnn import DualBranchSTGCNN


def build_model(
    num_segments: int,
    input_window: int,
    adjacency,
    cfg,
) -> DualBranchSTGCNN:
    return DualBranchSTGCNN(
        num_segments=num_segments,
        input_window=input_window,
        adjacency=adjacency,
        hidden_dim=cfg.hidden_dim,
        num_blocks=cfg.num_blocks,
        temporal_kernel_size=cfg.temporal_kernel_size,
        attention_dim=cfg.attention_dim,
        diffusion_hops=cfg.diffusion_hops,
        base_window=cfg.base_window,
        residual_topk=cfg.residual_topk,
        enforce_nonnegative=cfg.enforce_nonnegative,
    )