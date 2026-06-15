from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DataConfig:
    data_dir: Path = ROOT / "data"
    stib_filename: str = "stib_speeds.csv"
    reference_filename: str = "reference_speeds.csv"
    adjacency_filename: str = "adjacency.csv"

    @property
    def stib_path(self) -> Path:
        return self.data_dir / self.stib_filename

    @property
    def reference_path(self) -> Path:
        return self.data_dir / self.reference_filename

    @property
    def adjacency_path(self) -> Path:
        return self.data_dir / self.adjacency_filename


@dataclass(frozen=True)
class SplitConfig:
    strategy: str = "by_days"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    val_days: int = 2
    test_days: int = 3
    min_observed_targets: int = 1


@dataclass(frozen=True)
class FeatureConfig:
    recent_steps: int = 8
    daily_lags: int = 2
    weekly_lags: int = 1
    similar_days: int = 1
    similarity_metric: str = "cosine"
    pad_value: float = 0.0


@dataclass(frozen=True)
class NormalizationConfig:
    input: bool = True
    target: bool = True
    eps: float = 1e-6


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 64
    num_blocks: int = 2
    temporal_kernel_size: int = 3
    attention_dim: int = 64
    diffusion_hops: int = 2
    base_window: int = 4
    residual_topk: int = 2
    enforce_nonnegative: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    patience: int = 5
    min_delta: float = 0.0
    delta_loss_weight: float = 0.0
    prefer_mps: bool = True


@dataclass(frozen=True)
class OutputConfig:
    output_dir: Path = ROOT / "outputs"
    prediction_filename: str = "predictions.csv"
    checkpoint_filename: str = "model.pt"

    @property
    def prediction_path(self) -> Path:
        return self.output_dir / self.prediction_filename

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / self.checkpoint_filename


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


CFG = Config()
CFG.output.output_dir.mkdir(parents=True, exist_ok=True)