from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from training.losses import masked_mse_loss, temporal_delta_loss


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        delta_loss_weight: float = 0.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.delta_loss_weight = delta_loss_weight
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )

    def fit(
        self,
        train_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        val_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        epochs: int,
        patience: int,
        min_delta: float = 0.0,
    ) -> dict[str, float]:
        best_state = None
        best_val_loss = float("inf")
        stale_epochs = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(*train_data)
            val_loss = self.evaluate_loss(*val_data)

            print(f"Epoch {epoch:03d} | train={train_loss:.4f} | val={val_loss:.4f}")

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_state = deepcopy(self.model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1

            if stale_epochs >= patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {"best_val_loss": best_val_loss}

    def train_epoch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        self.model.train()

        total_loss = 0.0
        total_batches = 0

        # Batches are kept in chronological order so the temporal delta term
        # compares consecutive timestamps within each batch.
        for start in range(0, len(X), self.batch_size):
            end = start + self.batch_size

            X_t, y_t, mask_t = self._to_tensors(
                X[start:end], y[start:end], mask[start:end]
            )

            self.optimizer.zero_grad()

            y_hat = self.model(X_t)
            loss = self._loss(y_hat, y_t, mask_t)

            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_batches += 1

        return total_loss / max(total_batches, 1)

    def evaluate_loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        self.model.eval()

        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                end = start + self.batch_size

                X_t, y_t, mask_t = self._to_tensors(
                    X[start:end], y[start:end], mask[start:end]
                )

                y_hat = self.model(X_t)
                loss = self._loss(y_hat, y_t, mask_t)

                total_loss += float(loss.detach().cpu())
                total_batches += 1

        return total_loss / max(total_batches, 1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()

        predictions = []

        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                end = start + self.batch_size

                X_t = torch.as_tensor(
                    X[start:end],
                    dtype=torch.float32,
                    device=self.device,
                )
                y_hat = self.model(X_t)
                predictions.append(y_hat.detach().cpu().numpy())

        return np.concatenate(predictions, axis=0)

    def _loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        loss = masked_mse_loss(y_hat, y, mask)

        if self.delta_loss_weight > 0.0:
            loss = loss + self.delta_loss_weight * temporal_delta_loss(
                y_hat,
                y,
                mask,
            )

        return loss

    def _to_tensors(
        self,
        X: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.as_tensor(X, dtype=torch.float32, device=self.device),
            torch.as_tensor(y, dtype=torch.float32, device=self.device),
            torch.as_tensor(mask, dtype=torch.bool, device=self.device),
        )