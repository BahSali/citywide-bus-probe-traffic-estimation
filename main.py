from __future__ import annotations

import pandas as pd

from config import CFG
from data.loaders import load_adjacency_matrix, load_speed_csv
from evaluation.evaluator import aggregate_metrics, evaluate_per_segment
from evaluation.exporter import export_results
from model.factory import build_model
from preprocessing.prepare import prepare_dataset
from training.trainer import Trainer
from utils.checkpoint import save_checkpoint
from utils.device import get_device


def main() -> None:
    source = load_speed_csv(CFG.data.stib_path)
    reference = load_speed_csv(CFG.data.reference_path)

    dataset = prepare_dataset(
        source=source,
        reference=reference,
        cfg=CFG,
    )

    adjacency = load_adjacency_matrix(
        path=CFG.data.adjacency_path,
        ordered_nodes=dataset.segment_ids,
    )

    model = build_model(
        num_segments=len(dataset.segment_ids),
        input_window=dataset.input_window,
        adjacency=adjacency,
        cfg=CFG.model,
    )

    trainer = Trainer(
        model=model,
        device=get_device(CFG.training.prefer_mps),
        learning_rate=CFG.training.learning_rate,
        batch_size=CFG.training.batch_size,
        delta_loss_weight=CFG.training.delta_loss_weight,
    )

    history = trainer.fit(
        train_data=dataset.train,
        val_data=dataset.val,
        epochs=CFG.training.epochs,
        patience=CFG.training.patience,
        min_delta=CFG.training.min_delta,
    )

    predictions = trainer.predict(dataset.X_all)

    if dataset.target_scaler is not None:
        predictions = dataset.target_scaler.inverse_transform(predictions)
        target = dataset.target_scaler.inverse_transform(dataset.y_all)
    else:
        target = dataset.y_all

    prediction_df = pd.DataFrame(
        predictions,
        index=dataset.source.index,
        columns=dataset.segment_ids,
    )
    prediction_df.index.name = "timestamp"
    prediction_df.to_csv(CFG.output.prediction_path)

    metrics_df = evaluate_per_segment(
        y=target[dataset.test_idx],
        y_hat=predictions[dataset.test_idx],
        mask=dataset.mask_all[dataset.test_idx],
        segment_ids=dataset.segment_ids,
    )

    summary = aggregate_metrics(metrics_df)

    for metric, value in summary.items():
        print(f"{metric}: {value:.4f}")

    export_results(
        output_dir=CFG.output.output_dir,
        stib_df=dataset.source,
        reference_df=dataset.reference,
        prediction_df=prediction_df,
        metrics_df=metrics_df,
        test_idx=dataset.test_idx,
    )

    save_checkpoint(
        path=CFG.output.checkpoint_path,
        model=model,
        metadata={
            "best_val_loss": history["best_val_loss"],
            "num_segments": len(dataset.segment_ids),
            "input_window": dataset.input_window,
        },
    )


if __name__ == "__main__":
    main()