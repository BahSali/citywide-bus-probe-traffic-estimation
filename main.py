from config import CFG
from data.loaders import load_speed_csv, load_adjacency_matrix
from preprocessing.prepare import prepare_dataset
from models.factory import build_model
from training.trainer import Trainer
from evaluation.evaluator import evaluate_model
from evaluation.exporter import export_results
from utils.device import get_device
from utils.checkpoint import save_model_checkpoint


def main():
    stib = load_speed_csv(CFG.stib_csv)
    google = load_speed_csv(CFG.google_csv)

    dataset = prepare_dataset(stib, google, CFG)

    adjacency = load_adjacency_matrix(
        CFG.adjacency_csv,
        ordered_nodes=dataset.streets,
    )

    model = build_model(
        num_nodes=len(dataset.streets),
        input_window=dataset.input_window,
        adjacency=adjacency,
        cfg=CFG.model,
    )

    trainer = Trainer(
        model=model,
        device=get_device(CFG.use_mps),
        cfg=CFG.training,
    )

    history = trainer.fit(
        train_data=dataset.train,
        val_data=dataset.val,
    )

    predictions = trainer.predict(dataset.X_all)
    results = evaluate_model(dataset, predictions)

    export_results(results, CFG.output_dir)

    save_model_checkpoint(
        CFG.checkpoint_path,
        model,
        metadata={"best_val_loss": history.best_val_loss},
    )


if __name__ == "__main__":
    main()