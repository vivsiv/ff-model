import argparse
import logging
from datetime import datetime

import mlflow
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.modeling.data_builder import DATA_BUILDER_CONFIG_ARTIFACT_PATH, EVAL_SPLIT, TEST_SPLIT, TRAIN_SPLIT
from src.modeling.sequence.data_builder import SequenceDataset, SequenceModelDataBuilder
from src.modeling.sequence.inference import SequenceModelInference
from src.modeling.sequence.lstm import LSTMModel
from src.modeling.sequence.pipeline import LSTMPipeline
from src.modeling.sequence.preprocessor import SequencePreprocessor
from src.modeling.utils import build_base_model_name, setup_mlflow_run

logger = logging.getLogger(__name__)


class LSTMModelTrainer:
    """Trains and evaluates an LSTMModel on a builder's sequence splits,
    scoring only steps selected by loss_mask.

    - The supplied preprocessor is fitted on real training inputs on the first fit call.
    - Every split uses the same fitted preprocessing; real prepared steps must be finite.
    - NaN padding is allowed, and the builder's cached datasets are left unchanged.
    - The trainer retains the dataset's global padding rather than padding per batch.
    """

    def __init__(
        self,
        model: LSTMModel,
        data_builder: SequenceModelDataBuilder,
        preprocessor: SequencePreprocessor,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        tracking_dir: str = "mlruns",
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.data_builder = data_builder
        self.data = data_builder.get_data_splits()
        self.target = data_builder.target
        self.preprocessor = preprocessor
        self._prepared_data: dict[str, SequenceDataset] | None = None
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.tracking_dir = tracking_dir
        self.epochs_trained = 0

    @property
    def base_model_name(self) -> str:
        return build_base_model_name(self.target, self.data_builder.positions)

    @property
    def prepared_data(self) -> dict[str, SequenceDataset]:
        """Read cached prepared splits without fitting or transforming data."""
        if self._prepared_data is None:
            raise ValueError("Call fit() first; preprocessing has not been fitted.")

        return self._prepared_data

    def setup_mlflow(self, run_name: str | None = None, extra_params: dict | None = None) -> str:
        """Create a run in the base model's shared experiment and save the builder config."""
        run_id = setup_mlflow_run(
            experiment_name=self.base_model_name,
            run_name=run_name or f"lstm_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            tracking_dir=self.tracking_dir,
            tags={"model_family": "sequence", "model_type": "lstm", "phase": "train"},
            params=extra_params,
        )
        mlflow.log_dict(self.data_builder.config, DATA_BUILDER_CONFIG_ARTIFACT_PATH, run_id=run_id)

        return run_id

    def _fit_preprocessing(self) -> None:
        """Fit preprocessing once and cache transformed copies of all raw splits.

        Include real history even when excluded from loss; exclude padding from
        fitting and transformation. Entirely missing training features are imputed
        to zero without dropping their columns. Targets are not scaled.
        """
        if self._prepared_data is not None:
            return

        train = self.data[TRAIN_SPLIT]
        self.preprocessor.fit(train)

        prepared = {}
        for split, data in self.data.items():
            try:
                prepared[split] = self.preprocessor.transform(data)
            except ValueError as error:
                raise ValueError(f"{split}: {error}") from error

        self._prepared_data = prepared

    def fit(
        self,
        epochs: int = 20,
        evaluate_each_epoch: bool = True,
        run_id: str | None = None,
    ) -> list[dict[str, float | int]]:
        """Train on the builder's training split over a specified number of epochs.
        Optionally score its evaluation split each epoch. Requested evaluation
        requires at least one scored step; empty/unscored eval splits raise.

        - Training applies sampling weights when supplied by scaling error by sample weight.
          Batch loss is normalized by the batch's player count times the training
          set's average scored weight per player, not by each batch's weight sum.
          This preserves relative sample weights across uniformly sampled player batches.
        - Reported epoch MSE accounts for sample weights by aggregating squared errors and sample weights across all batches in an epoch.
        - Repeated calls to fit from the same Trainer object continue training where the last fit() call left off.
          Fitted preprocessing and prepared splits are reused along with optimizer state.
        - If run_id is supplied, log epoch MSE/RMSE and register the fitted pipeline.
        """
        if epochs < 1:
            raise ValueError("epochs must be positive.")

        if run_id is not None:
            mlflow.log_params({
                **LSTMPipeline.model_params(self.model),
                "batch_size": self.batch_size,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "device": str(self.device),
                "imputation_strategy": "mean",
                "keep_empty_features": True,
                "scaler": "StandardScaler",
            }, run_id=run_id)

        self._fit_preprocessing()
        train = self.prepared_data[TRAIN_SPLIT]
        train_loader = self._make_loader(train, training=True)

        eval_loader = None
        eval_data = self.prepared_data[EVAL_SPLIT]
        if evaluate_each_epoch:
            if not eval_data.loss_mask.any():
                raise ValueError("Cannot evaluate an empty 'eval' split with no scored steps.")
            eval_loader = self._make_loader(eval_data, training=False)

        history = []
        for epoch in range(1, epochs + 1):
            metrics = {"epoch": epoch, "train_mse": self._run_epoch(train_loader, training=True)}
            metrics["train_rmse"] = metrics["train_mse"] ** 0.5
            if eval_loader is not None:
                metrics["eval_mse"] = self._run_epoch(eval_loader, training=False)
                metrics["eval_rmse"] = metrics["eval_mse"] ** 0.5
            history.append(metrics)

            self.epochs_trained += 1
            if run_id is not None:
                logged_metrics = {key: value for key, value in metrics.items() if key != "epoch"}
                mlflow.log_metrics(logged_metrics, step=self.epochs_trained, run_id=run_id)

            logger.info("LSTM epoch metrics: %s", metrics)

        if run_id is not None:
            # TODO: log the eval dataset to mlflow
            self.log_model(run_id)

        return history

    def _make_loader(self, data: SequenceDataset, training: bool) -> DataLoader:
        if data.X.shape[-1] != self.model.lstm.input_size:
            raise ValueError("Feature count must match the model's input_size.")

        sample_weights = torch.ones_like(data.y)
        if training and data.sample_weight is not None:
            sample_weights = data.sample_weight

        # TODO: should we build a dataset that does the disk read directly so we can spawn multiple DataLoader procs (that do meaningful work)
        tensors = TensorDataset(data.X, data.lengths, data.y, data.loss_mask, sample_weights)
        return DataLoader(tensors, batch_size=self.batch_size, shuffle=training)

    def _run_epoch(self, loader: DataLoader, training: bool) -> float:
        # from nn.Module, activate dropout for training if configured, disables it for eval.
        self.model.train(training)
        total_error = total_sample_weight = 0.0
        if training:
            # The loader batches players uniformly. Use a split-wide normalizer so
            # low-weight batches retain less gradient influence, even at batch size 1.
            _, _, _, loss_mask, sample_weights = loader.dataset.tensors
            sample_weight_per_player = sample_weights[loss_mask].sum().item() / len(loader.dataset)

        # TODO: clean up code paths that run during training vs non training (i.e. loss doesnt need to be calculated during eval)
        # TODO: additional metrics like r^2
        with torch.set_grad_enabled(training):
            for X, lengths, y, loss_mask, sample_weights in loader:
                X, y = X.to(self.device), y.to(self.device)
                loss_mask, sample_weights = loss_mask.to(self.device), sample_weights.to(self.device)
                scored_sample_weights = sample_weights[loss_mask]
                sample_weight_sum = scored_sample_weights.sum()
                if sample_weight_sum.item() == 0:
                    continue

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                predictions = self.model(X, lengths)
                errors = (predictions[loss_mask] - y[loss_mask]).square()
                weighted_error = (errors * scored_sample_weights).sum()
                denominator = X.shape[0] * sample_weight_per_player if training else sample_weight_sum
                loss = weighted_error / denominator

                if not torch.isfinite(loss):
                    raise ValueError("Non-finite loss encountered.")

                if training:
                    loss.backward()
                    self.optimizer.step()

                total_error += weighted_error.detach().item()
                total_sample_weight += sample_weight_sum.item()

        return total_error / total_sample_weight

    def log_model(self, run_id: str):
        """Register one pyfunc model containing preprocessing, CPU weights, and feature order.

        Load with mlflow.pyfunc.load_model and predict with raw X/lengths arrays.
        Shapes allow variable batch sizes and padded sequence widths. This is an
        inference artifact, not an optimizer checkpoint for resuming training.
        """
        if self.epochs_trained == 0:
            raise ValueError("Call fit() before logging the pipeline.")

        train = self.data[TRAIN_SPLIT]
        input_example = {"X": train.X[:2].detach().cpu().numpy(), "lengths": train.lengths[:2].cpu().numpy()}

        return LSTMPipeline.log_pipeline(
            model=self.model,
            preprocessor=self.preprocessor,
            feature_columns=self.prepared_data[TRAIN_SPLIT].feature_columns,
            run_id=run_id,
            registered_model_name=f"{self.base_model_name}_lstm",
            input_example=input_example,
        )



def main():
    parser = argparse.ArgumentParser(description="Fits and evaluates an LSTM model against a gold dataset")
    parser.add_argument(
        "--tracking-dir", type=str, default="mlruns",
        help="MLflow tracking/registry directory, relative to the repo root (default: mlruns)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Parent directory containing gold data, relative to the repo root (default: data)",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to a SequenceModelDataBuilder YAML config defining target, splits, features, and sample weights",
    )

    # TODO: move to a lstm model config class
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden and cell state size (default: 64)")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of stacked LSTM layers (default: 1)")
    parser.add_argument("--dropout", type=float, default=0.0, help="Between-layer dropout probability (default: 0.0)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=32, help="Players per batch (default: 32)")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate (default: 0.001)")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu", help="Training device (default: cpu)")
    parser.add_argument(
        "--evaluate-test", action="store_true",
        help="Also evaluate the test split after training; omitted by default to keep test data held out",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    torch.manual_seed(42)

    data_builder = SequenceModelDataBuilder.from_config_file(data_dir=args.data_dir, config_path=args.config)
    train = data_builder.get_data_splits()[TRAIN_SPLIT]

    model = LSTMModel(
        input_size=train.X.shape[-1], hidden_size=args.hidden_size,
        num_layers=args.num_layers, dropout=args.dropout,
    )
    trainer = LSTMModelTrainer(
        model=model, data_builder=data_builder, preprocessor=SequencePreprocessor(), batch_size=args.batch_size,
        learning_rate=args.learning_rate, device=args.device, tracking_dir=args.tracking_dir,
    )

    run_id = trainer.setup_mlflow(extra_params={"config_path": args.config, "seed": 42, "epochs": args.epochs})
    logger.info("MLflow training run: %s", run_id)

    trainer.fit(epochs=args.epochs, run_id=run_id)
    if args.evaluate_test:
        inference = SequenceModelInference(
            pipeline=LSTMPipeline(model, trainer.preprocessor, trainer.prepared_data[TRAIN_SPLIT].feature_columns),
            data_builder=data_builder,
        )
        test_mse = inference.evaluate(split=TEST_SPLIT, run_id=run_id, step=trainer.epochs_trained)
        logger.info("LSTM test MSE: %s; RMSE: %s", test_mse, test_mse ** 0.5)


if __name__ == "__main__":
    main()
