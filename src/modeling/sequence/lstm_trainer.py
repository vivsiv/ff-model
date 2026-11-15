import argparse
import logging
from dataclasses import replace

import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.modeling.data_builder import EVAL_SPLIT, TEST_SPLIT, TRAIN_SPLIT
from src.modeling.sequence.data_builder import SequenceDataset, SequenceModelDataBuilder
from src.modeling.sequence.lstm import LSTMModel

logger = logging.getLogger(__name__)


class LSTMModelTrainer:
    """Trains and evaluates an LSTMModel on a builder's sequence splits,
    scoring only steps selected by loss_mask.

    - Mean imputation and scaling are fitted on real training inputs on the first fit call.
    - Every split uses the same fitted preprocessing; real prepared steps must be finite.
    - NaN padding is allowed, and the builder's cached datasets are left unchanged.
    - The trainer retains the dataset's global padding rather than padding per batch.
    """

    def __init__(
        self,
        model: LSTMModel,
        data_builder: SequenceModelDataBuilder,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        device: str = "cpu",
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.data_builder = data_builder
        self.data = data_builder.get_data_splits()
        self.target = data_builder.target
        self.preprocessor = Pipeline([
            ("imputer", SimpleImputer(strategy="mean", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ])
        self.prepared_data: dict[str, SequenceDataset] | None = None
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def fit(
        self,
        epochs: int = 20,
        evaluate_each_epoch: bool = True,
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
        """
        if epochs < 1:
            raise ValueError("epochs must be positive.")
        prepared = self._prepare_data()
        train = prepared[TRAIN_SPLIT]
        train_loader = self._make_loader(train, training=True)

        eval_loader = None
        eval_data = prepared[EVAL_SPLIT]
        if evaluate_each_epoch:
            if not eval_data.loss_mask.any():
                raise ValueError("Cannot evaluate an empty 'eval' split with no scored steps.")
            eval_loader = self._make_loader(eval_data, training=False)

        history = []
        for epoch in range(1, epochs + 1):
            metrics = {"epoch": epoch, "train_mse": self._run_epoch(train_loader, training=True)}
            if eval_loader is not None:
                metrics["eval_mse"] = self._run_epoch(eval_loader, training=False)
            history.append(metrics)
            logger.info("LSTM epoch metrics: %s", metrics)

        return history

    def evaluate(self, split: str = EVAL_SPLIT) -> float:
        """Return unweighted eval/test MSE using preprocessing learned by fit()."""
        if split not in (EVAL_SPLIT, TEST_SPLIT):
            raise ValueError("Evaluation split must be 'eval' or 'test'.")

        if self.prepared_data is None:
            raise ValueError("Preprocessing has not been fitted; call fit() before evaluate().")
        data = self.prepared_data[split]
        if not data.loss_mask.any():
            raise ValueError(f"Cannot evaluate an empty '{split}' split with no scored steps.")

        return self._run_epoch(self._make_loader(data, training=False), training=False)

    def _prepare_data(self) -> dict[str, SequenceDataset]:
        """Fit preprocessing once and cache transformed copies of all raw splits.

        Include real history even when excluded from loss; exclude padding from
        fitting and transformation. Entirely missing training features are imputed
        to zero without dropping their columns. Targets are not scaled.
        """
        if self.prepared_data is not None:
            return self.prepared_data
        train = self.data[TRAIN_SPLIT]
        training_inputs = train.X[train.real_step_mask].detach().cpu().numpy()
        self.preprocessor.fit(training_inputs)
        self.prepared_data = {
            split: self._transform_dataset(data, split) for split, data in self.data.items()
        }
        return self.prepared_data

    def _transform_dataset(self, data: SequenceDataset, split: str) -> SequenceDataset:
        """Transform only real features, preserving padding and all other fields."""
        real_mask = data.real_step_mask
        X = data.X.clone()
        if real_mask.any():
            inputs = data.X[real_mask].detach().cpu().numpy()
            transformed = self.preprocessor.transform(inputs)
            X[real_mask] = torch.as_tensor(transformed, dtype=X.dtype, device=X.device)
        if not torch.isfinite(X[real_mask]).all():
            raise ValueError(f"{split}: Real input steps must be finite after preprocessing.")
        return replace(data, X=X)

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


def main():
    parser = argparse.ArgumentParser(description="Fits and evaluates an LSTM model against a gold dataset")
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Parent directory containing gold data, relative to the repo root (default: data)",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to a SequenceModelDataBuilder YAML config defining target, splits, features, and sample weights",
    )
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden and cell state size (default: 64)")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of stacked LSTM layers (default: 1)")
    parser.add_argument("--dropout", type=float, default=0.0, help="Between-layer dropout probability (default: 0.0)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=32, help="Players per batch (default: 32)")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate (default: 0.001)")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu", help="Training device (default: cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for initialization and batch shuffling (default: 42)")
    parser.add_argument(
        "--evaluate-test", action="store_true",
        help="Also evaluate the test split after training; omitted by default to keep test data held out",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    torch.manual_seed(args.seed)
    data_builder = SequenceModelDataBuilder.from_config_file(data_dir=args.data_dir, config_path=args.config)
    train = data_builder.get_data_splits()[TRAIN_SPLIT]
    model = LSTMModel(
        input_size=train.X.shape[-1], hidden_size=args.hidden_size,
        num_layers=args.num_layers, dropout=args.dropout,
    )
    trainer = LSTMModelTrainer(
        model=model, data_builder=data_builder, batch_size=args.batch_size,
        learning_rate=args.learning_rate, device=args.device,
    )
    trainer.fit(epochs=args.epochs)
    if args.evaluate_test:
        test_mse = trainer.evaluate(split=TEST_SPLIT)
        logger.info("LSTM test MSE: %s; RMSE: %s", test_mse, test_mse ** 0.5)


if __name__ == "__main__":
    main()
