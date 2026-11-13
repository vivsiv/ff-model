import logging

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.modeling.data_builder import EVAL_SPLIT, TEST_SPLIT, TRAIN_SPLIT
from src.modeling.sequence.data_builder import SequenceDataset, SequenceModelDataBuilder
from src.modeling.sequence.lstm import LSTMModel

logger = logging.getLogger(__name__)


class LSTMModelTrainer:
    """Trains and evaluates an LSTMModel on a builder's sequence splits,
    scoring only steps selected by loss_mask.

    - Preprocessing is not performed here; real input steps must already be finite.
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
        Optionally score its evaluation split each epoch, if it has scored steps.

        - Training applies sampling weights when supplied by scaling error by sample weight.
        - Reported epoch MSE accounts for sampling weight by aggregating squared errors and weights across all batches in an epoch.
        - Repeated calls to fit from the same Trainer object continue training where the last fit() call left off.
        """
        if epochs < 1:
            raise ValueError("epochs must be positive.")
        train = self.data[TRAIN_SPLIT]
        train_loader = self._make_loader(train, training=True)

        eval_loader = None
        eval_data = self.data[EVAL_SPLIT]
        # A correctly built eval split can be empty when no seasons are reserved for it.
        if evaluate_each_epoch and eval_data.loss_mask.any():
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
        """Return unweighted eval/test MSE without updating model parameters."""
        if split not in (EVAL_SPLIT, TEST_SPLIT):
            raise ValueError("Evaluation split must be 'eval' or 'test'.")

        data = self.data[split]
        if not data.loss_mask.any():
            raise ValueError(f"Cannot evaluate an empty '{split}' split with no scored steps.")

        return self._run_epoch(self._make_loader(data, training=False), training=False)

    # TODO: in tabular models the trainer imputed missing/nan values because it was all part of the same pipeline. Here we need a custom preprocessor.
    # Decide if it goes here or in the builder (likely the builder). If it goes here the builder can't fail to build a dataset with NaN/missing feature values.
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
                loss = weighted_error / sample_weight_sum

                if not torch.isfinite(loss):
                    raise ValueError("Non-finite loss encountered.")

                if training:
                    loss.backward()
                    self.optimizer.step()

                total_error += weighted_error.detach().item()
                total_sample_weight += sample_weight_sum.item()

        return total_error / total_sample_weight
