import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.modeling.sequence.data_builder import SequenceDataset


class LSTMModel(nn.Module):
    """
    A unidirectional LSTM with a scalar regression head, for causal target prediction.
    Produces a prediction value for the target for each real step in the sequence.
    """

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        """
        Args:
            input_size (Required): The number of features in a step.
            hidden_size (default 64): The number of features in the hidden and cell states of an lstm cell.
                (input_size) inputs -> (hidden_size) activations -> (hidden_size) outputs
            num_layers (default 1): The number of lstm layers to have in the network.
            dropout (default 0.0): Zeroes n pct of outputs of hidden layers before they become input to the next hidden layer
                - Not applicable to the final layer, and set to 0 automatically for networks with 1 layer.
                - Only used during training
        PyTorch specific params:
            batch_first: If the data is ordered [BATCH, STEPS, FEATURES], always True.
            bidirectional: If the lstm is bidirectional (it processes the sequence forward and backward), always False.
        """

        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0, 
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, X: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Generate the prediction value from each step's history without reading future steps.
        Produces predictions for an entire batch of sequences in parallel (can be accelerated via gpu).

        Use Pytorch's PackedSequence to avoid passing padding values into the model.
            - Sorts sequences by descending length and tracks how many sequences are "alive" at each step.
            - Only updates hidden states for those sequences that are still "alive" at the current step.
            - Pad the Packed output sequence back to the size of the original padded input before sending it to the regression head.


        Args:
            X: [N, L, F] The data tensor from a SequenceDataset.
            lengths: [N] The lengths tensor from a SequenceDataset.

        Returns:
            [N, L] predictions, aligned with SequenceDataset targets and loss masks.
            Padded steps are zeroed and must be excluded from loss using the loss mask.
        """
        packed = pack_padded_sequence(X, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=X.shape[1])

        # output: [N, L, H], self.head(output): [N, L, 1], predictions: [N, L]
        predictions = self.head(output).squeeze(-1) 
        real_step_mask = SequenceDataset.build_real_step_mask(lengths.to(X.device), X.shape[1])

        return predictions.masked_fill(~real_step_mask, 0.0)
