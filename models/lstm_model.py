import torch
import torch.nn as nn

class MultiOutputLSTM(nn.Module):
    """
    A multi-output LSTM model for air quality prediction.
    Predicts a vector of pollutant concentrations.
    """
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(MultiOutputLSTM, self).__init__()
        
        self.output_size = output_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Pollutant-specific decoding heads; each head learns its own bias/scale
        self.heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(output_size)])

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        
        # Take the output from the last time step
        out = out[:, -1, :]

        # Run the shared representation through each pollutant head
        head_outputs = [head(out) for head in self.heads]
        out = torch.cat(head_outputs, dim=1)

        return out
