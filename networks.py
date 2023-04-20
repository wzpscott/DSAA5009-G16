import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, d_input, d_hidden, d_output, n_layers, bidirectional, agg_func='mean'):
        super().__init__()
        self.rnn = nn.LSTM(d_input, d_hidden, n_layers, batch_first=True, bidirectional=bidirectional)
        self.linear = nn.Linear(d_hidden, d_output)
        self.agg_func = agg_func
        
    def forward(self, x):
        """
        x: input tensor, [B, T, d_input]
        """
        x, _ = self.rnn(x)
        if self.agg_func == 'mean':
            x = x.mean(dim=1)
        elif self.agg_func == 'sum':
            x = x.sum(dim=1)
        x = self.linear(x)
        return x