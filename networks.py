import torch
import torch.nn as nn

class CNN_LSTM(nn.Module):
    def __init__(self, d_input, d_hidden, d_output, n_layers, bidirectional, kernel_size=5, agg_func='mean'):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(d_input, d_hidden//8, kernel_size=kernel_size),
            # nn.BatchNorm1d(d_hidden//4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(d_hidden//8, d_hidden//4, kernel_size=kernel_size),
            # nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.rnn = nn.LSTM(d_hidden//4, d_hidden, n_layers, batch_first=True, bidirectional=bidirectional)
        self.linear = nn.Linear(d_hidden, d_output)
        self.agg_func = agg_func
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        if self.agg_func == 'mean':
            x = x.mean(dim=1)
        elif self.agg_func == 'sum':
            x = x.sum(dim=1)
        x = self.linear(x)
        return x
    
class LSTM(nn.Module):
    def __init__(self, d_input, d_hidden, d_output, n_layers, bidirectional, agg_func='mean'):
        super().__init__()
        self.rnn = nn.LSTM(d_input, d_hidden, n_layers, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            d_hidden *= 2
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

class MLP(nn.Module):
    def __init__(self, d_input, d_hidden, d_output, n_layers):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.ReLU(),
            *[
                nn.Sequential(nn.Linear(d_hidden, d_hidden), nn.ReLU()) for _ in range(n_layers-2)
            ],
            nn.Linear(d_hidden, d_output),
        )
        
    def forward(self, x):
        return self.net(x)
    
class CNN(nn.Module):
    def __init__(self, d_input, d_hidden, d_output, n_layers, kernel_size=5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(d_input, d_hidden, kernel_size=5, padding='same'),
            # nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Conv1d(d_hidden, d_hidden, kernel_size=5, padding='same'),
                # nn.BatchNorm1d(d_hidden),
                nn.ReLU(),
            ) for _ in range(n_layers)]
        )
        self.linear = nn.Linear(d_hidden, d_output)
    def forward(self, x):
        # x: [N, L, C]
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.mean(dim=-1)
        x = self.linear(x)
        return x