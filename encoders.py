import torch
import torch.nn as nn

class IdentityEncoder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x
    
class PositionalEncoder(nn.Module):
    
    def __init__(self, d_input, n_freqs):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.d_output = d_input * (1 + 2 * self.freqs)
        self.embed_fns = [lambda x: x]
        
        freq_bands = torch.linspace(2.**0., 2.**(self.n_bands - 1), self.n_bands)
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))
            
    def forward(self, x):
        ret = torch.concat([fn(x) for fn in self.embed_fns], dim=-1)
        return ret