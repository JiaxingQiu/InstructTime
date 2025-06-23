import torch, math
import torch.nn as nn

class CrossAttnBlock(nn.Module):
    def __init__(self, width: int, heads: int = 8, dim_feedforward: int = 1024, drop: float = 0.):
        super().__init__()
        self.ln_q = nn.LayerNorm(width)
        self.ln_kv = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(
            width, heads, dropout=drop, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, dim_feedforward),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim_feedforward, width),
        )

    def forward(self, q, kv):
        # q : [B, 1, H]     kv : [B, 1, H]
        attn_out = self.attn(
            self.ln_q(q),
            self.ln_kv(kv),
            self.ln_kv(kv),
            need_weights=False,
        )[0]
        x = q + attn_out                   # residual
        x = x + self.mlp(x)                # FFN with residual
        return x


class SelfAttnBlock(nn.Module):
    """(Multi-head self-attention + FFN) with residual & layernorm."""
    def __init__(self, width: int, heads: int = 8, ffn_mult = 1, drop: float = 0.):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=drop, batch_first=True)
        self.ln2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * ffn_mult),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(width * ffn_mult, width),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)[0]
        x = x + self.ffn(self.ln2(x))
        return x


class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional information to a tensor of shape
    [batch_size, seq_len, d_model]. d_model = the number of features per position
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)                   # [max_len, d_model]
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)         # even indices
        pe[:, 1::2] = torch.cos(position * div_term)         # odd  indices
        pe = pe.unsqueeze(0)                                 # [1, max_len, d_model]

        self.register_buffer("pe", pe)                       # not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]                         # broadcast on batch dim
        return self.dropout(x)


class DiffusionRefiner(nn.Module):
    """
    Lightweight denoising refiner that improves ts_hat using iterative residual steps.
    Mimics diffusion-style denoising by progressively refining x₀ + noise.
    """
    def __init__(self, ts_dim, txt_dim, hidden_dim=768, n_steps=4, diff_txt_proj = True):
        
        super().__init__()
        self.n_steps = n_steps
        self.diff_txt_proj = diff_txt_proj
        if diff_txt_proj:
            self.proj_txt = nn.Linear(txt_dim, ts_dim)
            txt_dim = ts_dim
        self.step_blocks = nn.ModuleList([
            nn.Sequential(
                # nn.LayerNorm(ts_dim + txt_dim),
                nn.Linear(ts_dim + txt_dim, hidden_dim),  # [x || txt_emb]
                nn.GELU(),
                nn.Linear(hidden_dim, ts_dim)
            )
            for _ in range(n_steps)
        ])

    def forward(self, coarse_ts: torch.Tensor, txt_emb: torch.Tensor, noise_frac: float = 0.1):
        std = coarse_ts.detach().flatten(1).std(dim=1, keepdim=True)
        x = coarse_ts + noise_frac * std * torch.randn_like(coarse_ts)

        if self.diff_txt_proj:
            txt_emb = self.proj_txt(txt_emb)
        for block in self.step_blocks:
            h = torch.cat([x, txt_emb], dim=-1)  # [B, ts_dim + txt_dim]
            dx = block(h)
            x = x - dx  # residual denoising
        return x



class AddChannelDim(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.unsqueeze(1)

class CNNEncoder(nn.Module):
    def __init__(self, ts_dim, output_dim, 
                 num_channels=[64, 64, 128, 256], 
                 kernel_size=5, 
                 dropout=0.2):
        """
        CNN encoder for time series.
        
        Args:
            ts_dim (int): Input time series length
            hidden_dim (int): Final hidden dimension
            num_channels (list): Number of channels for each conv layer
            kernel_size (int): Kernel size for conv layers
            dropout (float): Dropout rate
        """
        super().__init__()

        self.ts_dim = ts_dim
        self.output_dim = output_dim
        
        # layers = [Lambda(lambda x: x.unsqueeze(1))]  # Add channel dimension
        layers = [AddChannelDim()]  # Add channel dimension
        in_channels = 1
        
        # Add conv blocks
        for out_channels in num_channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        layers.append(nn.Flatten())
        
        # Calculate output dimension
        with torch.no_grad():
            x = torch.zeros(2, ts_dim)
            for layer in layers:
                x = layer(x)
            conv_out_dim = x.shape[1]
        
        # Add final linear projection to match output_dim
        layers.append(nn.Linear(conv_out_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encoder(x)

# # CNN Encoder
# cnn = CNNEncoder(
#     ts_dim=300,
#     output_dim=128,
#     num_channels=[32, 64, 128],  # Three conv layers
#     kernel_size=5,
#     dropout=0.2
# )
# model = GeneralBinaryClassifier(cnn)
