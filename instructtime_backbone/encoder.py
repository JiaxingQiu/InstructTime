# from config import *
import torch
import torch.nn as nn

# -------- TS encoder --------
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



class PatchCNNTSEncoder(nn.Module):
    """
    Multi-resolution CNN encoder without attention.
    Each kernel-size branch produces a slice of the final embedding;
    the slices are concatenated to form the full `output_dim`.

    Args
    ----
    ts_dim : int
        Length (temporal dimension) of the input time series.
    output_dim : int
        Desired length of the final embedding. Must be divisible by len(kernel_sizes).
    kernel_sizes : list[int]
        Kernel sizes for the parallel CNN branches.
    hidden_num_channel : int
        Channel width for each CNNEncoder block.
    dropout : float
        Dropout rate used inside each CNN branch.
    """
    def __init__(
        self,
        ts_dim: int,
        output_dim: int,
        fracs: list[float] = [1, 2/3, 1/2, 1/3, 1/4, 1/6, 1/8, 1/10],#[1, 1/2, 1/4, 1/8],#[2/3, 1/2, 1/5, 1/10],
        hidden_num_channel: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()

        kernel_sizes = [int(ts_dim * frac) for frac in fracs]
        n_kernels = len(kernel_sizes)
        assert (
            output_dim % n_kernels == 0
        ), f"output_dim ({output_dim}) must be divisible by number of kernels ({n_kernels})."
        piece_dim = output_dim // n_kernels  # dimensional share per branch

        # One CNN branch per kernel size, each outputting `piece_dim`
        self.cnns = nn.ModuleList(
            [
                CNNEncoder(
                    ts_dim,
                    piece_dim,
                    num_channels=[hidden_num_channel],
                    kernel_size=ks,
                    dropout=dropout,
                )
                for ks in kernel_sizes
            ]
        )

        # Optional final LayerNorm for stability
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape [batch, ts_dim] or [batch, C, ts_dim] depending on CNNEncoder spec.

        Returns
        -------
        torch.Tensor
            Embedding of shape [batch, output_dim], where successive
            segments correspond to increasing kernel sizes.
        """
        # Collect slice-embeddings from each branch → list of [B, piece_dim]
        pieces = [cnn(x) for cnn in self.cnns]

        # Concatenate along feature dimension → [B, output_dim]
        embedding = torch.cat(pieces, dim=-1)

        # Normalise and return
        return self.layer_norm(embedding)

# ------- Text encoder -------
class _PatchMLP(nn.Module):
    def __init__(self, text_dim: int, piece_dim: int, hidden_mult: float = 0.5, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, int(text_dim * hidden_mult)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(text_dim * hidden_mult), piece_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:         # x: [B, text_dim]
        return self.mlp(x)                                      # [B, piece_dim]

class PatchMLPTextEncoder(nn.Module):
    def __init__(
        self,
        text_dim: int,
        output_dim: int,
        n_slices: int = 8,
        hidden_mult: float = 1.0, 
        dropout: float = 0.0,
    ):
        super().__init__()
        assert output_dim % n_slices == 0
        piece_dim = output_dim // n_slices
        # one MLP per slice (no weight sharing)
        self.slices = nn.ModuleList(
            [
                _PatchMLP(text_dim, piece_dim, hidden_mult, dropout)
                for _ in range(n_slices)
            ]
        )
    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        pieces = [mlp(text_tokens) for mlp in self.slices]  # list of (B, piece_dim)
        tx_emb = torch.cat(pieces, dim=-1)                 # (B, output_dim)
        return tx_emb
