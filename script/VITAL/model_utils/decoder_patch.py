import torch
import torch.nn as nn
from .helper import PositionalEncoding, SelfAttnBlock, CrossAttnBlock


class _SelfAttnTiny(nn.Module):
    def __init__(self, width: int, heads: int = 8, drop: float = 0.):
        super().__init__()
        self.ln = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=drop,
                                          batch_first=True)

    def forward(self, x):                               # x: [B, 1, D]
        return x + self.attn(self.ln(x), self.ln(x), self.ln(x),
                             need_weights=False)[0]
class PatchAttnDecoder(nn.Module):
    """
    Per-slice decoder with L stacked SelfAttnBlock layers,
    followed by mean-pool → LayerNorm → linear projection to ts_dim.
    """
    def __init__(
        self,
        piece_dim: int,
        ts_dim:   int,
        num_layers: int = 4,
        dropout:  float = 0.0
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                _SelfAttnTiny(
                    width=piece_dim,
                    drop=dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(piece_dim),
            nn.Linear(piece_dim, ts_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h : [B, piece_dim]
        returns: [B, ts_dim]
        """
        h = self.blocks(h)            # [B, piece_dim]
        ts_hat = self.out(h)          # [B, ts_dim]
        return ts_hat


class PatchConvDecoder(nn.Module):
    def __init__(
        self,
        piece_dim: int,
        ts_dim: int,
        hidden_dim: int | None = None,
        kernel_size: int = 1,
        mlptail: bool = False,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = ts_dim
        
        self.deconv = nn.ConvTranspose1d(
            in_channels=piece_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            bias=True,
        )
        self.flatten_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=ts_dim,
            kernel_size=kernel_size,      # consumes the whole length-16 window
            stride=1,
            padding=0,
            bias=True,
        )

        self.mlp = (
            nn.Sequential(
                nn.LayerNorm(ts_dim),
                nn.Linear(ts_dim, ts_dim),
                nn.GELU(),
                nn.Linear(ts_dim, ts_dim),
            )
            if mlptail else nn.Identity()
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h : [B, piece_dim]
        returns: [B, ts_dim]
        """
        if h.dim() == 2:
            h = h.unsqueeze(-1)                 # [B, piece_dim, 1]
        z = self.deconv(h)                  # [B, ts_dim, 16]
        z = self.flatten_conv(z)            # [B, ts_dim, 1]
        ts_hat = z.squeeze(-1)              # [B, ts_dim]
        return self.mlp(ts_hat)


class PatchMLPDecoder(nn.Module):
    def __init__(
        self,
        piece_dim: int,
        ts_dim: int,
        hidden_dim: int | None = None,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = ts_dim
        
        self.mlp = nn.Sequential(
                nn.LayerNorm(piece_dim),
                nn.Linear(piece_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, ts_dim),
            )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h : [B, piece_dim]
        returns: [B, ts_dim]
        """
        return self.mlp(h)


class _SimpleDiffusionTail(nn.Module):
    """
    Adds Gaussian noise x̃ = x + σ·ε during training and learns
    to predict (and subtract) that noise.  At eval time it is a no-op.

    Args
    ----
    dim          : feature dimension (ts_dim)
    max_sigma    : largest noise std-dev  (defaults: 0.2)
    hidden_mult  : width multiplier for the tiny denoiser MLP
    """
    def __init__(self, dim: int, max_sigma: float = 0.2, hidden_mult: int = 2):
        super().__init__()
        self.max_sigma = max_sigma
        hdim = dim * hidden_mult
        self.denoiser = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hdim),
            nn.GELU(),
            nn.Linear(hdim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:           # [B, dim]
        if not self.training:
            return x                                              # no noise at eval

        # sample noise scale σ ∈ (0, max_sigma]
        sigma = torch.rand(x.size(0), 1, device=x.device) * self.max_sigma

        # corrupt x
        noise = torch.randn_like(x)
        x_noisy = x + sigma * noise

        # predict the noise and reconstruct
        pred_noise = self.denoiser(x_noisy)
        x_denoised = x_noisy - pred_noise                         # ≈ x

        return x_denoised

class PatchDiffDecoder(nn.Module):
    def __init__(self, piece_dim: int, ts_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or ts_dim

        self.mlp = nn.Sequential(
            nn.LayerNorm(piece_dim),
            nn.Linear(piece_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, ts_dim),
        )

        self.diffusion_tail = _SimpleDiffusionTail(ts_dim, max_sigma=0.2)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h : [B, piece_dim]
        returns: [B, ts_dim]         (and optionally extra tensors when training)
        """
        x = self.mlp(h)                              # clean prediction

        # diffusion_tail handles train / eval branching
        return self.diffusion_tail(x)



class PatchResDecoder(nn.Module):
    def __init__(
        self,
        piece_dim: int,
        ts_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.5,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = ts_dim

        self.proj_in = nn.Sequential(
            nn.LayerNorm(piece_dim),
            nn.Linear(piece_dim, hidden_dim),
        )

        # Two residual MLP blocks
        self.block1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ts_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h : [B, piece_dim]
        returns: [B, ts_dim]
        """
        x = self.proj_in(h)          # [B, hidden_dim]

        x = x + self.block1(x)       # Residual block 1
        x = x + self.block2(x)       # Residual block 2

        return self.out(x)           # [B, ts_dim]

class PatchXDecoder(nn.Module):
    def __init__(
        self,
        piece_dim: int,
        ts_dim: int,
        num_layers: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.xattn_blocks = nn.Sequential(
            *[CrossAttnBlock(piece_dim, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(piece_dim),
            nn.Linear(piece_dim, ts_dim),
        )

    def forward(self, ts_piece: torch.Tensor, txt_piece: torch.Tensor) -> torch.Tensor:
        """
        ts_piece : [B, piece_dim]
        txt_piece : [B, piece_dim]
        returns: [B, ts_dim]
        """
        q = ts_piece.unsqueeze(1)      # [B, 1, piece_dim]
        kv = txt_piece.unsqueeze(1)    # [B, 1, piece_dim]
        for blk in self.xattn_blocks:
            q = blk(q, kv)             # still [B, 1, piece_dim]
        q = q.squeeze(1)               # [B, piece_dim]
        return self.mlp(q)             # [B, ts_dim]


class PatchX2Decoder(nn.Module):
    def __init__(
        self,
        piece_dim: int,
        ts_dim: int,
        num_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "ts2txt": CrossAttnBlock(piece_dim, nhead, dim_feedforward, dropout),
                "txt2ts": CrossAttnBlock(piece_dim, nhead, dim_feedforward, dropout),
            }) for _ in range(num_layers)
        ])
        self.mlp = nn.Sequential(
            nn.LayerNorm(piece_dim),
            nn.Linear(piece_dim, ts_dim),
        )

    def forward(self, ts_piece: torch.Tensor, txt_piece: torch.Tensor) -> torch.Tensor:
        """
        ts_piece : [B, piece_dim]
        txt_piece : [B, piece_dim]
        returns: [B, ts_dim]
        """
        q = ts_piece.unsqueeze(1)      # [B, 1, piece_dim]
        kv = txt_piece.unsqueeze(1)    # [B, 1, piece_dim]
        for layer in self.layers:
            q = layer["ts2txt"](q, kv)   # TS attends to TXT
            kv = layer["txt2ts"](kv, q)   # TXT attends back to TSs    
        q = q.squeeze(1)               # [B, piece_dim]
        return self.mlp(q)             # [B, ts_dim]

class PatchSelfDecoder(nn.Module):
    def __init__(
        self,
        piece_dim: int,
        ts_dim: int,
        num_layers: int = 6,
        nhead: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(piece_dim)
        self.blocks = nn.Sequential(
            *[SelfAttnBlock(piece_dim, nhead, ffn_mult, dropout) for _ in range(num_layers)]
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(piece_dim),
            nn.Linear(piece_dim, ts_dim),
        )

    def forward(self, ts_piece: torch.Tensor, txt_piece: torch.Tensor) -> torch.Tensor:
        """
        ts_piece : [B, piece_dim]
        txt_piece : [B, piece_dim]
        returns: [B, ts_dim]
        """
        ts = ts_piece.unsqueeze(1)      # [B, 1, piece_dim]
        tx = txt_piece.unsqueeze(1)    # [B, 1, piece_dim]
        tokens = torch.cat([ts, tx], dim=1)
        tokens = self.pos_encoder(tokens)
        h = self.blocks(tokens)
        ts_hat = self.mlp(h[:, 0])
        return ts_hat             # [B, ts_dim]

