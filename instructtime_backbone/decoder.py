import math
import torch
import torch.nn as nn


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


# ------- ts decoder_layers -------
class SelfAttnDecoder(nn.Module):
    def __init__(
        self,
        ts_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        nhead: int = 8,
        num_layers: int = 8,
        ffn_mult: int = 1,
        diffusion_steps: int = 0,
        diff_txt_proj: bool = True,
        p: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim or output_dim
        self.p = p
        if hidden_dim is not None and hidden_dim != output_dim:
            self.proj_ts = nn.Linear(output_dim, self.hidden_dim)
            self.proj_text = nn.Linear(output_dim, self.hidden_dim)
        self.pos_encoder = PositionalEncoding(self.hidden_dim)
        self.blocks = nn.Sequential(
            *[
                SelfAttnBlock(
                    width=self.hidden_dim,
                    heads=nhead,
                    drop=0.0,
                    ffn_mult=ffn_mult,
                )
                for _ in range(num_layers)
            ]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, ts_dim),
        )
        self.diffusion_steps = diffusion_steps
        if diffusion_steps > 0:
            self.diffusion_tail = DiffusionRefiner(
                ts_dim=ts_dim,
                txt_dim=output_dim,
                n_steps=diffusion_steps,
                diff_txt_proj=diff_txt_proj
            )

    @staticmethod
    def _as_sequence(x: torch.Tensor) -> torch.Tensor:
        """Ensure input is 3D: [B, L, E]."""
        return x.unsqueeze(1) if x.dim() == 2 else x

    def forward(self, ts_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ts_emb: [B, E] - time series embedding
            txt_emb: [B, E] - text embedding
        Returns:
            ts_hat: [B, ts_dim] - reconstructed time series
        """

        if self.training:
            # w ~ Bernoulli(p) per batch element
            w = torch.distributions.Bernoulli(self.p).sample((ts_emb.size(0), 1)).to(device=ts_emb.device, dtype=ts_emb.dtype) # (B, 1)
            ts_emb = w * ts_emb + (1.0 - w) * txt_emb

        if hasattr(self, "proj_ts"):
            tgt = self.proj_ts(ts_emb)
            memory = self.proj_text(txt_emb)
        else:
            tgt, memory = ts_emb, txt_emb
        tgt = self._as_sequence(tgt)
        memory = self._as_sequence(memory)
        # self-attention
        tokens = torch.cat([tgt, memory], dim=1)
        tokens = self.pos_encoder(tokens)
        h = self.blocks(tokens)
        ts_hat = self.out(h[:, 0])
        if self.diffusion_steps > 0:
            ts_hat = self.diffusion_tail(ts_hat, txt_emb)
        return ts_hat

