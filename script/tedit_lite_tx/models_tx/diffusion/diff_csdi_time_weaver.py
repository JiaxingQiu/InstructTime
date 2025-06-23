import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from linear_attention_transformer import LinearAttentionTransformer


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def get_linear_trans(heads=8,layers=1,channels=64,localheads=0,localwindow=0):

  return LinearAttentionTransformer(
        dim = channels,
        depth = layers,
        heads = heads,
        max_seq_len = 256,
        n_local_attn_heads = 0, 
        local_attn_window_size = 0,
    )

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class Diff_CSDI_TimeWeaver(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)
        
        # if "attr_proj_type" in config and config["attr_proj_type"] == "avg":
        #     self.attr_projector = AttrProjectorAvg(
        #         dim_in=config["attr_dim"], 
        #         dim_hid=config["channels"], 
        #         dim_out=config["channels"],
        #     )
        # else:
        self.attr_projector = AttrProjector(
            n_attrs=config["n_attrs"],
            dim_in=config["attr_dim"], 
            dim_hid=config["channels"], 
            dim_out=config["channels"], 
            n_heads=8, 
            n_layers=2)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, side_emb, attr_emb, diffusion_step):
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)
        x_in = x

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        attr_emb = self.attr_projector(attr_emb, length=L)
        attr_emb = attr_emb.permute(0,2,1)  # (B,L,C) -> (B,C,L)
        attr_emb = attr_emb.unsqueeze(2)  # (B,C,1,L)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x+attr_emb+x_in, side_emb, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, is_linear=False):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.side_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.is_linear = is_linear
        if is_linear:
            self.time_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
            self.feature_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
        else:
            self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)

        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if self.is_linear:
            y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, side_emb, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*(N+L))

        y = y.reshape(base_shape)
        y = y.reshape(B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, side_dim, _, _ = side_emb.shape
        side_emb = side_emb.reshape(B, side_dim, K * L)
        side_emb = self.side_projection(side_emb)  # (B,2*channel,K*L)
        y = y + side_emb

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)

        return (x + residual) / math.sqrt(2.0), skip


class AttrProjector_org(nn.Module):
    def __init__(self, n_attrs=3, dim_in=128, dim_hid=128, dim_out=128, n_heads=8, n_layers=2):
        super().__init__()
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.dim_out = dim_out

        self.proj_in = nn.Linear(n_attrs*self.dim_in, self.dim_hid)
        self.proj_mid = nn.Linear(self.dim_hid, self.dim_hid)
        self.proj_out = nn.Linear(self.dim_hid, self.dim_out)
        self.attns = nn.ModuleList(
            [torch.nn.MultiheadAttention(dim_hid, n_heads, batch_first=True) 
             for i in range(n_layers)])

    def forward(self, attr, length):
        B = attr.shape[0]
        attr = torch.reshape(attr, (B, -1))
        h = F.gelu(self.proj_in(attr))  # (B,d)
        h = F.gelu(self.proj_mid(h))
        h = h.unsqueeze(1)  # (B,d) -> (B,1,d)
        h = h.expand([-1, length, -1])  # (B,L,d)

        out = self.proj_out(h)
        return out


class AttrProjector_attn(nn.Module):
    def __init__(
        self,
        n_attrs=1,
        dim_in=768,
        dim_hid=128, 
        dim_out=768,
        n_heads=1,
        n_layers=2,
        dropout=0.0,
    ):
        super().__init__()

        self.n_attrs = n_attrs
        self.text_dim = dim_in
        self.output_dim = dim_out

        self.attn = nn.MultiheadAttention(
            embed_dim=dim_in,
            num_heads=n_heads,
            batch_first=True,
        )

        self.query = nn.Parameter(torch.randn(1, 1, dim_in))  # (1, 1, D)

        self.output_proj = nn.Linear(dim_in, dim_out)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, attr: torch.Tensor, length: int) -> torch.Tensor:
        B = attr.size(0)
        # reshape flat input → (B, n_attrs, dim_in)
        tokens = attr.view(B, self.n_attrs, self.text_dim)
        # trainable query (B, 1, dim_in)
        q = self.query.expand(B, -1, -1)
        # attention pooling: query attends to attribute tokens
        pooled, _ = self.attn(query=q, key=tokens, value=tokens)  # (B, 1, D)
        pooled = pooled.squeeze(1)  # (B, D)
        # project to output dim
        pooled = self.output_proj(pooled)
        pooled = self.gelu(pooled)
        pooled = self.dropout(pooled)
        # expand to (B, L, D)
        out = pooled.unsqueeze(1).expand(-1, length, -1)
        return out


class AttrProjectorAvg(nn.Module):
    def __init__(self, dim_in=128, dim_hid=128, dim_out=128):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.dim_out = dim_out

        self.proj_out = nn.Linear(self.dim_hid, self.dim_out)

    def forward(self, attr, length):
        B = attr.shape[0]
        h = torch.mean(attr, dim=1, keepdim=True)  # (B,1,d)
        h = h.expand([-1, length, -1])  # (B,L,d)

        out = self.proj_out(h)
        return out

    

class _PatchMLP(nn.Module):
    def __init__(self, text_dim: int, piece_dim: int, hidden_mult: float = 1.0, dropout: float = 0.0):
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

class AttrProjector(nn.Module):
    def __init__(
        self,
        n_attrs=1,
        dim_in=768,
        dim_hid=128, 
        dim_out=768,
        n_heads=1,
        n_layers=2,
        dropout=0.0,
    ):
        super().__init__()

        self.n_attrs = n_attrs
        self.text_dim = dim_in
        self.output_dim = dim_out
        self.n_heads = n_heads  # kept for compatibility
        self.n_layers = n_layers  # kept for compatibility

        self.n_slices = 4  # fixed or could be passed as argument if needed
        assert dim_out % self.n_slices == 0
        piece_dim = dim_out // self.n_slices

        self.slices = nn.ModuleList(
            [
                _PatchMLP(dim_in, piece_dim, hidden_mult=1.0, dropout=dropout)
                for _ in range(self.n_slices)
            ]
        )

    def forward(self, attr: torch.Tensor, length: int) -> torch.Tensor:
        B = attr.size(0)
        attr = attr.float()         
        tokens = attr.view(B, self.n_attrs, self.text_dim)  # (B, n_attrs, dim_in)
        pooled = tokens.mean(dim=1)                         # (B, dim_in)

        pieces = [mlp(pooled) for mlp in self.slices]       # list of (B, piece_dim)
        out = torch.cat(pieces, dim=-1)                     # (B, dim_out)
        out = out.unsqueeze(1).expand(-1, length, -1)       # (B, L, dim_out)
        return out

