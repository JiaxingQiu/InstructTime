import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from linear_attention_transformer import LinearAttentionTransformer
import numpy as np

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu", batch_first=True
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

class ResidualBlock(nn.Module):
    def __init__(self, side_dim, attr_dim, channels, diffusion_embedding_dim, nheads, is_linear=False, is_attr_proj=False):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        if is_attr_proj:
            self.attr_projection = nn.Linear(attr_dim, channels)
        else:
            self.attr_projection = nn.Identity()
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


    def forward_time(self, y, base_shape, attention_mask=None):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(0, 2, 1), mask=attention_mask).permute(0, 2, 1)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape, attention_mask=None):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if self.is_linear:
            y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(0, 2, 1), mask=attention_mask).permute(0, 2, 1)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, side_emb, attr_emb, diffusion_emb, attention_mask=None):
        B, channel, K, L = x.shape
        base_shape = x.shape

        # concatenate attr_emb and x
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb  # (B,C,K*(N+1))

        y = self.forward_time(y, base_shape, attention_mask)
        y = self.forward_feature(y, base_shape, attention_mask)  # (B,channel,K*(N+L))

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

class TsPatchEmbedding(nn.Module):
    def __init__(self, L_patch_len, channels, d_model, dropout):
        super(TsPatchEmbedding, self).__init__()
        self.L_patch_len = L_patch_len
        self.padding_patch_layer = nn.ReplicationPad2d((0, L_patch_len, 0, 0))
        self.value_embedding = nn.Sequential(
            nn.Linear(L_patch_len*channels, d_model),
            nn.ReLU(),
        )

    def forward(self, x_in):
        if x_in.shape[-1] % self.L_patch_len:
            x = self.padding_patch_layer(x_in)
        else:
            x = x_in
        x = x.unfold(dimension=3, size=self.L_patch_len, step=self.L_patch_len) # (B, C, n_var, Nl, Pl)
        B, C, n_var, Nl, Pl = x.shape
        x = x.permute(0, 3, 4, 2, 1).contiguous().reshape(B, 1, Nl, Pl*n_var*C)
        x = self.value_embedding(x) # (B, 1, Nl, D)
        x = x.permute(0, 3, 1, 2).contiguous() # (B, D, 1, Nl)
        return x

class CondPatchEmbedding(nn.Module):
    def __init__(self, L_patch_len, channels, d_model, dropout):
        super(CondPatchEmbedding, self).__init__()
        self.L_patch_len = L_patch_len
        self.padding_patch_layer = nn.ReplicationPad2d((0, L_patch_len, 0, 0))
        self.value_embedding = nn.Sequential(
            nn.Linear(L_patch_len*channels, d_model),
        )

    def forward(self, x_in):
        if x_in.shape[-1] % self.L_patch_len:
            x = self.padding_patch_layer(x_in)
        else:
            x = x_in
        x = x.unfold(dimension=3, size=self.L_patch_len, step=self.L_patch_len) # (B, C, n_var, Nl, Pl)
        B, C, n_var, Nl, Pl = x.shape
        x = x.permute(0, 3, 4, 2, 1).contiguous().reshape(B, 1, Nl, Pl*n_var*C)
        x = self.value_embedding(x) # (B, 1, Nl, D)
        x = x.permute(0, 3, 1, 2).contiguous() # (B, D, 1, Nl)
        return x

class PatchDecoder(nn.Module):
    def __init__(self, L_patch_len, n_var, d_model, channels):
        super().__init__()
        self.L_patch_len = L_patch_len
        self.channels = channels
        self.n_var = n_var
        self.linear = nn.Linear(d_model, L_patch_len*n_var*channels)

    def forward(self, x):
        B, D, _, Nl = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()# (B, 1, Nl, D)
        x = self.linear(x) #(B, 1, Nl, Pl*n_var*C)
        x = x.reshape(B, Nl, self.L_patch_len, self.n_var, self.channels).permute(0, 4, 3, 1, 2).contiguous() #(B, C, K, Nl, Pl)
        x = x.reshape(B, self.channels, self.n_var, Nl*self.L_patch_len)
        return x

class Diff_CSDI_Patch(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.ts_downsample = TsPatchEmbedding(L_patch_len=config["L_patch_len"], channels=inputdim, d_model=self.channels, dropout=0)
        self.side_downsample = CondPatchEmbedding(L_patch_len=config["L_patch_len"], channels=config["side_dim"], d_model=config["side_dim"], dropout=0)
        self.patch_decoder = PatchDecoder(L_patch_len=config["L_patch_len"], n_var=config["n_var"], d_model=self.channels, channels=1)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    attr_dim=config["attr_dim"],
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
        x = self.ts_downsample(x)
        side_emb = self.side_downsample(side_emb)
        B, _, Nk, Nl = x.shape

        diffusion_emb = self.diffusion_embedding(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, side_emb, attr_emb, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, Nk * Nl)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = x.reshape(B, self.channels, Nk, Nl) # (B, channel, Nk, Nl)
        x = self.patch_decoder(x) 
        x = x[:, :, :, :L] # (B, 1, K, L)
        x = x.reshape(B, K, L)
        return x
    
class Diff_CSDI_MultiPatch(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]
        self.multipatch_num = config["multipatch_num"]
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.ts_downsample = nn.ModuleList([])
        self.side_downsample = nn.ModuleList([])
        self.patch_decoder = nn.ModuleList([])
        for i in range(self.multipatch_num):
            self.ts_downsample.append(TsPatchEmbedding(L_patch_len=config["L_patch_len"]**i, channels=inputdim, d_model=self.channels, dropout=0))
            self.side_downsample.append(CondPatchEmbedding(L_patch_len=config["L_patch_len"]**i, channels=config["side_dim"], d_model=config["side_dim"], dropout=0))
            self.patch_decoder.append(PatchDecoder(L_patch_len=config["L_patch_len"]**i, n_var=config["n_var"], d_model=self.channels, channels=1))
        self.multipatch_mixer = nn.Linear(self.multipatch_num, 1)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    attr_dim=config["attr_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x_raw, side_emb_raw, attr_emb_raw, diffusion_step):
        B, inputdim, K, L = x_raw.shape
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        all_out = []
        for i in range(self.multipatch_num):
            x = self.ts_downsample[i](x_raw)
            side_emb = self.side_downsample[i](side_emb_raw)
            attr_emb = attr_emb_raw

            B, _, Nk, Nl = x.shape
            skip = []
            for layer in self.residual_layers:
                x, skip_connection = layer(x, side_emb, attr_emb, diffusion_emb)
                skip.append(skip_connection)

            x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
            x = x.reshape(B, self.channels, Nk * Nl)
            x = self.output_projection1(x)  # (B,channel,Nk*Nl)
            x = F.relu(x)
            x = x.reshape(B, self.channels, Nk, Nl) # (B, channel, Nk, Nl)
            x = self.patch_decoder[i](x) 
            x = x[:, :, :, :L] # (B, 1, K, L)
            all_out.append(x)
        all_out = torch.cat(all_out, dim=1) # (B, M, K, L)
        all_out = self.multipatch_mixer(all_out.permute(0, 2, 3, 1).contiguous()) # (B, K, L, 1)
        all_out = all_out.reshape(B, K, L)
        return all_out

class Diff_CSDI_MultiPatch_Weaver_Parallel(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]
        self.multipatch_num = config["multipatch_num"]
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.attention_mask_type = config["attention_mask_type"]
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.ts_downsample = nn.ModuleList([])
        self.side_downsample = nn.ModuleList([])
        self.patch_decoder = nn.ModuleList([])
        for i in range(self.multipatch_num):
            self.ts_downsample.append(TsPatchEmbedding(L_patch_len=config["L_patch_len"]**i, channels=inputdim, d_model=self.channels, dropout=0))
            self.side_downsample.append(CondPatchEmbedding(L_patch_len=config["L_patch_len"]**i, channels=config["side_dim"], d_model=config["side_dim"], dropout=0))
            self.patch_decoder.append(PatchDecoder(L_patch_len=config["L_patch_len"]**i, n_var=config["n_var"], d_model=self.channels, channels=1))
        self.multipatch_mixer = nn.Linear(self.multipatch_num, 1)
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    attr_dim=config["attr_dim"],
                    is_attr_proj=config["is_attr_proj"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                )
                for _ in range(config["layers"])
            ]
        )

        self.attr_projector = AttrProjector(
                n_attrs=config["n_attrs"],  # Since we're using a single attribute embedding
                dim_in=config["attr_dim"], 
                dim_hid=config["channels"], 
                dim_out=config["channels"],
                n_heads=8,  # Use same number of heads as in residual blocks
                n_layers=2  # Default number of attention layers
        )
    
    def forward(self, x_raw, side_emb_raw, attr_emb_raw, diffusion_step):
        B, inputdim, K, L = x_raw.shape
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        x_list = []
        side_list = []
        for i in range(self.multipatch_num):
            x = self.ts_downsample[i](x_raw)
            side_emb = self.side_downsample[i](side_emb_raw)
            x_list.append(x)
            side_list.append(side_emb)
        if self.attention_mask_type == "full":
            attention_mask = None
        elif self.attention_mask_type == "parallel":
            attention_mask = self.get_mask(attr_emb_raw.shape[1], [x_list[i].shape[-1] for i in range(len(x_list))])
            attention_mask = self.get_mask(0, [x_list[i].shape[-1] for i in range(len(x_list))], device=x_raw.device)
        
        x_in = torch.cat(x_list, dim=-1)
        side_in = torch.cat(side_list, dim=-1)
        B, _, Nk, Nl = x_in.shape

        _x_in = x_in

        attr_emb = self.attr_projector(attr_emb_raw, length=x_in.shape[-1])
        attr_emb = attr_emb.permute(0,2,1)  # (B,L,C) -> (B,C,L)
        attr_emb = attr_emb.unsqueeze(2)  # (B,C,1,L)

        skip = []
        for layer in self.residual_layers:
            x_in, skip_connection = layer(x_in+_x_in+attr_emb, side_in, attr_emb_raw, diffusion_emb, attention_mask=attention_mask)
            skip.append(skip_connection)        
        
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, Nk * Nl)
        x = self.output_projection1(x)  # (B,channel,Nk*Nl)
        x = F.relu(x)
        x = x.reshape(B, self.channels, Nk, Nl)

        start_id = 0
        all_out = []
        for i in range(len(x_list)):
            x_out = x[:,:,:,start_id:start_id+x_list[i].shape[-1]]
            x_out = self.patch_decoder[i](x_out)
            x_out = x_out[:, :, :, :L]
            all_out.append(x_out)
            start_id += x_list[i].shape[-1]

        all_out = torch.cat(all_out, dim=1) # (B, M, K, L)
        all_out = self.multipatch_mixer(all_out.permute(0, 2, 3, 1).contiguous()) # (B, K, L, 1)
        all_out = all_out.reshape(B, K, L)
        return all_out
    
    def get_mask(self, attr_len, len_list, device="cuda:0"):
        total_len = sum(len_list) + attr_len
        mask = torch.zeros(total_len, total_len, device=device) - float("inf")
        mask[:attr_len, :] = 0
        mask[:, :attr_len] = 0
        start_id = attr_len
        for i in range(len(len_list)):
            mask[start_id:start_id+len_list[i], start_id:start_id+len_list[i]] = 0
            start_id += len_list[i]
        return mask


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
        # input project
        B = attr.shape[0]
        attr = torch.reshape(attr, (B, -1))
        h = F.gelu(self.proj_in(attr))  # (B,d)
        h = F.gelu(self.proj_mid(h))
        h = h.unsqueeze(1)  # (B,d) -> (B,1,d)
        h = h.expand([-1, length, -1])  # (B,L,d)

        # out project
        out = self.proj_out(h)
        return out

    
class AttrProjector_attn(nn.Module):
    def __init__(self, n_attrs=3, dim_in=128, dim_hid=128, dim_out=128, n_heads=8, n_layers=2):
        super().__init__()
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.dim_out = dim_out
        self.n_attrs = n_attrs

        # Project input from (n_attrs * dim_in) to per-token dim_in
        self.proj_in = nn.Linear(dim_in, dim_hid)  # applied per token
        self.output_proj = nn.Linear(dim_hid, dim_out)

        # Trainable query for attention pooling
        self.query = nn.Parameter(torch.randn(1, 1, dim_hid))

        # Attention layer (TextEncoderAttention logic)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim_hid,
            num_heads=n_heads,
            batch_first=True
        )

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.0)

    def forward(self, attr, length):
        B = attr.shape[0]
        # Reshape to (B, n_attrs, dim_in)
        tokens = attr.view(B, self.n_attrs, self.dim_in)
        # Project each token (like text tokens)
        tokens = self.proj_in(tokens)  # (B, n_attrs, dim_hid)
        # Trainable query attention pooling
        q = self.query.expand(B, -1, -1)  # (B, 1, dim_hid)
        pooled, _ = self.attn(query=q, key=tokens, value=tokens)  # (B, 1, dim_hid)
        pooled = pooled.squeeze(1)  # (B, dim_hid)
        # Final MLP projection + activation
        pooled = self.output_proj(pooled)  # (B, dim_out)
        pooled = self.gelu(pooled)
        pooled = self.dropout(pooled)
        # Expand to (B, L, dim_out)
        out = pooled.unsqueeze(1).expand(-1, length, -1)
        return out



class AttrProjectorAvg(nn.Module):
    def __init__(self, dim_in=128, dim_hid=128, dim_out=128):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.dim_out = dim_out

        # Add input projection to handle the 768-dim input
        self.proj_in = nn.Linear(dim_in, dim_hid)
        self.proj_out = nn.Linear(dim_hid, dim_out)

    def forward(self, attr, length):
        # input project
        B = attr.shape[0]
        h = self.proj_in(attr)  # (B,dim_hid)
        h = h.unsqueeze(1)  # (B,1,dim_hid)
        h = h.expand([-1, length, -1])  # (B,L,dim_hid)

        # out project
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

