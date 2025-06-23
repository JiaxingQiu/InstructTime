import torch
import torch.nn as nn


class SideEncoder(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.device = configs["device"]

        self.time_emb_dim = configs["time_emb"]
        self.total_emb_dim = self.time_emb_dim

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, tp):
        B, L = tp.shape

        time_emb = self.time_embedding(tp, self.time_emb_dim)  # (B,L,emb)
        side_emb = time_emb.unsqueeze(2)  # (B,L,1,emb)
        side_emb = side_emb.permute(0, 3, 2, 1)  # (B,*,V,L)
        return side_emb