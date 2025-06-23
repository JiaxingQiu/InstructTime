import torch
import torch.nn as nn
import numpy as np


# class AttributeEncoder(nn.Module):
#     def __init__(self, configs) -> None:
#         super().__init__()
#         self.device = configs["device"]
#         self.emb_dim = configs["attr_dim"]
#         self.n_attrs = configs["n_attrs"]
#         # Project text embeddings to desired dimension
#         self.proj = nn.Sequential(
#             nn.Linear(768, self.emb_dim),  # MPNET base has 768-dimensional embeddings
#             nn.GELU(),
#             nn.Linear(self.emb_dim, self.emb_dim)
#         )

#     def forward(self, attrs, replace_with_empty=False):
#         """
#         Args:
#             attrs: Pre-computed text embeddings (B, 768)
#             replace_with_empty: whether use zero embeddings.
#         """
#         if replace_with_empty:
#             return torch.zeros((attrs.shape[0], self.emb_dim), device=self.device)
#         # Convert to float and project
#         attrs = attrs.float()
#         return self.proj(attrs)

#     def get_all_embs(self):
#         """This method is kept for backward compatibility but may not be meaningful for text embeddings"""
#         return [torch.zeros(1, self.emb_dim, device=self.device)]

    
class AttributeEncoder(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.device = configs["device"]
        self.emb_dim = configs["attr_dim"]
        self.n_attrs = configs["n_attrs"]

    def forward(self, attrs, replace_with_empty=False):
        """
        Args:
            attrs: Pre-computed text embeddings (B, 768)
            replace_with_empty: whether to use zero embeddings.
        """
        if replace_with_empty:
            return torch.zeros((attrs.shape[0], attrs.shape[1]), device=self.device)
        return attrs

    def get_all_embs(self):
        return []
