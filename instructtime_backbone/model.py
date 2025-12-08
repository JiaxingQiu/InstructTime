import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchinfo import summary as nn_summary
from encoder import *
from decoder import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# InstructTime model architecture (variational design is not used)
class VITAL(nn.Module):
    def __init__(self, 
                 ts_dim: int, 
                 text_dim: int, 
                 output_dim: int,
                 beta: float = 1.0,
                 ts_encoder = None,
                 text_encoder = None,
                 ts_decoder = None,
                 variational = False,
                 clip_mu = False,
                 gen_w_src_text = False,
                 linear_interpolation = True):
        super().__init__()
        # ts_encoder
        self.ts_encoder = TSEncoder(ts_dim, output_dim, encoder_layers = ts_encoder, variational = variational)
        # text_encoder
        self.text_encoder = TextEncoder(text_dim, output_dim, encoder_layers = text_encoder)
        # ts_decoder
        self.ts_decoder = TSDecoder(ts_dim = ts_dim, output_dim = output_dim, decoder_layers = ts_decoder)
        
        self.device = device
        self.beta = beta
        self.clip_mu = clip_mu
        self.variational = variational
        self.gen_w_src_text = gen_w_src_text
        self.linear_interpolation = linear_interpolation
        self.to(device)
        print(nn_summary(self))
    
    def clip(self, ts_embedded, text_embedded):
        logits = torch.matmul(ts_embedded, text_embedded.T) 
        return logits
    
    def forward(self, ts, text_features):
        """
        Args:
            ts: torch.Tensor [B, ts_dim] - Time series in raw scale (ts_dim = sequence length, e.g., 300)
            text_features: torch.Tensor [B, text_dim] - Text pretrained embeddings (features) from pretrained text encoder (text_dim, e.g., 768)
        Returns:
            logits: [B, B] - CLIP similarity matrix
            ts_hat: [B, ts_dim] - Reconstructed time series
            mean: [B, output_dim] - Encoder mean (L2 normalized)
            log_var: [B, output_dim] - Encoder log variance
        """
        # ---- (V)AE encoder ----
        ts_emb, mean, log_var = self.ts_encoder(ts) # ts in raw scale
        # --- Text encoder forward pass ---
        text_embedded = self.text_encoder(text_features)
        # --- CLIP forward pass ---
        if self.clip_mu:
            logits = self.clip(mean, text_embedded)
        else:
            logits = self.clip(ts_emb, text_embedded)
        # --- VAE decoder forward pass ---
        ts_hat = self.ts_decoder(ts_emb, text_embedded, ts, text_embedded) # during trining, only use text_embedded as source text embedding
        return logits, ts_hat, mean, log_var
    
    def interpolate(self, emb0, emb1, angle_ratio):
        if self.linear_interpolation:
            emb_tgt = (1 - angle_ratio) * emb0 + angle_ratio * emb1
            return F.normalize(emb_tgt, dim=1)
        else:
            # Calculate current angle between vectors
            cos_theta = torch.sum(emb0 * emb1, dim=1, keepdim=True)
            current_angle = torch.acos(torch.clamp(cos_theta, -1, 1))
            
            # Handle edge parallel cases
            sin_theta = torch.sin(current_angle)
            parallel_mask = torch.abs(sin_theta) <  1e-8
            if parallel_mask.any():
                # Linear interpolation for parallel case
                emb_tgt = (1 - angle_ratio) * emb0 + angle_ratio * emb1
            else:
                # Slerp for non-parallel case
                w1 = torch.sin((1 - angle_ratio) * current_angle) / sin_theta
                w2 = torch.sin(angle_ratio * current_angle) / sin_theta
                emb_tgt = w1 * emb0 + w2 * emb1
            return F.normalize(emb_tgt, dim=1)
    
    def generate(self, w, ts, tx_f_tgt, tx_f_src):
        """
        Generate time series by interpolating between source time series and target text.
        
        Args:
            w: float or [B] - Interpolation weight (0=reconstruction, 1=target text)
            ts: [B, ts_dim] - Source time series (ts_dim = sequence length, e.g., 300)
            tx_f_tgt: [B, text_dim] - Target text pretrained embeddings (features) from pretrained text encoder (text_dim, e.g., 768)
            tx_f_src: [B, text_dim] - Source text pretrained embeddings (features) from pretrained text encoder (not used)
        Returns:
            ts_hat: [B, ts_dim] - Generated time series
            ts_emb_tgt: [B, output_dim] - Interpolated time series embedding
            tx_emb_tgt: [B, output_dim] - Target text embedding
            tx_emb_src: [B, output_dim] - Source text embedding
        """
        # embedding of target text conditions
        tx_emb_tgt = self.text_encoder(tx_f_tgt)
        # embedding of source time series 
        ts_emb_src, _, _ = self.ts_encoder(ts)
        # embedding of source text conditions
        tx_emb_src = self.text_encoder(tx_f_src)
        # target time series embeddings interpolated from source time series and target text conditions
        # ts_emb_tgt = (1-w)*ts_emb_src + w*tx_emb_tgt # interpolation of ts_emb and tx_emb
        ts_emb_tgt = self.interpolate(ts_emb_src, tx_emb_tgt, w)
        # (IMPORTANT) if generate with source text condition, overwrite tx_emb_tgt with tx_emb_src
        if self.gen_w_src_text: tx_emb_tgt = tx_emb_src
        ts_hat = self.ts_decoder(ts_emb_tgt, tx_emb_tgt) # during generation, use tx_emb_src as source text embedding

        return ts_hat, ts_emb_tgt, tx_emb_tgt, tx_emb_src


class LocalNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
    
    def forward(self, x):
        # Compute mean and std along feature dimension
        mean = x.mean(dim=1, keepdim=True)  # [batch_size, 1]
        std = x.std(dim=1, keepdim=True)    # [batch_size, 1]
        
        # Normalize
        x_norm = (x - mean) / (std + self.eps)
        
        return x_norm, mean, std

class TSEncoder(nn.Module):
    def __init__(self, ts_dim: int, output_dim: int, encoder_layers = None, variational: bool = False):
        super().__init__()
        self.variational = variational
        self.local_norm = LocalNorm()
        if encoder_layers is None:
            # default encoder layers
            self.encoder_layers = PatchCNNTSEncoder(ts_dim = ts_dim,
                                                    output_dim=output_dim)
        else:
            self.encoder_layers = encoder_layers # pass an instance of custom encoder layers from classes in the encoder module
        
        if self.variational:
            self.mean_layer = nn.Linear(output_dim, output_dim)
            self.logvar_layer = nn.Linear(output_dim, output_dim)
    
    def reparameterization(self, mean, log_var, ep=1):
        var = ep * torch.exp(0.5 * log_var) # slower than using log_var directly
        # var = log_var
        epsilon = torch.randn_like(var).to(device)      
        z = mean + var*epsilon  # Using var directly
        # z = F.softmax(z, dim=1) # This variable follows a Dirichlet distribution
        return z
    
    def forward(self, x):
        #  ---- encode -----
        x_encoded = self.encoder_layers(x)

        if self.variational:
            mean = self.mean_layer(x_encoded)
            mean = F.normalize(mean, dim=1)
            log_var = self.logvar_layer(x_encoded)
            z = self.reparameterization(mean, log_var)
        else:
            mean = x_encoded
            mean = F.normalize(mean, dim=1)
            log_var = torch.full_like(mean, -1e2)  # effectively 0 variance
            z = mean
        
        return z, mean, log_var

import inspect
class TSDecoder(nn.Module):
    def __init__(self, ts_dim: int, output_dim: int, decoder_layers = None):
        super().__init__()
        if decoder_layers is None:
            self.decoder = SelfAttnDecoder(ts_dim=ts_dim,
                                output_dim=output_dim,
                                num_layers=8,
                                diffusion_steps = 0
                            )
        else:
            self.decoder = decoder_layers
        
        # ---------- figure out how many positional args it needs --------
        sig = inspect.signature(self.decoder.forward)
        # skip the first ("self") parameter
        required_params_count = 0
        for param in sig.parameters.values():
            if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and param.default == inspect.Parameter.empty:
                required_params_count += 1
        self._n_required_args = required_params_count
    
    def forward(self, ts_emb, txt_emb, ts = None, src_txt_emb = None):
        x_hat = self.decoder(ts_emb, txt_emb) 
        x_hat = x_hat.squeeze(1) if x_hat.dim() == 3 and x_hat.size(1) == 1 else x_hat         # [B, ts_dim]
        return x_hat

class TextEncoder(nn.Module):
    def __init__(self, text_dim: int, output_dim: int, encoder_layers = None):
        """Text encoder that can use either MLP or CNN architecture.
        
        Args:
            text_dim (int): Input text embedding dimension
            output_dim (int): Output embedding dimension
            text_encoder_type (str): Type of encoder to use ('mlp' or 'cnn')
        """
        super().__init__()
        if encoder_layers is None:
            self.encoder_layers = PatchMLPTextEncoder(
                text_dim=text_dim,
                output_dim=output_dim
            )   
        else:
            self.encoder_layers = encoder_layers # pass an instance of custom encoder layers from classes in the encoder module
        
    def forward(self, text_features):
        tx_emb = self.encoder_layers(text_features)
        tx_emb = F.normalize(tx_emb, dim=1)
        return tx_emb

