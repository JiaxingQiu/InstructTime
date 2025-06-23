import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchinfo import summary as nn_summary
from config import *
from model_utils.encoder import *
from model_utils.decoder import *
from vital2d import *

# Verbally Instructed Time series Augmentation Learning (VITAL)  (3d)
class VITAL3D(nn.Module):
    def __init__(self, 
                 ts_dim: int, 
                 text_dim: int, 
                 n_text: int,
                 output_dim: int,
                 temperature: float = 0.01,
                 ts_encoder = None,
                 text_encoder = None,
                 ts_decoder = None,
                 clip_mu = True,
                 concat_embeddings = True,
                 variational = True):
        super().__init__()
        
        # Handle ts_encoder
        if ts_encoder is None:
            self.ts_encoder = TSVAEEncoder(ts_dim, output_dim) #default encoder
        elif isinstance(ts_encoder, type):
            self.ts_encoder = ts_encoder(ts_dim, output_dim)
        else:
            self.ts_encoder = ts_encoder
        
        # Handle text_encoder with attention
        if text_encoder is None:
            self.text_encoder = TextEncoderWithAttention(text_dim, n_text, output_dim)
        elif isinstance(text_encoder, type):
            self.text_encoder = text_encoder(text_dim, n_text, output_dim)
        else:
            self.text_encoder = text_encoder
        
        if concat_embeddings:
            decode_dim = 2*output_dim
        else:
            decode_dim = output_dim
        if ts_decoder is None:
            self.ts_decoder = TSVAEDecoder(ts_dim = ts_dim, output_dim = decode_dim) #default decoder
        elif isinstance(ts_decoder, type):
            self.ts_decoder = ts_decoder(ts_dim = ts_dim, output_dim = decode_dim)
        else:
            self.ts_decoder = ts_decoder
        
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        self.device = device
        self.to(device)
        print(nn_summary(self))
        self.clip_mu = clip_mu
        self.concat_embeddings = concat_embeddings
        self.variational = variational
    def clip(self, ts_embedded, text_embedded):
        # ts_embedded = F.normalize(ts_embedded, dim=1)
        # text_embedded = F.normalize(text_embedded, dim=1)
        logits = torch.matmul(ts_embedded, text_embedded.T) #* torch.exp(self.temperature)
        return logits


    def forward(self, ts, text_features_list):
        """Forward pass of the VITAL model.
        
        Args:
            ts (torch.Tensor): Globally normalized time series of shape [batch_size, ts_dim].
            text_features_list (List[torch.Tensor]): List of text features from a pretrained encoder.
                Each tensor in the list has shape [batch_size, text_dim].
                
        Returns:
            tuple:
                - logits (torch.Tensor): Similarity scores of shape [batch_size, batch_size]
                - x_hat (torch.Tensor): Reconstructed time series of shape [batch_size, ts_dim]
                - mean (torch.Tensor): Latent mean of shape [batch_size, output_dim]
                - log_var (torch.Tensor): Latent log variance of shape [batch_size, output_dim]
        """

        # ---- VAE encoder ----
        # Encode time series
        z, mean, log_var = self.ts_encoder(ts) # ts in raw scale
        if not self.variational: # if not variational (AE instead of VAE), use the mean as the latent variable, 
            z = mean
        # --- Text encoder forward pass ---
        text_embedded, _ = self.text_encoder(text_features_list)

        # --- CLIP forward pass ---
        if self.clip_mu:
            logits = self.clip(mean, text_embedded)
        else:
            logits = self.clip(z, text_embedded)
        
        # --- VAE decoder forward pass ---
        if self.concat_embeddings:
            decode_input = torch.cat([z, text_embedded], dim=1)
        else:
            decode_input = z
        ts_hat = self.ts_decoder(decode_input, x_mean, x_std)

        
        return logits, ts_hat, mean, log_var

# default text encoder with attention on text splits
class TextEncoderWithAttention(nn.Module):
    def __init__(self, text_dim, n_text, output_dim, encoder_config=None):
        super().__init__()

        # Default encoder configurations
        self.encoder_config = {
            '5': {
                'hidden_dim': 128,
                'n_heads': 1,
                'n_layers': 1,
                'dropout': 0.1
            },
            '4': {
                'hidden_dim': 256,
                'n_heads': 1,
                'n_layers': 1,
                'dropout': 0.1
            },
            '3': {
                'hidden_dim': 512,# 256
                'n_heads': 4,#2
                'n_layers': 1,
                'dropout': 0.1
            },
            '2': {
                'hidden_dim': 1024,
                'n_heads': 4,
                'n_layers': 2,
                'dropout': 0.1
            },
            '1': {
                'hidden_dim': 1024,
                'n_heads': 8,
                'n_layers': 3,
                'dropout': 0.1
            }
        } if encoder_config is None else encoder_config
        
        # Select configuration based on n_text
        self.config = self.encoder_config[str(min(n_text, 5))]

        # Create n_text copies of the encoder
        self.text_encoders = nn.ModuleList([
            self._single_text_encoder(text_dim, output_dim, self.config) 
            for _ in range(n_text)
        ])
        
        # Attention mechanism to combine text features
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=1,  # Single head for clear attention weights
            batch_first=True
        )
        
        # Trainable query vector
        self.query = nn.Parameter(torch.randn(1, 1, output_dim))
        
    
    def _single_text_encoder(self, text_dim, output_dim, config, hidden_dim=512):#256
        layers = [
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(config['dropout'])
        ]
        # Add transformer blocks
        for _ in range(config['n_layers']):
            layers.append(
                TransformerBlock(
                    dim=hidden_dim,
                    hidden_dim=config['hidden_dim'],
                    num_heads=config['n_heads'],
                    dropout=config['dropout']
                )
            )
        layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)
    

    def forward(self, text_features):
        # Encode each text input
        encoded_texts = []
        for i, text in enumerate(text_features):
            encoded = self.text_encoders[i](text)  # [batch_size, output_dim]
            encoded_texts.append(encoded)
        
        # Stack encoded texts [batch_size, n_text, output_dim]
        encoded_texts = torch.stack(encoded_texts, dim=1)
        
        # Expand query for batch size
        query = self.query.expand(encoded_texts.size(0), -1, -1)
        
        # Apply attention
        attended_output, attention_weights = self.attention(
            query=query,  # [batch_size, 1, output_dim]
            key=encoded_texts,  # [batch_size, n_text, output_dim]
            value=encoded_texts  # [batch_size, n_text, output_dim]
        )
        
        # Remove the query dimension
        combined_output = attended_output.squeeze(1)  # [batch_size, output_dim]
        
        return combined_output, attention_weights
    
# # Initialize
# text_encoder = TextEncoderWithAttention(
#     text_dim=768,    # BERT embedding size
#     n_text=3,        # Number of text inputs
#     output_dim=128   # Final output dimension
# )

# # Forward pass
# text_features = [text1, text2, text3]  # List of text tensors
# combined_output, attention_weights = text_encoder(text_features)

# # Visualize attention weights if needed
# import matplotlib.pyplot as plt
# plt.imshow(attention_weights[0].detach().cpu())
# plt.colorbar()
# plt.show()


