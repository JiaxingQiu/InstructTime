import torch, math
import torch.nn as nn
import torch.nn.functional as F
from .helper import PositionalEncoding, SelfAttnBlock, DiffusionRefiner
from torch.utils.checkpoint import checkpoint

# ------- custom ts decoder_layers -------

class WeightedFusionDecoder(nn.Module):
    """
    Decoder that learns a per-dimension weight vector w using a single SelfAttnBlock,
    then decodes the element-wise fused embedding to reconstruct the time series.
    """
    def __init__(
        self,
        ts_dim: int,
        output_dim: int,
        nhead: int = 8,
        ffn_mult: int = 1,
        dropout: float = 0.0,
        diffusion_steps: int = 0,
        diff_txt_proj: bool = True,
    ):
        super().__init__()
        self.hidden_dim = output_dim

        self.pos_encoder = PositionalEncoding(self.hidden_dim)
        self.mu_logsigma_block = SelfAttnBlock(
            width=self.hidden_dim,
            heads=nhead,
            ffn_mult=ffn_mult,
            drop=dropout,
        )

        self.out_block = SelfAttnBlock(
            width=self.hidden_dim,
            heads=nhead,
            ffn_mult=ffn_mult,
            drop=dropout,
        )
        self.out = nn.Linear(self.hidden_dim, ts_dim)

        self.diffusion_steps = diffusion_steps
        if diffusion_steps > 0:
            self.diffusion_tail = DiffusionRefiner(
                ts_dim=ts_dim,
                txt_dim=output_dim,
                n_steps=diffusion_steps,
                diff_txt_proj=diff_txt_proj,
            )

    @staticmethod
    def _as_sequence(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1) if x.dim() == 2 else x

    def forward(self, ts_emb: torch.Tensor, txt_emb: torch.Tensor):
        """
        Args:
            ts_emb: [B, D]
            txt_emb: [B, D]
        Returns:
            ts_hat: [B, ts_dim]
        """
        tokens = torch.cat([self._as_sequence(ts_emb), self._as_sequence(txt_emb)], dim=1) # [B, 2, D]
        tokens = self.pos_encoder(tokens)
        h = self.mu_logsigma_block(tokens)               # [B, 2, D]
        mu = h[:, 0] # [B, D]
        log_sigma = h[:, 1] # [B, D]
        sigma = F.softplus(log_sigma) + 1e-6 # [B, D]

        eps = torch.randn_like(sigma) # [B, D]
        z = mu + sigma * eps # [B, D]
        w = torch.sigmoid(z) # [B, D]

        blended = w * ts_emb + (1.0 - w) * txt_emb # [B, D] 

        tokens = torch.cat([self._as_sequence(ts_emb),
                    self._as_sequence(txt_emb),
                    self._as_sequence(blended)], dim=1)   # [B,3,D]

        h_out  = self.out_block(tokens)                          # [B,3,D]
        ts_hat = self.out(h_out[:, 2])                           # use blended token

        if self.diffusion_steps > 0:
            ts_hat = self.diffusion_tail(ts_hat, txt_emb)

        return ts_hat



class ResNetDecoder(nn.Module):
    def __init__(self, ts_dim, output_dim, hidden_dim=128, num_blocks=2, dropout=0.1):
        super().__init__()
        self.ts_dim = ts_dim
        self.output_dim = output_dim
        
        
        layers = [
            nn.Linear(output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        ]
        
        # Add residual blocks
        for i in range(num_blocks):
            layers.append(
                ResidualBlock(
                    hidden_dim, 
                    hidden_dim,  # Keep dimension constant
                    dropout=dropout*(1+i/num_blocks)
                )
            )
        
        # Final projection
        layers.extend([
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, ts_dim),
            nn.LeakyReLU(0.2)
        ])

        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.decoder(x)
# # old default decoder layers
# nn.Sequential(
#     nn.Linear(output_dim, 256), # +2 2 for x_mean and x_std
#     nn.LeakyReLU(0.2),
#     nn.Linear(256, 256),
#     nn.LeakyReLU(0.2),
#     nn.Linear(256, ts_dim)
# )

class CNNDecoder(nn.Module):
    def __init__(self, ts_dim, output_dim, num_channels=[64, 64, 128, 256], kernel_size=5, dropout=0.2):
        """
        CNN decoder for time series.
        
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
            x = torch.zeros(2, output_dim)
            for layer in layers:
                x = layer(x)
            conv_out_dim = x.shape[1]
        
        # Add final linear projection to match output_dim
        layers.append(nn.Linear(conv_out_dim, ts_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.decoder(x)

class MLPDecoder(nn.Module):
    def __init__(self, ts_dim, output_dim, hidden_dim=128, num_hidden_layers=6, dropout=0.2):
        """
        Multi-layer perceptron decoder.
        
        Args:
            ts_dim (int): Input time series length
            output_dim (int): Output dimension
            hidden_dim (int): Hidden dimension
            num_hidden_layers (int): Number of hidden layers
            dropout (float): Dropout rate
        """
        super().__init__()
        
        self.ts_dim = ts_dim
        self.output_dim = output_dim

        layers = []
        
        if num_hidden_layers > 0:
            # First hidden layer (input projection)
            layers.extend([
                nn.Linear(output_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
        
            # Hidden layers
            if num_hidden_layers > 1:
                for _ in range(num_hidden_layers - 1):
                    layers.extend([
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(dropout)
                    ])
        else:
            hidden_dim = output_dim
        
        # Final projection - extend instead of append for list of layers
        layers.extend([
            nn.Linear(hidden_dim, ts_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        ])
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.decoder(x)

class LSTMDecoder(nn.Module):
    def __init__(self, ts_dim, output_dim, hidden_dim=128, num_layers=2, dropout=0.1, bidirectional=False):
        super().__init__()


        self.ts_dim = ts_dim
        self.output_dim = output_dim
        
        self.lstm = nn.LSTM(
            input_size=ts_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Account for bidirectional in final dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.projection = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, ts_dim)
        # If input is (batch_size, ts_dim), unsqueeze to add sequence dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        # Get LSTM output
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state from all layers
        if self.lstm.bidirectional:
            # Concatenate forward and backward last hidden states
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1]
            
        # Project to final dimension
        output = self.projection(hidden)
        
        return output

class MultiLSTMDecoder(nn.Module):
    def __init__(self, 
                 ts_dim, 
                 output_dim, 
                 hidden_dims=[128, 256, 64],  # Multiple LSTM sizes
                 num_layers=2, 
                 dropout=0.1, 
                 bidirectional=False,
                 mask=-1): # it is the indicator of masked values in the time series (i.e. -1), default to -1
        super().__init__()
        
        self.ts_dim = ts_dim
        self.output_dim = output_dim

        # Create multiple LSTM modules
        self.lstms = nn.ModuleList([
            nn.LSTM(
                input_size=output_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional
            ) for hidden_dim in hidden_dims
        ])
        
        # Total dimension from all LSTMs
        total_lstm_dim = sum(hidden_dim * 2 if bidirectional else hidden_dim 
                           for hidden_dim in hidden_dims)
        
        # Projection layers
        self.projection = nn.Sequential(
            nn.Linear(total_lstm_dim, total_lstm_dim // 2),
            nn.BatchNorm1d(total_lstm_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(total_lstm_dim // 2, ts_dim),
            nn.BatchNorm1d(ts_dim)
        )
        
        self.bidirectional = bidirectional
        self.mask = mask
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, ts_dim)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        if self.mask is not None:
            # Create mask for values != -1
            mask = (x != self.mask).float()
            # Apply mask to input
            x = x * mask # zero out masked values
            
        # Process through each LSTM
        lstm_outputs = []
        for lstm in self.lstms:
            _, (hidden, _) = lstm(x)
            
            if self.bidirectional:
                # Concatenate forward and backward last hidden states
                hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            else:
                hidden = hidden[-1]
                
            lstm_outputs.append(hidden)
        
        # Concatenate all LSTM outputs
        combined = torch.cat(lstm_outputs, dim=1)
        
        # Project to final dimension
        output = self.projection(combined)
        
        return output

class TransformerDecoderTS(nn.Module):
    def __init__(self, ts_dim: int, output_dim: int, 
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        """Transformer-based decoder for time series reconstruction.
        
        Args:
            ts_dim: Input time series dimension
            output_dim: Latent space dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout probability
        """
        super().__init__()
        
        # Project latent vector to transformer input dimension
        self.input_projection = nn.Linear(output_dim, ts_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(ts_dim, dropout)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=ts_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(ts_dim, ts_dim)
        
    def forward(self, z):
        # # Project latent vector to transformer input dimension
        x = self.input_projection(z)  # [batch_size, ts_dim]
        # x = z
        # Add sequence dimension and repeat for transformer
        x = x.unsqueeze(1)  # [batch_size, 1, ts_dim]
        # Add positional encoding
        x = self.pos_encoder(x)
        # Create memory (same as input for auto-regressive generation)
        memory = x
        # Generate output sequence
        output = self.transformer_decoder(x, memory)
        # Project back to original dimension
        output = self.output_projection(output.squeeze(1))
        return output
# old default decoder layers
# TransformerDecoder(ts_dim = ts_dim, 
#                   output_dim = output_dim, 
#                   nhead = 4,
#                   num_layers = 6,
#                   dim_feedforward = 512,
#                   dropout = 0.0)

class AttentionDecoder(nn.Module):
    def __init__(self, ts_dim: int, output_dim: int, 
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """Simple attention-based decoder for time series reconstruction.
        
        Args:
            ts_dim: Input time series dimension
            output_dim: Latent space dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        
        # Multi-head attention for reconstruction
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Trainable query for attention
        self.query = nn.Parameter(torch.randn(1, 1, output_dim))
        
        # Output projection
        self.output_proj = nn.Linear(output_dim, ts_dim)
        
        # Normalization and dropout
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, z):
        k = v = self.norm(z.unsqueeze(1))          # [B,1,E]
        q = self.query.expand(k.size(0), -1, -1)   # [B,1,E]

        attn_out, _ = self.attention(q, k, v)      # [B,1,E]
        out = self.output_proj(attn_out.squeeze(1))  # [B,E] ->[B,ts_dim]
        return out


class TransformerDecoderTXTS(nn.Module):
    """
    Decoder that takes
      • ts_emb  → target  (tgt)
      • txt_emb → memory
    With positional encoding.
    
    Args:
        ts_dim: original time-series dimension
        output_dim: shared embedding size of ts_emb / txt_emb
        nhead: number of attention heads
        num_layers: number of transformer layers
        dim_feedforward: dimension of feedforward network
        dropout: dropout probability
        project_input: whether to project output_dim to ts_dim at the beginning
    """

    def __init__(
        self,
        ts_dim: int,           # original time-series dimension
        output_dim: int,          # shared embedding size of ts_emb / txt_emb
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        project_input: bool = False,  # whether to project output_dim to ts_dim at the beginning
    ):
        super().__init__()
        hidden_dim = output_dim

        self.project_input = project_input
        if project_input:
            self.input_projection = nn.Linear(output_dim, ts_dim)
            hidden_dim = ts_dim
            
        # Add positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output_projection = nn.Linear(hidden_dim, ts_dim)
        
        # # Replace simple linear projection with feed-forward network
        # self.output_ln = nn.LayerNorm(hidden_dim)
        # self.output_ffn = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim * 4),  # Expand dimension
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim * 4, ts_dim)  # Project to final dimension
        # )

    @staticmethod
    def _as_sequence(x: torch.Tensor) -> torch.Tensor:
        """Accept [B,E] or [B,L,E] and return [B,L,E]."""
        return x.unsqueeze(1) if x.dim() == 2 else x

    def forward(self, ts_emb: torch.Tensor, txt_emb: torch.Tensor):
        """
        ts_emb : [B,E] time-series embeddings  (tgt)
        txt_emb: [B,E] text embeddings        (memory)
        """
        tgt = self._as_sequence(ts_emb) # [B, 1, E]
        memory = self._as_sequence(txt_emb) # [B, 1, E]
        
        # Project input if needed
        if self.project_input:
            tgt = self.input_projection(tgt)
            memory = self.input_projection(memory)

        # Add positional encoding
        tgt = self.pos_encoder(tgt)
        memory = self.pos_encoder(memory)

        dec_out = self.decoder(tgt=tgt, memory=memory) # [B, 1, E]
        ts_hat = self.output_projection(dec_out) # ts_hat: [B, 1, ts_dim]
        
        # # Apply feed-forward network with layer norm
        # dec_out = self.output_ln(dec_out)
        # ts_hat = self.output_ffn(dec_out) # ts_hat: [B, 1, ts_dim]
        return ts_hat

# # old default decoder layers
# TransformerDecoderTXTS(
#                 ts_dim     = ts_dim,
#                 output_dim = output_dim,
#                 nhead = 8,
#                 num_layers = 6,
#                 dim_feedforward = 512,
#                 dropout = 0.0)

class TransformerDecoderDiff(nn.Module):
    """
    Decoder that takes
      • ts_emb  → target  (tgt)
      • txt_emb → memory
    With positional encoding.
    
    Args:
        ts_dim: original time-series dimension
        output_dim: shared embedding size of ts_emb / txt_emb
        nhead: number of attention heads
        num_layers: number of transformer layers
        dim_feedforward: dimension of feedforward network
        dropout: dropout probability
        project_input: whether to project output_dim to ts_dim at the beginning
    """

    def __init__(
        self,
        ts_dim: int,           # original time-series dimension
        output_dim: int,          # shared embedding size of ts_emb / txt_emb
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        project_input: bool = False,  # whether to project output_dim to ts_dim at the beginning
    ):
        super().__init__()
        hidden_dim = output_dim

        self.project_input = project_input
        if project_input:
            self.input_projection = nn.Linear(output_dim, ts_dim)
            hidden_dim = ts_dim
            
        # Add positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        #  self.output_projection = nn.Linear(hidden_dim, ts_dim)
        
        # Replace simple linear projection with feed-forward network
        self.output_ln = nn.LayerNorm(hidden_dim)
        self.output_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),  # Expand dimension
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, ts_dim)  # Project to final dimension
        )
        
    @staticmethod
    def _as_sequence(x: torch.Tensor) -> torch.Tensor:
        """Accept [B,E] or [B,L,E] and return [B,L,E]."""
        return x.unsqueeze(1) if x.dim() == 2 else x

    def forward(self, ts_emb: torch.Tensor, txt_emb: torch.Tensor):
        """
        ts_emb : [B,E] time-series embeddings  (tgt)
        txt_emb: [B,E] text embeddings        (memory)
        """
        tgt = self._as_sequence(ts_emb) # [B, 1, E]
        memory = self._as_sequence(txt_emb) # [B, 1, E]
        
        # Project input if needed
        if self.project_input:
            tgt = self.input_projection(tgt)
            memory = self.input_projection(memory)

        # Add positional encoding
        tgt = self.pos_encoder(tgt)
        memory = self.pos_encoder(memory)

        dec_out = self.decoder(tgt=tgt, memory=memory) # [B, 1, E]
        # ts_hat = self.output_projection(dec_out) # ts_hat: [B, 1, ts_dim]
        
        # Apply feed-forward network with layer norm
        dec_out = self.output_ln(dec_out)
        ts_hat = self.output_ffn(dec_out) # ts_hat: [B, 1, ts_dim]
        return ts_hat


class TransformerDecoderTXTS2(nn.Module):
    def __init__(
        self,
        ts_dim: int,
        output_dim: int,      # E
        nhead: int = 4,
        num_layers: int = 1, # one layer, higher dim_forward tend to work better
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        project_input: bool = False,
    ):
        super().__init__()

        hidden_dim = ts_dim if project_input else output_dim
        self.project_input = project_input
        if project_input:
            self.input_projection = nn.Linear(output_dim, ts_dim)

        # --- shared positional encoding ---
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        # --- full-size decoder for ts→txt ---
        full_layer  = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder_ts2tx = nn.TransformerDecoder(full_layer,
                                                   num_layers=num_layers)

        # --- *smaller* decoder for txt→txt -------------
        small_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,                 # keep width so shapes align
            nhead=max(1, nhead // 2),           # fewer heads
            dim_feedforward=max(256, dim_feedforward // 2),
            dropout=dropout,
            batch_first=True,
        )
        self.decoder_tx2tx = nn.TransformerDecoder(
            small_layer,         # independent weights
            num_layers=max(1, num_layers // 2) 
        )

        # project concat([dec_out1, dec_out2]) → ts_dim
        self.output_projection = nn.Linear(hidden_dim * 2, ts_dim)

    # ------------------------------------------------------------------
    @staticmethod
    def _as_sequence(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1) if x.dim() == 2 else x

    def forward(self,
                ts_emb: torch.Tensor,
                txt_emb: torch.Tensor,
                src_txt_emb: torch.Tensor | None = None) -> torch.Tensor:

        if src_txt_emb is None:
            src_txt_emb = txt_emb

        ts_in      = self._as_sequence(ts_emb)
        tx_in      = self._as_sequence(txt_emb)
        src_tx_in  = self._as_sequence(src_txt_emb)

        if self.project_input:
            ts_in      = self.input_projection(ts_in)
            tx_in      = self.input_projection(tx_in)
            src_tx_in  = self.input_projection(src_tx_in)

        ts_in      = self.pos_encoder(ts_in)
        tx_in      = self.pos_encoder(tx_in)
        src_tx_in  = self.pos_encoder(src_tx_in)

        dec_out1   = self.decoder_ts2tx(tgt=ts_in, memory=src_tx_in)  # [B,1,E]
        dec_out2   = self.decoder_tx2tx(tgt=tx_in, memory=src_tx_in)  # [B,1,E]  (lighter)

        fused      = torch.cat([dec_out1, dec_out2], dim=-1)          # [B,1,2E]
        ts_hat     = self.output_projection(fused.squeeze(1))         # [B,ts_dim]
        return ts_hat

class TransformerDecoderPreAuto(nn.Module):

    def __init__(
        self,
        ts_dim: int,           # original time-series dimension
        output_dim: int,          # shared embedding dim of ts_emb / txt_emb
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        k = 10, # number of condition tokens per embedding
        use_checkpoint: bool = True,  # Enable gradient checkpointing
    ):
        super().__init__()
        self.k = k
        self.d_model = nhead * 4
        self.use_checkpoint = use_checkpoint

        # ---------- Conditioning projections ----------
        self.start_token = nn.Linear(output_dim * 2, 1)
        self.cond_proj = nn.Sequential(nn.Linear(output_dim, k * self.d_model), nn.GELU())
        self.mem_proj  = nn.Sequential(nn.Linear(output_dim * 2, self.d_model), nn.GELU())
        self.tgt_proj  = nn.Sequential(nn.Linear(1, self.d_model), nn.GELU())

        # ---------- Decoder stack ----------
        self.register_buffer("strict_causal_mask", torch.triu(torch.ones(5000,5000),1).bool())
        self.decoder_layer = nn.TransformerDecoderLayer(
                                d_model=self.d_model, 
                                nhead=nhead,
                                dim_feedforward=dim_feedforward,
                                dropout=dropout,
                                batch_first=True)
        
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(self.d_model, 1)
        
    def _decoder_forward(self, tgt, memory, tgt_mask):
        """Helper function for checkpointing"""
        return self.decoder(tgt, memory, tgt_mask=tgt_mask)
        
    def forward(self, ts_emb:torch.Tensor, txt_emb:torch.Tensor, ts:torch.Tensor):
        """
        Returns ts_hat of shape [B, T] (same length as input series).
        """
        B, T_full = ts.shape
        if T_full < 1:
            raise ValueError("need at least one step")

        # 1) teacher-forcing shift
        # sos   = ts[:, :1]                       # [B,1] raw first value
        sos = self.start_token(torch.cat([ts_emb, txt_emb], dim=1))  # [B,1]
        ts_in = torch.cat([sos, ts[:, :-1]], 1) # [B,T]
        T = ts_in.size(1)

        # 2) build per-time-step memory (same as before, but for T+2k)
        T_plus = T + 2 * self.k
        ts_emb_t = ts_emb.unsqueeze(1).expand(-1, T_plus, -1)
        txt_emb_t = txt_emb.unsqueeze(1).expand(-1, T_plus, -1)

        memory = self.mem_proj(torch.cat([ts_emb_t, txt_emb_t], 2))  # [B,T+2k,d]

        # 3) prepend 2k condition tokens to the decoder input
        cond_ts = self.cond_proj(ts_emb).view(B, self.k, self.d_model)  # [B,k,d]
        cond_txt = self.cond_proj(txt_emb).view(B, self.k, self.d_model)  # [B,k,d]
        tgt_ts = self.tgt_proj(ts_in.unsqueeze(-1))  # [B,T,d]
        tgt = torch.cat([cond_ts, cond_txt, tgt_ts], 1)  # [B,T+2k,d]

        # 4) causal mask (allow reading cond tokens + past)
        causal_mask = self.strict_causal_mask[:T_plus, :T_plus].to(ts.device)

        # 5) decode with checkpointing if enabled
        if self.use_checkpoint and self.training:
            dec_out = checkpoint(self._decoder_forward, tgt, memory, causal_mask)
        else:
            dec_out = self.decoder(tgt, memory, tgt_mask=causal_mask)
            
        ts_hat = self.output_projection(dec_out[:, 2*self.k:]).squeeze(-1)  # drop cond tokens → [B,T]

        return ts_hat
    
    @torch.no_grad()
    def forecast(
        self,
        txt_emb:  torch.Tensor,            # [B, E]  text embedding
        horizon:  int,                     # how many steps to produce
        ts_emb:   torch.Tensor| None = None,            # [B, E]  ts embedding
        ts_ctx:   torch.Tensor | None = None,  # optional observed prefix [B, T_ctx]
    ) -> torch.Tensor:
        if ts_emb is None:
            ts_emb = txt_emb.clone()

        device, dtype = ts_emb.device, ts_emb.dtype
        B             = ts_emb.size(0)
        seq = torch.empty(B, 0, device=device, dtype=dtype) if ts_ctx is None \
              else ts_ctx.to(device=device, dtype=dtype).clone()

        for _ in range(horizon):
            T_cur = seq.size(1)
            dummy_ts = torch.zeros(B, max(1, T_cur), device=device, dtype=dtype)
            if T_cur:
                dummy_ts[:, :T_cur] = seq
            ts_hat = self(ts_emb, txt_emb, dummy_ts)
            seq    = torch.cat([seq, ts_hat[:, -1:]], dim=1)

        return seq  # [B, T_ctx + horizon]    
    


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def timestep_embed(t: torch.Tensor, dim: int):
    """Sinusoidal t-embedding   t: [B] → [B, dim]"""
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=t.device) * -(math.log(10_000.0) / half))
    ang   = t[:, None] * freqs[None]
    emb   = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
    return emb if dim % 2 == 0 else F.pad(emb, (0, 1))

class AttnEpsNet(nn.Module):
    """
    Input  tokens:  [x_t]  (ts_dim)           ← noisy target
                    [c_ts] (emb_dim)          ← ts_emb
                    [c_tx] (emb_dim)          ← txt_emb
                    [t]    (time_dim)         ← timestep embedding
    A small Transformer operates on this 4-token sequence.
    Output: ε̂ prediction shaped like x_t.
    """
    def __init__(
        self,
        ts_dim: int,
        emb_dim: int,
        time_dim: int   = 128,
        depth: int      = 4,
        heads: int      = 8,
        drop: float     = 0.0,
    ):
        super().__init__()
        # Remove projections and use original dimensions
        self.ts_dim = ts_dim
        self.emb_dim = emb_dim
        self.time_dim = time_dim

        self.pos_encoder = PositionalEncoding(d_model=emb_dim, dropout=drop)
        
        # Only project timestep embedding
        self.proj_xt = nn.Linear(ts_dim, emb_dim)
        self.proj_t = nn.Linear(time_dim, emb_dim)

        # Use emb_dim as the width for transformer blocks
        self.blocks = nn.Sequential(*[
            ResidualAttentionBlock(emb_dim, heads=heads, drop=drop) for _ in range(depth)
        ])

        self.out = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, ts_dim)
        )

    def forward(self, x_t, ts_emb, txt_emb, t):
        B = x_t.size(0)
        t_emb = timestep_embed(t, self.time_dim)        # [B, time_dim]
        t_emb = self.proj_t(t_emb)                      # [B, emb_dim]
        x_t = self.proj_xt(x_t)                          # [B, emb_dim]

        # Stack tokens with their original dimensions
        tokens = torch.stack([
            x_t,                    # noisy token [B, emb_dim]
            ts_emb,                 # cond-ts token [B, emb_dim]
            txt_emb,                # cond-txt token [B, emb_dim]
            t_emb                   # time token [B, emb_dim]
        ], dim=1)                   # → [B, 4, emb_dim]
        tokens = self.pos_encoder(tokens)

        h = self.blocks(tokens)     # Transformer
        eps_hat = self.out(h[:, 0]) # take the first token's output
        return eps_hat


# ---------------------------------------------------------------------------
# Diffusion decoder using the new ε-net
# ---------------------------------------------------------------------------
class DiffusionDecoderAtt(nn.Module):
    """
    Same DDIM sampler as before, but ε-net now = tiny Transformer with attention.
    """
    def __init__(
        self,
        ts_dim: int,
        emb_dim: int,
        steps: int = 20,
        eta: float = 0.0,
        depth: int = 4,
        heads: int = 8,
    ):
        super().__init__()
        self.steps = steps
        self.eta   = eta

        betas  = torch.linspace(1e-4, 0.02, steps)
        alphas = 1. - betas
        acp    = torch.cumprod(alphas, dim=0)

        self.register_buffer("alphas_cumprod",                acp)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1 - acp).sqrt())
        self.register_buffer("sqrt_recip_alphas_cumprod",     (1 / acp).sqrt())

        self.eps_net = AttnEpsNet(ts_dim, emb_dim,depth=depth, heads=heads)

    # --- one DDIM reverse step ------------------------------------------------
    def ddim_step(self, x_t, t, ts_emb, txt_emb):
        a_t      = self.alphas_cumprod[t].view(-1, 1)
        sqrt1m   = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        recip_sqrt = self.sqrt_recip_alphas_cumprod[t].view(-1, 1)

        eps      = self.eps_net(x_t, ts_emb, txt_emb, t)                    # ε̂
        x0_pred  = (x_t - sqrt1m * eps) / recip_sqrt                        # predict x₀

        if (t == 0).all():                                                  # final step
            return x0_pred

        t_prev  = torch.clamp(t - 1, min=0)
        a_prev  = self.alphas_cumprod[t_prev].view(-1, 1)
        sigma   = self.eta * ((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)).sqrt()

        dir_xt  = ((1 - a_prev - sigma**2).sqrt()) * eps
        noise   = sigma * torch.randn_like(x_t) if self.eta > 0 else 0
        return a_prev.sqrt() * x0_pred + dir_xt + noise

    # --- full reverse process --------------------------------------------------
    def forward(self, ts_emb: torch.Tensor, txt_emb: torch.Tensor):
        B   = ts_emb.size(0)
        # x_t = torch.randn(B, ts_emb.size(1), device=ts_emb.device)          # same ts_dim
        x_t = torch.randn(B, self.eps_net.proj_xt.in_features, device=ts_emb.device)


        for i in reversed(range(self.steps)):
            t  = torch.full((B,), i, dtype=torch.long, device=ts_emb.device)
            x_t = self.ddim_step(x_t, t, ts_emb, txt_emb)

        return x_t    # ts_hat


# ---------------------------------------------------------------------------
# Residual attention decoder that takes:
#   • ts_emb  → target
#   • txt_emb → context
# ---------------------------------------------------------------------------
class ResidualAttentionDecoderTXTS(nn.Module):
    """
    Residual-attention decoder that takes:
      • ts_emb  → target
      • txt_emb → context
    """
    def __init__(
        self,
        ts_dim: int,
        output_dim: int,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        project_input: bool = False,
    ):
        super().__init__()
        hidden_dim = output_dim

        self.project_input = project_input
        if self.project_input:
            self.proj_ts = nn.Linear(output_dim, ts_dim)
            self.proj_text = nn.Linear(output_dim, ts_dim)
            hidden_dim = ts_dim

        self.pos_encoder = PositionalEncoding(hidden_dim)

        self.blocks = nn.Sequential(*[
            ResidualAttentionBlock(
                width=hidden_dim,
                heads=nhead,
                ffn_mult=dim_feedforward // hidden_dim,
                drop=dropout
            ) for _ in range(num_layers)
        ])

        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ts_dim)
        )

    @staticmethod
    def _as_sequence(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1) if x.dim() == 2 else x  # [B, 1, E]

    def forward(self, ts_emb: torch.Tensor, txt_emb: torch.Tensor):
        B = ts_emb.size(0)

        tgt    = self._as_sequence(ts_emb)
        memory = self._as_sequence(txt_emb)

        if self.project_input:
            tgt    = self.proj_ts(tgt)
            memory = self.proj_text(memory)

        tokens = torch.cat([tgt, memory], dim=1)  # [B, 2, hidden_dim]
        tokens = self.pos_encoder(tokens)

        h = self.blocks(tokens)
        ts_hat = self.out(h[:, 0])  # return first token's prediction
        return ts_hat.unsqueeze(1)  # shape: [B, 1, ts_dim]


# ---------------------------------------------------------------------------
# Residual attention decoder with a diffusion-style refining tail
# ---------------------------------------------------------------------------
class ResAttDiffDecoder(nn.Module):
    """
    Residual-attention decoder with optional diffusion denoising tail.
    Produces a coarse prediction and optionally refines it with learned noise steps.
    """
    def __init__(
        self,
        ts_dim: int,
        output_dim: int,
        nhead: int = 8,
        num_layers: int = 3,
        ffn_mult = 1,
        dropout: float = 0.1,
        project_input: bool = False,
        diffusion_steps: int = 10,  # if > 0, apply diffusion tail
        diff_txt_proj: bool = False,
    ):
        super().__init__()
        hidden_dim = output_dim
        self.diffusion_steps = diffusion_steps

        self.project_input = project_input
        if self.project_input:
            self.proj_ts = nn.Linear(output_dim, ts_dim)
            self.proj_text = nn.Linear(output_dim, ts_dim)
            hidden_dim = ts_dim

        self.pos_encoder = PositionalEncoding(hidden_dim)

        self.blocks = nn.Sequential(*[
            ResidualAttentionBlock(
                width=hidden_dim,
                heads=nhead,
                ffn_mult=ffn_mult,
                drop=dropout
            ) for _ in range(num_layers)
        ])

        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ts_dim)
        )

        # Optional diffusion tail
        if diffusion_steps > 0:
            self.diffusion_tail = DiffusionRefiner(
                ts_dim=ts_dim,
                txt_dim=hidden_dim,
                n_steps=diffusion_steps,
                diff_txt_proj=diff_txt_proj
            )

    @staticmethod
    def _as_sequence(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1) if x.dim() == 2 else x  # [B, 1, E]

    def forward(self, ts_emb: torch.Tensor, txt_emb: torch.Tensor, refine: bool = True):
        B = ts_emb.size(0)

        tgt    = self._as_sequence(ts_emb)
        memory = self._as_sequence(txt_emb)

        if self.project_input:
            tgt    = self.proj_ts(tgt)
            memory = self.proj_text(memory)

        tokens = torch.cat([tgt, memory], dim=1)  # [B, 2, hidden_dim]
        tokens = self.pos_encoder(tokens)

        h = self.blocks(tokens) # [B, 2, hidden_dim]
        ts_hat = self.out(h[:, 0])  # [B, ts_dim]

        if self.diffusion_steps > 0 and refine:
            # ts_hat = ts_hat + self.diffusion_tail(ts_hat, txt_emb)
            ts_hat = self.diffusion_tail(ts_hat, txt_emb)

        return ts_hat.unsqueeze(1)  # [B, 1, ts_dim]

class ResAttVDecoder(nn.Module):
    """
    Residual-attention decoder with optional diffusion denoising tail.
    Produces a coarse prediction and optionally refines it with learned noise steps.
    """
    def __init__(
        self,
        ts_dim: int,
        output_dim: int,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 768,
        dropout: float = 0.1,
        project_input: bool = False
    ):
        super().__init__()
        hidden_dim = output_dim 

        self.project_input = project_input
        if self.project_input:
            self.proj_ts = nn.Linear(output_dim, ts_dim)
            self.proj_text = nn.Linear(output_dim, ts_dim)
            hidden_dim = ts_dim

        self.pos_encoder = PositionalEncoding(hidden_dim)

        self.blocks = nn.Sequential(*[
            ResidualAttentionBlock(
                width=hidden_dim,
                heads=nhead,
                ffn_mult=dim_feedforward // hidden_dim,
                drop=dropout
            ) for _ in range(num_layers)
        ])

        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ts_dim)
        )
    
    @staticmethod
    def _as_sequence(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1) if x.dim() == 2 else x  # [B, 1, E]

    def forward(self, ts_emb: torch.Tensor, txt_emb: torch.Tensor):
        tgt    = self._as_sequence(ts_emb)
        memory = self._as_sequence(txt_emb)

        if self.project_input:
            tgt    = self.proj_ts(tgt)
            memory = self.proj_text(memory)

        tokens = torch.cat([tgt, memory], dim=1)  # [B, 2, hidden_dim]
        tokens = self.pos_encoder(tokens)

        h = self.blocks(tokens) # [B, 2, hidden_dim]
        ts_hat_mean = self.out(h[:, 0])  # [B, ts_dim]
        ts_hat_std = self.out(h[:, 1])  # [B, ts_dim]

        # Add noise with learned variance
        noise = torch.randn_like(ts_hat_mean) * ts_hat_std
        ts_hat = ts_hat_mean + noise

        return ts_hat.unsqueeze(1)  # [B, 1, ts_dim]


