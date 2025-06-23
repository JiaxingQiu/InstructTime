# from config import *
import torch
import torch.nn as nn
from .helper import ResidualAttentionBlock

# ------- custom ts encoder_layers (for VITAL embedding generation) -------
class TransformerBlock(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Pre-norm attention
        normed = self.norm1(x)
        attended, _ = self.attention(normed, normed, normed)
        x = x + attended
        
        # Pre-norm MLP
        normed = self.norm2(x)
        x = x + self.mlp(normed)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(
            # First transformation
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),  # Dropout after activation
            
            # Second transformation
            nn.Linear(hidden_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.Dropout(dropout)   # Dropout before residual connection
        )
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        return self.activation(x + self.proj(x))

class ResNetEncoder(nn.Module):
    def __init__(self, ts_dim, output_dim, hidden_dim=128, num_blocks=2, dropout=0.1):
        super().__init__()
        self.ts_dim = ts_dim
        self.output_dim = output_dim
        # layers = [
        #     # Initial projection
        #     nn.Linear(ts_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(dropout)
        # ]
        layers = [
            ResidualBlock(
                    ts_dim, 
                    ts_dim,
                    dropout=0.1
                ),
            nn.Linear(ts_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
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
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.LeakyReLU(0.2)
        ])

        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encoder(x)

# resnet_encoder = ResNetEncoder(ts_dim=300, output_dim=128)
# model = GeneralBinaryClassifier(resnet_encoder)
# model = CLIPModel(ts_encoder=resnet_encoder, text_encoder=None)



# class Lambda(nn.Module):
#     def __init__(self, func):
#         super().__init__()
#         self.func = func
    
#     def forward(self, x):
#         return self.func(x) 

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

# # CNN Encoder
# cnn = CNNEncoder(
#     ts_dim=300,
#     output_dim=128,
#     num_channels=[32, 64, 128],  # Three conv layers
#     kernel_size=5,
#     dropout=0.2
# )
# model = GeneralBinaryClassifier(cnn)

class MultiCNNEncoder(nn.Module):
    def __init__(self, ts_dim, output_dim, kernel_sizes=[80, 50, 20, 5], hidden_num_channel=16, dropout=0.0):
        """
        Multi-resolution CNN encoder with attention mechanism.
        
        Args:
            ts_dim (int): Input time series length
            output_dim (int): Output embedding dimension
            kernel_sizes (list): Different kernel sizes for multi-resolution analysis
            hidden_num_channel (int): Number of channels in CNN layers
            dropout (float): Dropout rate
        """
        super().__init__()
        
        self.ts_dim = ts_dim
        self.output_dim = output_dim

        # Create CNNs for different resolutions
        self.cnns = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.cnns.append(CNNEncoder(ts_dim, 
                                      output_dim,
                                      num_channels=[hidden_num_channel],
                                      kernel_size=kernel_size, 
                                      dropout=dropout))
        
        # Attention mechanism to combine different resolutions
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=1,  # Single head for clear attention weights
            batch_first=True
        )
        
        # Trainable query vector for attention
        self.query = nn.Parameter(torch.randn(1, 1, output_dim))
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        # Get embeddings from each CNN
        cnn_embeddings = []
        for cnn in self.cnns:
            embedding = cnn(x)  # [batch_size, output_dim]
            cnn_embeddings.append(embedding)
        
        # Stack embeddings from different resolutions [batch_size, n_kernels, output_dim]
        cnn_embeddings = torch.stack(cnn_embeddings, dim=1)
        
        # Apply layer normalization
        cnn_embeddings = self.layer_norm(cnn_embeddings)
        
        # Expand query for batch size
        query = self.query.expand(cnn_embeddings.size(0), -1, -1)
        
        # Apply attention to combine different resolutions
        attended_output, _ = self.attention(
            query=query,  # [batch_size, 1, output_dim]
            key=cnn_embeddings,  # [batch_size, n_kernels, output_dim]
            value=cnn_embeddings  # [batch_size, n_kernels, output_dim]
        )
        
        # Remove the query dimension and get final embedding
        combined_output = attended_output.squeeze(1)  # [batch_size, output_dim]
        
        return combined_output


# old default encoder layers
# MultiCNNEncoder(ts_dim = ts_dim,
#                 output_dim=output_dim,
#                 kernel_sizes=[150, 100, 50, 10],
#                 hidden_num_channel=16,
#                 dropout=0)

class MultiCNNEncoderScalar(nn.Module):
    """
    Multi-resolution CNN encoder that fuses branch embeddings with a **learned
    softmax weight per kernel size** (“scalar gating”).

    Each branch → (B, D); stack → (B, n_k, D); softmax over a trainable
    vector `alpha` gives weights that are applied and summed.
    """

    def __init__(
        self,
        ts_dim: int,
        output_dim: int,
        kernel_sizes=[80, 50, 20, 5],
        hidden_num_channel: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.cnns = nn.ModuleList(
            [
                CNNEncoder(
                    ts_dim,
                    output_dim,
                    num_channels=[hidden_num_channel],
                    kernel_size=k,
                    dropout=dropout,
                )
                for k in kernel_sizes
            ]
        )

        self.layer_norm = nn.LayerNorm(output_dim)

        # one score per branch, initialised to zero → uniform weights
        self.alpha = nn.Parameter(torch.zeros(len(kernel_sizes)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_embeds = torch.stack([cnn(x) for cnn in self.cnns], dim=1)  # (B,n_k,D)
        branch_embeds = self.layer_norm(branch_embeds)

        w = torch.softmax(self.alpha, 0)                                   # (n_k,)
        combined = (branch_embeds * w.view(1, -1, 1)).sum(1)              # (B,D)

        return combined


class MultiCNNEncoderMaxPool(nn.Module):
    """
    Multi-resolution CNN encoder that takes the **feature-wise maximum**
    across kernel branches (parameter-free).

    Good when informative patterns appear strongly in just one scale.
    """

    def __init__(
        self,
        ts_dim: int,
        output_dim: int,
        kernel_sizes=[80, 50, 20, 5],
        hidden_num_channel: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.cnns = nn.ModuleList(
            [
                CNNEncoder(
                    ts_dim,
                    output_dim,
                    num_channels=[hidden_num_channel],
                    kernel_size=k,
                    dropout=dropout,
                )
                for k in kernel_sizes
            ]
        )

        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_embeds = torch.stack([cnn(x) for cnn in self.cnns], dim=1)  # (B,n_k,D)
        branch_embeds = self.layer_norm(branch_embeds)

        combined, _ = branch_embeds.max(dim=1)                             # (B,D)

        return combined




class MLPEncoder(nn.Module):
    def __init__(self, ts_dim, output_dim, hidden_dim=128, num_hidden_layers=6, dropout=0.2):
        """
        Multi-layer perceptron encoder.
        
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
                nn.Linear(ts_dim, hidden_dim),
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
            hidden_dim = ts_dim
        
        # Final projection - extend instead of append for list of layers
        layers.extend([
            nn.Linear(hidden_dim, output_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        ])
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encoder(x)

# # Linear Encoder
# mlp = MLPEncoder(
#     ts_dim=300,
#     output_dim=128,   
#     hidden_dim=128,
#     num_hidden_layers=6,
#     dropout=0.2
# )
# # Create classifiers
# model = GeneralBinaryClassifier(mlp)

class LSTMEncoder(nn.Module):
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

# Usage:
# lstm_encoder = LSTMEncoder(ts_dim=300, output_dim=128)
# model = GeneralBinaryClassifier(lstm_encoder)
# model = CLIPModel(ts_encoder=lstm_encoder, text_encoder=None)



class MultiLSTMEncoder(nn.Module):
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
                input_size=ts_dim,
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
            nn.Linear(total_lstm_dim // 2, output_dim),
            nn.BatchNorm1d(output_dim)
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

# Usage:
# multi_lstm_encoder = MultiLSTMEncoder(
#     ts_dim=300, 
#     output_dim=128,
#     hidden_dims=[128, 256, 64],  # Three LSTMs with different sizes
#     num_layers=2
# )
# model = GeneralBinaryClassifier(multi_lstm_encoder)
# model = CLIPModel(ts_encoder=multi_lstm_encoder, text_encoder=None)


# # old default encoder layers
# nn.Sequential(
#     MultiLSTMEncoder(
#         ts_dim=ts_dim, 
#         output_dim=256,
#         hidden_dims=[512, 512, 256, 256],  # LSTMs with different sizes
#         num_layers=2,
#         dropout=0,
#         bidirectional=False,
#         mask=0  # mask 0 with 0 to suppress the effect of masked values
#     ),
#     nn.LayerNorm(256),  # Add LayerNorm at the end
#     nn.Linear(256, 512),
#     nn.LeakyReLU(0.2),
#     nn.LayerNorm(512),
#     nn.Linear(512, output_dim),
#     nn.LeakyReLU(0.2),
#     nn.LayerNorm(output_dim)
# )


class ResAttEncoder(nn.Module):
    """
    Pure Transformer encoder for **univariate** time series.
    Args
    ----
    ts_dim         : int   – length of the series (token count L)
    output_dim      : int   – embedding size D returned by the encoder
    n_layers        : int   – number of ResidualAttentionBlocks
    width           : int   – internal width (defaults to output_dim)
    pool            : 'mean' | 'max' | 'cls'
    """
    def __init__(
        self,
        ts_dim: int,
        output_dim: int,
        n_layers: int = 3,
        heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        width: int | None = None,
        pool: str = "mean",
    ):
        super().__init__()
        self.width = width = output_dim if width is None else width
        self.pool  = pool = pool.lower()

        # 1) scalar-to-vector projection (value embedding)
        self.val_proj = nn.Linear(1, width)

        # 2) optional CLS token
        if pool == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, width))
            ts_dim += 1

        # 3) learnable positional embedding
        self.pos = nn.Parameter(torch.randn(1, ts_dim, width))

        # 4) transformer blocks
        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, ffn_mult, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(width)
        self.out_proj = nn.Linear(width, output_dim)

    def _pool(self, x):                       # x: [B,L,D] after norm
        if self.pool == "cls":
            return x[:, 0]                    # CLS token
        elif self.pool == "mean":
            return x.mean(dim=1)              # average over time
        elif self.pool == "max":
            return x.max(dim=1).values        # max over time
        else:
            raise ValueError("pool must be 'cls', 'mean', or 'max'")

    def forward(self, x):
        """
        x : Tensor [B, ts_dim]  – raw univariate series
        """
        B, L = x.shape
        x = x.unsqueeze(-1)                   # [B,L,1]
        x = self.val_proj(x)                  # [B,L,D]

        if self.pool == "cls":
            cls = self.cls_token.expand(B, -1, -1)  # [B,1,D]
            x   = torch.cat([cls, x], dim=1)

        x = x + self.pos[:, : x.size(1)]      # add positions

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)                      # [B,L,D]
        x = self._pool(x)                     # [B,D]
        return self.out_proj(x)               # [B,output_dim]
    


class TSAttentionEncoder(nn.Module):
    """
    Simple attention-based encoder for univariate time series.
    Pools a sequence of scalars (time series) into a fixed-size embedding
    using a single trainable query and multi-head attention.

    Args:
        sequence_length: Length of the time series (number of time steps)
        output_dim: Output embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    def __init__(self, ts_dim, output_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.sequence_length = ts_dim
        self.output_dim = output_dim
        self.value_proj = nn.Linear(1, output_dim)  # project scalar to embedding
        self.attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, output_dim))
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, sequence_length] (univariate time series)
        if x.dim() != 2 or x.size(1) != self.sequence_length:
            raise ValueError(f"Input must be of shape [B, {self.sequence_length}]")
        x = x.unsqueeze(-1)  # [B, sequence_length, 1]
        x = self.value_proj(x)  # [B, sequence_length, output_dim]
        x = self.norm(x)
        q = self.query.expand(x.size(0), -1, -1)  # [B, 1, output_dim]
        pooled, _ = self.attn(q, x, x)  # [B, 1, output_dim]
        pooled = pooled.squeeze(1)  # [B, output_dim]
        return self.dropout(pooled)



# ------- custom text encoder_layers (for VITAL embedding generation) -------
class IdenticalEncoder(nn.Module):
    def __init__(self, text_dim: int, output_dim: int):
        super().__init__()
        if text_dim != output_dim:
            raise ValueError(
                f"Identity mapping requires text_dim == output_dim "
                f"(got {text_dim} vs {output_dim})."
            )
    def forward(self, text_features):
        return text_features

# MLP text encoder
class TextEncoderMLP(nn.Module):
    def __init__(self, 
                 text_dim: int, 
                 output_dim: int):
        super().__init__()
        
        # Simple MLP with 3 layers
        self.encoder = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, text_features):
        tx_emb = self.encoder(text_features)
        return tx_emb


# CNN text encoder
class TextEncoderCNN(nn.Module):
    def __init__(self, 
                 text_dim: int,
                 output_dim: int,
                 num_channels=[64],#, 128, 256
                 kernel_size=50,
                 dropout=0):
        """
        CNN encoder for text features.
        
        Args:
            output_dim (int): Output embedding dimension
            num_channels (list): Number of channels for each conv layer
            kernel_size (int): Kernel size for conv layers
            dropout (float): Dropout rate
        """
        super().__init__()
        
        layers = [AddChannelDim()]  # Add channel dimension
        in_channels = 1
        
        # Add conv blocks
        for out_channels in num_channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
                nn.GELU(),
                nn.BatchNorm1d(out_channels),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        layers.append(nn.Flatten())
        
        # Calculate output dimension
        with torch.no_grad():
            x = torch.zeros(2, text_dim)
            for layer in layers:
                x = layer(x)
            conv_out_dim = x.shape[1]
        
        # Add final linear projection to match output_dim
        layers.append(nn.Linear(conv_out_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, text_features):
        return self.encoder(text_features)




class TextEncoderMultiCNN(nn.Module):
    def __init__(self, 
                 text_dim: int,
                 output_dim: int,
                 kernel_sizes=[500, 250, 100, 50],  # Different context windows
                 hidden_num_channel=16,
                 dropout=0.0):
        """
        Multi-resolution CNN encoder for text features with attention mechanism.
        
        Args:
            text_dim (int): Input text embedding dimension
            output_dim (int): Output embedding dimension
            kernel_sizes (list): Different kernel sizes for multi-resolution analysis
            hidden_num_channel (int): Number of channels in CNN layers
            dropout (float): Dropout rate
        """
        super().__init__()
        
        # Create CNNs for different resolutions
        self.cnns = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.cnns.append(
                TextEncoderCNN(
                    text_dim=text_dim,
                    output_dim=output_dim,
                    num_channels=[hidden_num_channel],
                    kernel_size=kernel_size,
                    dropout=dropout
                )
            )
        
        # Attention mechanism to combine different resolutions
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=1,  # Single head for clear attention weights
            batch_first=True
        )
        
        # Trainable query vector for attention
        self.query = nn.Parameter(torch.randn(1, 1, output_dim))
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, text_features):
        # Get embeddings from each CNN
        cnn_embeddings = []
        for cnn in self.cnns:
            embedding = cnn(text_features)  # [batch_size, output_dim]
            cnn_embeddings.append(embedding)
        
        # Stack embeddings from different resolutions [batch_size, n_kernels, output_dim]
        cnn_embeddings = torch.stack(cnn_embeddings, dim=1)
        
        # Apply layer normalization
        cnn_embeddings = self.layer_norm(cnn_embeddings)
        
        # Expand query for batch size
        query = self.query.expand(cnn_embeddings.size(0), -1, -1)
        
        # Apply attention to combine different resolutions
        attended_output, _ = self.attention(
            query=query,  # [batch_size, 1, output_dim]
            key=cnn_embeddings,  # [batch_size, n_kernels, output_dim]
            value=cnn_embeddings  # [batch_size, n_kernels, output_dim]
        )
        
        # Remove the query dimension and get final embedding
        combined_output = attended_output.squeeze(1)  # [batch_size, output_dim]
        
        return combined_output

# Usage example:
# old default text encoder layer 
# TextEncoderMultiCNN(text_dim = text_dim,
#                     output_dim=output_dim,
#                     kernel_sizes=[768, 384, 192],  # Different context windows
#                     hidden_num_channel=16,
#                     dropout=0.0)

class TextEncoderAttention(nn.Module):
    """
    A minimal text encoder that summarizes a sequence of token embeddings
    with a single trainable–query multi-head attention layer.

    Args
    ----
    text_dim   : int  – dimension of the incoming token embeddings
    output_dim : int  – dimension of the fixed-length text representation
    num_heads  : int  – number of attention heads (≥ 1)
    dropout    : float
    """

    def __init__(
        self,
        text_dim: int,
        output_dim: int,
        num_heads: int = 1, # single head
        dropout: float = 0.0,
    ):
        super().__init__()

        # Single-layer (self-)attention that pools the whole sequence.
        self.attn = nn.MultiheadAttention(
            embed_dim=text_dim,  # Use text_dim directly
            num_heads=num_heads,
            batch_first=True,  # (B, L, D) I/O layout
        )

        # Single trainable query that "asks" the sequence for information.
        self.query = nn.Parameter(torch.randn(1, 1, text_dim))  # Match text_dim

        # Project to output dimension after attention
        self.output_proj = nn.Linear(text_dim, output_dim)
        self.gelu = nn.GELU()

        # Normalization & dropout for a touch of stability.
        # self.norm = nn.LayerNorm(text_dim)  # Normalize text_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        text_features : Tensor, shape (batch, text_dim) or (batch, seq_len, text_dim)
            Token embeddings (e.g. from a pretrained word / sentence encoder).
            If 2D, will be treated as a single-token sequence.

        Returns
        -------
        pooled_repr : Tensor, shape (batch, output_dim)
            Fixed-length embedding summarizing the whole sequence.
        """
        # Handle 2D input by adding sequence length dimension
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)  # (B, 1, D)

        # Normalize input tokens
        tokens = text_features # self.norm(text_features)  # (B, L, D)

        # Expand the shared query to the current batch size.
        q = self.query.expand(tokens.size(0), -1, -1)  # (B, 1, D)

        # Attention pooling.
        pooled, _ = self.attn(query=q, key=tokens, value=tokens)  # (B, 1, D)
        pooled = pooled.squeeze(1)                               # (B, D)

        # Project to output dimension
        pooled = self.output_proj(pooled)  # (B, output_dim)
        pooled = self.gelu(pooled)
        pooled = self.dropout(pooled)

        return pooled


# usage:
# text_encoder = TextEncoderAttention(
#     text_dim=768,  # e.g., BERT embedding dimension
#     output_dim=128,
#     num_heads=4,
#     dropout=0.1
# )


# # default text encoder with attention
# class TextEncoder(nn.Module):
#     def __init__(self, 
#                  text_dim: int, 
#                  output_dim: int,
#                  dropout: float = 0.0):
#         """Text encoder using transformer blocks
        
#         Args:
#             text_dim (int): Input text embedding dimension
#             output_dim (int): Output embedding dimension
#         """
#         super().__init__()
#         self.encoder = nn.Sequential(
#             # Initial projection
#             nn.Linear(text_dim, 512),
#             # nn.LayerNorm(512),
#             nn.GELU(),
#             nn.Dropout(dropout),
            
#             # Transformer blocks
#             TransformerBlock(dim=512, hidden_dim=1024, num_heads=8, dropout=dropout),
#             TransformerBlock(dim=512, hidden_dim=1024, num_heads=8, dropout=dropout),
#             TransformerBlock(dim=512, hidden_dim=1024, num_heads=8, dropout=dropout),
            
#             # Final projection
#             nn.Linear(512, output_dim)
#         )

#     def forward(self, text_features):
#         """Forward pass
        
#         Args:
#             text_features (torch.Tensor): Input tensor of shape [batch_size, text_dim]
            
#         Returns:
#             torch.Tensor: Encoded text of shape [batch_size, output_dim]
#         """

#         tx_emb = self.encoder(text_features)
#         tx_emb = F.normalize(tx_emb, dim=1)
#         return tx_emb
# # test text encoder
# # Initialize text encoder
# text_dim = 768  # typical BERT dimension
# output_dim = 10
# text_encoder = TextEncoder(text_dim, output_dim)
# # Create a single random vector
# single_vector = torch.randn(text_dim)
# # Create a batch of repeated vectors
# batch_size = 4
# repeated_batch = single_vector.unsqueeze(0).repeat(batch_size, 1)
# # Create a mixed batch where first element is single_vector and rest are random
# mixed_batch = torch.cat([
#     single_vector.unsqueeze(0),  # First element is single_vector
#     torch.randn(batch_size - 1, text_dim)  # Rest are random
# ], dim=0)
# # Process single vector
# single_output = text_encoder(single_vector.unsqueeze(0))[0]  # Take first element since encoder expects batch
# # Process repeated batch
# repeated_output = text_encoder(repeated_batch)
# # Process mixed batch
# mixed_output = text_encoder(mixed_batch)
# print(single_output)
# print(repeated_output[0])
# print(mixed_output[0])