import torch
import matplotlib.pyplot as plt
import numpy as np
from data import get_features, get_features3d
from config import *

def plot_reconstructions(model,
                         df, 
                         config_dict,
                         num_samples=20,
                         title="Data Reconstructions"):
    """
    Plot original and reconstructed time series from a VAE model.
    
    Args:
        model: The VAE model
        df: Dataframe containing the dataset
        config_dict: Config dictionary
        text_col: Text column name
        text_col_ls: List of text column names
        num_samples: Number of samples to plot (default=20)
        start_idx: Starting index for samples (default=0)
        title: Title for the plot (default="Data Reconstructions")
    """
    model.eval()
    # get sample without replacement num_samples
    df = df.sample(n=num_samples, replace=False)
    # # get row start_idx:start_idx + num_samples
    # df = df.iloc[start_idx:(start_idx + num_samples)]  
    
    if config_dict['3d']:
        ts_f, tx_f_ls, _ = get_features3d(df, config_dict, text_col_ls = config_dict['text_col_ls'])
        ts_f = ts_f.to(device)
        tx_f_ls = [tx_f.to(device) for tx_f in tx_f_ls]
        _, ts_hat, _, _ = model(ts_f, tx_f_ls)
    else:
        ts_f, tx_f, _ = get_features(df, config_dict, text_col = 'text')
        ts_f = ts_f.to(device)
        tx_f = tx_f.to(device)
        _, ts_hat, _, _ = model(ts_f, tx_f)


    rows = num_samples // 5
    fig, axes = plt.subplots(rows, 5, figsize=(20, 2*rows), facecolor='white')

    with torch.no_grad():
        for i in range(num_samples):
            # Plot
            row, col = i//5, i%5
            axes[row, col].plot(ts_f[i].cpu().numpy(), label='Original')
            axes[row, col].plot(ts_hat[i].cpu().detach().numpy(), label='Reconstructed')
            axes[row, col].legend()
            axes[row, col].set_title(f'Sample {i}')

    plt.tight_layout()
    fig.suptitle(title, y=1.02, fontsize=16)
    plt.show()



def plot_reconstruction_from_distances(model, 
                                       df, 
                                       config_dict,
                                       sample_idx=1, 
                                       distances=[0, 5e-4, 7.5e-4, 1e-3, 2e-3]):
    """
    Visualize how different levels of latent space noise affect reconstruction quality.
    
    Args:
        model: The VAE model
        df: Dataframe containing the dataset
        config_dict: Config dictionary
        text_col: Text column name
        text_col_ls: List of text column names
        sample_idx: Index of the sample to analyze (default=1)
        distances: List of noise variances to add to latent space (default=[0, 5e-4, 7.5e-4, 1e-3, 2e-3])
    """
    model.eval()
    # x, _, _ = dataloader.dataset[sample_idx]

    # get row start_idx:start_idx + num_samples
    df = df.iloc[[sample_idx]]  
    if config_dict['3d']:
        ts_f, tx_f_ls, _ = get_features3d(df, config_dict, text_col_ls = config_dict['text_col_ls'])
        ts_f = ts_f.to(device)
        tx_f_ls = [tx_f.to(device) for tx_f in tx_f_ls]
        tx_emb, _ = model.text_encoder(tx_f_ls)
        x = ts_f[0]
    else:
        ts_f, tx_f, _ = get_features(df, config_dict, text_col = 'text')
        ts_f = ts_f.to(device)
        tx_f = tx_f.to(device)
        tx_emb = model.text_encoder(tx_f)
        x = ts_f[0]


    # Create figure and subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Sample {sample_idx} vs Reconstructions from distances', fontsize=14)
    axs = axs.flatten()

    # Plot original signal in first subplot
    axs[0].plot(x.cpu().detach().numpy(), 'b-', label='Original')
    axs[0].set_title('Original Signal')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Value')
    axs[0].grid(True)
    axs[0].legend()

    with torch.no_grad():
        # Plot reconstructions with different distances
        for i, distance in enumerate(distances, 1):
            if x.dim() == 1:
                x = x.unsqueeze(0) # add batch dimension
            # Get embeddings and reconstruction
            _, z_mean, z_log_var = model.ts_encoder(x)
            z = model.ts_encoder.reparameterization(z_mean, z_log_var + distance)
            x_hat = model.ts_decoder(z, tx_emb, x, tx_emb)
            
            _, z_mean_hat, _ = model.ts_encoder(x_hat)
            
            # Calculate Euclidean distance between original and reconstructed latents
            z_mean = z_mean.cpu().detach().numpy()
            z_mean_hat = z_mean_hat.cpu().detach().numpy()
            euc_dist = np.sqrt(np.sum((z_mean - z_mean_hat) ** 2))

            # Plot reconstruction
            if x_hat.dim() == 2:
                x_hat = x_hat[0]
                x = x[0]
            axs[i].plot(x_hat.cpu().detach().numpy(), 'r-', label='Reconstruction')
            axs[i].plot(x.cpu().detach().numpy(), 'b--', alpha=0.5, label='Original')
            axs[i].set_title(f'Noise var={distance:.1e}\nLatent space distance={euc_dist:.4f}')
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel('Value')
            axs[i].grid(True)
            axs[i].legend()

    plt.tight_layout()
    plt.show()
# Usage example:
# plot_reconstruction_noise_analysis(model, train_dataloader)
# 
# # With custom parameters:
# plot_reconstruction_noise_analysis(
#     model, 
#     train_dataloader,
#     sample_idx=5,
#     distances=[0, 1e-4, 1e-3, 1e-2]
# )