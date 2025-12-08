import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import matplotlib.pyplot as plt
import numpy as np
import torch
from config import *
from data import get_features3d, get_features


# ----------------- Augment Time Series by Text Instructions --------------------------------
def interpolate_ts_tx(df, model, config_dict, text_cols, w):

    model.eval() # 2d vital model
    ts_f, tx_f_ls, _ = get_features3d(df, config_dict, text_col_ls = text_cols)
    ts_f = ts_f.to(device)
    tx_f_ls = [tx_f.to(device) for tx_f in tx_f_ls]
    _, tx_f_src, _ = get_features(df, config_dict, text_col = 'text')
    tx_f_src = tx_f_src.to(device)

    # ----- ts_embeddings -----
    ts_emb, ts_emb_mean, _ = model.ts_encoder(ts_f)
    if not model.variational:
        ts_emb = ts_emb_mean

    ts_hat_ls = {}
    for txid in range(len(tx_f_ls)):
        ts_hat, _, _, _ = model.generate(w, ts_f, tx_f_ls[txid], tx_f_src)
        text_condition = df[text_cols[txid]]
        ts_hat_ls[text_cols[txid]] = list(zip(text_condition, ts_hat))
    
    return ts_hat_ls



# ----------------- Visualize augmentation of a single time series (input as df) --------------------------------
def interpolate_ts_tx_single(df, model, config_dict, text_cols, w, label = False):
    model.eval() # 2d vital model
    ts_f, tx_f_ls, _ = get_features3d(df, config_dict, text_col_ls = text_cols)
    ts_f = ts_f.to(device)
    raw_ts = ts_f[0].detach().cpu().numpy()
    tx_f_ls = [tx_f.to(device) for tx_f in tx_f_ls]
    _, tx_f_src, _ = get_features(df, config_dict, text_col = 'text')
    tx_f_src = tx_f_src.to(device)
    
    ts_hat_ls = {}
    simi = {}
    l1 = {}
    l2 = {}
    for txid in range(len(tx_f_ls)):
        ts_hat, ts_emb_tgt, tx_emb_tgt, _ = model.generate(w, ts_f, tx_f_ls[txid], tx_f_src)
        # # plot the ts_hat
        # plt.plot(ts_hat[0].detach().cpu().numpy())
        # plt.show()
        if label:
            text_condition = df[text_cols[txid]].iloc[0] # string version of text condition
        else:
            text_condition = text_cols[txid]

        ts_hat_ls[text_condition] = ts_hat[0].detach().cpu().numpy()
        logits = torch.matmul(ts_emb_tgt, tx_emb_tgt.T) 
        simi[text_condition] = torch.diag(logits).detach().cpu().numpy()# torch.exp()
        l2[text_condition] = torch.norm(ts_emb_tgt - tx_emb_tgt, dim=1, p=2).detach().cpu().numpy()  
        l1[text_condition] = torch.norm(ts_emb_tgt - tx_emb_tgt, dim=1, p=1).detach().cpu().numpy()
        
    ts2tx_distances = {'simi': simi, 'l1': l1, 'l2': l2}
    
    return ts2tx_distances, ts_hat_ls, raw_ts


def plot_interpolate_ts_tx_ws(df, model, config_dict, text_cols, w_values=None, label = True):
    """
    Plot interpolated time series for multiple w values in a grid layout.
    Creates a separate figure for each text condition.
    
    Args:
        df: Input DataFrame
        model: VITAL model
        config_dict: Configuration dictionary
        text_cols: List of text conditions
        w_values: List of interpolation weights (default: np.arange(0, 1.1, 0.1))
    """
    if w_values is None:
        w_values = np.arange(0, 1.1, 0.1)
    
    # Get ground truth reconstruction       
    _, ts_hats, raw_ts = interpolate_ts_tx_single(df, model, config_dict, 
                                         text_cols = ['text'], # ground truth text condition
                                         w=0, # reconstruction only on ground truth text condition
                                         label = False)
    
    # Collect all results first
    all_results = {}
    for w in w_values:
        dists, ts_hats, _ = interpolate_ts_tx_single(df, model, config_dict, text_cols, w=w, label=label)
        all_results[w] = {'dists': dists, 'ts_hats': ts_hats}
    
    # Create a figure for each text condition
    for text_condition in list(dists['simi'].keys()):
        # Calculate grid dimensions for this text condition
        n_cols = 5
        n_rows = int(np.ceil(len(w_values) / n_cols))
        
        # Create figure and subplots
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
        fig.suptitle(f'Augmenting towards: {text_condition}', fontsize=18)
        
        # Flatten axes array if needed
        if n_rows == 1:
            axs = axs.reshape(1, -1)
        
        # Plot for each w value
        for idx, w in enumerate(w_values):
            row = idx // n_cols
            col = idx % n_cols
            
            # Get results for this w value
            dists = all_results[w]['dists']
            ts_hats = all_results[w]['ts_hats']
            ts_hat = ts_hats[text_condition]
            
            # Plot reconstruction and ground truth
            axs[row, col].plot(ts_hat, 'r-', label='Augmented') #np.round(ts_hat)
            axs[row, col].plot(raw_ts, 'b--', alpha=0.5, label='Raw')
            
            # Add distance metrics to title
            simi_val = float(dists['simi'][text_condition]) if isinstance(dists['simi'][text_condition], np.ndarray) else dists['simi'][text_condition]
            l1_val = float(dists['l1'][text_condition]) if isinstance(dists['l1'][text_condition], np.ndarray) else dists['l1'][text_condition]
            l2_val = float(dists['l2'][text_condition]) if isinstance(dists['l2'][text_condition], np.ndarray) else dists['l2'][text_condition]
            
            w_title = f'w = {w:.1f}' if w != 0 else 'w = 0 (reconstruction)'
            title = f'{w_title}\nCosine: {simi_val:.3f}\nL2: {l2_val:.3f}' # \nL1: {l1_val:.3f}
            axs[row, col].set_title(title, pad=15, fontsize=10)
            axs[row, col].set_xlabel('Time Steps', fontsize=9)
            axs[row, col].set_ylabel('Value', fontsize=9)
            axs[row, col].grid(True, linestyle='--', alpha=0.3)
            axs[row, col].legend()
            
            # Remove top and right spines
            axs[row, col].spines['top'].set_visible(False)
            axs[row, col].spines['right'].set_visible(False)
        
        # Hide any empty subplots
        for idx in range(len(w_values), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axs[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()


def viz_generation_marginal(df, model, config_dict, tid=0, w_values = np.arange(0.4, 1.2, 0.2),
                               sampling = False, b=200, ep=1, ylims = None):
    model.eval()
    w_values = np.concatenate([[0], w_values])
    for y_col in config_dict['txt2ts_y_cols']:
        y_levels = list(df[y_col].unique())
        for i in range(len(y_levels)):
            df_level = df[df[y_col] == y_levels[i]].reset_index(drop=True).iloc[[tid]].copy()
            print(df_level[y_col])
            for j in range(len(y_levels)):
                df_level['text' + str(j)] = y_levels[j]
            text_cols = ['text' + str(j) for j in range(len(y_levels))]
            if sampling:
                plot_interpolate_ts_tx_ws_sampling(df_level, model, config_dict, text_cols, w_values = w_values, 
                                                   label = True, b=b, ep=ep, ylims = ylims)
            else:
                plot_interpolate_ts_tx_ws(df_level, model, config_dict, text_cols, w_values = w_values, label = True)


def viz_generation_conditional(df, model, config_dict, tid=0, w_values = np.arange(0.4, 1.2, 0.2),
                               sampling = False, b=200, ep=1, ylims = None):
    model.eval()
    w_values = np.concatenate([[0], w_values])
    for y_col in config_dict['txt2ts_y_cols']:
        y_levels = list(df[y_col].unique())
        for i in range(len(y_levels)):
            df_level = df[df[y_col] == y_levels[i]].reset_index(drop=True).iloc[[tid]].copy()
            print(df_level[y_col])
            for j in range(len(y_levels)):
                # find the substring in raw_text(df_level['text'].values[0]) that is the same as any of the elemnet in y_levels
                # replace the substring y_levels[j] 
                modified_text = df_level['text'].values[0]
                for level in y_levels:
                    if level in modified_text:
                        modified_text = modified_text.replace(level, y_levels[j])
                df_level['text' + str(j)] = modified_text
            text_cols = ['text' + str(j) for j in range(len(y_levels))]
            if sampling:
                plot_interpolate_ts_tx_ws_sampling(df_level, model, config_dict, text_cols, w_values = w_values, 
                                                   label = True, b=b, ep=ep, ylims = ylims)
            else:
                plot_interpolate_ts_tx_ws(df_level, model, config_dict, text_cols, w_values = w_values, label = True)


def interpolate_ts_tx_single_sampling(df, model, config_dict, text_cols, w, 
                                      label=False, b=100, ep=1, plot = False):
    model.eval() # 2d vital model

    ts_f, tx_f_ls, _ = get_features3d(df, config_dict, text_col_ls = text_cols)
    ts_f = ts_f.to(device)
    raw_ts = ts_f[0].detach().cpu().numpy()
    tx_f_ls = [tx_f.to(device) for tx_f in tx_f_ls]
    
    # Get source text features
    _, tx_f_src, _ = get_features(df, config_dict, text_col = 'text')
    tx_f_src = tx_f_src.to(device)
    tx_f_src = tx_f_src.repeat(b, 1)  # Repeat for batch size

    # ----- ts_embeddings -----
    _, ts_emb_mean, ts_emb_log_var = model.ts_encoder(ts_f)
    ts_emb_mean = ts_emb_mean.expand(b, -1)
    if not model.variational:
        ts_emb_log_var = torch.zeros_like(ts_emb_mean) - 1e2   
    else:
        ts_emb_log_var = ts_emb_log_var.expand(b, -1)
    ts_emb = model.ts_encoder.reparameterization(ts_emb_mean, ts_emb_log_var, ep=ep) # (b, dim)
    ts_f = ts_f.repeat(b, 1)  # [b, seq_length] raw time series

    ts_hat_ls = {}
    for txid in range(len(tx_f_ls)):
        tx_f_tgt = tx_f_ls[txid]
        tx_f_tgt = tx_f_tgt.repeat(b, 1)

        ts_hat, _, _, _ = model.generate(w, ts_f, tx_f_tgt, tx_f_src)

        if plot:
            ts_hat_r = torch.quantile(ts_hat, 0.975, dim=0) 
            ts_hat_q = torch.quantile(ts_hat, 0.5, dim=0)
            ts_hat_i = torch.quantile(ts_hat, 0.025, dim=0)
            plt.plot(ts_hat_r.detach().cpu().numpy(), '--', color='red', linewidth=0.5)
            plt.plot(ts_hat_q.detach().cpu().numpy(), color='red', linewidth=0.5)
            plt.plot(ts_hat_i.detach().cpu().numpy(), '--', color='red', linewidth=0.5)
            plt.show()
            
        if label:
            text_condition = df[text_cols[txid]].iloc[0]
        else:
            text_condition = text_cols[txid]
        # Move ts_hat to CPU and convert to numpy to free GPU memory
        ts_hat_ls[text_condition] = ts_hat.detach().cpu()

        # Clean up CUDA memory
        del tx_f_tgt
        if plot:
            del ts_hat_r, ts_hat_q, ts_hat_i
        torch.cuda.empty_cache()
    
    # Clean up remaining CUDA tensors
    del ts_f, ts_emb_mean, ts_emb_log_var, ts_emb, tx_f_src
    for tx_f in tx_f_ls:
        del tx_f
    torch.cuda.empty_cache()
        
    return ts_hat_ls, raw_ts



def plot_interpolate_ts_tx_ws_sampling(df, model, config_dict, text_cols, w_values=None, 
                                       label = True, plot = False, b=200, ep=1, ylims = None,
                                       return_essentials = False):
    if w_values is None:
        w_values = np.arange(0, 1.1, 0.1)
    median_dict = {}    
    # Get ground truth reconstruction
    ts_hats, raw_ts = interpolate_ts_tx_single_sampling(df, model, config_dict, 
                                         text_cols = ['text'], # ground truth text condition
                                         w=0, # reconstruction only on ground truth text condition
                                         label = False, plot = plot, b=b, ep=ep)
    
    # ------------------------- Collect all results first -------------------------
    all_results = {}
    for w in w_values:
        ts_hats, _ = interpolate_ts_tx_single_sampling(df, model, config_dict, text_cols, 
                                                       w=w, label=label, plot = plot, b=b, ep=ep)
        # Store results as CPU tensors
        all_results[w] = {'ts_hats': {k: v for k, v in ts_hats.items()}}
        torch.cuda.empty_cache()
    
    # --------- NEW: Compute global min/max across all text_conditions and w ---------
    global_min, global_max = float('inf'), float('-inf')
    for text_condition in list(ts_hats.keys()):
        for w in w_values:
            if w in all_results:
                ts_hats = all_results[w]['ts_hats']
                ts_hat = ts_hats[text_condition]
                global_min = min(global_min, ts_hat.min().item())
                global_max = max(global_max, ts_hat.max().item())
    # Also include raw_ts in global min/max
    global_min = min(global_min, raw_ts.min())
    global_max = max(global_max, raw_ts.max())

    white_space = np.round((global_max - global_min) * 0.1, 2)
    if ylims is None:
        ylims = (global_min - white_space, global_max + white_space)
    # ------------------------------------------------------------------------------
    for text_condition in list(ts_hats.keys()):
        
        n_cols = 5
        n_rows = int(np.ceil(len(w_values) / n_cols))
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
        fig.suptitle(f'Augmenting towards: {text_condition}', fontsize=18)
        if n_rows == 1:
            axs = axs.reshape(1, -1)
        
        for w in w_values:
            if w in all_results:
                ts_hats = all_results[w]['ts_hats']
                ts_hat = ts_hats[text_condition]
                ts_hat_r = torch.quantile(ts_hat, 0.975, dim=0).numpy()
                ts_hat_q = torch.quantile(ts_hat, 0.5, dim=0).numpy()
                ts_hat_i = torch.quantile(ts_hat, 0.025, dim=0).numpy()

                if text_condition not in median_dict:
                    median_dict[text_condition] = {}
                median_dict[text_condition][float(w)] = ts_hat_q
                
        
        # --- Plotting ---
        for idx, w in enumerate(w_values):
            row = idx // n_cols
            col = idx % n_cols
            ts_hats = all_results[w]['ts_hats']
            ts_hat = ts_hats[text_condition]
            
            # Calculate quantiles and immediately convert to numpy
            ts_hat_r = torch.quantile(ts_hat, 0.975, dim=0).numpy()
            ts_hat_q = torch.quantile(ts_hat, 0.5, dim=0).numpy()
            ts_hat_i = torch.quantile(ts_hat, 0.025, dim=0).numpy()

            axs[row, col].plot(ts_hat_q, 'r-', label='Augmented') 
            axs[row, col].plot(ts_hat_r, 'r--')
            axs[row, col].plot(ts_hat_i, 'r--')
            axs[row, col].plot(raw_ts, 'b-', alpha=0.7, label='Raw')
            w_title = f'w = {w:.1f}' if w != 0 else 'w = 0 (reconstruction)'
            title = f'{w_title}'
            axs[row, col].set_title(title, pad=15, fontsize=10)
            axs[row, col].set_xlabel('Time Steps', fontsize=9)
            axs[row, col].set_ylabel('Value', fontsize=9)
            axs[row, col].grid(True, linestyle='--', alpha=0.3)
            axs[row, col].legend()
            axs[row, col].spines['top'].set_visible(False)
            axs[row, col].spines['right'].set_visible(False)
            axs[row, col].set_ylim(*ylims)
            
            # Clean up temporary arrays
            del ts_hat_r, ts_hat_q, ts_hat_i

        # Hide any empty subplots
        for idx in range(len(w_values), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axs[row, col].axis('off')

        plt.tight_layout()
        plt.show()
        
        # Clean up the figure
        plt.close(fig)
    
    # Final cleanup
    del all_results, ts_hats
    torch.cuda.empty_cache()

    # ---------- 3. pack & return (optional)------------------------------------------ #
    if return_essentials:
        return {
            'raw_ts': raw_ts,
            'median': median_dict,
            'ylims': ylims,
            'w_values': list(map(float, w_values))
        }

