
import torch
import pandas as pd
import matplotlib.pyplot as plt
from data import get_features3d, get_features
from config import *


def vital_contrast_infer(df, model, config_dict,
                         text_cols = ['text', 'text1', 'text2'], # i.e text1 text2 columns as 'low consecutive increases' and 'moderate consecutive increases'
                         var_ratio = 2,
                         K=100):
    model.eval() # 2d vital model
    ts_f, tx_f_ls, _ = get_features3d(df, config_dict, text_col_ls = text_cols)
    ts_f = ts_f.to(device)
    tx_f_ls = [tx_f.to(device) for tx_f in tx_f_ls]

    ts_f_repeated = ts_f.repeat(K, 1)  # Adjust dimensions as needed
    tx_f_ls_repeated = [tx_f.repeat(K, 1) for tx_f in tx_f_ls]

    # Forward pass for the batch
    _, z_mean, z_log_var = model.ts_encoder(ts_f_repeated)
    z = model.ts_encoder.reparameterization(z_mean, z_log_var, var_ratio)
    
    ts_hat_ls = []
    logits_ls = []
    # get the logits for each text (condition)
    for tx_f in tx_f_ls_repeated:
        tx_embedded = model.text_encoder(tx_f)
        ts_hat = model.ts_decoder(z, tx_embedded, ts_f, tx_embedded)

        logits = model.clip(z, tx_embedded)
        logits = torch.diag(logits).reshape(-1, 1)
        logits_ls.append(logits)
        ts_hat_ls.append(ts_hat)

    # get the softmax probabilities
    logits_ls = torch.cat(logits_ls, dim=1)
    exp_preds = torch.exp(logits_ls)
    softmax_probs = exp_preds / exp_preds.sum(dim=1, keepdim=True) # each row corresponds to a z, each column corresponds to the probability of a text condition given z

    text_conditioned_ts_hat = {}
    text_conditioned_ts_hat_probs = {}
    # For each text condition, get the time series where it has the highest probability
    for i, text_col in enumerate(text_cols):
        ts_hat = ts_hat_ls[i]
        # Get the probabilities for this text condition
        probs = softmax_probs[:, i]
        # Get the indices where this text condition has the highest probability
        mask = torch.argmax(softmax_probs, dim=1) == i
        # Store the time series and their corresponding probabilities
        text_col_string = df[text_col].iloc[0]
        text_conditioned_ts_hat[text_col_string] = ts_hat[mask]
        text_conditioned_ts_hat_probs[text_col_string] = probs[mask]

    return text_conditioned_ts_hat, text_conditioned_ts_hat_probs

def plot_vital_contrast_reconstructions(raw_ts, text_conditioned_ts_hat, text_conditioned_ts_hat_probs, n=3, title=''):
    """
    Plot the top n time series reconstructions for each text condition.
    
    Args:
        raw_ts (torch.Tensor): Original time series
        text_conditioned_ts_hat (dict): Dictionary mapping text conditions to their corresponding time series
        text_conditioned_ts_hat_probs (dict): Dictionary mapping text conditions to their probabilities
        n (int): Number of top reconstructions to plot for each condition
        title (str): Title for the plot
    """
    n_conditions = len(text_conditioned_ts_hat)
    n_cols = n
    n_rows = n_conditions
    
    # Create figure and subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
    fig.suptitle(title, fontsize=18)
    
    # If only one condition, make axs 2D
    if n_conditions == 1:
        axs = axs.reshape(1, -1)
    
    # Plot reconstructions for each text condition
    for i, (text_condition, ts_hats) in enumerate(text_conditioned_ts_hat.items()):
        probs = text_conditioned_ts_hat_probs[text_condition]
        
        # Sort by probability and get top n
        sorted_indices = torch.argsort(probs, descending=True)
        top_n_indices = sorted_indices[:n]
        top_n_ts_hats = ts_hats[top_n_indices]
        top_n_probs = probs[top_n_indices]
        
        # Plot each of the top n reconstructions
        for j in range(n):
            if j < len(top_n_ts_hats):  # Check if we have enough reconstructions
                ts_hat = top_n_ts_hats[j]
                prob = top_n_probs[j].item()
                
                # Plot reconstruction
                if ts_hat.dim() == 2:
                    ts_hat = ts_hat[0]
                axs[i, j].plot(ts_hat.cpu().detach().numpy(), 'r-', label='Reconstruction')
                axs[i, j].plot(raw_ts, 'b--', alpha=0.5, label='Original')
                axs[i, j].set_title(f'{text_condition}\nprob={prob:.3f}')
                axs[i, j].set_xlabel('Time')
                axs[i, j].set_ylabel('Value')
                axs[i, j].set_ylim(50, 200)  # Adjust y-limits as needed
                axs[i, j].grid(True)
                axs[i, j].legend()
            else:
                # Hide empty subplots
                axs[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()

def cal_embeddings_distances(df, text_cols, model, config_dict):
    
    model.eval() # 2d vital model
    ts_f, tx_f_ls, _ = get_features3d(df, config_dict, text_col_ls = text_cols)
    ts_f = ts_f.to(device)
    tx_f_ls = [tx_f.to(device) for tx_f in tx_f_ls]

    # ----- ts_embeddings -----
    ts_emb, ts_emb_mean, _ = model.ts_encoder(ts_f)
        

    # ----- ts_embeddings distances to tx_embeddings -----
    simi = {}
    l1 = {}
    l2 = {}
    for txid in range(len(tx_f_ls)):
        tx_emb = model.text_encoder(tx_f_ls[txid])
        
        # calculate dot product similarity between all ts and this txt_emb
        logits = torch.matmul(ts_emb_mean, tx_emb.T) 
        simi[df[text_cols[txid]].iloc[0]] = torch.exp(torch.diag(logits))
        
        # calculate L1/L2 norm distance between all ts and this txt_emb
        l2[df[text_cols[txid]].iloc[0]] = torch.norm(ts_emb_mean - tx_emb, dim=1, p=2)
        l1[df[text_cols[txid]].iloc[0]] = torch.norm(ts_emb_mean - tx_emb, dim=1, p=1)
    # # nicer print dict
    # for key, value in simi.items():
    #     print(f'{key}: {value}')
    # for key, value in l1.items():
    #     print(f'{key}: {value}')
    # for key, value in l2.items():
    #     print(f'{key}: {value}')
    ts2tx_distances = {'simi': simi, 'l1': l1, 'l2': l2}

        
    # ----- pairwise similary / l2/ l1 distances between all embeddings (concate ts and tx_unqiue) -----
    tx_emb = None
    for txid in range(len(tx_f_ls)):
        tx_emb_1 = model.text_encoder(tx_f_ls[txid])[0].reshape(1,-1) # [1, tx_emb_dim]
        if tx_emb is None:
            tx_emb = tx_emb_1
        else:
            tx_emb = torch.cat([tx_emb, tx_emb_1], dim=0)

    # concate txt embeddings (one for each text) and all ts embeddings
    all_emb = torch.cat([tx_emb, ts_emb_mean], dim=0)
    # calculate pairwise similary / l2/ l1 distance between all embeddings
    simi_mat = torch.exp(torch.matmul(all_emb, all_emb.T))
    l1_dist_mat = torch.cdist(all_emb, all_emb, p=1) # L1 norm
    l2_dist_mat = torch.cdist(all_emb, all_emb, p=2) # L2 norm

    pairwise_distances = {'simi': simi_mat, 'l1': l1_dist_mat, 'l2': l2_dist_mat}

    return pairwise_distances, ts2tx_distances

def plot_embeddings_graph(adj_mat, k = 2, title = '', subtitle = ''):
    # Create a network graph
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    
    # zero out lower than 50 percentile
    adj_mat[adj_mat < np.percentile(adj_mat, 25)] = 0
    np.fill_diagonal(adj_mat, 0)  # Remove self-loops

    # Create and draw network
    G = nx.from_numpy_array(adj_mat)
    pos = nx.spring_layout(G, k=0.5, iterations=5000)

    plt.figure(figsize=(6, 4))

    # Define a list of contrasting colors
    color_list = ['darkgreen', 'blue', 'red', 'purple', 'orange', 
                 'brown', 'pink', 'gray', 'olive', 'cyan',
                 'magenta', 'yellow', 'teal', 'coral', 'navy',
                 'maroon', 'lime', 'indigo', 'gold', 'silver']
    
    # Use the first k colors from the list
    colors = color_list[:k]

    # Draw edges first
    nx.draw_networkx_edges(G, pos,
                          edge_color='grey',
                          width=0.1)

    # Draw first k nodes as triangles
    first_k_nodes = list(G.nodes())[:k]
    nx.draw_networkx_nodes(G, pos,
                          nodelist=first_k_nodes,
                          node_color=colors,
                          node_shape='^',
                          node_size=100)

    # Draw remaining nodes as circles, distributed among k categories
    remaining_nodes = list(G.nodes())[k:]
    n_remaining = len(remaining_nodes)
    nodes_per_category = n_remaining // k
    
    for i in range(k):
        start_idx = i * nodes_per_category
        end_idx = (i + 1) * nodes_per_category if i < k-1 else n_remaining
        category_nodes = remaining_nodes[start_idx:end_idx]
        
        nx.draw_networkx_nodes(G, pos,
                              nodelist=category_nodes,
                              node_color=colors[i],
                              node_size=10)
    
    # Add node indices as labels
    labels = {i: str(i) for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=3)
    
    # add title
    plt.title(title)
    # subtitle
    plt.suptitle(subtitle)
    plt.show()

def net_emb(df,
            model, 
            config_dict, 
            top = 100,
            y_col = None,
            text_levels = None):
   # network the ts embeddings with predicted text conditions
    if y_col is None:
        y_col = config_dict['y_col']
    if text_levels is None:
        text_levels = config_dict['y_levels']
    
    df_ls = []
    for i in range(len(text_levels)):
        df_sub = df[df[y_col].str.contains(text_levels[i], case=False, na=False)].reset_index(drop=True) # if text_levels[i] is a substring of each row df[y_col]
        df_sub = df_sub.iloc[range(top)].copy()
        df_ls.append(df_sub)
    df = pd.concat(df_ls, ignore_index=True)

    text_cols = []
    for i in range(len(text_levels)):
        df['text'+str(i)] = text_levels[i]
        text_cols.append('text'+str(i))

    pairwise_distances, ts2tx_distances = cal_embeddings_distances(df, text_cols, model, config_dict)


    adj_mat = pairwise_distances['l2'].detach().cpu().numpy()
    adj_mat = 1/(adj_mat+1e-8)
    plot_embeddings_graph(adj_mat, k=len(text_levels), title = y_col, subtitle = '1 / l2')


    adj_mat = pairwise_distances['simi'].detach().cpu().numpy()
    plot_embeddings_graph(adj_mat, k=len(text_levels), title = y_col, subtitle = 'similarity')

    return pairwise_distances, ts2tx_distances

def cal_embeddings_distances_w_text(df, 
                                    text_cols, # predicted text conditions
                                    model, 
                                    config_dict):
    
    model.eval() # 2d vital model
    
    # ----- text conditions -----
    ts_f, tx_f_ls, _ = get_features3d(df, config_dict, text_col_ls = text_cols)
    ts_f = ts_f.to(device) 
    tx_f_ls = [tx_f[0].reshape(1,-1).to(device) for tx_f in tx_f_ls]
    tx_f_condi = torch.cat(tx_f_ls, dim=0)
    tx_emb = model.text_encoder(tx_f_condi) # dim = [k_levels, tx_emb_dim]

    # ----- orginial text description/caption -----
    _, tx_f_raw, _ = get_features(df, config_dict, text_col = 'text') # text is the default caption column
    tx_f_raw = tx_f_raw.to(device)
    tx_emb_raw = model.text_encoder(tx_f_raw) # dim = [k_levels * top, tx_emb_dim]
        
    # ----- ts_embeddings -----
    ts_emb, ts_emb_mean, _ = model.ts_encoder(ts_f)  # dim = [k_levels * top, ts_emb_dim]
    
        
    # ----- pairwise similary / l2/ l1 distances between all embeddings (concate ts and tx_unqiue) -----
    # concate txt embeddings (one for each text) and all ts embeddings
    all_emb = torch.cat([tx_emb, ts_emb_mean, tx_emb_raw], dim=0)
    # calculate pairwise similary / l2/ l1 distance between all embeddings
    simi_mat = torch.exp(torch.matmul(all_emb, all_emb.T))
    l1_dist_mat = torch.cdist(all_emb, all_emb, p=1) # L1 norm
    l2_dist_mat = torch.cdist(all_emb, all_emb, p=2) # L2 norm

    pairwise_distances = {'simi': simi_mat, 'l1': l1_dist_mat, 'l2': l2_dist_mat}

    return pairwise_distances

def plot_embeddings_graph_w_text(adj_mat, k = 2, title = '', subtitle = ''):
    # Create a network graph
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    
    # zero out lower than 50 percentile
    adj_mat[adj_mat < np.percentile(adj_mat, 25)] = 0
    np.fill_diagonal(adj_mat, 0)  # Remove self-loops

    # Create and draw network
    G = nx.from_numpy_array(adj_mat)
    pos = nx.spring_layout(G, k=0.5, iterations=5000)

    plt.figure(figsize=(9, 6))

    # Define a list of contrasting colors
    color_list = ['darkgreen', 'blue', 'red', 'purple', 'orange', 
                 'brown', 'pink', 'gray', 'olive', 'cyan',
                 'magenta', 'yellow', 'teal', 'coral', 'navy',
                 'maroon', 'lime', 'indigo', 'gold', 'silver']
    
    # Use the first k colors from the list
    colors = color_list[:k]

    # Draw edges first
    nx.draw_networkx_edges(G, pos,
                          edge_color='grey',
                          width=0.1)

    # Draw first k nodes as larger triangles
    first_k_nodes = list(G.nodes())[:k]
    nx.draw_networkx_nodes(G, pos,
                          nodelist=first_k_nodes,
                          node_color=colors,
                          node_shape='^',
                          node_size=150)

    # For the remaining nodes, first half are time series embeddings, second half are text embeddings
    remaining_nodes = list(G.nodes())[k:]
    n_remaining = len(remaining_nodes)
    half_nodes = n_remaining // 2
    
    # First half: time series embeddings (circles)
    ts_nodes = remaining_nodes[:half_nodes]
    nodes_per_category = half_nodes // k
    
    for i in range(k):
        start_idx = i * nodes_per_category
        end_idx = (i + 1) * nodes_per_category if i < k-1 else half_nodes
        category_nodes = ts_nodes[start_idx:end_idx]
        
        nx.draw_networkx_nodes(G, pos,
                              nodelist=category_nodes,
                              node_color=colors[i],
                              node_size=50,
                              node_shape='o')  # circles for time series
    
    # Second half: text embeddings (triangles)
    text_nodes = remaining_nodes[half_nodes:]
    nodes_per_category = (n_remaining - half_nodes) // k
    
    for i in range(k):
        start_idx = i * nodes_per_category
        end_idx = (i + 1) * nodes_per_category if i < k-1 else (n_remaining - half_nodes)
        category_nodes = text_nodes[start_idx:end_idx]
        
        nx.draw_networkx_nodes(G, pos,
                              nodelist=category_nodes,
                              node_color=colors[i],
                              node_size=50,
                              node_shape='^')  # triangles for text
    
    # Add node indices as labels with proper formatting
    labels = {}
    # Add labels for first k nodes (text conditions)
    for i, node in enumerate(first_k_nodes):
        labels[node] = ''
    
    # Add labels for time series nodes (circles)
    for i, node in enumerate(ts_nodes):
        labels[node] = f'ts{i+1}'
    # Add labels for text nodes (triangles)
    for i, node in enumerate(text_nodes):
        labels[node] = f'tx{i+1}'
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    # add title
    plt.title(title)
    # subtitle
    plt.suptitle(subtitle)
    plt.show()

def net_emb_w_text(df,
            model, 
            config_dict, 
            top = 100,
            y_col = None,
            text_levels = None):
   # network the ts and text embeddings with predicted text conditions
    if y_col is None:
        y_col = config_dict['y_col']
    if text_levels is None:
        text_levels = config_dict['y_levels']
    
    df_ls = []
    for i in range(len(text_levels)):
        df_sub = df[df[y_col].str.contains(text_levels[i], case=False, na=False)].reset_index(drop=True) # if text_levels[i] is a substring of each row df[y_col]
        df_sub = df_sub.iloc[range(top)].copy()
        df_ls.append(df_sub)
    df = pd.concat(df_ls, ignore_index=True)

    text_cols = []
    for i in range(len(text_levels)):
        df['text'+str(i)] = text_levels[i]
        text_cols.append('text'+str(i))

    pairwise_distances = cal_embeddings_distances_w_text(df, text_cols, model, config_dict)


    adj_mat = pairwise_distances['l2'].detach().cpu().numpy()
    adj_mat = 1/(adj_mat+1e-8)
    plot_embeddings_graph_w_text(adj_mat, k=len(text_levels), title = y_col, subtitle = '1 / l2')


    adj_mat = pairwise_distances['simi'].detach().cpu().numpy()
    plot_embeddings_graph_w_text(adj_mat, k=len(text_levels), title = y_col, subtitle = 'similarity')

    return pairwise_distances

