# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
from tslearn.metrics import dtw, lcss, cdist_dtw
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
import numpy as np
import gc
import matplotlib.pyplot as plt
import pandas as pd
from generation import interpolate_ts_tx
from numpy.lib.stride_tricks import sliding_window_view   # NumPy ≥ 1.20

# Add the tedit_lite and tedit_lite_tx folders to the system path
import sys, os
# Path for tedit_lite
tedit_attr_path = os.path.abspath("../tedit_lite")
if tedit_attr_path not in sys.path:
    sys.path.append(tedit_attr_path)
from tedit_generation import tedit_generate_ts_tx as tedit_generate_ts_tx

# Path for tedit_lite_tx
tedit_tx_path = os.path.abspath("../tedit_lite_tx")
if tedit_tx_path not in sys.path:
    sys.path.append(tedit_tx_path)
from tedit_tx_generation import tedit_generate_ts_tx as tedit_tx_generate_ts_tx


def _split_patches(ts, *, n_patches=8):
    ts = np.ascontiguousarray(ts)   # uniform stride‑layout
    n  = ts.size
    if n_patches <= 0:
        raise ValueError("n_patches must be positive")

    patch_len = max(1, n // n_patches)          # floor division
    # Build a 2‑D view of all length‑patch_len windows, then hop by patch_len
    view = sliding_window_view(ts, patch_len)[::patch_len]
    return [view[i] for i in range(view.shape[0])]


# # patchify
# def _split_patches(ts, patch_len=None, step=None):
#     ts = np.asarray(ts)
#     n  = len(ts)
#     patch_len = n // 4 if patch_len is None else int(patch_len)
#     step      = n // 8 if step      is None else int(step)
#     patch_len = max(1, patch_len)
#     step      = max(1, step)
#     if patch_len > n:
#         raise ValueError("patch_len cannot exceed series length")
#     return [ts[i : i + patch_len] for i in range(0, n - patch_len + 1, step)]

def _patch_lcss(ref_p, aug_p): # lcss similarity
    mat = np.zeros((len(ref_p), len(aug_p)))
    for r, rp in enumerate(ref_p):
        for a, ap in enumerate(aug_p):
            mat[r, a] = lcss(rp, ap)
    return np.median(mat) 

def _patch_dtw(ref_p, aug_p, n_jobs=None):
    ref_arr = np.vstack(ref_p)        # shape (n_ref_p, patch_len)
    aug_arr = np.vstack(aug_p)        # shape (n_aug_p, patch_len)
    mat = cdist_dtw(ref_arr, aug_arr,
                    n_jobs=n_jobs,
                    verbose=0)        # still returns *distances*
    return np.median(mat)

def _patch_mse(ref_p, aug_p):
    """Median patch‑level mean‑squared‑error."""
    ref_arr = np.vstack(ref_p)          # (R, L)
    aug_arr = np.vstack(aug_p)          # (A, L)
    diff    = ref_arr[:, None, :] - aug_arr[None, :, :]
    mse     = np.mean(diff ** 2, axis=-1)   # (R, A)
    return np.median(mse)

def _patch_mae(ref_p, aug_p):
    """Median patch‑level mean‑squared‑error."""
    ref_arr = np.vstack(ref_p)          # (R, L)
    aug_arr = np.vstack(aug_p)          # (A, L)
    diff    = ref_arr[:, None, :] - aug_arr[None, :, :]
    mae     = np.mean(np.abs(diff), axis=-1)
    return np.median(mae)


def _process_pair(i, j, ref_ts, aug_ts, n_patches):
    """Return DTW‑similarity, LCSS‑similarity, MSE‑similarity for one pair."""
    if n_patches is None:                 # ── global comparison ──
        dtw_dist  = dtw(ref_ts, aug_ts)
        lcss_simi = lcss(ref_ts, aug_ts)          # already similarity
        mse_dist  = np.mean((ref_ts - aug_ts) ** 2)
        mae_dist  = np.mean(np.abs(ref_ts - aug_ts))
    else:                                 # ── patch‑wise comparison ──
        ref_p = _split_patches(ref_ts, n_patches=n_patches)
        aug_p = _split_patches(aug_ts, n_patches=n_patches)
        dtw_dist  = _patch_dtw(ref_p,  aug_p)
        lcss_simi = _patch_lcss(ref_p, aug_p)
        mse_dist  = _patch_mse(ref_p,  aug_p)
        mae_dist  = _patch_mae(ref_p,  aug_p)
    # dtw_simi = 1 / (1 + dtw_dist)         # convert distance → similarity
    dtw_simi = dtw_dist # use distance instead of similarity
    # mse_simi = 1 / (1 + mse_dist)
    # mae_simi = 1 / (1 + mae_dist)
    return i, j, dtw_simi, lcss_simi, mse_dist, mae_dist



def calculate_similarity_parallel(ref_ts_list, aug_ts_list, n_jobs=None, n_patches=None):
    """
    Calculate DTW and LCSS distance matrices between reference and augmented time series in parallel.
    
    Args:
        ref_ts_list (list): List of reference time series
        aug_ts_list (list): List of augmented time series
        n_jobs (int): Number of parallel jobs. If None, uses 70% of available CPUs.
        patch_params (dict): Dictionary with keys 'patch_len' and 'step'. If None, no patching is done.
        
    Returns:
        tuple: (dtw_matrix, lcss_matrix, dtw_summaries, lcss_summaries)
        where each matrix has shape (n_ref, n_aug) and summaries are lists of length n_aug
    """
    if n_jobs is None:
        n_jobs = max(1, int(multiprocessing.cpu_count() * 0.7))
    
    n_ref = len(ref_ts_list)
    n_aug = len(aug_ts_list)
    dtw_matrix = np.zeros((n_ref, n_aug))
    lcss_matrix = np.zeros((n_ref, n_aug))
    mse_matrix = np.zeros((n_ref, n_aug))
    mae_matrix = np.zeros((n_ref, n_aug))

    jobs = [(i, j, ref_ts, aug_ts, n_patches)
            for i, ref_ts in enumerate(ref_ts_list)
            for j, aug_ts in enumerate(aug_ts_list)]

    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_process_pair)(*job) for job in tqdm(jobs, desc="Similarity")
    )

    # collect results
    for i, j, dtw_simi, lcss_simi, mse_dist, mae_dist in results:
        dtw_matrix[i, j] = dtw_simi
        lcss_matrix[i, j] = lcss_simi
        mse_matrix[i, j] = mse_dist
        mae_matrix[i, j] = mae_dist
    
    # summaries (unchanged)
    def _summ(col):
        return {
            'mean': col.mean().round(4),
            'std' : col.std().round(4),
            'min' : col.min().round(4),
            'max' : col.max().round(4),
            **{f"q{p}": np.quantile(col, p/100).round(4)
               for p in (5, 25, 50, 75, 95)}
        }
    dtw_summaries  = [_summ(dtw_matrix[:, j])  for j in range(n_aug)]
    lcss_summaries = [_summ(lcss_matrix[:, j]) for j in range(n_aug)]
    mse_summaries = [_summ(mse_matrix[:, j]) for j in range(n_aug)]
    mae_summaries = [_summ(mae_matrix[:, j]) for j in range(n_aug)]

    return dtw_matrix, lcss_matrix, mse_matrix, mae_matrix, dtw_summaries, lcss_summaries, mse_summaries, mae_summaries


def _stratified_bootstrap(group, b):
    n = len(group)
    replace = n < b
    return group.sample(n=b, replace=replace, random_state=333)

def _scale_to_range(ts, ths=(None, None)):
    if ths[0] is None and ths[1] is None:
        return ts
    
    # Convert tuple to list for modification
    ths_list = list(ths)
    
    # Get current min and max
    current_min = ts.min()
    current_max = ts.max()
        
    # Scale to [0,1] first
    ts_scaled = (ts - current_min) / (current_max - current_min)
    if ths_list[0] is None:
        ths_list[0] = current_min 
    if ths_list[1] is None:
        ths_list[1] = current_max
    # Then scale to desired range
    ts_final = ts_scaled * (ths_list[1] - ths_list[0]) + ths_list[0]
    
    return ts_final

# ----------------- TS distances evaluation on synthetic data --------------------------------

def eval_ts_similarity(df, # df can be df_train / df_test
                      model, config_dict, w,  y_col, 
                      conditions = None, # a list of tuples of (y_col, y_level) to filter the df (should not filter y_col)
                      b=200,
                      ths = (None, None),
                      round_to = 4,
                      n_patches=None,
                      aug_type="conditional",
                      meta = None, # tedit meta
                      configs = None, # teidt configs
                      model_type = None # if None, will be auto-detected
                      ):
    
    # Auto-detect model type if not specified
    if model_type is None:
        if meta is None:
            model_type = 'vital'
        else:
            if 'level_maps' in meta:
                model_type = 'tedit'
            elif 'attr_emb_dim' in meta:
                model_type = 'tedit_tx'
            else:
                raise ValueError("Could not determine model type from meta dictionary")
       
    if config_dict['ts_global_normalize']:
        global_mean = config_dict['ts_normalize_mean']
        global_std  = config_dict['ts_normalize_std']
    
    if conditions is not None:
        for condition in conditions:
            df = df[df[condition[0]] == condition[1]]

    df = (
        df.groupby(y_col, group_keys=False)
        .apply(_stratified_bootstrap, b=b)
        .reset_index(drop=True)
    )

    model.eval()
    df_dists = pd.DataFrame()
    y_levels = list(df[y_col].unique())
    if config_dict['open_vocab']:
        aug_level_map = df.groupby(y_col).agg({f"{y_col}_aug": lambda x: list(x.unique())}).to_dict()[f"{y_col}_aug"]

    for i in range(len(y_levels)):
        # agument data with original reference text y_levels[i]
        ref_level = y_levels[i]
        df_level = df[df[y_col] == ref_level].reset_index(drop=False).copy() #
        # agument towards new text conditions 
        for j in range(len(y_levels)):
            if not config_dict['open_vocab']:
                if aug_type == 'marginal':
                    df_level['text' + str(j)] = y_levels[j] # marginal augmentation
                elif aug_type == "conditional":
                    df_level['text' + str(j)] = df_level['text'].str.replace(ref_level, y_levels[j]) # sub ref_text with y_levels[j] in 'text'
            else:
                if aug_type == 'marginal':
                    df_level['text' + str(j)] = np.random.choice(aug_level_map[y_levels[j]], size=len(df_level))
                elif aug_type == "conditional":
                    # change the original augmented y_col to a new level (also augmented)
                    df_level[y_col+'_aug'] = np.random.choice(aug_level_map[y_levels[j]], size=len(df_level))
                    df_level['text' + str(j)]  = ''
                    for str_col in [col+'_aug' for col in config_dict['txt2ts_y_cols']]:
                        df_level['text' + str(j)] += ' ' + df_level[str_col]
                    df_level['text' + str(j)] = df_level['text' + str(j)].str.strip()
        
        # mapping text_col to y_level
        col_level_map = dict(zip(['text' + str(j) for j in range(len(y_levels))], y_levels))
    
        # only augment towards NEW text conditions except original aug_text
        new_text_cols = ['text' + str(j) for j in range(len(y_levels)) if y_levels[j] != ref_level]
        
        # Choose model based on model_type
        if model_type == 'tedit_tx':
            new_level_col_map = {k: v for k, v in col_level_map.items() if k in new_text_cols}
            ts_hat_ls = tedit_tx_generate_ts_tx(df_level,
                                               meta,
                                               config_dict,
                                               configs,
                                               y_col,
                                               new_level_col_map)
        elif model_type == 'tedit':
            new_level_col_map = {k: v for k, v in col_level_map.items() if k in new_text_cols}
            ts_hat_ls = tedit_generate_ts_tx(df_level,
                                            meta,
                                            config_dict,
                                            configs,
                                            y_col,
                                            new_level_col_map)
        else:  # vital model
            ts_hat_ls = interpolate_ts_tx(df_level, model, config_dict, new_text_cols, w)
    
        for text_col, pairs in ts_hat_ls.items():
            aug_level = col_level_map[text_col]

            # augmented time series
            aug_df = pd.DataFrame(pairs, columns=['aug_text', 'ts_hat'])
            aug_df['ts_hat'] = aug_df['ts_hat'].apply(lambda x: x.cpu().detach().numpy())
            aug_ts_list = [aug_df['ts_hat'][i] for i in range(len(aug_df))]
            
            # target time series
            tgt_df = df[df[y_col] == aug_level]
            tgt_ts_list = [tgt_df[[str(i+1) for i in range(config_dict['seq_length'])]].to_numpy()[i] for i in range(len(tgt_df))]
            
            # orginal time series (reference time series) 
            ref_df = df[df[y_col] == ref_level]   
            ref_ts_list = [ref_df[[str(i+1) for i in range(config_dict['seq_length'])]].to_numpy()[i] for i in range(len(ref_df))]
            
            if config_dict['ts_global_normalize']:
                tgt_ts_list = [(tgt - global_mean) / global_std for tgt in tgt_ts_list]
                ref_ts_list = [(ref_ts - global_mean) / global_std for ref_ts in ref_ts_list]
            else:
                if ths[0] is not None or ths[1] is not None:
                    aug_ts_list = [_scale_to_range(aug_ts, ths) for aug_ts in aug_ts_list]
                if round_to is not None:
                    ref_ts_list = [np.round(ref_ts, round_to) for ref_ts in ref_ts_list]
                    aug_ts_list = [np.round(aug_ts, round_to) for aug_ts in aug_ts_list]

            # plot 20 time series in aug_ts_list
            fig, axs = plt.subplots(4, 5, figsize=(20, 8))
            for i in range(20):
                axs[i//5, i%5].plot(aug_ts_list[i], label='augmented', color='red')
                axs[i//5, i%5].plot(tgt_ts_list[i], label='target', color='black')
                axs[i//5, i%5].plot(ref_ts_list[i], label='original', color='darkgrey')
                axs[i//5, i%5].legend()
            plt.suptitle(ref_level+" -> "+aug_level, fontsize=15)
            plt.tight_layout()
            plt.show()

            _, _, _, _, dtw_aug2tgt, lcss_aug2tgt, mse_aug2tgt, mae_aug2tgt = calculate_similarity_parallel(aug_ts_list, tgt_ts_list, n_patches=n_patches)
            _, _, _, _, dtw_ref2tgt, lcss_ref2tgt, mse_ref2tgt, mae_ref2tgt = calculate_similarity_parallel(ref_ts_list, tgt_ts_list, n_patches=n_patches)
            
            aug_df['dtw_aug2tgt'] = dtw_aug2tgt
            aug_df['lcss_aug2tgt'] = lcss_aug2tgt
            aug_df['mse_aug2tgt'] = mse_aug2tgt
            aug_df['mae_aug2tgt'] = mae_aug2tgt
            aug_df['dtw_ref2tgt'] = dtw_ref2tgt
            aug_df['lcss_ref2tgt'] = lcss_ref2tgt
            aug_df['mse_ref2tgt'] = mse_ref2tgt
            aug_df['mae_ref2tgt'] = mae_ref2tgt

            aug_df['ref_y_level'] = ref_level
            aug_df['aug_y_level'] = aug_level
            aug_df['ref_y_col'] = y_col
            df_dists = pd.concat([df_dists, aug_df], ignore_index=True)
   
    return df_dists




def eng_dists(df_dist, 
              ref_y_level, 
              aug_y_level, 
              metrics = ['lcss', 'dtw', 'mse', 'mae'], 
              plot = False):
    import matplotlib.pyplot as plt
   
    df = df_dist[
                    (df_dist['ref_y_level'].str.contains(ref_y_level, case=False, regex=False)) & 
                    (df_dist['aug_y_level'].str.contains(aug_y_level, case=False, regex=False))
                ]
    for metric in metrics:
        df['aug_'+metric] = df[metric+'_aug2tgt'].apply(lambda x: x['q50'])
        df['ref_'+metric] = df[metric+'_ref2tgt'].apply(lambda x: x['q50'])
        df['diff_'+metric] = df['aug_'+metric] - df['ref_'+metric]

    if plot:
        for metric in metrics:
            plt.hist(df['aug_'+metric], bins=20, alpha=0.5, label='augmented')
            plt.hist(df['ref_'+metric], bins=20, alpha=0.5, label='reference')
            plt.title(f'{metric} similarity')
            plt.legend()
            plt.show()
    
    return df




def eng_dists_multiple(df_dists, base_aug_dict, metric='lcss', aug_type="conditional"):

    df_dists = df_dists[df_dists['aug_type'] == aug_type]

    df_ls = []
    for _, pairs in base_aug_dict.items():
        n_cols = 4  # Always use 4 columns
        n_pairs = len(pairs)
        n_rows = int(np.ceil(n_pairs / n_cols))
        
        # Create figure with 3 rows per pair: boxplot of differences, boxplot of values, and histogram
        fig, axes = plt.subplots(3 * n_rows, n_cols, figsize=(n_cols*5, 8*n_rows))
        
        # Handle single row case properly
        if n_rows == 1:
            axes = axes.reshape(3, n_cols)  # Make it 2D array if only one row
        else:
            axes = axes.reshape(3 * n_rows, n_cols)  # Ensure proper shape for multiple rows
        
        for idx, (ref, aug) in enumerate(pairs):
            row = idx // n_cols
            col = idx % n_cols
            
            df = eng_dists(df_dists, ref, aug, metrics = [metric])
            df['ref'] = ref
            df['aug'] = aug
            df_ls.append(df)

            # first row boxplot 'diff_'+metric + horizontal line at 0
            diff_df = pd.DataFrame()
            diff_df[metric] = df['diff_'+metric]
            box = axes[2*row, col].boxplot(
                [diff_df[col] for col in diff_df.columns],
                labels=diff_df.columns,
                showfliers=False,
                patch_artist=False  # No fill, just lines
            )
            for i, metric in enumerate(diff_df.columns):
                color = 'black' #if metric == aug_matrix else 'grey'
                # Set box color
                for line in [
                    box['boxes'][i],
                    box['whiskers'][2*i], box['whiskers'][2*i+1],
                    box['caps'][2*i], box['caps'][2*i+1],
                    box['medians'][i]
                ]:
                    line.set_color(color)
                    line.set_linewidth(1)
            # axes[2*row, col].set_ylim(-1, 1)
            axes[2*row, col].axhline(0, color='black', linewidth=0.5)
            axes[2*row, col].set_title(f'{ref}\n→\n{aug}', pad=20)
            if col == 0:  # Only add ylabel to first subplot
                axes[2*row, col].set_ylabel('similarity increase')
            
            # second row boxplot 'aug_'+metric + 'ref_'+metric + horizontal line at 'ref_'+metric .median()
            box_df = pd.DataFrame()
            box_df['augmented'] = df['aug_' + metric]
            box_df['reference'] = df['ref_' + metric]
            box = axes[2*row+1, col].boxplot(
                [box_df[col] for col in box_df.columns],
                labels=box_df.columns,
                showfliers=False,
                patch_artist=False  # No fill, just lines
            )
            for i, x in enumerate(box_df.columns):
                color = 'black' if x == 'augmented' else 'grey'
                # Set box color
                for line in [
                    box['boxes'][i],
                    box['whiskers'][2*i], box['whiskers'][2*i+1],
                    box['caps'][2*i], box['caps'][2*i+1],
                    box['medians'][i]
                ]:
                    line.set_color(color)
                    line.set_linewidth(1)
            # axes[2*row+1, col].set_ylim(-0.1, 1)
            axes[2*row+1, col].axhline(box_df['reference'].median(), color='black', linewidth=0.5)
            if col == 0:  # Only add ylabel to first subplot
                axes[2*row+1, col].set_ylabel(metric + ' similarity')
            
            # --- third row histogram ---
            aug_stat = df['aug_'+metric].values.tolist()
            bas_stat = df['ref_'+metric].values.tolist()
            
            # Calculate shared bins based on combined data range
            all_data = aug_stat + bas_stat
            bins = np.linspace(min(all_data), max(all_data), 20)
            
            axes[2*row+2, col].hist(aug_stat, bins=bins, alpha=0.5, label='Augmented')
            axes[2*row+2, col].hist(bas_stat, bins=bins, alpha=0.5, label='Base')
            axes[2*row+2, col].set_xlabel(metric + ' similarity')
            if col == 0:
                axes[2*row+2, col].set_ylabel('Count')
            axes[2*row+2, col].legend()
        
        # Hide any empty subplots
        total_plots = n_rows * n_cols
        for idx in range(n_pairs, total_plots):
            row = idx // n_cols
            col = idx % n_cols
            axes[2*row, col].axis('off')
            axes[2*row+1, col].axis('off')
            axes[2*row+2, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    res_df = pd.concat(df_ls, ignore_index=True)
    # reformat
    res_df['score'] = res_df['diff_'+metric]
    res_df['metric'] = 'delta_'+metric
    res_df.rename(columns={'aug_y_level': 'tgt_level',
                           'ref_y_level': 'src_level'}, inplace=True)
    res_df = res_df[['aug_type', 'attr', 'src_level', 'tgt_level', 'metric', 'score']]
    return res_df







# ----------------- point-wise distance evaluation on synthetic with ground truth data --------------------------------
# only used for synthetic_gt data
def eval_pw_dist(df, model, config_dict, w, aug_type="conditional",
                 meta = None, # tedit meta
                 configs = None, # teidt configs
                 model_type = None # if None, will be auto-detected
                ):
    # Auto-detect model type if not specified
    if model_type is None:
        if meta is None:
            model_type = 'vital'
        else:
            if 'level_maps' in meta:
                model_type = 'tedit'
            elif 'attr_emb_dim' in meta:
                model_type = 'tedit_tx'
            else:
                raise ValueError("Could not determine model type from meta dictionary")

    model.eval()
    text_pairs = config_dict['text_config']['text_pairs']

    if config_dict['ts_global_normalize']:
        global_mean = config_dict['ts_normalize_mean']
        global_std  = config_dict['ts_normalize_std']

    pw_df_all = pd.DataFrame()
    # attribute to be modified
    for att_idx, att in enumerate(text_pairs):  
        if config_dict['open_vocab']:
            # aug_level_map = aug_level_map_all[att_idx]
            y_col = config_dict['txt2ts_y_cols'][att_idx]
            aug_level_map = df.groupby(y_col).agg({f"{y_col}_aug": lambda x: list(x.unique())}).to_dict()[f"{y_col}_aug"]

        # source level to be modified
        for src_level, _ in tqdm(att):   
            df_src_level = df.loc[df["segment"+str(att_idx+1)] == src_level,:]
            tgt_levels = [tgt_level for (tgt_level, _) in att if tgt_level != src_level]
            for tgt_id, tgt_level in enumerate(tgt_levels): 
                if not config_dict['open_vocab']:
                    if aug_type == 'marginal':
                        df_src_level['text'+str(tgt_id+1)] = tgt_level # marginal augmentation 
                    elif aug_type == "conditional":
                        df_src_level['text'+str(tgt_id+1)] = df_src_level['text'].str.replace(src_level, tgt_level) # sub src_level with tgt_level  in 'text'+str(tgt_id+1)
                else:
                    if aug_type == 'marginal':
                        df_src_level['text'+str(tgt_id+1)] = np.random.choice(aug_level_map[tgt_level], size=len(df_src_level))
                    elif aug_type == "conditional":
                        # change the original augmented y_col to a new level (also augmented)
                        df_src_level[y_col+'_aug'] = np.random.choice(aug_level_map[tgt_level], size=len(df_src_level))
                        df_src_level['text'+str(tgt_id+1)]  = ''
                        for str_col in [col+'_aug' for col in config_dict['txt2ts_y_cols']]:
                            df_src_level['text'+str(tgt_id+1)] += ' ' + df_src_level[str_col]
                        df_src_level['text'+str(tgt_id+1)] = df_src_level['text'+str(tgt_id+1)].str.strip()
                    
            new_text_cols = ['text'+str(tgt_id+1) for tgt_id, _ in enumerate(tgt_levels)]
            # mapping text_col to y_level
            col_level_map = dict(zip(['text' + str(j+1) for j in range(len(tgt_levels))], tgt_levels))

            # Choose model based on model_type
            if model_type == 'tedit_tx':
                new_level_col_map = {k: v for k, v in col_level_map.items() if k in new_text_cols}
                ts_hat_ls = tedit_tx_generate_ts_tx(df_src_level,
                                                   meta,
                                                   config_dict,
                                                   configs,
                                                   "segment"+str(att_idx+1),
                                                   new_level_col_map)
            elif model_type == 'tedit':
                new_level_col_map = {k: v for k, v in col_level_map.items() if k in new_text_cols}
                ts_hat_ls = tedit_generate_ts_tx(df_src_level,
                                                meta,
                                                config_dict,
                                                configs,
                                                "segment"+str(att_idx+1),
                                                new_level_col_map)
            else:  # vital model
                ts_hat_ls = interpolate_ts_tx(df_src_level, model, config_dict, new_text_cols, w)
            
            # target level to modify towards
            for text_col,tgt_level in zip(new_text_cols, tgt_levels):
                # target time series
                df_tgt = df.loc[df["segment"+str(att_idx+1)] == tgt_level,:]
                tgt_ts_list = [df_tgt[[str(i+1) for i in range(config_dict['seq_length'])]].to_numpy()[i] for i in range(len(df_tgt))]
                if config_dict['ts_global_normalize']:
                    tgt_ts_list = [(tgt - global_mean) / global_std for tgt in tgt_ts_list] # globally standardize time series

                # augmented time series
                pairs = ts_hat_ls[text_col]
                aug_df = pd.DataFrame(pairs, columns=['aug_text', 'ts_hat'])
                aug_df['ts_hat'] = aug_df['ts_hat'].apply(lambda x: x.cpu().detach().numpy())
                aug_ts_list = [aug_df['ts_hat'][i] for i in range(len(aug_df))]

                # point-wise performance by mse and mae
                mse = []
                mae = []
                for tgt, aug in zip(tgt_ts_list, aug_ts_list):
                    diff = tgt - aug
                    mse.append(np.mean(diff ** 2))
                    mae.append(np.mean(np.abs(diff)))
                pw_df = pd.DataFrame({ "MSE": mse, "MAE": mae})
                pw_df['src_level'] = src_level
                pw_df['tgt_level'] = tgt_level
                pw_df['attr'] = 'segment'+str(att_idx+1)
                pw_df['aug_type'] = aug_type
                pw_df_all = pd.concat([pw_df_all, pw_df], ignore_index=True)

    # reformat
    mse_df = pw_df_all[['aug_type', 'attr', 'src_level', 'tgt_level', 'MSE']] 
    mse_df.rename(columns={'MSE': 'score'}, inplace=True)
    mae_df = pw_df_all[['aug_type', 'attr', 'src_level', 'tgt_level', 'MAE']] 
    mae_df.rename(columns={'MAE': 'score'}, inplace=True)
    mse_df['metric'] = 'mse'
    mae_df['metric'] = 'mae'

    res_df = pd.concat([mse_df, mae_df], ignore_index=True)
    res_df = res_df[['aug_type', 'attr', 'src_level', 'tgt_level', 'metric', 'score']]
    return res_df