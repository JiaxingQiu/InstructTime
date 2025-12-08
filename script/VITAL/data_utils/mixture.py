
# ------------------------------------------------------------------
# mix time series without ground truth (large-random synthetic dataset)
# ------------------------------------------------------------------


# utilities to mix attributes of time series and text 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def mix_2tstxt(df, config_dict, text1, text2, n = None, w = 1.0, plot = False):
    """
    Mix two time series patterns by sampling with replacement n times.
    Vectorized version for better performance.
    """
    df1 = df[df.text == text1].reset_index(drop=True)
    df2 = df[df.text == text2].reset_index(drop=True)
    ts_cols = [str(i) for i in range(1, config_dict['seq_length']+1)]
    
    if n is None:
        n = len(df1)
    
    # Sample indices with replacement
    idx1 = np.random.randint(0, len(df1), size=n)
    idx2 = np.random.randint(0, len(df2), size=n)
    
    # Get all series at once
    ts1_all = df1.loc[idx1, ts_cols].values
    ts2_all = df2.loc[idx2, ts_cols].values
    
    # Standardize all series at once
    ts1_std = (ts1_all - ts1_all.mean(axis=1, keepdims=True)) / ts1_all.std(axis=1, keepdims=True)
    ts2_std = (ts2_all - ts2_all.mean(axis=1, keepdims=True)) / ts2_all.std(axis=1, keepdims=True)
    
    # Mix all series at once
    mixed_series = ts1_std + w * ts2_std
    
    # Create descriptions
    descriptions = [text1 + ' ' + text2] * n
    
    # Plot first 10 examples if requested
    if plot:
        for i in range(min(3, n)):
            plt.figure(figsize=(15, 5))
            plt.plot(ts2_std[i], 'b--', linewidth=1, label='Series 2')
            plt.plot(ts1_std[i], 'g--', linewidth=1, label='Series 1')
            plt.plot(mixed_series[i], 'r-', linewidth=2, label='Mixed Series')
            plt.title(text1 + ' ' + text2)
            plt.legend()
            plt.show()
    
    # mixed_series make them as the  ts_cols of a dataframe
    # descriptions it a column of 'text'
    df_mixed = pd.DataFrame(mixed_series, columns=ts_cols)
    df_mixed['text'] = descriptions
    return df_mixed



def mix_multiple_tstxt(df, config_dict, text_ls, weights=None, n=None, plot=False):
    """
    Mix multiple time series patterns by sampling with replacement n times.
    Vectorized version for better performance.
    
    Args:
        df: DataFrame containing time series data
        config_dict: Configuration dictionary containing sequence length
        text_ls: List of pattern descriptions to mix
        weights: List of weights for each pattern (default: equal weights)
        n: Number of samples to generate
        plot: Whether to plot the mixed series
        
    Returns:
        List of mixed time series and their descriptions
    """
    # Get time series columns
    ts_cols = [str(i) for i in range(1, config_dict['seq_length']+1)]
    
    # Set default weights if not provided
    if weights is None:
        weights = [1.0] * len(text_ls)
    # normalize the weights
    weights = [w / sum(weights) for w in weights]
    
    # Get dataframes for each text
    dfs = [df[df.text == text].reset_index(drop=True) for text in text_ls]
    
    if n is None:
        n = min(len(df) for df in dfs)
    
    # Sample indices with replacement for each pattern
    idxs = [np.random.randint(0, len(df), size=n) for df in dfs]
    
    # Get all series at once for each pattern
    series_all = [df.loc[idx, ts_cols].values for df, idx in zip(dfs, idxs)]
    
    # # Standardize all series at once
    # series_std = [(s - s.mean(axis=1, keepdims=True)) / s.std(axis=1, keepdims=True) 
    #               for s in series_all]
    
    # center all series at once
    series_std = [(s - s.mean(axis=1, keepdims=True)) for s in series_all]
    
    # Mix all series with weights
    mixed_series = np.zeros_like(series_std[0])
    for s, w in zip(series_std, weights):
        mixed_series += w * s
    
    # Create descriptions
    description = ' '.join(text_ls)
    descriptions = [description] * n
    
    # Plot first 10 examples if requested
    if plot:
        for i in range(min(2, n)):
            plt.figure(figsize=(15, 5))
            # Plot individual components
            for j, s in enumerate(series_std):
                plt.plot(s[i], '--', linewidth=1, label=f'Series {j+1}')
            # Plot mixed series
            plt.plot(mixed_series[i], '-', linewidth=2, label='Mixed Series', color='black')
            plt.title(description)
            plt.legend()
            plt.show()
    
    # mixed_series make them as the  ts_cols of a dataframe
    # descriptions it a column of 'text'
    df_mixed = pd.DataFrame(mixed_series, columns=ts_cols)
    df_mixed['text'] = descriptions
    for i, text in enumerate(text_ls):
        df_mixed['segment'+str(i+1)] = text
    return df_mixed



def add_y_col(df, config_dict):
    df[config_dict['y_col']] = ''
    for y_level in config_dict['y_levels']:
        # if df.text contain string y_level, then df[config_dict['y_col']] = y_level
        df.loc[df['text'].str.contains(y_level), config_dict['y_col']] = y_level
    return df



def mix_w_counter(df, config_dict, n=None, plot=False):
    """
    Mix time series based on text pairs where each text has its own scaling weight.
    
    Args:
        df: the dataframe to mix
        config_dict: the config dictionary containing text pairs and weights
        n: the number of samples to mix
        plot: whether to plot the mixed time series
        
    Returns:
        pd.DataFrame: DataFrame containing mixed time series
    """
    text_pairs = config_dict['text_config']['text_pairs']
    
    # Generate all possible combinations
    from itertools import product
    all_combinations = list(product(*text_pairs))
    
    df_mixed = pd.DataFrame()
    for comb in all_combinations:
        # Extract texts and their corresponding weights
        text_ls = [text for text, _ in comb]
        weights = [weight for _, weight in comb]
        
        # Mix the time series with the corresponding weights
        df_sub = mix_multiple_tstxt(df, config_dict, 
                                   text_ls=text_ls,
                                   weights=weights,
                                   n=n,
                                   plot=plot)
        df_mixed = pd.concat([df_mixed, df_sub], ignore_index=True)
    
    return df_mixed

# from itertools import combinations, product
# for comb in combinations(text_pairs, 2):
#     print(comb)
#     print(list(product(*comb)))





# ------------------------------------------------------------------
# mix time series with ground truth
# ------------------------------------------------------------------

# mix time series with ground truth
def mix_multiple_tstxt_gt(df, config_dict, id_dict, text_ls, weights, plot = False):
    # normalize the weights
    weights = [w / sum(weights) for w in weights]
    ts_cols = [str(i) for i in range(1, config_dict['seq_length']+1)]
    # dfs = [df[df['id'].isin(id_dict[text])].reset_index(drop=True) for text in text_ls] # not mapping correctly to the id_dict
    dfs = []
    for text in text_ls:
        ids = id_dict[text]
        df_sub = (
            df.set_index("id")         # make id the index
            .loc[ids]                 # pull rows **in that order**
            .reset_index()            # restore id as a column
        )
        dfs.append(df_sub)
    series_all = [df[ts_cols].values for df in dfs]
    series_centered = [(s - s.mean(axis=1, keepdims=True)) for s in series_all] # center all series at once
    mixed_series = np.zeros_like(series_centered[0])
    for s, w in zip(series_centered, weights):
        mixed_series += w * s
    description = ' '.join(text_ls)
    descriptions = [description] * len(id_dict[text_ls[0]])
    # Plot first 10 examples if requested
    if plot:
        for i in range(min(2, len(id_dict[text_ls[0]]))):
            plt.figure(figsize=(15, 5))
            # Plot individual components
            for j, s in enumerate(series_centered):
                plt.plot(s[i], '--', linewidth=1, label=f'Series {j+1}')
            # Plot mixed series
            plt.plot(mixed_series[i], '-', linewidth=2, label='Mixed Series', color='black')
            plt.title(description)
            plt.legend()
            plt.show()
    df_mixed = pd.DataFrame(mixed_series, columns=ts_cols)
    df_mixed['text'] = descriptions
    for i, text in enumerate(text_ls):
        df_mixed['segment'+str(i+1)] = text
        df_mixed['segment'+str(i+1)+'_srcid'] = id_dict[text]
    return df_mixed  
        

 
def mix_w_counter_gt(df, config_dict, id_dict, plot=False):
    text_pairs = config_dict['text_config']['text_pairs']
    n = config_dict['text_config']['n']
    # Generate all possible combinations
    from itertools import product
    all_combinations = list(product(*text_pairs))
    df_mixed = pd.DataFrame()
    # for each comb
    for comb in all_combinations:
        # Extract texts and their corresponding weights
        text_ls = [text for text, _ in comb]
        weights = [weight for _, weight in comb]
        df_sub = mix_multiple_tstxt_gt(df, config_dict, id_dict, text_ls, weights, plot = plot)
        df_mixed = pd.concat([df_mixed, df_sub], ignore_index=True)
  
    return df_mixed


def _split_id_dict(id_dict): 
    id_dict_train, id_dict_test, id_dict_left = {}, {}, {}
    for level, ids in id_dict.items():
        n_total = len(ids)
        n_train = int(0.7 * n_total)          # 70 %
        n_test = int(0.2 * n_total)          # 20 %
        # the remaining (≈10 %) go to test

        id_dict_train[level] = ids[:n_train]
        id_dict_test[level] = ids[n_train : n_train + n_test]
        id_dict_left [level] = ids[n_train + n_test :]
    return id_dict_train, id_dict_test, id_dict_left


def gt_train_test_left(df, config_dict):
    text_pairs = config_dict['text_config']['text_pairs']
    n = config_dict['text_config']['n'] if config_dict['text_config']['n'] is not None else 300

    # Generate id list for each level of attributes
    id_dict = {}                     
    for att_idx, att in enumerate(text_pairs):           
        for level, weight in att:   
            df_sub = df.loc[df["ts_description"] == level, ["id"]]
            replace = n > len(df_sub)
            sampled_ids = df_sub["id"].sample(n=n, replace=replace, random_state=config_dict['random_state']).tolist()  # reproducible
            id_dict[level] = sampled_ids 

    id_dict_train, id_dict_test, id_dict_left = _split_id_dict(id_dict)

    # quick sanity‑check
    print({k: (len(id_dict_train[k]), len(id_dict_test[k]), len(id_dict_left[k]))
        for k in id_dict.keys()})

    df_train = mix_w_counter_gt(df, config_dict, id_dict_train, plot=False)
    df_test = mix_w_counter_gt(df, config_dict, id_dict_test)
    df_left = mix_w_counter_gt(df, config_dict, id_dict_left)

    return df_train, df_test, df_left