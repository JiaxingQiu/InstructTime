# ---------------------------------- Point-wise performance vs GT ----------------------------------
if 'df_pw_dists_all' in locals():
    # prepare data and labels
    metrics = ['mse', 'mae']
    types_ = ['conditional']
    data = [
        df_pw_dists_all[(df_pw_dists_all.metric == m) & 
                        (df_pw_dists_all.aug_type == t)].score
        for m in metrics for t in types_
    ]
    labels = [f"{m.upper()} ({t})" for m in metrics for t in types_]
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=labels)
    ax.set_ylabel('Distance to ground truth')
    ax.set_ylim(-1, 20)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()


#---------------------------------- time series distance ----------------------------------
# # Define the base augmentation pairs
# df_dists_raw = df_dists_ls[0]
# df_dists = pd.DataFrame()
# for metric in ['dtw', 'lcss']: # 'mse', 'mae', , 'lcss'
#     df = eng_dists_multiple(df_dists_raw, base_aug_dict, metric = metric, aug_type='marginal')
#     df_dists = pd.concat([df_dists, df], ignore_index=True)
# df_dists_margi = df_dists
# print('-'*80)
# Define the base augmentation pairs
if 'df_dists_ls' in locals():
    df_dists_raw = df_dists_ls[1]
    df_dists = pd.DataFrame()
    for metric in ['dtw', 'lcss']: # 'mse', 'mae',, 'lcss' 
        df = eng_dists_multiple(df_dists_raw, base_aug_dict, metric = metric, aug_type='conditional')
        df_dists = pd.concat([df_dists, df], ignore_index=True)
    df_dists_condi = df_dists
    print('-'*80)
    # df_dists_all = pd.concat([df_dists_margi, df_dists_condi], ignore_index=True)
    df_dists_all = df_dists_condi


# # ---------------------------------- Math properties ----------------------------------
# df_stats_condi = eng_math_diff_multiple(df_stats_all, base_aug_dict, aug_type='conditional')# 
# # df_stats_margi = eng_math_diff_multiple(df_stats_all, base_aug_dict, aug_type='marginal')# 
# df_stats_condi['aug_type'] = 'conditional'
# # df_stats_margi['aug_type'] = 'marginal'
# # df_stats = pd.concat([df_stats_condi, df_stats_margi], ignore_index=True)
# df_stats = df_stats_condi


#---------------------------------- RaTS ----------------------------------
# df_rats_margi = df_rats_ls[0]
# df_rats_margi = df_rats_margi[df_rats_margi['aug_type'] == 'marginal']
if 'df_rats_ls' in locals():
    df_rats_condi = df_rats_ls[0]
    df_rats_condi = df_rats_condi[df_rats_condi['aug_type'] == 'conditional']
    # df_rats_all = pd.concat([df_rats_margi, df_rats_condi], ignore_index=True)
    df_rats_all = df_rats_condi

    df_rats_all.dropna(inplace=True)
    fig = plot_rats(df_rats_all, metrics = ['RaTS'], figsize=(12, 3))
    plt.show()
    # fig = plot_rats(df_rats_all, metrics = ['RaTS_preserved'], figsize=(12, 3))
    # plt.show()







# ---------------------------------- Summary functions ----------------------------------
def summarize_scores(df_all, aug_type= 'conditional', mean_sd = True):
    df_conditional = df_all[df_all['aug_type'] == aug_type]

    stats_table = df_conditional.groupby('metric')['score'].agg(
        mean='mean',
        std='std',
        q25=lambda x: x.quantile(0.25),
        q50=lambda x: x.quantile(0.50),
        q75=lambda x: x.quantile(0.75)
    ).round(3)
    
    if mean_sd:
        stats_table['final_score'] = stats_table.apply(
            lambda row: f"{row['mean']:.2f} ({row['std']:.2f})",
            axis=1
        )
    else:
        stats_table['final_score'] = stats_table.apply(
            lambda row: f"{row['q50']:.2f} [{row['q25']:.2f}, {row['q75']:.2f}]",
            axis=1
        )

    final_score_row = stats_table['final_score'].to_frame().T
    final_score_row.index = ['final_score']
    # Rename columns if necessary
    if 'mse' in final_score_row.columns:
        final_score_row = final_score_row.rename(columns={
            'mse': 'Point-wise MSE ↓',
            'mae': 'Point-wise MAE ↓',
            'delta_dtw': 'DTW distance decrease ↓',
            'RaTS': 'RaTS ↑',
            'RaTS_preserved': '|RaTS (preserved)|↓'
        })
    else:
        final_score_row = final_score_row.rename(columns={
            'delta_dtw': 'DTW distance decrease ↓',
            'RaTS': 'RaTS ↑',
            'RaTS_preserved': '|RaTS (preserved)|↓'
        })
    if 'delta_lcss' in final_score_row.columns:
        final_score_row = final_score_row.rename(columns={
            'delta_lcss': 'LCSS similarity increase ↑'
        })

    # Reorder columns (only keep those that exist)
    desired_order = [
        'Point-wise MSE ↓',
        'Point-wise MAE ↓',
        'DTW distance decrease ↓',
        'LCSS similarity increase ↑',
        'RaTS ↑',
        '|RaTS (preserved)|↓'
    ]
    final_score_row = final_score_row[[col for col in desired_order if col in final_score_row.columns]]
    return final_score_row

import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

def _draw_subplot(ax, subdf, title):
    width = 0.35
    x     = list(range(len(metrics)))

    # prepare data
    data_m = [subdf[(subdf.metric==m)&(subdf.aug_type=='marginal')]['score']
              for m in metrics]
    data_c = [subdf[(subdf.metric==m)&(subdf.aug_type=='conditional')]['score']
              for m in metrics]

    pos_m = [xi - width/2 for xi in x]
    pos_c = [xi + width/2 for xi in x]

    # marginal (solid)
    ax.boxplot(data_m, positions=pos_m, widths=width, **base_kwargs)

    # conditional (dashed)
    dashed = base_kwargs.copy()
    dashed.update({
        'boxprops':    dict(color="black", linestyle="--"),
        'medianprops': dict(color="red",   linestyle="--", linewidth=2),
        'whiskerprops':dict(color="black", linestyle="--"),
        'capprops':    dict(color="black", linestyle="--"),
    })
    ax.boxplot(data_c, positions=pos_c, widths=width, **dashed)
    # styling
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=0)
    ax.set_title(title)
    ax.set_ylabel("Score (symlog)")
    
    ax.set_yscale('symlog', linthresh=1e-2)
    ax.set_ylim(-1, 10)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(0, color='gray', linestyle='-', linewidth=1)

def plot_summary(df_all):
    # 1) Combine & clean
    attr_levels = df_all['attr'].unique().tolist()
    metrics     = df_all['metric'].unique().tolist()

    # 2) Shared box‐style kwargs
    base_kwargs = dict(
        notch=True,
        showmeans=True,
        patch_artist=False,
        boxprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        medianprops=dict(color="red", linewidth=2),
        flierprops=dict(markeredgecolor="black",
                        markerfacecolor="black",
                        marker="o",
                        markersize=3)
    )

    # 3) Build a GridSpec with 2 columns
    n_attrs  = len(attr_levels)
    n_cols   = 2
    n_rows   = 1 + math.ceil(n_attrs / n_cols)
    fig = plt.figure(figsize=(12, 3*n_rows))
    gs  = fig.add_gridspec(n_rows, n_cols, hspace=0.4)

    # Top row: merged across both columns
    ax0 = fig.add_subplot(gs[0, :])
    _draw_subplot(ax0, df_all, title="All attributes combined")

    # Remaining rows: one attr per cell, left→right top→bottom
    for idx, attr in enumerate(attr_levels):
        row = 1 + (idx // n_cols)
        col = idx % n_cols
        ax  = fig.add_subplot(gs[row, col])
        sub = df_all[df_all['attr']==attr]
        _draw_subplot(ax, sub, title=f"Attribute = {attr}")

    # 4) Legend on the top plot
    legend_items = [
        Line2D([0],[0], color='black', lw=2, linestyle='-',  label='Marginal'),
        Line2D([0],[0], color='black', lw=2, linestyle='--', label='Conditional')
    ]
    ax0.legend(handles=legend_items, loc='lower right')

    plt.tight_layout()
    plt.show()


if 'df_pw_dists_all' in locals():
    df_all = pd.concat([df_rats_all, df_dists_all, df_pw_dists_all], ignore_index=True).dropna(subset=['score'])
else:
    df_all = pd.concat([df_rats_all, df_dists_all], ignore_index=True).dropna(subset=['score'])
# summarize_scores(df_all)


res_df_msd = summarize_scores(df_all)
res_df_iqr = summarize_scores(df_all, mean_sd = False)
if meta is None:
    res_df_msd.to_csv(os.path.join(config_dict['output_dir'], 'res_df_msd'+eval_suffix+'.csv'), index=False)
    res_df_iqr.to_csv(os.path.join(config_dict['output_dir'], 'res_df_iqr'+eval_suffix+'.csv'), index=False)