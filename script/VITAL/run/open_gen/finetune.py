# ------------------------------------------------------------
# load original model
# ------------------------------------------------------------
overwrite = False
with open('run/model.py', 'r') as file:
    exec(file.read())

# ------------------------------------------------------------
# viz original model
# ------------------------------------------------------------
model.eval()
cdict = config_dict
df_tmp = df_left.copy()
tgt_level = loo_text
cdict['y_levels'] = list(set(cdict['y_levels'] + [tgt_level]))
cdict['y_pred_levels'] = list(set(cdict['y_pred_levels'] + [tgt_level]))
for y_col in config_dict['txt2ts_y_cols']:
    try:
        df_tmp[y_col] = df_tmp[y_col].str.replace(loo_text, tgt_level)
        text_levels = list(df_tmp[y_col].unique())
        _ = net_emb(df_tmp, model, cdict,
                    top=100,
                    y_col = y_col,
                    text_levels = text_levels)
    except Exception as e:
        print(f"Error plot network embedding for {y_col}")
        continue
model.eval()
w_values = [0, 0.3, 0.5, 0.7, 0.9]
y_col = config_dict['y_col']
ref_level = loo_text
for tgt_level in list(set(cdict['y_levels']) - set([ref_level])):
    df_level = df_left[df_left[y_col] == ref_level].reset_index(drop=True).iloc[[0]].copy()
    df_level['new_text'] = tgt_level
    plot_interpolate_ts_tx_ws_sampling(df_level, model, config_dict, text_cols=['new_text'], w_values = w_values, label = True, b=1, ep=1, ylims = None)

# ------------------------------------------------------------
# prepare finetune data
# ------------------------------------------------------------
config_dict['alpha_init'] = 1e-3
cdict = config_dict
cdict['y_levels'] = list(set(cdict['y_levels'] + [tgt_level]))
cdict['y_pred_levels'] = list(set(cdict['y_pred_levels'] + [tgt_level]))
df_loo = df_left[df_left['text'].str.strip().isin([loo_text])].sample(n=sample_n)
# df_loo = df_left.groupby('text').sample(n=10, random_state=config_dict['random_state'])
# target_loo = gen_target(df_loo, cdict['custom_target_cols'])
# ts_f_loo, tx_f_loo, labels_loo = get_features(df_loo, cdict)
# loo_dataloader = VITALDataset(ts_f_loo, tx_f_loo, labels_loo, target_loo).dataloader(batch_size=cdict['batch_size'])
# augment df_loo to observed conditions
df_aug = pd.DataFrame()
df_tmp = df_loo.copy()
for ref_text in list(set(config_dict['y_levels']) - set([loo_text])):
    tmp_all = pd.DataFrame()
    for w in [0.5, 0.7, 0.9]: #  0, 0.3,
        df_tmp['new_text'] = ref_text
        ts_hat_ls = interpolate_ts_tx(df_tmp, model, cdict, text_cols=['new_text'], w=w)
        tmp = pd.DataFrame(ts_hat_ls["new_text"], columns=["aug_text", "ts_hat"])
        tmp["ts_hat"] = tmp["ts_hat"].apply(lambda x: x.cpu().detach().numpy())
        tmp_all = pd.concat([tmp_all, tmp])
    
    ts_str_cols = [str(i + 1) for i in range(config_dict["seq_length"])] 
    df_ref = df_left[df_left['text'].str.strip().isin([ref_text])].sample(n=len(tmp_all))
    df_ref[ts_str_cols] = np.vstack(tmp_all["ts_hat"].to_numpy())
    df_ref.reset_index(drop=True, inplace=True)
    # plot_ts(df_ref, 0, 200)
    df_aug = pd.concat([df_aug, df_ref], ignore_index=True)

df_loo = pd.concat([df_loo, df_aug], ignore_index=True)

target_loo = gen_target(df_loo, cdict['custom_target_cols'])
ts_f_loo, tx_f_loo, labels_loo = get_features(df_loo, cdict)
loo_dataloader = VITALDataset(ts_f_loo, tx_f_loo, labels_loo, target_loo).dataloader(batch_size=cdict['batch_size'])

# ------------------------------------------------------------
# finetune model
# ------------------------------------------------------------
for param in model.parameters():
    param.requires_grad = True
# for param in model.text_encoder.parameters():
#     param.requires_grad = True
model.train()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config_dict['init_lr'],
    weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',
    factor=0.9,         
    patience=config_dict['patience'],    
    min_lr=1e-20,        
    threshold=1e-4,      
    cooldown=20          
)
_, _, _ = train_vital(model, 
                    loo_dataloader,
                    loo_dataloader, 
                    optimizer, 
                    scheduler,
                    num_epochs = 1000, 
                    train_type = 'joint',
                    alpha_init = config_dict['alpha_init'],
                    es_patience = 100,
                    beta = 0.0
)

# ------------------------------------------------------------
# viz finetuned model
# ------------------------------------------------------------
# plot embedding again
model.eval()
cdict = config_dict
df_tmp = df_left.copy()
tgt_level = loo_text
cdict['y_levels'] = list(set(cdict['y_levels'] + [tgt_level]))
cdict['y_pred_levels'] = list(set(cdict['y_pred_levels'] + [tgt_level]))
for y_col in config_dict['txt2ts_y_cols']:
    try:
        df_tmp[y_col] = df_tmp[y_col].str.replace(loo_text, tgt_level)
        text_levels = list(df_tmp[y_col].unique())
        _ = net_emb(df_tmp, model, cdict,
                    top=100,
                    y_col = y_col,
                    text_levels = text_levels)
    except Exception as e:
        print(f"Error plot network embedding for {y_col}")
        continue
# plot generation examples
model.eval()
w_values = [0, 0.3, 0.5, 0.7, 0.9] # 
y_col = config_dict['y_col']
for tgt_level in list(set(cdict['y_levels'] + [tgt_level])):
    for ref_level in list(set(cdict['y_levels'] + [tgt_level])):
        df_level = df_left[df_left[y_col] == ref_level].reset_index(drop=True).iloc[[0]].copy()
        df_level['new_text'] = tgt_level
        plot_interpolate_ts_tx_ws_sampling(df_level, model, config_dict, text_cols=['new_text'], w_values = w_values, label = True, b=1, ep=1, ylims = None)
