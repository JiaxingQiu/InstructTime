if attr_suffix.startswith('_at_level'):
    config_dict = update_config(config_dict, level_maps = get_level_maps(df_train, config_dict["txt2ts_y_cols"]))

# ---- prepare target matrix ----
if len(config_dict['custom_target_cols']) > 0:
    target_train = gen_target(df_train, config_dict['custom_target_cols'])
    target_test = gen_target(df_test, config_dict['custom_target_cols'])
    target_left = gen_target(df_left, config_dict['custom_target_cols'])
else:
    target_train = None
    target_test = None
    target_left = None

# -- assign global normalization mean and std --
if config_dict['ts_global_normalize']:
    train_mean_std = {'mean': np.nanmean(df_train[[str(i+1) for i in range(config_dict['seq_length'])]].values), 
                      'std': np.nanstd(df_train[[str(i+1) for i in range(config_dict['seq_length'])]].values, ddof=0) }
    config_dict = update_config(config_dict, 
                                ts_normalize_mean = train_mean_std['mean'],
                                ts_normalize_std = train_mean_std['std'])
    print("standardization mean and std: ", config_dict['ts_normalize_mean'], config_dict['ts_normalize_std'])


if overwrite:
    # ------------------------- ready eval inputs for CLIP -------------------------
    # 1. ts to txt prediction evaluation (truei and texti are reserved names) 
    n_levels = len(config_dict['y_levels'])
    for i in range(n_levels):
        df_train[f'true{i+1}'] = df_train[config_dict['y_col']].apply(lambda x: 1 if x == config_dict['y_levels'][i] else 0)
        df_train[f'text{i+1}'] = config_dict['y_pred_levels'][i]
    for i in range(n_levels):
        df_test[f'true{i+1}'] = df_test[config_dict['y_col']].apply(lambda x: 1 if x == config_dict['y_levels'][i] else 0)
        df_test[f'text{i+1}'] = config_dict['y_pred_levels'][i]
    evalclipts2txt_train = EvalCLIPTS2TXT(df_train, 
                                            config_dict,
                                            y_true_cols = [f'true{i+1}' for i in range(n_levels)],
                                            y_pred_cols = [f'text{i+1}' for i in range(n_levels)],
                                            y_pred_cols_ls = config_dict['y_pred_cols_ls'])
    evalclipts2txt_test = EvalCLIPTS2TXT(df_test, 
                                            config_dict,
                                            y_true_cols = [f'true{i+1}' for i in range(n_levels)],
                                            y_pred_cols = [f'text{i+1}' for i in range(n_levels)],
                                            y_pred_cols_ls = config_dict['y_pred_cols_ls'])
    # 2. txt to ts prediction evaluation (caption is reserved name)
    txt_tsid_mapping_train = []
    for col in config_dict['txt2ts_y_cols']:  
        txt_tsid_mapping_train_sub = gen_txt_tsid_mapping(df_train, col, k=500, n_neg=3)
        txt_tsid_mapping_train.extend(txt_tsid_mapping_train_sub)
    txt_tsid_mapping_test = []
    for col in config_dict['txt2ts_y_cols']: 
        txt_tsid_mapping_test_sub = gen_txt_tsid_mapping(df_test, col, k=500, n_neg=3)
        txt_tsid_mapping_test.extend(txt_tsid_mapping_test_sub)
    evalcliptxt2ts_train = EvalCLIPTXT2TS(df_train, txt_tsid_mapping_train, config_dict)
    evalcliptxt2ts_test = EvalCLIPTXT2TS(df_test, txt_tsid_mapping_test, config_dict)

    # ------------------------- ready dataloaders ------------------------- 
    if config_dict['3d']:
        ts_f_train, tx_f_train_ls, labels_train = get_features3d(df_train, 
                                                                config_dict,
                                                                text_col_ls = config_dict['text_col_ls'])
        train_dataloader = VITAL3DDataset(ts_f_train, tx_f_train_ls, labels_train, target_train).dataloader(batch_size=config_dict['batch_size'])
        ts_f_test, tx_f_test_ls, labels_test = get_features3d(df_test, 
                                                                config_dict,
                                                                text_col_ls = config_dict['text_col_ls'])
        test_dataloader = VITAL3DDataset(ts_f_test, tx_f_test_ls, labels_test, target_test).dataloader(batch_size=config_dict['batch_size'])
        ts_f_left, tx_f_left_ls, labels_left = get_features3d(df_left, 
                                                                config_dict,
                                                                text_col_ls = config_dict['text_col_ls'])
        left_dataloader = VITAL3DDataset(ts_f_left, tx_f_left_ls, labels_left, target_left).dataloader(batch_size=config_dict['batch_size'])
    else: 
        ts_f_train, tx_f_train, labels_train = get_features(df_train, config_dict)
        if target_train is None: 
            target_train = gen_text_similarity_target(tx_f_train)
        train_dataloader = VITALDataset(ts_f_train, tx_f_train, labels_train, target_train).dataloader(batch_size=config_dict['batch_size'])
        ts_f_test, tx_f_test, labels_test = get_features(df_test, config_dict)
        if target_test is None: 
            target_test = gen_text_similarity_target(tx_f_test)
        test_dataloader = VITALDataset(ts_f_test, tx_f_test, labels_test, target_test).dataloader(batch_size=config_dict['batch_size'])
        ts_f_left, tx_f_left, labels_left = get_features(df_left, config_dict)
        if target_left is None: 
            target_left = gen_text_similarity_target(tx_f_left)
        left_dataloader = VITALDataset(ts_f_left, tx_f_left, labels_left, target_left).dataloader(batch_size=config_dict['batch_size'])
        
# ------------------------- ready input features dimension -------------------------
# get the dimension of input features
if 'ts_f_dim' not in locals():
    # just to get the dimension out
    if config_dict['3d']:
        ts_f_dim, tx_f_dim_ls, labels_dim = get_features3d(df_train.iloc[:1,:], config_dict, text_col_ls = config_dict['text_col_ls'])
    else:
        ts_f_dim, tx_f_dim, labels_dim = get_features(df_train.iloc[:1,:], config_dict)
