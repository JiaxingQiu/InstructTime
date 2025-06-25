if 'model_name' not in locals(): model_name = 'air_at' 

config_dict = update_config(config_dict,
    
    # Eval settings (clip)
    # ts2txt
    y_col = 'season_str',
    y_levels = ['The season is spring.', 'The season is summer.', 'The season is fall.', 'The season is winter.'],
    y_pred_levels = ['The season is spring.', 'The season is summer.', 'The season is fall.', 'The season is winter.'],
    # txt2ts
    txt2ts_y_cols = ['season_str'],
    # open vocabulary
    open_vocab_dict_path = "../../data/air_quality/aug_text.json",
    
    # Data settings
    seq_length = 168,
    custom_target_cols = ['season_str', 'label'], 
    ts_global_normalize = True, 

    # Model settings
    model_name = model_name,
    
    # Train settings
)
