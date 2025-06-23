if 'model_name' not in locals(): model_name = 'air_at' 

config_dict = update_config(config_dict,
    
    # Eval settings (clip)
    # ts2txt
    y_col = 'city_str',
    y_levels = ['This is air quality in Beijing.', 'This is air quality in London.'],
    y_pred_levels = ['This is air quality in Beijing.', 'This is air quality in London.'],
    # txt2ts
    txt2ts_y_cols = ['city_str', 'season_str', 'year_str'],
    # open vocabulary
    open_vocab_dict_path = "../../data/air_quality/aug_text.json",
    
    # Data settings
    seq_length = 168,
    custom_target_cols = ['city_str', 'season_str', 'year_str', 'label'], 
    ts_global_normalize = True, 

    # Model settings
    model_name = model_name,
    alpha = 1/100,
    
    # Train settings
)
