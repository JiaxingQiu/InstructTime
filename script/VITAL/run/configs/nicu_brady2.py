if 'model_name' not in locals(): model_name = 'nicu_at' 


text_config = {
    'cl': {
        'die7d': True,
        'fio2': False
    },
    'demo': {
        'ga_bwt': True,
        'gre': False, # gender_race_ethnicity
        'apgar_mage': False
    },
    'ts': {
        # dynamic
        'ts_event': {
            'sumb': True, # sum_brady
            'sumd': False, # sum_desat
            'simple': False,
            'full': False,
            'event1': False},
        # static and categorical
        'succ_inc': True,
        'succ_unc': False,
        'histogram': True
    },
    'split': False
}


# text_config['cl']['die7d'] = True # udpate text_config here if needed
# text_config['ts']['succ_unc'] = False

config_dict = update_config(config_dict,
    
    # Eval settings (clip)
    # ts2txt
    y_col = 'description_ts_event_binary',
    y_levels = ['No events.', 'Bradycardia events happened.'], 
    y_pred_levels = ['No events.', 'Bradycardia events happened.'], 
    txt2ts_y_cols = ['description_ts_event_binary'], # , 'description_histogram', 'description_ts_event_binary'
    # open vocabulary
    open_vocab_dict_path = "../../data/nicu/aug_text.json",
    
    # Data settings
    seq_length = 300,
    downsample = True,
    downsample_size = 20000,
    downsample_levels = ['No events.', 'Bradycardia events happened.'], 
    custom_target_cols = ['description_ts_event_binary', 'label'], # 'description_histogram', 'description_ts_event_binary', 
    ts_global_normalize = True, 
    
    # Model settings
    model_name = model_name,
    
    # Train settings
    
    # Text configuration
    text_config = text_config
)
