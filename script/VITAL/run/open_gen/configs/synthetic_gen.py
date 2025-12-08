text1 = ('flat trend',1)
text3 = ('no shift of mean',1)
text4 = ("moderate variability", 1)
counter_text11 = ('upward trend', 1)
counter_text12 = ('downward trend',1)
counter_text31 = ('upward shift of mean',1)
counter_text32 = ('downward shift of mean',1)
counter_text41 = ("low variability",1)
counter_text42 = ("high variability",1)


text_config = {'text_pairs': [
                    [text1, counter_text11, counter_text12],
                    [],# space holder for seasonality
                    [text3, counter_text31, counter_text32],
                    [text4, counter_text41, counter_text42]
                ],  'n': None, 'gt': False}

config_dict = update_config(config_dict,
    
    # Eval settings (clip)
    # ts2txt
    y_col = 'segment'+str(attr_id),
    y_levels = list(set([t[0]for t in text_config['text_pairs'][attr_id-1]])-set([loo_text])),
    y_pred_levels = list(set([t[0]for t in text_config['text_pairs'][attr_id-1]])-set([loo_text])),
    # txt2ts
    txt2ts_y_cols = ['segment'+str(attr_id)], # 
    # open vocabulary
    open_vocab_dict_path = "../../data/synthetic/aug_text_gen.json",
    
    # Data settings
    seq_length = 200,
    custom_target_cols = ['label'],
    ts_global_normalize = False, 
    
    # Model settings
    model_name = model_name,
    
    # Train settings
    
    # Text configuration
    text_config = text_config
)
