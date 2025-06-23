if 'model_name' not in locals(): model_name = 'syn_gt_at'

text1 = ('No trend.',1)
text2 = ('No seasonal pattern.',1)
text3 = ('No sharp shifts.',1)
text4 = ("The time series exhibits low variability.", 1)
counter_text11 = ('The time series shows upward linear trend.',1)
counter_text12 = ('The time series shows downward linear trend.',1)
counter_text13 = ('The time series shows upward quadratic trend.',1)
counter_text14 = ('The time series shows downward quadratic trend.',1)
counter_text2 = ('The time series exhibits a seasonal pattern.',1)
counter_text31 = ('The mean of the time series shifts upwards.',1)
counter_text32 = ('The mean of the time series shifts downwards.',1)
counter_text4 = ("The time series exhibits high variability.",1)


text_config = {'text_pairs': [
                    [text1, counter_text11, counter_text12, counter_text13, counter_text14],
                    [text2, counter_text2],
                    [text3, counter_text31, counter_text32],
                    [text4, counter_text4]
                ],  'n': None, 'gt': True}

attr_id = 3 # y_col by the third attribute (third element in the text_config['text_pairs'])
config_dict = update_config(config_dict,
    
    # Eval settings (clip)
    # ts2txt
    y_col = 'segment'+str(attr_id),
    y_levels = [t[0]for t in text_config['text_pairs'][attr_id-1]],
    y_pred_levels =[t[0]for t in text_config['text_pairs'][attr_id-1]],
    # txt2ts
    txt2ts_y_cols = ['segment1', 'segment2', 'segment3', 'segment4'], # 
    # open vocabulary
    open_vocab_dict_path = "../../data/synthetic/aug_text.json",
    
    # Data settings
    seq_length = 200,
    custom_target_cols = ['segment1', 'segment2', 'segment3', 'segment4', 'label'], # if text based, overwrite with ['label']
    ts_global_normalize = False, 
    
    # Model settings
    model_name = model_name,
    
    # Train settings
    
    # Text configuration
    text_config = text_config
)
