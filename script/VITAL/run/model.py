# --- prepare saving paths ---
import os
config_dict = update_config(config_dict, output_dir = os.path.abspath('./results/' + model_name))
print(config_dict['output_dir'])
model_path = config_dict['output_dir']+'/model.pth' 
config_path = config_dict['output_dir']+'/config.pth'


# if encoder and decoder not defined otherwise, use the default ones
if 'ts_encoder' not in locals():
    ts_encoder = None
if 'ts_decoder' not in locals():
    ts_decoder = None
if 'text_encoder' not in locals():
    text_encoder = None

# customize model
if overwrite:  
    
    # ------------------------- initialize model -------------------------
    if config_dict['3d']:
        model = VITAL3D(
                    ts_dim=ts_f_dim.shape[1],
                    text_dim=tx_f_dim_ls[0].shape[1],
                    n_text=len(tx_f_dim_ls),
                    output_dim=config_dict['embedded_dim'],
                    ts_encoder = ts_encoder,
                    ts_decoder = ts_decoder,
                    clip_mu = config_dict['clip_mu']
                )
    else:
        model = VITAL(
                    ts_dim=ts_f_dim.shape[1],
                    text_dim=tx_f_dim.shape[1],
                    output_dim=config_dict['embedded_dim'],
                    ts_encoder=ts_encoder,
                    ts_decoder=ts_decoder,
                    text_encoder = text_encoder,
                    clip_mu = config_dict['clip_mu'],
                    variational = config_dict['variational'],
                    gen_w_src_text = config_dict['gen_w_src_text']
                )
    config_dict = update_config(config_dict, model_init = model)
    train_eval_metrics_ts2txt_list = []
    test_eval_metrics_ts2txt_list = []
    train_eval_metrics_txt2ts_list = []
    test_eval_metrics_txt2ts_list = []
    train_losses = []
    test_losses = []
    # ------------------------- ready output directory -------------------------
    import os
    import shutil
    keep_file_list = [
        os.path.join( config_dict['output_dir'], 'model_clip.pth'),
        # Add more files as needed
    ]
    if os.path.exists(config_dict['output_dir']):
        for filename in os.listdir(config_dict['output_dir']):
            file_path = os.path.join(config_dict['output_dir'], filename)
            if file_path not in keep_file_list:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
    else:
        os.makedirs(config_dict['output_dir'])
    torch.save(config_dict, config_path)
else:
    config_dict = torch.load(config_path, map_location=torch.device(device), weights_only=False) # If a variable is assigned anywhere in the function, Python treats it as local
    config_dict = update_config(config_dict, output_dir = os.path.abspath('./results/' + model_name)) # overwrite with current eval_dir
    if 'open_vocab' in locals():
        config_dict = update_config(config_dict, open_vocab = open_vocab)
    model = config_dict['model_init']
    print(nn_summary(model))
    model.device = device
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device), weights_only=False))
    

    