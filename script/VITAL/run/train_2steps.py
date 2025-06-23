# # run this script to train clip first then decoder.
config_dict_org = config_dict.copy()

# train clip (once)
if overwrite: # only load clip encoders during training phase
    if not os.path.exists(config_dict_org['output_dir']+'/model_clip.pth'):
        config_dict['train_type'] = 'clip'   
        with open('run/train.py', 'r') as file:
            exec(file.read())
        torch.save(model.state_dict(), config_dict_org['output_dir']+'/model_clip.pth')
    else:
        state_dict = torch.load(config_dict_org['output_dir'] + '/model_clip.pth', map_location=torch.device(device))
        ts_encoder_state = {k.replace('ts_encoder.', ''): v for k, v in state_dict.items() if k.startswith('ts_encoder.')}
        text_encoder_state = {k.replace('text_encoder.', ''): v for k, v in state_dict.items() if k.startswith('text_encoder.')}
        model.ts_encoder.load_state_dict(ts_encoder_state)
        model.text_encoder.load_state_dict(text_encoder_state)
    
config_dict['train_type'] = 'joint'
config_dict['es_patience'] = config_dict_org['num_epochs']
with open('run/train.py', 'r') as file:
    exec(file.read()) 

config_dict = config_dict_org.copy()