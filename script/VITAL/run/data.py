# --- model name as the saving path ----------------------------------------------------------
if 'open_vocab' in locals():
    if open_vocab: suffix = "_open" 
model_name = ''.join([dataset_name, attr_suffix, suffix]) 
# attr_suffix: '' for text-based, '_at' for attribute-based
# suffix: special name for different model architectures
# dataset_name: name of the dataset, can be 'syn_gt', 'syn', 'air', 'nicu'

# --- configs, default attribute-based version ----------------------------------------
if dataset_name == 'syn_gt':
    exec(open('run/configs/synthetic_w_gt.py', 'r').read())
elif dataset_name == "syn":
    exec(open('run/configs/synthetic.py', 'r').read())
elif dataset_name == "air":
    exec(open('run/configs/air_quality.py', 'r').read())
elif dataset_name == "nicu":
    exec(open('run/configs/nicu.py', 'r').read())
if attr_suffix == '':
    config_dict = update_config(config_dict, custom_target_cols = ['label'])
if attr_suffix == '_at':
    config_dict = update_config(config_dict, custom_target_cols = ['label'])
if 'open_vocab' in locals():
    config_dict = update_config(config_dict, open_vocab = open_vocab)
if 'alpha_init' in locals(): 
    config_dict = update_config(config_dict, alpha_init = alpha_init)
    
# --- prepare train, test, left dataframes --------------------------------------------
if dataset_name == 'syn_gt':
    exec(open('run/prepare_datasets/synthetic.py', 'r').read())
elif dataset_name == "syn":
    exec(open('run/prepare_datasets/synthetic.py', 'r').read())
elif dataset_name == "air":
    exec(open('run/prepare_datasets/air_quality.py', 'r').read())
elif dataset_name == "nicu":
    exec(open('run/prepare_datasets/nicu.py', 'r').read())

# --- prepare tensor model inputs -----------------------------------------------------
with open('run/inputs.py', 'r') as file:
    exec(file.read())