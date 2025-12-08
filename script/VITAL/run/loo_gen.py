from IPython.display import clear_output 
if 'num_epochs' not in locals(): num_epochs = 1000

dataset_name = 'syn_gt'
attr_suffix = ''
open_vocab = False
suffix = '_gen'+str(attr_id)+'/'+loo_text.lower().replace(' ', '_') 

# config 
exec(open('run/settings.py', 'r').read())
model_name = ''.join([dataset_name, attr_suffix, suffix]) 
exec(open('run/open_gen/configs/synthetic_gen.py', 'r').read())
config_dict = update_config(config_dict, open_vocab = open_vocab)
config_dict = update_config(config_dict, alpha_init = 1e-2) # pretraining alpha_init
config_dict = update_config(config_dict, num_epochs = num_epochs) # pretraining alpha_init
config_dict = update_config(config_dict, custom_target_cols = ['label'])
config_dict = update_config(config_dict, output_dir = os.path.abspath('./results/' + model_name))


# prepare data
exec(open('run/open_gen/prepare_datasets/synthetic_gen.py', 'r').read())
exec(open('run/inputs.py', 'r').read())

# pretrain
exec(open('run/model.py', 'r').read())
exec(open('run/train_2steps.py', 'r').read())

config_dict_org = config_dict.copy()

if loo_text == "baseline":
    
    config_dict = config_dict_org.copy() # # config of the pretrain model
    print(config_dict['output_dir'])
    print(config_dict['y_levels'])
    for w in [0.9]:
        exec(open('run/eval.py', 'r').read()) # eval classifier will be saved under config_dict['output_dir'], but result dataframe will be saved under subfolder output_dir
        exec(open('run/eng_eval.py', 'r').read())
    clear_output(wait=True)
        
else:
    for sample_n in [0, 1, 2, 5, 10]:
        
        config_dict = config_dict_org.copy() # # config of the pretrain model
        output_dir = config_dict['output_dir'] + "/" + str(sample_n) # saving path of finetuned model performances
        os.makedirs(output_dir, exist_ok=True)
        print(output_dir)

        # finetune
        if sample_n > 0:
            exec(open('run/open_gen/finetune.py', 'r').read())
        
        # eval
        config_dict['y_levels'] = config_dict['y_levels'] + [loo_text]
        config_dict['y_pred_levels'] = config_dict['y_pred_levels'] + [loo_text]
        for w in [0.9]:
            exec(open('run/eval.py', 'r').read()) # eval classifier will be saved under config_dict['output_dir'], but result dataframe will be saved under subfolder output_dir
            exec(open('run/eng_eval.py', 'r').read())
            res_df_msd.to_csv(os.path.join(output_dir, 'res_df_msd'+str(w)+'.csv'), index=False)
            res_df_iqr.to_csv(os.path.join(output_dir, 'res_df_iqr'+str(w)+'.csv'), index=False)
        
        clear_output(wait=True)
    