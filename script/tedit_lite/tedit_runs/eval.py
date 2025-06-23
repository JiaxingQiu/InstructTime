from tedit_data import *
from tedit_generation import *
import sys, os
if vital_path not in sys.path: 
    sys.path.append(vital_path)
from eval import *

# prepare meta, model, configs, output_dir
_, meta = TEditDataset(df_train, df_test, df_left, config_dict, split="test").get_loader(batch_size=128)
model, configs = load_model(meta, dataset_name=dataset_name, mdl_name = tedit_mdl)
output_dir = configs['train']['output_folder'] 

# run the same evaluation script on vital model
exec(open(os.path.join(vital_path, 'run/eval.py')).read())
exec(open(os.path.join(vital_path, 'run/eng_eval.py')).read())


res_df_msd = summarize_scores(df_all)
res_df_iqr = summarize_scores(df_all, mean_sd = False)
res_df_msd.to_csv(os.path.join(output_dir, 'res_df_msd.csv'), index=False)
res_df_iqr.to_csv(os.path.join(output_dir, 'res_df_iqr.csv'), index=False)