from tedit_data import TEditDataset
from tedit_trainer import Trainer
from tedit_model import *
from tedit_generation import tedit_generate
import yaml


######################################
# config
######################################
_, meta = TEditDataset(df_train, df_test, df_left, config_dict, split="train").get_loader(batch_size=128)
configs = yaml.safe_load(open("tedit_configs/"+tedit_mdl+"/"+dataset_name+".yaml")) 
configs['model']['attrs']['num_attr_ops'] = meta['attr_n_ops']
configs['train']['output_folder'] = "tedit_save/"+tedit_mdl+"/"+dataset_name
if resume:
    configs['train']['model_path'] = configs['train']['output_folder'] + "/ckpts/model_best.pth"  # uncomment to resume training


######################################
# training
######################################
if train:
    train_loader, _ = TEditDataset(df_train, df_test, df_left, config_dict, split="train").get_loader(batch_size=128)
    valid_loader, _  = TEditDataset(df_train, df_test, df_left, config_dict, split="valid").get_loader(batch_size=128)
    model = ConditionalGenerator(configs['model'])
    trainer = Trainer(configs['train'], model, train_loader, valid_loader)
    trainer.train()

######################################
# generate (take a peak)
######################################
plot_n = 20
df_left_tmp = df_left.sample(plot_n)
test_loader, _ = TEditDataset(df_train, df_test, df_left_tmp, config_dict, split="test").get_loader(batch_size=128, shuffle=False)
pred_cg, pred_te, src, tgt = tedit_generate(
    configs,
    test_loader=test_loader,
    n_samples = 1,
    sampler = "ddpm-ddim",
    plot_n=plot_n
)
pred_cg, pred_te, src, tgt = tedit_generate(
    configs,
    test_loader=test_loader,
    n_samples = 1,
    sampler = "ddpm-ddim", # used in paper
    plot_n=plot_n
)