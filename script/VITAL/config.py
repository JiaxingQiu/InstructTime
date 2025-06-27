# configurations goes here
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the config dictionary with default values
config_dict = {
    # Device
    'device': device,
    'random_state': 333,

    # Eval settings
    # clip ts2txt
    'y_col': '', # column name for the classification outcome, ie. 'cl_event'
    'y_levels': [], # important: only y_levels are kept in the data after downsampling. (i.e. high, low, moderate need to be explicitly listed here)
    'y_pred_levels': [],
    'y_pred_cols_ls': None,
    # clip txt2ts + attribute specification
    'txt2ts_y_cols': [],
    # open vocabulary evaluation
    'open_vocab': False,



    # Data settings
    # 'text_col_ls': ['demo', 'cl_event', 'ts_description'], #['cl_event', 'ts_description', 'demo_ga', 'demo_weight', 'demo_apgar', 'demo_mother'] # for 3d
    'seq_length': 300,
    # ts features to be embedded
    'downsample': False,
    'downsample_levels': [], # levels to be downsampled, can be a subset of y_levels
    'downsample_size': 1000,
    'ts_aug': False,
    'ts_aug_max_size': None,
    'balance': True,
    'ts_subseq': False,
    'ts_subseq_n': 1,
    'ts_subseq_min_length_ratio':1/6,
    'ts_subseq_max_length_ratio': 2/3,
    'ts_subseq_step_size_ratio': 1/30,
    'ts_augsub': False,
    # text features
    'attr_suffix': '',
    # target matrix settings
    'block_label': True, # only block or diagonal two types of clip targets
    'custom_target_cols': [], # 'label' is the same as the default "by_label" target
    

    # Data loader settings
    'batch_size': 512,
    'text_encoder_name': 'sentence-transformers/paraphrase-mpnet-base-v2',#'sentence-transformers/all-mpnet-base-v2',
    'ts_encoder_name': 'hr_vae_linear_medium',
    'ts_global_normalize': False, 
    'ts_local_normalize': False, # shared with ts_subseq settings to fill na
    'ts_normalize_mean': None,
    'ts_normalize_std': None,

    # Model settings
    'model_name': None,
    '3d': False, # **{'3d': False/True} to change in update_config
    'embedded_dim': 768,
    'model_init': None,
    'clip_mu': False,
    'variational': False,
    'train_type': 'joint', # or 'vae', 'clip'
    'clip_target_type': 'by_target', # or 'by_label'
    'gen_w_src_text': False, # generate w_src_text for interpolation

    # Training settings
    'init_lr': 1e-4,
    'patience': 500,
    'num_saves': 1,
    'num_epochs': 2000,
    'alpha_init': None, # initial alpha, if None, will be recalibrated after 50 epochs
    'beta': 0.0, # weight of kl loss
    'es_patience': 200, # early stopping patience
    'target_ratio': 10, # target ratio (clip loss over rc loss)

    # Text configuration
    'text_config': None
}


def set_seed(seed: int = 333) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value. Default is 42.
    """
    import random
    import numpy as np
    import torch
    from sklearn.utils import check_random_state
    
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    # Deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Scikit-learn
    check_random_state(seed)
    
    print(f"Random seed set to {seed}")

set_seed(config_dict['random_state'])  # or any other seed value

def update_config(config_dict, **kwargs):
    """Update configuration with new values"""
    config_dict.update(kwargs)
    print(config_dict['model_name'])
    return config_dict


# def get_config_dict():
#     """Get current configuration"""
    
#     # Create a copy to avoid modifying the original
#     config = config_dict.copy()
    
#     # Validate y_levels and y_pred_levels match
#     assert len(config['y_levels']) == len(config['y_pred_levels'])
#     # n_levels = len(config['y_levels'])

#     # # Generate y_pred_cols_ls for 3D models
#     # if config['3d']:
#     #     pred_cols = [f'text{i+1}' for i in range(n_levels)]
#     #     y_pred_cols_ls = []
#     #     for text_col in pred_cols:
#     #         new_cols = [text_col if x == config['y_col'] else x 
#     #                    for x in config['text_col_ls']]
#     #         y_pred_cols_ls.append(new_cols)
#     # else:
#     #     y_pred_cols_ls = None

#     # config['y_pred_cols_ls'] = y_pred_cols_ls

#     if not config['variational']:
#         config['beta'] = 0.0 # no kl loss for non-variational models
    
#     assert config['model_name'] is not None, "model_name is required"
#     print(config['model_name'])
    
#     return config



# text_encoders = [
#     # BERT-base
#     'bert-base-uncased',
#     'bert-base-cased',
#     'bert-large-uncased',
#     'bert-large-cased',
#     'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
#     'allenai/scibert_scivocab_uncased',
#     'dmis-lab/biobert-base-cased-v1.2',
#     # Standard RoBERTa
#     'roberta-base',             # 12-layer
#     'roberta-large',            # 24-layer
#     'allenai/biomed_roberta_base',  # Biomedical domain
#     'roberta-base-openai-detector', # OpenAI's version
#     'microsoft/roberta-base-openai-detector',  # Microsoft's version
#     # DistilBERT
#     'distilbert-base-uncased',  # General purpose, uncased
#     'distilbert-base-cased',    # General purpose, cased
#     'distilroberta-base',       # Distilled version of RoBERTa
#     'distilbert-base-multilingual-cased',  # Multilingual version
#     # MPNet
#     'microsoft/mpnet-base',     # Base model
#     'microsoft/mpnet-large',    # Large model
#     # Sentence Transformers
#     'sentence-transformers/all-mpnet-base-v2',  # Optimized for sentence embeddings
#     'sentence-transformers/paraphrase-mpnet-base-v2',  # Optimized for paraphrase detection
#     'sentence-transformers/all-MiniLM-L6-v2',  # Optimized for sentence embeddings
#     'sentence-transformers/all-MiniLM-L12-v2'  # Optimized for sentence embeddings
# ]

# ts_encoders = [
#     'hr_vae_linear_medium',
#     'sp_vae_linear_medium'
# ]


# def get_true_components(config):
#     """Get all components that are True in the config."""
#     true_components = []
    
#     # Flatten nested dictionary and get True values
#     for category, params in config.items():
#         if category == 'split':
#             if params:
#                 true_components += ['split']
#         else:
#             true_components += [
#                 key for key, value in params.items() 
#                 if value is True
#             ]
    
#     return true_components