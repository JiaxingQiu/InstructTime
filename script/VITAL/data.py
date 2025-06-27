import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from config import *

from data_utils.mixture import *
from data_utils.preprocessor import *
from data_utils.masker import *
# from data_utils.augmentor import *
# from data_utils.describer import *

class VITALDataset(Dataset):
    def __init__(self, ts_features, text_features, labels, targets = None):
        """
        Args:
            ts_features: time series features tensor [N, ts_dim]
            text_features: text features tensor [N, text_dim]
            labels: class labels tensor [N]
        """
        # Verify inputs are tensors
        if not isinstance(ts_features, torch.Tensor):
            ts_features = torch.FloatTensor(ts_features)
        if not isinstance(text_features, torch.Tensor):
            text_features = torch.FloatTensor(text_features)
        if not isinstance(labels, torch.Tensor):
            labels = torch.LongTensor(labels)
        
        assert len(ts_features) == len(text_features) == len(labels), "All inputs must have the same length"
        self.ts_features = ts_features
        self.text_features = text_features
        self.labels = labels
        self.targets = targets
        self.ts_features = self.ts_features.to(device)
        self.text_features = self.text_features.to(device)
        self.labels = self.labels.to(device)
        if targets is None:
            labels_equal = (self.labels.unsqueeze(0) == self.labels.unsqueeze(1))
            self.targets = labels_equal.float()
        else:
            self.targets = targets.float()
        self.targets = self.targets.to(device)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            idx,
            self.ts_features[idx],
            self.text_features[idx],
            self.labels[idx],
            self.targets[idx]
        )
    def dataloader(self, batch_size=32):
        return DataLoader(self, 
                          batch_size=batch_size, 
                          shuffle=True,
                          num_workers=0,
                          pin_memory=False) # they will already be in GPU
    

class VITAL3DDataset(Dataset):
    def __init__(self, ts_features, text_features_list, labels, targets = None):
        """
        Dataset for VITAL3D model handling multiple text features per time series.
        
        Args:
            ts_features: time series features tensor [N, ts_dim]
            text_features_list: list of text features tensors, each [N, text_dim]
            labels: class labels tensor [N]
        """
        # Convert inputs to tensors if needed
        self.ts_features = torch.FloatTensor(ts_features) if not isinstance(ts_features, torch.Tensor) else ts_features
        self.text_features_list = [
            torch.FloatTensor(text_features) if not isinstance(text_features, torch.Tensor) else text_features
            for text_features in text_features_list
        ]
        self.labels = torch.LongTensor(labels) if not isinstance(labels, torch.Tensor) else labels
        
        # Verify dimensions
        n_samples = len(self.ts_features)
        assert all(len(text_features) == n_samples for text_features in self.text_features_list), \
            "All text features must have the same number of samples as ts_features"
        assert len(self.labels) == n_samples, \
            "Labels must have the same number of samples as features"
        
        self.ts_features = self.ts_features.to(device)
        self.text_features_list = [t.to(device) for t in self.text_features_list]
        self.labels = self.labels.to(device)
        if targets is None:
            labels_equal = (self.labels.unsqueeze(0) == self.labels.unsqueeze(1))
            self.targets = labels_equal.float()
        else:
            self.targets = targets.float()
        self.targets = self.targets.to(device)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Returns:
            tuple: (
                idx: sample index,
                ts_features: time series features [ts_dim],
                text_features: list of text features, each [text_dim],
                label: class label,
                target_org: similarity target
            )
        """
        return (
            idx,
            self.ts_features[idx],
            [text_features[idx] for text_features in self.text_features_list],
            self.labels[idx],
            self.targets[idx]
        )
    
    def dataloader(self, batch_size=32, shuffle=True):
        """
        Creates a DataLoader for this dataset.
        
        Args:
            batch_size: int, number of samples per batch
            shuffle: bool, whether to shuffle the data
            num_workers: int, number of worker processes for data loading
        
        Returns:
            DataLoader: PyTorch DataLoader object
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
            drop_last=False
        )
    

class TSFeature(Dataset):
    # mean, std can be computed from this object
    # get the normalized / encoded time series features of raw time series
    def __init__(self, ts_df, normalize_mean=None, normalize_std=None, global_norm = True, local_norm = False):
        """Initialize dataset with optional normalization
        
        Args:
            ts_df (pd.DataFrame): Time series dataframe with shape [obs, time]
            normalize_mean (float, optional): Pre-computed mean for normalization
            normalize_std (float, optional): Pre-computed std for normalization
        """
        # Copy the original dataframe
        self.ts_df = ts_df.copy().astype(float)
        data = self.ts_df.values

        # ---- normalize the time series data ----
        # default is raw scaled data
        data_scaled = data 
        # do a global normalization
        if global_norm: 
            if normalize_mean is None:
                self.normalize_mean = data.mean()
            else:
                self.normalize_mean = normalize_mean
            if normalize_std is None:
                self.normalize_std = data.std()
            else:
                self.normalize_std = normalize_std
            data_scaled = (data_scaled - self.normalize_mean) / (self.normalize_std + 1e-8)
        # continue with a local normalization
        if local_norm: 
            obs_mean = np.nanmean(data_scaled, axis=1)
            osb_std = np.nanstd(data_scaled, axis=1)
            osb_std = np.where(osb_std == 0, 1e-8, osb_std)
            data_scaled_T = (data_scaled.T - obs_mean.T) / osb_std.T
            data_scaled = data_scaled_T.T
        
            
        
        # Store normalized data
        self.norm_ts_df = pd.DataFrame(
            data_scaled,
            index=ts_df.index,
            columns=ts_df.columns
        )
        self.norm_ts_df = self.norm_ts_df.astype(float)
        self.features = self.norm_ts_df.values # numpy array shape: [obs, time]
        self.features = torch.tensor(self.features, dtype=torch.float32)

    def __len__(self):
        return len(self. ts_df)
    
    def __getitem__(self, idx):
        ts_f = torch.tensor(self.features[idx], dtype=torch.float32).to(device)
        return ts_f


class TXTFeature(Dataset):
    def __init__(self, txt_ls, encoder_model_name):
        self.txt_ls = txt_ls # list of strings
        self.encoder_model_name = encoder_model_name
        self.encoder = TXTEncoder(model_name = encoder_model_name)
        self.features = self.encoder.encode_text_list(txt_ls) # tensor shape: [obs, embed_dim]
        # Convert to tensor if not already
        if not isinstance(self.features, torch.Tensor):
            self.features = torch.tensor(self.features)
        # # Global min-max normalization (across all values)
        # min_val = torch.min(self.features)
        # max_val = torch.max(self.features)
        # self.features = (self.features - min_val) / (max_val - min_val + 1e-8)  # Add small epsilon to avoid division by zero

    def __len__(self):
        return len(self.txt_ls)
    
    def __getitem__(self, idx):
        txt_f = torch.tensor(self.features[idx], dtype=torch.float32).to(device)
        return txt_f



def get_ts_txt_org(df, text_col = 'text', seq_length = 300):
    # raise error if any of the columns are not in the dataframe
    df.columns = df.columns.astype(str)
    required_columns = ['1', 'text', 'label']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is required but not found in the dataframe.")
    
    if 'VitalID' in df.columns:
        df = df.assign(id_time='id_' + df['VitalID'].astype(str) + '_' + df['VitalTime'].astype(str))
        df = df.set_index('id_time')
        df = df.drop(columns=['VitalID', 'VitalTime'])
    
    # get normalized time series dataframe norm_ts_df
    ts_df = df.loc[:,'1':str(seq_length)]
    # get text list txt_ls
    txt_ls = df.loc[:,text_col].tolist()
    labels = df.loc[:,'label'].tolist() # id of target matrix, indicating observations with the same id.

    return ts_df, txt_ls, labels


def get_features(df, 
                 config_dict,
                 text_col='text'):
    ts_df, txt_ls, labels = get_ts_txt_org(df, text_col = text_col, seq_length = config_dict['seq_length'])
    # --- ts_f ---
    ts_f = TSFeature(ts_df,
                    normalize_mean=config_dict['ts_normalize_mean'], 
                    normalize_std=config_dict['ts_normalize_std'], 
                    global_norm = config_dict['ts_global_normalize'], 
                    local_norm = config_dict['ts_local_normalize']).features

    # --- tx_f ---
    if 'level_maps' in config_dict.keys():
        tx_f = encode_attributes(df, config_dict['level_maps'])
    else:
        tx_f = TXTFeature(txt_ls, 
                        encoder_model_name=config_dict['text_encoder_name']).features

    # --- labels ---
    labels = torch.tensor(labels)
    
    return ts_f, tx_f, labels


def get_features3d(df, 
                 config_dict,
                 text_col_ls=['demo', 'cl_event', 'ts_description']):
    
    ts_df, _, labels = get_ts_txt_org(df, seq_length = config_dict['seq_length'])
    ts_f = TSFeature(ts_df, 
                     normalize_mean=config_dict['ts_normalize_mean'],
                     normalize_std=config_dict['ts_normalize_std'],
                     global_norm = config_dict['ts_global_normalize'], 
                     local_norm = config_dict['ts_local_normalize']).features
    labels = torch.tensor(labels)
    tx_f_list = []
    for text_col in text_col_ls:
        _, txt_ls, _ = get_ts_txt_org(df, text_col = text_col, seq_length = config_dict['seq_length'])
        if 'level_maps' in config_dict.keys():
            tx_f = encode_attributes(df, config_dict['level_maps'])
        else:
            tx_f = TXTFeature(txt_ls, 
                            encoder_model_name=config_dict['text_encoder_name']).features
        tx_f_list.append(tx_f)
    return ts_f, tx_f_list, labels


def gen_target(df, cluster_cols=['label']):
    if len(cluster_cols) == 0:
        return None
    target_sum = None
    for cluster_col in cluster_cols:
        label_mapping = {cat: idx+1 for idx, cat in enumerate(sorted(df[cluster_col].unique()))}
        df['cluster'] = df[cluster_col].map(label_mapping).astype(int)
        labels = torch.tensor(df['cluster'].values).to(device)
        labels_equal = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        if target_sum is None:
            target_sum = labels_equal
        else:
            target_sum += labels_equal
    target_normalized = target_sum / len(cluster_cols)
    target_normalized = target_normalized.to(device)
    return target_normalized

def gen_text_similarity_target(text_features):
    """
    Generate pairwise target matrix based on cosine similarity between text features.
    
    Args:
        text_features: Text features tensor of shape [N, D] where N is number of samples and D is feature dimension
        
    Returns:
        target: Normalized similarity matrix of shape [N, N] with values between 0 and 1
    """
    text_features_norm = torch.nn.functional.normalize(text_features, p=2, dim=1)
    similarity = torch.mm(text_features_norm, text_features_norm.t())
    similarity = (similarity + 1) / 2
    
    return similarity


import json, random
np.random.seed(config_dict['random_state'])
def gen_open_vocab_text(df_train, df_test, df_left, config_dict):
    df_train['text_attr'] = df_train['text']
    df_test['text_attr'] = df_test['text']
    df_left['text_attr'] = df_left['text']
    with open(config_dict['open_vocab_dict_path'], "r") as f:
        aug_text_dict = json.load(f)
    df_train['text_aug'] = ''
    df_test['text_aug'] = ''
    df_left['text_aug'] = ''
    for attr_col in config_dict['txt2ts_y_cols']:
        df_train[attr_col+'_aug'] = ''
        df_test[attr_col+'_aug'] = ''
        df_left[attr_col+'_aug'] = ''
        for attr_level in aug_text_dict[attr_col]:
            aug_levels = aug_text_dict[attr_col][attr_level] # list of augmented strings
            aug_levels_in = aug_levels[0:int(0.7*len(aug_levels))] # first half goes to training set
            aug_levels_out = aug_levels[int(0.3*len(aug_levels)):] # second half goes to left out set

            train_idx = df_train[attr_col] == attr_level
            test_idx = df_test[attr_col] == attr_level
            left_idx = df_left[attr_col] == attr_level

            df_train.loc[train_idx, attr_col+'_aug'] = random.choices(aug_levels_in, k=train_idx.sum())
            df_test.loc[test_idx, attr_col+'_aug'] = random.choices(aug_levels_in, k=test_idx.sum())
            df_left.loc[left_idx, attr_col+'_aug'] = random.choices(aug_levels_out, k=left_idx.sum())
        df_train['text_aug'] += ' ' + df_train[attr_col+'_aug']
        df_test['text_aug'] += ' ' + df_test[attr_col+'_aug']
        df_left['text_aug'] += ' ' + df_left[attr_col+'_aug']
    df_train['text'] = df_train['text_aug']
    df_test['text'] = df_test['text_aug']
    df_left['text'] = df_left['text_aug']
    return df_train, df_test, df_left
#  --- test ---
# for attr_col in config_dict['txt2ts_y_cols']:
#     col_aug = attr_col + '_aug'
#     left_unique = set(df_left[col_aug].unique())
#     train_unique = set(df_train[col_aug].unique())
#     test_unique = set(df_test[col_aug].unique())

#     # Remove empty strings if present
#     left_unique.discard('')
#     train_unique.discard('')
#     test_unique.discard('')

#     # Check for intersection
#     overlap_train = left_unique & train_unique
#     overlap_test = left_unique & test_unique

#     print(f"Column: {col_aug}")
#     print(f"  Overlap with train: {overlap_train}")
#     print(f"  Overlap with test: {overlap_test}")
#     if not overlap_train and not overlap_test:
#         print("  PASS: df_left is disjoint from df_train and df_test for this column.")
#     else:
#         print("  FAIL: Overlap found!")



def get_level_maps(df, categorical_columns, sort_levels=True):
    level_maps = {}
    
    for col in categorical_columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in dataframe")
            continue
            
        levels = df[col].dropna().unique()
        if sort_levels:
            levels = sorted(levels)
        
        level2idx = {lvl: i for i, lvl in enumerate(levels)}
        level_maps[col] = level2idx
        
        print(f"Column '{col}': {len(levels)} unique levels")
    
    return level_maps

# def encode_attributes(df, level_maps, onehot = True):   
#     attrs_idx = []
    
#     for col, level2idx in level_maps.items():
#         if col not in df.columns:
#             print(f"Warning: Column '{col}' not found in dataframe")
#             continue
            
#         # Encode the column
#         attr_arr = (
#             df[col]
#             .map(level2idx)          # map strings → ints
#             .fillna(-1)              # unseen level or NaN
#             .astype(int)
#             .to_numpy()
#         )
#         attrs_idx.append(attr_arr)
    
#     # Stack all attributes into a single array
#     if attrs_idx:
#         attrs_idx = np.stack(attrs_idx, axis=1)
#     else:
#         attrs_idx = np.array([]).reshape(len(df), 0)
    
#     return  torch.tensor(attrs_idx, dtype=torch.float32)

def encode_attributes(df, level_maps, onehot=True):
    """
    Encode categorical columns.

    Parameters
    ----------
    df : pandas.DataFrame
    level_maps : dict[str, dict[str, int]]
        column → {level → index}.
    onehot : bool, default True
        True  → concatenated one-hot vectors
        False → raw integer indices (same shape as before).

    Returns
    -------
    torch.Tensor
        (N, ΣKi) when onehot=True, else (N, C); dtype float32.
    """
    int_cols   = []          # per-column integer indices
    n_classes  = []          # Ki per column

    for col, level2idx in level_maps.items():
        if col not in df.columns:
            print(f"Warning: column '{col}' not found")
            continue

        arr = df[col].map(level2idx).astype(int).to_numpy()
        int_cols.append(arr)
        n_classes.append(len(level2idx))            # Ki

    if not int_cols:                                # no valid columns
        return torch.empty(len(df), 0, dtype=torch.float32)

    idx_matrix = np.stack(int_cols, axis=1)         # (N, C)

    if not onehot:                                  # raw indices
        return torch.tensor(idx_matrix, dtype=torch.float32)

    # --- one-hot per column, then concatenate ---------------------------------
    onehot_parts = [
        F.one_hot(torch.tensor(idx_matrix[:, i]), num_classes=Ki).float()
        for i, Ki in enumerate(n_classes)
    ]
    return torch.cat(onehot_parts, dim=1)           # (N, ΣKi)
