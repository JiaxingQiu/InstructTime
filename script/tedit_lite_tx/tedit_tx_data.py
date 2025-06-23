import numpy as np
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd

class TEditDataset(Dataset):
    """
    A dataset class that initializes directly from dataframes without saving to local files.
    """
    def __init__(self, df_train, df_test, df_left, config_dict, split="train"):
        self.split = split
        
        # Initialize sentence transformer model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2').to(self.device)
        self.text_model.eval()  # Set to evaluation mode
        
        # Calculate normalization stats
        train_mean_std = {
            'mean': np.nanmean(df_train[[str(i+1) for i in range(config_dict['seq_length'])]] .values), 
            'std': np.nanstd(df_train[[str(i+1) for i in range(config_dict['seq_length'])]] .values, ddof=0)
        }
        
        # Get arrays for each split
        train_ts, train_attrs = self._get_arrays_with_text_emb(df_train, config_dict)
        valid_ts, valid_attrs = self._get_arrays_with_text_emb(df_test, config_dict)
        test_ts, test_attrs = self._get_arrays_with_text_emb(df_left, config_dict)
        
        # Normalize
        if config_dict["ts_global_normalize"]:
            m, s = train_mean_std["mean"], train_mean_std["std"]
            s = s if s > 0 else 1e-8
        else:
            m, s = 0, 1
            
        # Calculate dimensions for model configs
        self.meta = {
            'train_mean_std': {'mean': m, 'std': s},
            'attr_emb_dim': train_attrs.shape[-1],  # Should be 768 for mpnet
            'side_dim': train_ts.shape[1],  # Sequence length
            'attr_dim': train_attrs.shape[-1],  # Should be 768 for mpnet
            'n_attrs': 1  # Since we're using a single text embedding
        }
        
        train_ts = (train_ts - m) / s
        valid_ts = (valid_ts - m) / s
        test_ts = (test_ts - m) / s
        
        # Set data based on split
        if split == "all":
            self.ts = np.concatenate([train_ts, valid_ts, test_ts], axis=0)
            self.attrs = np.concatenate([train_attrs, valid_attrs, test_attrs], axis=0)
        elif split == "train":
            self.ts = train_ts
            self.attrs = train_attrs
        elif split == "valid":
            self.ts = valid_ts
            self.attrs = valid_attrs
        elif split == "test":
            self.ts = test_ts
            self.attrs = test_attrs
            
        self.time_points = np.arange(self.ts.shape[1])
        print(f"Loaded {split} split – {len(self)} samples.")
        print(f"Model dimensions - side_dim: {self.meta['side_dim']}, attr_dim: {self.meta['attr_dim']}, n_attrs: {self.meta['n_attrs']}")

    def _get_text_embeddings(self, texts):
        """Get text embeddings from sentence transformer model."""
        # Get embeddings
        with torch.no_grad():
            embeddings = self.text_model.encode(texts, convert_to_numpy=True)
        return embeddings

    def _get_arrays_with_text_emb(self, df, config_dict):
        """Extract time series and compute text embeddings from dataframe."""
        # Get time series
        ts = df[[str(i) for i in range(1, config_dict['seq_length'] + 1)]].values
        
        # Get text embeddings
        text_embeddings = self._get_text_embeddings(df['text'].values)
        
        return ts, text_embeddings

    def __len__(self):
        return self.ts.shape[0]

    def __getitem__(self, idx):
        return {
            "x": self.ts[idx][..., None],    # (L, 1)
            "attrs": self.attrs[idx],        # (emb_dim,)
            "tp": self.time_points           # (L,)
        }
    
    def get_loader(self, batch_size=128, shuffle=False, num_workers=0):
        """Create a DataLoader for this dataset instance."""
        return DataLoader(self,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         num_workers=num_workers), self.meta

# # usage example
# loader = TEditDataset(df_train, df_test, df_left, config_dict, split="test").get_loader(batch_size=128)
# for batch in loader:
#     print(batch["x"].shape, batch["attrs"].shape, batch["tp"].shape)
#     break



class TEditDataset_lite(Dataset):
    """
    Lightweight version of TEditDataset initialized from precomputed meta and a single dataframe.
    This version generates text embeddings on the fly using SentenceTransformer.
    
    Args:
        df_input (pd.DataFrame): Input dataframe containing time series and text
        meta (dict): Precomputed metadata containing normalization stats and embedding dimension
        config_dict (dict): Configuration dictionary containing sequence length
    """
    def __init__(self, df_input, meta, config_dict):
        # Validate inputs
        if not isinstance(df_input, pd.DataFrame):
            raise TypeError("df_input must be a pandas DataFrame")
        if not isinstance(meta, dict) or 'train_mean_std' not in meta:
            raise ValueError("meta must be a dictionary containing 'train_mean_std'")
        if not isinstance(config_dict, dict) or 'seq_length' not in config_dict:
            raise ValueError("config_dict must contain 'seq_length'")
            
        self.meta = meta
        self.config_dict = config_dict
        
        # Initialize sentence transformer model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2').to(self.device)
        self.text_model.eval()  # Set to evaluation mode
        
        # Validate required columns exist
        required_ts_cols = [str(i) for i in range(1, config_dict['seq_length'] + 1)]
        missing_ts_cols = [col for col in required_ts_cols if col not in df_input.columns]
        if missing_ts_cols:
            raise ValueError(f"Missing time series columns: {missing_ts_cols}")
        if 'text' not in df_input.columns:
            raise ValueError("DataFrame must contain a 'text' column")
            
        # Get arrays
        ts, attrs = self._get_arrays(df_input, self.config_dict)

        # Normalize using precomputed train mean and std
        m, s = meta['train_mean_std']['mean'], meta['train_mean_std']['std']
        s = s if s > 0 else 1e-8  # Prevent division by zero
        self.ts = (ts - m) / s
        self.attrs = attrs
        self.time_points = np.arange(self.ts.shape[1])
        
        print(f"Loaded lite dataset – {len(self)} samples with embedding dimension {self.attrs.shape[-1]}.")
        print(f"Model dimensions - side_dim: {self.meta['side_dim']}, attr_dim: {self.meta['attr_dim']}, n_attrs: {self.meta['n_attrs']}")

    def _get_text_embeddings(self, texts):
        """Get text embeddings from sentence transformer model."""
        # Get embeddings
        with torch.no_grad():
            embeddings = self.text_model.encode(texts, convert_to_numpy=True)
        return embeddings

    def _get_arrays(self, df: pd.DataFrame, config_dict: dict) -> tuple:
        """
        Extract time series and compute text embeddings from dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            config_dict (dict): Configuration dictionary
            
        Returns:
            tuple: (time_series_array, text_embeddings_array)
        """
        # Get time series
        ts = df[[str(i) for i in range(1, config_dict['seq_length'] + 1)]].values
        
        # Get text embeddings
        text_embeddings = self._get_text_embeddings(df['text'].values)
        
        return ts, text_embeddings

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.ts.shape[0]

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            dict: Dictionary containing:
                - x: Time series data of shape (L, 1)
                - attrs: Text embeddings of shape (emb_dim,)
                - tp: Time points of shape (L,)
        """
        return {
            "x": self.ts[idx][..., None],    # (L, 1)
            "attrs": self.attrs[idx],        # (emb_dim,)
            "tp": self.time_points           # (L,)
        }

    def get_loader(self, batch_size: int = 128, shuffle: bool = False, num_workers: int = 0) -> tuple:
        """
        Create a DataLoader for this lite dataset instance.
        
        Args:
            batch_size (int): Number of samples per batch
            shuffle (bool): Whether to shuffle the data
            num_workers (int): Number of worker processes for data loading
            
        Returns:
            tuple: (DataLoader instance, metadata dictionary)
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True  # Enable faster data transfer to GPU
        ), self.meta
