# import torch
# from torch.utils.data import Dataset
# import numpy as np
# from tqdm import tqdm
# import pandas as pd
# import matplotlib.pyplot as plt
# from joblib import Parallel, delayed
# import multiprocessing

# from config import *
# from .preprocessor import *
# from .describer import generate_descriptions_parallel

# # Create class vitalsign timeseries which is a child of Dataset class from torch.util.data
# class ScaledTSDataset(Dataset):

#     def __init__(self, ts_df):
#         self.ts_df = ts_df # Shape: [obs, time]
#         self.ts_scaled, self.ts_means, self.ts_stds = self.scale_ts(ts_df)
 

#     def scale_ts(self, ts_df):
#         data = ts_df.values
#         obs_mean = np.nanmean(data, axis=1)
#         osb_std = np.nanstd(data, axis=1)
#         osb_std = np.where(osb_std == 0, 1e-8, osb_std)
#         data_t = (data.T - obs_mean.T) / osb_std.T
#         data_scaled = data_t.T
#         return data_scaled, obs_mean, osb_std
        
#     def __len__(self):
#         return len(self.ts_df)
    
#     def __getitem__(self, idx):
#         # Get single time series and convert to tensor
#         ts = torch.tensor(self.ts_scaled[idx], dtype=torch.float32).to(device)
#         ts_mean = torch.tensor(self.ts_means[idx], dtype=torch.float32).to(device)
#         ts_std = torch.tensor(self.ts_stds[idx], dtype=torch.float32).to(device)
#         return ts, ts_mean, ts_std



# # def augment_ts_df(ts_df, pretrained_model_path, K = 50, dist = 5e-4):

# #     # df to return
# #     df_aug = pd.DataFrame()

# #     # K number of augmentations for each sample
# #     model = VAE_Linear_Medium().to(device)
# #     model.load_state_dict(torch.load(pretrained_model_path))
# #     model.eval()
# #     ts_dataset = ScaledTSDataset(ts_df)

# #     for i in tqdm(range(len(ts_dataset))):
# #         x, ts_mean, ts_std = ts_dataset[i]
# #         ts_mean = ts_mean.cpu().detach().numpy()
# #         ts_std = ts_std.cpu().detach().numpy()
# #         ts_hat_ls = []
# #         euc_dist_ls = []
# #         for k in range(K): 
# #             # augment x k times
# #             distance = np.random.uniform(0, dist)
# #             z_mean, z_log_var = model.encode(x)
# #             z = model.reparameterization(z_mean, z_log_var + distance)
# #             x_hat = model.decode(z) # length of 300
# #             z_mean_hat, _ = model.encode(x_hat)

# #             # Calculate Euclidean distance
# #             z_mean = z_mean.cpu().detach().numpy()
# #             z_mean_hat = z_mean_hat.cpu().detach().numpy()
# #             euc_dist = np.sqrt(np.sum((z_mean - z_mean_hat) ** 2))

# #             x_hat = x_hat.cpu().detach().numpy()
# #             ts_hat = x_hat * ts_std + ts_mean
# #             ts_hat_ls.append(ts_hat)
# #             euc_dist_ls.append(euc_dist)

# #         # Convert to numpy array with shape (K, 300)
# #         ts_hat_ls = np.array(ts_hat_ls)
# #         euc_dist_ls = np.array(euc_dist_ls)

# #         #  the dataframe
# #         df_aug_i = pd.DataFrame(ts_hat_ls, columns=[str(i) for i in range(1, 301)])  # '1' to '300'
# #         # make all cells integer 
# #         df_aug_i = df_aug_i.round().astype(int)
# #         df_aug_i.insert(0, 'rowid', [i] * K)  # Add rowid at the beginning
# #         df_aug_i['euc_dist'] = euc_dist_ls     # Add euc_dist at the end

# #         df_aug = pd.concat([df_aug, df_aug_i], ignore_index=True)


# #     return df_aug


# def augment_ts_df(ts_df, pretrained_model_path, K=50, dist=5e-4):
#     # df to return
#     df_aug = pd.DataFrame()

#     # K number of augmentations for each sample
#     model = VAE_Linear_Medium().to(device)
#     model.load_state_dict(torch.load(pretrained_model_path))
#     model.eval()
#     ts_dataset = ScaledTSDataset(ts_df)

#     for i in tqdm(range(len(ts_dataset))):
#         x, ts_mean, ts_std = ts_dataset[i]  # Already on GPU from dataset
#         ts_hat_ls = []
#         euc_dist_ls = []
        
#         for k in range(K): 
#             # augment x k times
#             distance = torch.tensor(np.random.uniform(0, dist), device=device)
#             z_mean, z_log_var = model.encode(x)
#             z = model.reparameterization(z_mean, z_log_var + distance)
#             x_hat = model.decode(z)
#             z_mean_hat, _ = model.encode(x_hat)

#             # Calculate Euclidean distance on GPU
#             euc_dist = torch.sqrt(torch.sum((z_mean - z_mean_hat) ** 2))
            
#             # Scale back on GPU
#             ts_hat = x_hat * ts_std + ts_mean
            
#             # Store results (only convert to CPU at the end)
#             ts_hat_ls.append(ts_hat)
#             euc_dist_ls.append(euc_dist)

#             # Cleanup intermediate tensors
#             del z_mean, z_log_var, z, x_hat, z_mean_hat
#             torch.cuda.empty_cache()

#         # Stack tensors on GPU
#         ts_hat_tensor = torch.stack(ts_hat_ls)
#         euc_dist_tensor = torch.stack(euc_dist_ls)

#         # Only convert to numpy at the final step
#         ts_hat_np = ts_hat_tensor.round().cpu().numpy()
#         euc_dist_np = euc_dist_tensor.cpu().numpy()

#         # Create DataFrame
#         df_aug_i = pd.DataFrame(ts_hat_np, columns=[str(i) for i in range(1, 301)])
#         df_aug_i = df_aug_i.astype(int)
#         df_aug_i.insert(0, 'rowid', [i] * K)
#         df_aug_i['euc_dist'] = euc_dist_np

#         df_aug = pd.concat([df_aug, df_aug_i], ignore_index=True)

#         # Cleanup
#         del ts_hat_ls, euc_dist_ls, ts_hat_tensor, euc_dist_tensor, df_aug_i
#         torch.cuda.empty_cache()

#     # Final cleanup
#     del model, ts_dataset
#     torch.cuda.empty_cache()
#     return df_aug


# def process_single_sample(i, x, ts_mean, ts_std, model, K, dist):
#     """Process a single sample for augmentation"""
#     ts_mean = ts_mean.cpu().detach().numpy()
#     ts_std = ts_std.cpu().detach().numpy()
#     ts_hat_ls = []
#     euc_dist_ls = []
    
#     for k in range(K):
#         # augment x k times
#         distance = np.random.uniform(0, dist)
#         z_mean, z_log_var = model.encode(x)
#         z = model.reparameterization(z_mean, z_log_var + distance)
#         x_hat = model.decode(z)
#         z_mean_hat, _ = model.encode(x_hat)

#         # Calculate Euclidean distance
#         z_mean = z_mean.cpu().detach().numpy()
#         z_mean_hat = z_mean_hat.cpu().detach().numpy()
#         euc_dist = np.sqrt(np.sum((z_mean - z_mean_hat) ** 2))

#         x_hat = x_hat.cpu().detach().numpy()
#         ts_hat = x_hat * ts_std + ts_mean
#         ts_hat_ls.append(ts_hat)
#         euc_dist_ls.append(euc_dist)

#     # Convert to numpy array
#     ts_hat_ls = np.array(ts_hat_ls)
#     euc_dist_ls = np.array(euc_dist_ls)

#     # Create dataframe for this sample
#     df_aug_i = pd.DataFrame(ts_hat_ls, columns=[str(j) for j in range(1, 301)])
#     df_aug_i = df_aug_i.round().astype(int)
#     df_aug_i.insert(0, 'rowid', [i] * K)
#     df_aug_i['euc_dist'] = euc_dist_ls

#     del ts_hat_ls, euc_dist_ls
#     return df_aug_i

# def augment_ts_df_parallel(ts_df, pretrained_model_path, K=50, dist=5e-4):
    
#     # Clear CUDA cache 
#     torch.cuda.empty_cache()

#     # Determine optimal number of cores
#     total_cores = multiprocessing.cpu_count()
#     n_cores = max(1, int(total_cores * 0.5))  # Use 50% of available cores
#     print(f"Total cores available: {total_cores}")
#     print(f"Using {n_cores} cores for parallel processing")

#     # Initialize model
#     model = VAE_Linear_Medium().to(device)
#     model.load_state_dict(torch.load(pretrained_model_path))
#     model.eval()
#     ts_dataset = ScaledTSDataset(ts_df)

#     # Process samples in parallel
#     results = Parallel(n_jobs=n_cores, verbose=1)(
#             delayed(process_single_sample)(
#                 i, 
#                 ts_dataset[i][0],  # x
#                 ts_dataset[i][1],  # ts_mean
#                 ts_dataset[i][2],  # ts_std
#                 model, 
#                 K, 
#                 dist
#             ) for i in range(len(ts_dataset))
#         )
#     # Combine results
#     df_aug = pd.concat(results, ignore_index=True)
#     # Cleanup
#     del model, ts_dataset, results
#     torch.cuda.empty_cache()
#     return df_aug


# def plot_augmentor(x, pretrained_model_path):

#     # x has to be a output of scaledtsdataset
    
#     # K number of augmentations for each sample
#     model = VAE_Linear_Medium().to(device)
#     model.load_state_dict(torch.load(pretrained_model_path))
#     model.eval()

    
#     #  Create figure and subplots
#     fig, axs = plt.subplots(2, 3, figsize=(15, 8))
#     fig.suptitle('Original Signal vs Reconstructions with Different Distances', fontsize=14)

#     # Flatten axs for easier iteration
#     axs = axs.flatten()

#     # Plot original signal in first subplot
#     axs[0].plot(x.detach().numpy(), 'b-', label='Original')
#     axs[0].set_title('Original Signal')
#     axs[0].set_xlabel('Time')
#     axs[0].set_ylabel('Value')
#     axs[0].grid(True)
#     axs[0].legend()

#     # Plot reconstructions with different distances
#     distances = [0, 5e-4, 7.5e-4, 1e-3, 2e-3]
#     for i, distance in enumerate(distances, 1):
#         # Get embeddings and reconstruction
#         z_mean, z_log_var = model.encode(x)
#         z = model.reparameterization(z_mean, z_log_var + distance)
#         x_hat = model.decode(z)
#         z_mean_hat, z_log_var_hat = model.encode(x_hat)

#         # Calculate Euclidean distance
#         z_mean = z_mean.cpu().detach().numpy()
#         z_mean_hat = z_mean_hat.cpu().detach().numpy()
#         euc_dist = np.sqrt(np.sum((z_mean - z_mean_hat) ** 2))

#         # Plot reconstruction
#         axs[i].plot(x_hat.detach().numpy(), 'r-', label=f'Reconstruction')
#         axs[i].plot(x.detach().numpy(), 'b--', alpha=0.5, label='Original')
#         axs[i].set_title(f'Larger Gaussian var={distance:.1e}\nEuclidean dist between latents={euc_dist:.4f}')
#         axs[i].set_xlabel('Time')
#         axs[i].set_ylabel('Value')
#         axs[i].grid(True)
#         axs[i].legend()

#     # Adjust layout
#     plt.tight_layout()
#     plt.show()
        




# def augment_ts_n_desc(df_sub,
#                     config_dict, 
#                     pretrained_model_path='./data_utils/pretrained/hr_vae_linear_medium.pth',
#                     K = 50):
#     ts_df = df_sub.loc[:, '1':'300'].astype(float)
#     df_aug_sub = augment_ts_df_parallel(ts_df, pretrained_model_path, K = K)
#     df_aug_desc = generate_descriptions_parallel(ts_df = df_aug_sub.loc[:, '1':'300'], id_df = df_aug_sub.loc[:, ['rowid']])
#     df_aug_sub = df_aug_sub.merge(df_aug_desc, on='rowid', how='left')
#     del ts_df, df_aug_desc


#     if 'rowid' in df_sub.columns:
#         # maintain the original rowid
#         df_rowid = df_sub[['rowid']].copy()
#         df_rowid['raw_rowid'] = df_rowid['rowid']
#         df_rowid.reset_index(drop=True, inplace=True)
#         df_rowid['rowid'] = df_rowid.index.to_series()
#         df_aug_sub = df_aug_sub.merge(df_rowid, on='rowid', how='left')
#         df_aug_sub['rowid'] = df_aug_sub['raw_rowid']
#         df_aug_sub.drop(columns=['raw_rowid'], inplace=True)
#         del df_rowid


#     columns_to_keep = ['rowid'] + list(df_sub.columns[~df_sub.columns.isin(df_aug_sub.columns)])
#     df_raw = df_sub[columns_to_keep]
#     df_aug_sub = df_aug_sub.merge(df_raw, on='rowid', how='left')
#     del df_raw

#     df_aug_sub = text_gen_input_column(df_aug_sub, config_dict)
#     # df_sub = pd.concat([df_sub, df_aug_sub])
#     print(f"Augmented {len(df_aug_sub)} rows")
#     return df_aug_sub


# def augment_balance_data(df_sub, 
#                          txt_ls_org,
#                          y_col, 
#                          config_dict, 
#                          pretrained_model_path='./data_utils/pretrained/hr_vae_linear_medium.pth', 
#                          K = 50): # augement each time series K times
    
#     if not config_dict['balance']:
#         aug_data = augment_ts_n_desc(df_sub, config_dict, pretrained_model_path, K)
#         return pd.concat([df_sub, aug_data], ignore_index=True)
    
#     # Get original class sizes and calculate needed augmentations
#     class_sizes = df_sub[y_col].value_counts()
#     max_size = config_dict['ts_aug_max_size']
#     if max_size is None:
#         max_size = class_sizes.max()
#     else:
#         max_size = min(max_size, class_sizes.max())
    
#     print("\n\n\nOriginal class distribution:")
#     for class_label in txt_ls_org:
#         print(f"Class {class_label}: {class_sizes.get(class_label, 0)}")
#     print(f"\nTarget size per class: {max_size}")
    
#     final_dfs = [df_sub]  # Start with original data
    
#     # Only augment classes that need it
#     for class_label in txt_ls_org:
#         if class_label not in class_sizes or class_sizes[class_label] < max_size:
#             df_class = df_sub[df_sub[y_col] == class_label]
#             needed_samples = max_size - class_sizes.get(class_label, 0)
            
#             # Calculate exact K needed to avoid over-generation
#             k = int(np.ceil(needed_samples / len(df_class)))
            
#             if k > 0:
#                 aug_data = augment_ts_n_desc(df_class, config_dict, pretrained_model_path, K=k)
#                 if len(aug_data) > needed_samples:
#                     aug_data = aug_data.head(needed_samples)
#                 final_dfs.append(aug_data)
    
#     df_balanced = pd.concat(final_dfs, ignore_index=True)
    
#     print("\nFinal class distribution:")
#     final_dist = df_balanced[y_col].value_counts()
#     for class_label in txt_ls_org:
#         print(f"Class {class_label}: {final_dist.get(class_label, 0)}")
    
#     # group by rowid, add an augid, indicating the order of each row within each rowid
#     df_balanced = df_balanced.groupby('rowid').apply(lambda x: x.assign(augid=range(len(x)))).reset_index(drop=True)
#     df_balanced['raw_aug_id'] = df_balanced['rowid'].astype(str) + '_' + df_balanced['augid'].astype(str)
#     df_balanced = df_balanced.set_index('raw_aug_id', inplace=False, drop=True).rename_axis(None)
    
#     return df_balanced
        


# def downsample_neg_levels(df, config_dict, random_state=333):
#     neg_sample_size = config_dict['downsample_size']
#     down_levels = config_dict['downsample_levels']
#     # y_levels = df[config_dict['y_col']].unique()
#     y_levels = config_dict['y_levels'] # important: only y_levels are kept in the data after downsampling. (i.e. high, low, moderate need to be explicitly listed in the config)
#     keep_levels = [level for level in y_levels if level not in down_levels]
#     df_kept = df[df[config_dict['y_col']].isin(keep_levels)]
#     # Downsample specified negative levels and combine
#     df_downsampled = pd.concat([
#         df[df[config_dict['y_col']] == level].sample(
#             n=min(neg_sample_size, len(df[df[config_dict['y_col']] == level])), # min(neg_sample_size, len(df[df[config_dict['y_col']] == level]))
#             replace=False, 
#             random_state=random_state
#         ) for level in down_levels
#     ])
#     df = pd.concat([df_kept, df_downsampled])
#     # Print class distribution after downsampling
#     print("After downsampling:")
#     print(df[config_dict['y_col']].value_counts())
#     return df



# def add_linear_trend(df, slope=0.5, text_condition="High amount of consecutive increases."):
#     """
#     Add a linear trend y = slope * x to time series columns for rows matching text_condition
    
#     Args:
#         df (pd.DataFrame): Input dataframe with time series columns '1' to '300'
#         slope (float): Slope of the linear trend to add (default 0.5)
#         text_condition (str): Text condition to match in 'text' column
        
#     Returns:
#         pd.DataFrame: Modified copy of input dataframe
#     """
#     # Create a copy of the input dataframe
#     df_modified = df.copy()
    
#     # Create x values array [1, 2, ..., 300]
#     x = np.arange(1, 301)
    
#     # Calculate linear trend
#     y = np.round(slope * x)
    
#     # Add trend to matching rows
#     mask = df_modified['text'] == text_condition
#     for col in map(str, range(1, 301)):
#         df_modified.loc[mask, col] = df_modified.loc[mask, col] + y[int(col)-1]
    
#     return df_modified