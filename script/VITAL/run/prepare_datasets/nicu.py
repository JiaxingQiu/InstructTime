# ------------------------------------------------------------------------------------------------
# prepare dataset and arguments for training
# ------------------------------------------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
import time


# ---- ready original dataframes ----
# Train Data
df = pd.read_excel('../../data/nicu/PAS Challenge HR Data.xlsx', engine="calamine")
df.columns = df.columns.astype(str)
df['unique_values'] = df.loc[:, '1':'300'].apply(lambda x: x.nunique(), axis=1)# filter out rows where unique_values <= 5
df = df[df['unique_values'] > 5]
df = df.reset_index(drop=True)
# rolling smoothing with window size 5
original_data = df.loc[:, '1':'300'].copy().astype(float)
smoothed_data = original_data.apply(lambda row: row.rolling(window=3, min_periods=1, center=True).mean(), axis=1)
df.loc[:, '1':'300'] = smoothed_data
del original_data, smoothed_data
df_y = pd.read_excel('../../data/nicu/PAS Challenge Outcome Data.xlsx', engine="calamine")[['VitalID', 'DiedNICU', 'DeathAge']]
df_demo = pd.read_excel('../../data/nicu/PAS Challenge Demographic Data.xlsx', engine="calamine")
df_x = pd.read_excel('../../data/nicu/PAS Challenge Model Data.xlsx', engine="calamine")
df = df.merge(df_x[['VitalID', 'VitalTime', 'Age']], on=['VitalID', 'VitalTime'], how='left')
df = label_death7d(df, df_y, id_col='VitalID')
df = df.merge(df_demo, on='VitalID', how='left')
df_desc = generate_descriptions_parallel(ts_df = df.loc[:, '1':'300'], id_df = df.loc[:, ['VitalID', 'VitalTime']])
df = df.merge(df_desc, on=['VitalID', 'VitalTime'], how='left')
df = text_gen_input_column(df, config_dict)
df['text'] = ''
for str_col in config_dict['txt2ts_y_cols']:
    df['text'] += ' ' + df[str_col].apply(lambda x: x.strip())   
df['text'] = df['text'].str.strip()
df['rowid'] = df.index.to_series() 
df_train = df
if config_dict['downsample']:
    df_train = downsample_neg_levels(df_train, config_dict, config_dict['random_state'])
# option 1: split first data
df_train, df_temp = train_test_split(df_train, test_size=0.3, stratify=df_train[config_dict['y_col']], random_state=config_dict['random_state'])
df_test, df_left = train_test_split(df_temp, test_size=1/3,stratify=df_temp[config_dict['y_col']],random_state=config_dict['random_state'])
print("train, test, left: ", len(df_train), len(df_test), len(df_left))

# # Test Data
# df_y_test = pd.read_excel('../../data/nicu/Test Data/Test Demographic Key.xlsx', sheet_name=0, engine="calamine")
# df_test = pd.read_excel('../../data/nicu/Test Data/Test HR Data.xlsx', sheet_name=0, engine="calamine") # test hr with description
# df_test.columns = df_test.columns.astype(str)
# df_test['unique_values'] = df_test.loc[:, '1':'300'].apply(lambda x: x.nunique(), axis=1)# filter out rows where unique_values <= 5
# df_test = df_test[df_test['unique_values'] > 5]
# df_test = df_test.reset_index(drop=True)
# # rolling smoothing with window size 5
# original_data = df_test.loc[:, '1':'300'].copy().astype(float)
# smoothed_data = original_data.apply(lambda row: row.rolling(window=3, min_periods=1, center=True).mean(), axis=1)
# df_test.loc[:, '1':'300'] = smoothed_data
# del original_data, smoothed_data
# df_test = label_death7d(df_test, df_y_test, id_col='TestID')
# df_demo_test = pd.read_excel('../../data/nicu/Test Data/Test Demographic Data.xlsx', sheet_name=0, engine="calamine")
# df_test = df_test.merge(df_demo_test, on='TestID', how='left')
# df_test['rowid'] = df_test.index.to_series()
# df_test['VitalTime'] = df_test['Age']*24*60*60 # convert to second since birth
# df_test['VitalTime'] = df_test['VitalTime'].astype(int)
# rename_dict = {'TestID': 'VitalID'}
# df_test = df_test.rename(columns=rename_dict)
# df_desc_test = generate_descriptions_parallel(ts_df = df_test.loc[:, '1':'300'], id_df = df_test.loc[:, ['VitalID', 'VitalTime']])
# df_test = df_test.merge(df_desc_test, on=['VitalID', 'VitalTime'], how='left')
# df_test = text_gen_input_column(df_test, config_dict)
# df_test['text'] = ''
# for str_col in config_dict['txt2ts_y_cols']:
#     df_test['text'] += ' ' + df_test[str_col].apply(lambda x: x.strip()) 
# df_test_org = df_test[df.columns]
# # option 2: second data as test and left
# if config_dict['downsample']:
#     df_test_org = downsample_neg_levels(df_test_org, config_dict, config_dict['random_state'])
# df_test, df_left = train_test_split(df_test_org, test_size=0.5, stratify=df_test_org[config_dict['y_col']], random_state=config_dict['random_state']) 
# print("train, test, left: ", len(df_train), len(df_test), len(df_left))

# # ---- augment + balance train data----
# target_event_rate = len(df_test[df_test[config_dict['y_col']]==config_dict['y_levels'][0]])/len(df_test)
if config_dict['ts_aug']:
    df_train = augment_balance_data(df_train, 
                                    config_dict['y_levels'], 
                                    config_dict['y_col'], 
                                    config_dict, 
                                    pretrained_model_path='./pretrained/hr_vae_linear_medium.pth')
    df_test['augid'] = 0
    df_left['augid'] = 0

# ---- created masked subsequences of each time series ----
if config_dict['ts_subseq']:
    df_train = subseq_raw_df(df_train,config_dict)
    df_test = subseq_raw_df(df_test,config_dict)
    df_left = subseq_raw_df(df_left,config_dict)

    # default fill na with preset global mean
    df_train_na = df_train.loc[:, '1':'300'].copy()
    df_test_na = df_test.loc[:, '1':'300'].copy()
    df_left_na = df_left.loc[:, '1':'300'].copy()

    if config_dict['ts_local_normalize']:
        # Fill NaN with row-wise means using np.nanmean
        df_train_na = df_train_na.apply(lambda x: x.fillna(np.nanmean(x)), axis=1)
        df_test_na = df_test_na.apply(lambda x: x.fillna(np.nanmean(x)), axis=1)
        df_left_na = df_left_na.apply(lambda x: x.fillna(np.nanmean(x)), axis=1)
    else:
        # Fill NaN with global preset mean
        df_train_na = df_train_na.fillna(config_dict['ts_normalize_mean'])
        df_test_na = df_test_na.fillna(config_dict['ts_normalize_mean'])
        df_left_na = df_left_na.fillna(config_dict['ts_normalize_mean'])

    df_train.loc[:, '1':'300'] = df_train_na
    df_test.loc[:, '1':'300'] = df_test_na
    df_left.loc[:, '1':'300'] = df_left_na
    # remove df_train_na and df_test_na from memory
    del df_train_na, df_test_na, df_left_na

# ---- block or not ----
if not config_dict['block_label']:
    # reset index but keep the original index as a column 'str_index'
    df_train['str_index'] = df_train.index.to_series()
    df_test['str_index'] = df_test.index.to_series()
    df_left['str_index'] = df_left.index.to_series()
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    df_left = df_left.reset_index(drop=True)
    df_train['label'] = df_train.index.to_series()
    df_test['label'] = df_test.index.to_series()
    df_left['label'] = df_left.index.to_series()
else:
    if 'subid' in df_train.columns:
        if config_dict['ts_augsub']:
            # = get a subsequence then augment it multiple times
            df_train['label'] = df_train['rowid'].astype(int)*100 + df_train['subid'].astype(int)
            df_test['label'] = df_test['rowid'].astype(int)*100 + df_test['subid'].astype(int)
            df_left['label'] = df_left['rowid'].astype(int)*100 + df_left['subid'].astype(int)
        else:
            # get unqiue subsequences for each augmentation
            df_train['label'] = df_train['rowid'].astype(int)*10000 + df_train['augid'].astype(int)*100 + df_train['subid'].astype(int)
            df_test['label'] = df_test['rowid'].astype(int)*10000 + df_test['augid'].astype(int)*100 + df_test['subid'].astype(int)
            df_left['label'] = df_left['rowid'].astype(int)*10000 + df_left['augid'].astype(int)*100 + df_left['subid'].astype(int)
    else:
        df_train['label'] = df_train['rowid'].astype(int)
        df_test['label'] = df_test['rowid'].astype(int)
        df_left['label'] = df_left['rowid'].astype(int)


if config_dict['open_vocab']:
    df_train, df_test, df_left = gen_open_vocab_text(df_train, df_test, df_left, config_dict)

print('\n\nfinal distribution of text prediction')
print(df_train[config_dict['y_col']].value_counts())
print(df_test[config_dict['y_col']].value_counts())
print(df_left[config_dict['y_col']].value_counts())
print(df_train['text'].value_counts())
print(df_test['text'].value_counts())
print(df_left['text'].value_counts())




# ------------------------------------------------------------------------------------------------
# prepare arguments for evaluation
# ------------------------------------------------------------------------------------------------
df_eval = df_left
w = 0.8 # stength of augmentation

math = True
ts_dist = True
rats = True

# argument dictionary {y_col:conditions}
args0 = {#'description_succ_inc': None,
        'description_histogram': None,
        'description_ts_event_binary': None,
        # 'description_ts_event': None
        }

args1 = {'description_histogram': [('description_ts_event_binary', "No events.")],
        'description_ts_event_binary': [('description_histogram', "Low variability.")]
        }

args_ls = [args0, args1]

# Define the base augmentation pairs
base_aug_dict = {'description_histogram': [('Low variability.', 'High variability.'),
                                            ('High variability.', 'Low variability.')],
                'description_ts_event_binary': [('No events.', 'Bradycardia events happened.'),
                                                ('Bradycardia events happened.', 'No events.')],
                }


