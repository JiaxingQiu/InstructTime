
# if dataset_name == 'syn_gt':
#     df_train = pd.read_csv('../../data/synthetic/train_gt.csv.zip', compression='zip')
#     df_test = pd.read_csv('../../data/synthetic/test_gt.csv.zip', compression='zip')
#     df_left = pd.read_csv('../../data/synthetic/left_gt.csv.zip', compression='zip')
# else:
#     df_train = pd.read_csv('../../data/synthetic/train.csv.zip', compression='zip')
#     df_test = pd.read_csv('../../data/synthetic/test.csv.zip', compression='zip')
#     df_left = pd.read_csv('../../data/synthetic/left.csv.zip', compression='zip')

# if config_dict['open_vocab']:
#     df_train, df_test, df_left = gen_open_vocab_text(df_train, df_test, df_left, config_dict)

# # update text
# df_train['text'] = ''
# df_test['text'] = ''
# df_left['text'] = ''
# for str_col in config_dict['txt2ts_y_cols']:
#     df_train['text'] += ' ' + df_train[str_col]
#     df_test['text'] += ' ' + df_test[str_col]
#     df_left['text'] += ' ' + df_left[str_col]


# # left one out textual condition
# # df_train = df_train[df_train['text'].str.strip() != loo_text.strip()]
# df_train = df_train[df_train['text'].str.strip().isin(config_dict['y_levels'])]
# df_left = pd.concat([df_test, df_left], axis=0, ignore_index=True)
# df_left = df_left[df_left['text'].str.strip().isin(config_dict['y_levels'] + [loo_text])]

# print('\n\nfinal distribution of text prediction')
# print(df_train['text'].value_counts())
# print(df_left['text'].value_counts())
from sklearn.model_selection import train_test_split
df = pd.read_csv('../../data/synthetic/data_open_gen.csv.zip', compression='zip')
df.columns = df.columns.astype(str)
df['text'] = df['ts_description']
df['text'] = df['text'].str.strip()
df = df.reset_index(drop=True)
df[config_dict['y_col']] = df['text']
df_train, df_left = train_test_split(df, test_size=0.3, stratify=df[config_dict['y_col']], random_state=config_dict['random_state'])
df_train['label'] = df_train.index.to_series()
df_left['label'] = df_left.index.to_series()


df_train = df_train[df_train['text'].str.strip().isin(config_dict['y_levels'])]
df_left = df_left[df_left['text'].str.strip().isin(config_dict['y_levels'] + [loo_text])]
df_test = df_left[df_left['text'].str.strip().isin(config_dict['y_levels'])]

if config_dict['open_vocab']:
    df_train, df_test, df_left = gen_open_vocab_text(df_train, df_test, df_left, config_dict)

print('\n\nfinal distribution of text prediction')
print(df_train['text'].value_counts())
print(df_left['text'].value_counts())

# ------------------------------------------------------------------------------------------------
# prepare arguments for evaluation
# ------------------------------------------------------------------------------------------------
df_eval = df_left #df_test if 'df_left' not in locals() else df_left
w = 0.8 # stength of augmentation

# Matrices
math = False
ts_dist = False
rats = True

# argument dictionary used for ts_dist and rats
args0 = {'segment1': None,
        # 'segment2': None,
        'segment3': None,
        'segment4':None
        }

args_ls = [args0]

# Define the base augmentation pairs used in math and ts_dist
base_aug_dict = {'segment1': [('flat trend', 'upward trend'), 
                              ('flat trend', 'downward trend'),
                              ('upward trend', 'flat trend'),
                              ('upward trend', 'downward trend'),
                              ('downward trend', 'flat trend'),
                              ('downward trend', 'upward trend')],
                # 'segment2': [('No seasonal pattern.', 'The time series exhibits a seasonal pattern.'),
                #             ('The time series exhibits a seasonal pattern.', 'No seasonal pattern.')],
                'segment3': [('no shift of mean', 'upward shift of mean'),
                             ('no shift of mean', 'downward shift of mean'),
                             ('upward shift of mean', 'no shift of mean'),
                             ('upward shift of mean', 'downward shift of mean'),
                             ('upward shift of mean', 'no shift of mean'),
                             ('downward shift of mean', 'no shift of mean')],
                'segment4': [("low variability", "high variability"),
                             ('low variability', 'moderate variability'),
                             ('moderate variability', 'low variability'),
                             ('moderate variability', 'high variability'),
                             ('high variability', 'moderate variability'),
                            ('high variability', "low variability")]
                }

base_aug_dict = {k: v for k, v in base_aug_dict.items() if k in config_dict['txt2ts_y_cols']}
