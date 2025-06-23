# ------------------------------------------------------------------------------------------------
# prepare dataset and arguments for training
# ------------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

df = pd.read_csv('../../data/air_quality/air_quality.csv.zip', compression='zip')
df.columns = df.columns.astype(str)
df['text'] = ''
for str_col in config_dict['txt2ts_y_cols']:
    df['text'] += ' ' + df[str_col]
df['text'] = df['text'].str.strip()
df_train, df_temp = train_test_split(df, test_size=0.3, stratify=df[config_dict['y_col']], random_state=config_dict['random_state'])
df_test, df_left = train_test_split(df_temp, test_size=1/3,stratify=df_temp[config_dict['y_col']],random_state=config_dict['random_state'])
        
if config_dict['downsample']:
    df_train = downsample_neg_levels(df_train, config_dict, config_dict['random_state'])
    df_test = downsample_neg_levels(df_test, config_dict, config_dict['random_state'])
    df_left = downsample_neg_levels(df_left, config_dict, config_dict['random_state'])
    

df_train['label'] = df_train.index.to_series()
df_test['label'] = df_test.index.to_series()
df_left['label'] = df_left.index.to_series()


if config_dict['open_vocab']:
    df_train, df_test, df_left = gen_open_vocab_text(df_train, df_test, df_left, config_dict)

print('\n\nfinal distribution of text prediction')
print(df_train['text'].value_counts())
print(df_test['text'].value_counts())
print(df_left['text'].value_counts())




# ------------------------------------------------------------------------------------------------
# prepare arguments for evaluation
# ------------------------------------------------------------------------------------------------
df_eval = df_left
w = 0.8 # stength of augmentation

math = False
ts_dist = True
rats = True

# argument dictionary {y_col:conditions}
args0 = {'city_str': None,
        'season_str': None
        }

args1 = {'city_str': [('year_str', 'It is measured in 2017.')],
        'season_str': [('city_str', 'This is air quality in Beijing.'), ('year_str', 'It is measured in 2017.')]
        }

args2 = {'city_str': [('year_str', 'It is measured in 2017.')],
        'season_str': [('city_str', 'This is air quality in London.'), ('year_str', 'It is measured in 2017.')]
        }
args_ls = [args0, args1, args2]

# Define the base augmentation pairs used in math and ts_dist
base_aug_dict = {'city_str': [('This is air quality in Beijing.', 'This is air quality in London.'), 
                            ('This is air quality in London.', 'This is air quality in Beijing.')],
                'season_str': [('The season is winter.', 'The season is summer.'),
                               ('The season is winter.', 'The season is spring.'),
                               ('The season is winter.', 'The season is fall.'),
                               ('The season is spring.', 'The season is summer.'),
                               ('The season is spring.', 'The season is fall.'),
                               ('The season is summer.', 'The season is winter.'),
                               ('The season is summer.', 'The season is spring.'),
                               ('The season is summer.', 'The season is fall.'),
                               ('The season is summer.', 'The season is winter.'),
                               ('The season is fall.', 'The season is winter.'),
                               ('The season is fall.', 'The season is spring.'),
                               ('The season is fall.', 'The season is summer.')]
                }

