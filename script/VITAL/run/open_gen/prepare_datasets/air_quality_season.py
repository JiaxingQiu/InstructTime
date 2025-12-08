# read 
df_train = pd.read_csv('../../data/air_quality/train.csv.zip', compression='zip')
df_test = pd.read_csv('../../data/air_quality/test.csv.zip', compression='zip')
df_left = pd.read_csv('../../data/air_quality/left.csv.zip', compression='zip')

if config_dict['open_vocab']:
    df_train, df_test, df_left = gen_open_vocab_text(df_train, df_test, df_left, config_dict)

df_train['text'] = ''
df_test['text'] = ''
df_left['text'] = ''
for str_col in config_dict['txt2ts_y_cols']:
    df_train['text'] += ' ' + df_train[str_col]
    df_test['text'] += ' ' + df_test[str_col]
    df_left['text'] += ' ' + df_left[str_col]



# left one out textual condition
df_train = df_train[df_train['text'].str.strip() != loo_text.strip()]
df_left = pd.concat([df_test, df_left], axis=0, ignore_index=True)

df_train = df_train[df_train['city'] == "Beijing"]
df_left = df_left[df_left['city'] == "Beijing"]

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
ts_dist = False
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
base_aug_dict = {'season_str': [('The season is winter.', 'The season is summer.'),
                               ('The season is winter.', 'The season is spring.'),
                               ('The season is winter.', 'The season is fall.'),
                               ('The season is spring.', 'The season is summer.'),
                               ('The season is spring.', 'The season is fall.'),
                               ('The season is spring.', 'The season is winter.'),
                               ('The season is summer.', 'The season is spring.'),
                               ('The season is summer.', 'The season is fall.'),
                               ('The season is summer.', 'The season is winter.'),
                               ('The season is fall.', 'The season is winter.'),
                               ('The season is fall.', 'The season is spring.'),
                               ('The season is fall.', 'The season is summer.')]
                }

