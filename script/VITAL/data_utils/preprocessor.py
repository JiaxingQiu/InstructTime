import pandas as pd
import numpy as np
import torch.nn as nn
import re
from config import *

# ------- pretrained ts and txt encoders (for feature generation) -------
class TXTEncoder():
    def __init__(self, model_name):
       
        # model_name = 'all-mpnet-base-v2', 'all-MiniLM-L6-v2', etc
        self.model_name = model_name
        self.device = device
        

    def encode_text_list(self, text_list):
        # sentence transformer
        if self.model_name in ['sentence-transformers/all-mpnet-base-v2',  # Optimized for sentence embeddings
                          'sentence-transformers/paraphrase-mpnet-base-v2',  # Optimized for paraphrase detection
                          'sentence-transformers/all-MiniLM-L6-v2',  # Optimized for sentence embeddings
                          'sentence-transformers/all-MiniLM-L12-v2']:
            
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.model_name, trust_remote_code=True).to(device)
            output = model.encode(text_list) # tensor shape: [obs, embed_dim]
            return output
        
        # BERT-base
        elif self.model_name in ['bert-base-uncased',
                            'bert-base-cased',
                            'bert-large-uncased',
                            'bert-large-cased',
                            'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
                            'allenai/scibert_scivocab_uncased',
                            'dmis-lab/biobert-base-cased-v1.2']:

            from transformers import BertTokenizer, BertModel
            tokenizer = BertTokenizer.from_pretrained(self.model_name)
            model = BertModel.from_pretrained(self.model_name)
            model = model.to(device)
            encoded_input = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True, max_length=512)
            # Move all input tensors to the same device as the model
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            output = model(**encoded_input)
            return output.pooler_output #output.last_hidden_state[:, 0, :]
            
        # RoBERTa
        elif self.model_name in ['roberta-base',             # 12-layer
                            'roberta-large',            # 24-layer
                            'allenai/biomed_roberta_base',  # Biomedical domain
                            'roberta-base-openai-detector', # OpenAI's version
                            'microsoft/roberta-base-openai-detector']:

            from transformers import RobertaTokenizer, RobertaModel
            tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
            model = RobertaModel.from_pretrained(self.model_name)
            model = model.to(device)
            encoded_input = tokenizer(text_list, return_tensors='pt', padding=True)
            # Move all input tensors to the same device as the model
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            output = model(**encoded_input)
            return output.pooler_output #output.last_hidden_state[:, 0, :]
            
        
        # DistilBERT
        elif self.model_name in ['distilbert-base-uncased',  # General purpose, uncased
                            'distilbert-base-cased',    # General purpose, cased
                            'distilroberta-base',       # Distilled version of RoBERTa
                            'distilbert-base-multilingual-cased']:
            from transformers import DistilBertTokenizer, DistilBertModel
            tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
            model = DistilBertModel.from_pretrained(self.model_name)
            model = model.to(device)
            encoded_input = tokenizer(text_list, return_tensors='pt', padding=True)
            # Move all input tensors to the same device as the model
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            output = model(**encoded_input)
            return output.last_hidden_state.mean(dim=1) # output.last_hidden_state[:, 0, :]
            
        # MPNet
        elif self.model_name in ['microsoft/mpnet-base',     # Base model
                            'microsoft/mpnet-large']:
            from transformers import AutoTokenizer, MPNetModel
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = MPNetModel.from_pretrained(self.model_name)
            model = model.to(device) 
            encoded_input = tokenizer(text_list, return_tensors='pt', padding=True)
            # Move all input tensors to the same device as the model
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            output = model(**encoded_input)
            return output.pooler_output
        

class TSEncoder():
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = device
        
        # load pretrained TS encoder
        if model_name == 'hr_vae_linear_medium':
            self.model = VAE_Linear_Medium().to(device)
            self.model.load_state_dict(torch.load('./pretrained/'+model_name+'.pth' ))
        if model_name == 'sp_vae_linear_medium':
            self.model = VAE_Linear_Medium().to(device)
            self.model.load_state_dict(torch.load('./pretrained/'+model_name+'.pth' ))
        

class VAE_Linear_Medium(nn.Module):
    def __init__(self, seq_len=300):
        # super(VAE_Linear_Medium, self).__init__()
        super().__init__() 

        self.seq_len = seq_len

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2)
        )
        
        # Latent mean and variance 
        self.mean_layer = nn.Linear(128, 32)
        self.logvar_layer = nn.Linear(128, 32)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, seq_len)
        )
     
    def encode(self, x):
        # Linear encoding
        x = self.encoder(x)
        
        # Get latent parameters
        mean, log_var = self.mean_layer(x), self.logvar_layer(x)
        return mean, log_var
    
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)      
        z = mean + var*epsilon
        return z

    def decode(self, z):
        # Linear decoding
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, log_var)
        x_hat = self.decode(z)
        return x_hat, mean, log_var


# --------------------------general data engineering----------------------------------
def label_death7d(df, df_y, id_col='VitalID'):
    
    # df_y must have columns ID, DiedNICU, DeathAge
    if id_col not in df_y.columns or 'DiedNICU' not in df_y.columns or 'DeathAge' not in df_y.columns:
        raise ValueError("df_y must have columns ID, DiedNICU, and DeathAge")
    # df must have columns ID and Age
    if id_col not in df.columns or 'Age' not in df.columns:
        raise ValueError("df must have columns ID and Age")
    
    # Create a copy of df_day to avoid modifying original
    df_labeled = df.copy()

    # Initialize the death label column with 0
    df_labeled['Died'] = 0

    # Filter patients who died in NICU
    died_patients = df_y[df_y['DiedNICU'] == 1]

    # For each patient who died
    for _, patient in died_patients.iterrows():
        id = patient[id_col]
        death_age = patient['DeathAge']
        
        # Find records for this patient within 7 days of death
        mask = (
            (df_labeled[id_col] == id) & 
            (df_labeled['Age'] >= death_age - 7) & 
            (df_labeled['Age'] <= death_age)
        )
        
        # Label these records as 1
        df_labeled.loc[mask, 'Died'] = 1
    # Check distribution by TestID
    print("\nSample of patients with positive labels:")
    print(df_labeled[df_labeled['Died'] == 1].groupby(id_col).size().sort_values(ascending=False).head(5))

    return df_labeled

# --------------------------text input processing----------------------------------
def text_extract_event1(text):
    """
    Extract the first event description from the text.
    """
    # Find start of Event 1
    start_marker = " Event 1: "
    end_marker = " Event 2: "
    
    # initialize event1_str
    event1_str = ""

    # Get start position
    start_pos = text.find(start_marker)
    if start_pos == -1:  # If Event 1 not found
        return event1_str
    start_pos += len(start_marker)
    # Get end position
    end_pos = text.find(end_marker)
    if end_pos == -1:  # If Event 2 not found, take until end
        return text[start_pos:].strip()
    
    # Return text between markers
    event1_str = text[start_pos:end_pos].strip()
    event1_str = re.sub(r'\s+', ' ', event1_str).strip() # remove extra spaces

    return event1_str

def text_summarize_brady(text):
    """
    Count events and identify the single event type in the text.
    
    Args:
        text (str): Input text containing event descriptions
        
    Returns:
        str: Summary like "2 Bradycardia90 (heart rate below 90) events happened."
              or "1 Bradycardia90 (heart rate below 90) event happened."
    """
    if not isinstance(text, str):
        return ""#"No Bradycardia events."
    
    # Count events
    event_count = text.count("\n Event")
    if event_count == 0:
        return ""#"No Bradycardia events."
    
    # Find the event name and threshold
    import re
    brady_match = re.search(r'Bradycardia (\d+)', text)

    if brady_match:
        event_name = brady_match.group(0)  # e.g., "Bradycardia90"
        threshold = brady_match.group(1)    # e.g., "90"
        event_word = "event" if event_count == 1 else "events"
        return f"{event_name} event (heart rate below {threshold}) happened." #"{event_count} {event_name} {event_word} (heart rate below {threshold}) happened."
    
def text_summarize_desat(text):
    """
    Count events and identify the single event type in the text.
    
    Args:
        text (str): Input text containing event descriptions
        
    Returns:
        str: Summary like "2 Desaturation90 (spo2 below 90) events happened." or "1 Desaturation90 (spo2 below 90) event happened."
    """
    if not isinstance(text, str):
        return ""#"No Desaturation events."
    
    # Count events
    event_count = text.count("\n Event")
    if event_count == 0:
        return ""#"No Desaturation events."
    
    # Find the event name and threshold
    import re
    desat_match = re.search(r'Desaturation(\d+)', text)

    if desat_match:
        event_name = desat_match.group(0)  # e.g., "Desaturation90"
        threshold = desat_match.group(1)    # e.g., "90"
        event_word = "event" if event_count == 1 else "events"
        return f"{event_count} {event_name} {event_word} (spo2 below {threshold}) happened."
    else:
        event_word = "event" if event_count == 1 else "events"
        return f"{event_count} Desaturation {event_word} happened."

def text_gen_demo(row,
             ga_bwt=True,
             gre=False, # gender_race_ethnicity
             apgar_mage=False,
             split=False):
    """
    Generate a demographic description string from a single row.
    
    Args:
        row (pd.Series): A row containing demographic information:
            - EGA: Gestational age in weeks
            - BWT: Birth weight in grams
            - Male: 0 for female, 1 for male
            - Black: 0 for white, 1 for black
            - Hispanic: 0 for non-Hispanic, 1 for Hispanic
            - Apgar5: APGAR score at 5 minutes
            - Maternal Age: Mother's age in years
            
    Returns:
        str: Formatted demographic description
    """

    def is_valid_number(value):
        """Check if value is a valid number (not NaN, None, or invalid)"""
        if value is None:
            return False
        if pd.isna(value):  # Catches pandas NA, NaN
            return False
        if np.isnan(value) if isinstance(value, float) else False:  # Catches numpy NaN
            return False
        try:
            float(value)  # Try converting to float
            return True
        except (ValueError, TypeError):
            return False
        
    ga_str = ""
    weight_str = ""
    gender_str = ""
    race_str = ""
    ethnicity_str = ""
    apgar_str = ""
    mother_str = ""

    # Format each component
    if ga_bwt:
        if is_valid_number(row['EGA']): 
            ga_str = f"This infant has gestational age {int(round(row['EGA'], 1))} weeks."
        if is_valid_number(row['BWT']):
            weight_str = f"Birth weight is {int(round(row['BWT'], 1))} grams."
    
    if gre:
        gender_str = "This infant is Male " if row['Male'] == 1 else "This infant is Female "
        race_str = "Black " if row['Black'] == 1 else "non-Black "
        ethnicity_str = "Hispanic." if row['Hispanic'] == 1 else "non-Hispanic."
    if apgar_mage:
        if is_valid_number(row['Apgar5']):
            apgar_str = f"The Apgar5 scores {int(round(row['Apgar5'], 1))}. "
        if is_valid_number(row['Maternal Age']):
            mother_str = f"Mother is {int(round(row['Maternal Age'], 1))} years old."
    
    
    if not split:
        if not (ga_str or weight_str or gender_str or race_str or ethnicity_str or apgar_str or mother_str):
            return ""
        else:
            return_str = f"{ga_str} {weight_str} {gender_str} {race_str} {ethnicity_str} {apgar_str} {mother_str}"
            return_str = return_str.strip()
            return_str = re.sub(r'\s+', ' ', return_str).strip() # remove extra spaces
            return return_str
    else:
        return [ga_str, weight_str, gender_str, race_str, ethnicity_str, apgar_str, mother_str]

def text_gen_cl_event(row,
                 die7d=True,
                 fio2=False,
                 split=False):
    """
    Generate a clinical event description string from a single row.
    
    Args:
        - row (pd.Series): A row containing clinical information.
        - die7d (bool): if ture, return death in 7 days.
        - fio2 (bool): if ture, return the FiO2 event.
    """
    die7d_str = ""
    fio2_str = ""

    if die7d:
        die7d_str = 'This infant will die in 7 days.' if row['Died'] == 1 else 'This infant will survive.'
    
    if fio2:
        fio2_str = f"The FIO2 is {int(round(row['FIO2']))}."
    
    return_str = f"{die7d_str} {fio2_str}"
    return_str = return_str.strip()
    return_str = re.sub(r'\s+', ' ', return_str).strip() # remove extra spaces
    if not split:
        return return_str
    else:
        return [die7d_str, fio2_str]

def text_gen_ts_event(row, 
                      sumb=True, # sum_brady
                      sumd=True, # sum_desat
                      simple=True,
                      full=False,
                      event1=False):
    """
    Generate a time series event description string from a single row.

    Args:
        - row (pd.Series): A row containing time series information.
        - sumb (bool): if ture, return summarize of count and definition of Bradycardia event.
        - sumd (bool): if ture, return summarize of count and definition of Desaturation event.
        - simple (bool): if ture, return simplified time series event description.
        - full (bool): if ture, return full event description.
        - event1 (bool): if ture, return the first event description.
    """


    sum_str = ""
    simple_str = ""
    full_str = ""
    event1_str = ""
    if sumb:
        sum_str = text_summarize_brady(row['description_ts_event'])
        # sum_str = sum_str + " "
    if sumd:
        sum_str = text_summarize_desat(row['description_ts_event'])
        # sum_str = sum_str + " "
    if simple:
        x = row['description_ts_event']
        simple_str = '\n'.join(x.split('\n')[1:]) if isinstance(x, str) else x
        # simple_str = simple_str + " "
    if full:
        full_str = row['description_ts_event']
        # full_str = full_str + " "
    if event1:
        event1_str = text_extract_event1(row['description_ts_event'])
        # event1_str = event1_str + " "
    text_str = f"{sum_str} {simple_str} {event1_str} {full_str}"
    text_str = text_str.strip()
    text_str = re.sub(r'\s+', ' ', text_str).strip() # remove extra spaces
    if text_str == "":
        return "No events."
    return text_str


def text_gen_ts_description(row,
                            succ_inc=True,
                            succ_unc=True,
                            histogram=True):
    
    ts_event_str = ""
    histogram_str = ""
    succ_inc_str = ""
    succ_unc_str = ""

    ts_event_str = row['description_ts_event'] # already engineered by text_gen_ts_event
    if histogram:
        histogram_str = row['description_histogram']
        # histogram_str = histogram_str + " "
    if succ_inc:
        succ_inc_str = row['description_succ_inc']
        # succ_inc_str = succ_inc_str + " "
    if succ_unc:
        succ_unc_str = row['description_succ_unc']
        # succ_unc_str = succ_unc_str + " "
    
    text_str = f"{histogram_str} {succ_inc_str} {succ_unc_str} {ts_event_str}"
    text_str = text_str.strip()
    text_str = re.sub(r'\s+', ' ', text_str).strip() # remove extra spaces
    return text_str

def text_gen_input_column(df, config_dict):

    text_config = config_dict['text_config']
    df.columns = df.columns.astype(str)
    # demographic description
    df['demo'] = df.apply(text_gen_demo, axis=1, **text_config['demo'])
    # clinical events
    df['cl_event'] = df.apply(text_gen_cl_event, axis=1, **text_config['cl'])
    # time series events
    df['description_ts_event'] = df.apply(text_gen_ts_event, axis=1, 
                                         sumb=text_config['ts']['ts_event']['sumb'],
                                         sumd=text_config['ts']['ts_event']['sumd'],
                                         simple=text_config['ts']['ts_event']['simple'],
                                         full=text_config['ts']['ts_event']['full'],
                                         event1=text_config['ts']['ts_event']['event1'])
    df['description_ts_event_binary'] = df['description_ts_event'].apply(lambda x: x if x == 'No events.' else 'Bradycardia events happened.')
    # time series description
    df['ts_description'] = df.apply(text_gen_ts_description, axis=1, 
                                    succ_inc=text_config['ts']['succ_inc'],
                                    succ_unc=text_config['ts']['succ_unc'],
                                    histogram=text_config['ts']['histogram'])
    
    # # create reserved text column for 2d clip
    # df['text'] = df['cl_event'] + ' ' + df['demo'] + ' ' + df['ts_description'] # default text for NICU data.
    # df['text'] = df['text'].apply(lambda x: x.strip())
    # # if not config_dict['3d']: # suppress
    # #     print("replace 'text' with: ", config_dict['text_col'])
    # #     df['text'] = df[config_dict['text_col']]
    # print(df['text'].value_counts().nlargest(5)) # head 5 only
    
    if text_config['split']:
        # Generate demographic descriptions
        demo_parts = df.apply(text_gen_demo, axis=1, split=True, **text_config['demo'])
        if text_config['demo']['ga_bwt']:
            df['demo_ga'] = [parts[0] for parts in demo_parts]
            df['demo_weight'] = [parts[1] for parts in demo_parts]
        if text_config['demo']['gre']:
            df['demo_gender'] = [parts[2] for parts in demo_parts]
            df['demo_race'] = [parts[3] for parts in demo_parts]
            df['demo_ethnicity'] = [parts[4] for parts in demo_parts]
        if text_config['demo']['apgar_mage']:
            df['demo_apgar'] = [parts[5] for parts in demo_parts]
            df['demo_mother'] = [parts[6] for parts in demo_parts]

        # Generate clinical event descriptions
        cl_parts = df.apply(text_gen_cl_event, axis=1, split=True, **text_config['cl'])
        if text_config['cl']['die7d']:
            df['cl_die7d'] = [parts[0] for parts in cl_parts]
        if text_config['cl']['fio2']:
            df['cl_fio2'] = [parts[1] for parts in cl_parts]

        print("\nAvailable text columns:")
        text_cols = [col for col in df.columns if col.startswith(('demo_', 'cl_', 'ts_'))]
        print(text_cols)
    
    
    return df

def plot_ts(df, idx, len=300):
    """
    Plot a single time series from the DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing time series data
    idx : int
        Index of the time series to plot
    """
    import matplotlib.pyplot as plt
    
    # Get time series data
    ts_cols = [str(i) for i in range(1, len+1)]
    ts = df.loc[idx, ts_cols].values
    
    # Create plot
    plt.figure(figsize=(10, 5))
    plt.plot(ts, 'b-', linewidth=2)
    plt.title(f'ID: {idx}, caption: {df.loc[idx, "text"]}')
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Heart Rate')
    # plt.ylim(50, 200)
    plt.grid(True)
    plt.show()
    
## Example usage:
# ts = plot_ts(df_train, 2)