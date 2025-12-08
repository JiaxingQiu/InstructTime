# Clear cache
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# Suppress warnings
import warnings
warnings.filterwarnings('ignore') # all warnings

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import io
import gzip

# Import all the necessary modules
from config import *
from data import *
from train import *
from eval import *
from generation import *
from vital2d import *
print("using device: ", device)
