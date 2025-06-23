# Set R environment variables using the conda environment path if needed
import os
# r_home = '/sfs/gpfs/tardis/home/jq2uw/llm_nicu_vitalsigns/clip_env/lib/R'
r_home = '/sfs/weka/scratch/jq2uw/instruct_time/instime/lib/R'

if os.path.exists(r_home):
    os.environ['R_HOME'] = r_home
    os.environ['R_LIBS'] = f"{r_home}/library"
    os.environ['R_LIBS_USER'] = os.path.expanduser('~/R/goolf/4.3')
    os.environ['LD_LIBRARY_PATH'] = f"{r_home}/lib:" + os.environ.get('LD_LIBRARY_PATH', '')

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
