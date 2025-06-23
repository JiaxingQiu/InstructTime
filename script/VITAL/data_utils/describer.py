import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import STAP
import rpy2.robjects as ro
ro.r('options(warn=-1)')  # Suppress all warnings
import os
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import gc
from tqdm import tqdm

# Enable automatic conversion between numpy and R arrays
numpy2ri.activate()

def load_r_functions(path="../Data/utils/"):
    """Load all R functions from files in the path"""
    # Get all R files in the directory
    r_files = []
    for file in os.listdir(path):
        if file.endswith('.R'):
            r_files.append(os.path.join(path, file))
    
    # Read all R files
    r_code = ""
    for file in r_files:
        with open(file, 'r') as f:
            r_code += f.read() + "\n"
    
    return STAP(r_code, "r_functions")

# Load R functions
r_functions = load_r_functions()



def generate_descriptions(ts_df, id_df, type=0):
    """
    Generate various descriptions for time series data.
    
    Parameters:
    -----------
    ts_data : pandas.DataFrame
        Input dataframe where the first id_cols columns are identifiers and the rest are time series data
    id_cols : int, default=2
        Number of identifier columns at the start of the dataframe
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the original identifiers and generated descriptions
    """
    # Split data into identifiers and time series
    df_desc = id_df.copy()
    # Generate successive increment summaries
    df_desc['description_succ_inc'] = [
        describe_succ_inc_summ(row.values) for _, row in ts_df.iterrows()
    ]
    # Generate successive unchange summaries
    df_desc['description_succ_unc'] = [
        describe_succ_unc_summ(row.values) for _, row in ts_df.iterrows()
    ]
    # Generate histogram descriptions
    df_desc['description_histogram'] = [
        describe_hr_histogram(row.values) for _, row in ts_df.iterrows()
    ]
    
    # Generate brady events descriptions for different thresholds
    events80 = [describe_brady_events(row.values, th=80, plot=False, type=type) for _, row in ts_df.iterrows()]
    events90 = [describe_brady_events(row.values, th=90, plot=False, type=type) for _, row in ts_df.iterrows()]
    events100 = [describe_brady_events(row.values, th=100, plot=False, type=type) for _, row in ts_df.iterrows()]
    
    # Get most severe events - using enumerate to get the correct index
    df_desc['description_ts_event'] = [
        get_most_severe_event_py(idx, events80, events90, events100) 
        for idx in range(len(ts_df))
    ]
    
    return df_desc




def generate_descriptions_parallel(ts_df, id_df, type=2):
    """
    Generate various descriptions for time series data using parallel processing.
    """
    # Split data into identifiers and time series
    df_desc = id_df.copy()
    
    # Define helper functions for parallel processing
    def process_row(row):
        # remove nan values from row.values
        x = row.values[~np.isnan(row.values)]
        succ_inc = describe_succ_inc_summ(x)
        succ_unc = describe_succ_unc_summ(x)
        histogram = describe_hr_histogram(x)
        events80 = describe_brady_events(x, th=80, plot=False, type=type)
        events90 = describe_brady_events(x, th=90, plot=False, type=type)
        events100 = describe_brady_events(x, th=100, plot=False, type=type)
        return {
            'succ_inc': succ_inc,
            'succ_unc': succ_unc,
            'histogram': histogram,
            'events': (events80, events90, events100)
        }
    
    # determine the number of cores to use by checking the number of available cores
    try:
        total_cores = multiprocessing.cpu_count()
        n_cores = max(1, int(total_cores * 0.5)) 
    except:
        n_cores = 4

    # # Process all rows in parallel
    # results = Parallel(n_jobs=n_cores, verbose=1)(
    #     delayed(process_row)(row) for _, row in ts_df.iterrows()
    # )
    # Set up batch processing
    batch_size = 500  # Adjust based on your system's memory
    results = []
    
    # Calculate total number of batches for progress bar
    n_batches = (len(ts_df) + batch_size - 1) // batch_size
    parallel = Parallel(n_jobs=n_cores, batch_size=10, prefer='processes')  # Send 10 tasks to each worker at a time
    
    # Process in batches with progress bar
    for batch_start in tqdm(range(0, len(ts_df), batch_size), 
                          total=n_batches,
                          desc="Processing descriptions"):
        batch_end = min(batch_start + batch_size, len(ts_df))
        batch_df = ts_df.iloc[batch_start:batch_end]
        
        # Process batch in parallel
        batch_results = parallel(
            delayed(process_row)(row) for _, row in batch_df.iterrows()
        )
        results.extend(batch_results)
        
        # Force garbage collection after each batch
        gc.collect()
    
    # Extract results
    df_desc['description_succ_inc'] = [r['succ_inc'] for r in results]
    df_desc['description_succ_unc'] = [r['succ_unc'] for r in results]
    df_desc['description_histogram'] = [r['histogram'] for r in results]
    
    # Process events
    events80 = [r['events'][0] for r in results]
    events90 = [r['events'][1] for r in results]
    events100 = [r['events'][2] for r in results]
    
    # Get most severe events
    df_desc['description_ts_event'] = [
        get_most_severe_event_py(idx, events80, events90, events100) 
        for idx in range(len(ts_df))
    ]

    # Final cleanup
    ro.r('gc()')

    return df_desc

def describe_brady_events(x, th=80, direction="<", plot=False, type=1):
    """Python wrapper to call R's describe_brady_event function"""
    x = x[~np.isnan(x)]
    r_vector = robjects.FloatVector(x)
    description = r_functions.describe_brady_event(
        r_vector, th, direction, plot, type
    )
    return str(description[0]) if description else ""

def get_most_severe_event_py(row_index, events80, events90, events100):
    """
    Find the most severe event for a given row index.
    
    Args:
        row_index: Row index or row values from DataFrame
        events80 (list/Series): Event descriptions for threshold 80
        events90 (list/Series): Event descriptions for threshold 90
        events100 (list/Series): Event descriptions for threshold 100
    
    Returns:
        str: Most severe event description or empty string if no events
    """
    # If row_index is a row from DataFrame.iterrows(), get the index
    if hasattr(row_index, 'name'):
        row_index = row_index.name
    
    # Get events for this row index
    event80 = str(events80[row_index])
    event90 = str(events90[row_index])
    event100 = str(events100[row_index])
    
    # Check if all events are empty
    if all(event == "" or event == "nan" for event in [event80, event90, event100]):
        return ""
        
    # Return most severe event (priority: 80 > 90 > 100)
    if event80 != "" and event80 != "nan":
        return event80
    if event90 != "" and event90 != "nan":
        return event90
    if event100 != "" and event100 != "nan":
        return event100
    

    

def describe_hr_histogram(x):
    """
    Python wrapper to call R's describe_hr_histogram function
    
    Args:
        x (np.ndarray): Time series data
    
    Returns:
        str: Description of heart rate variability
    """
    x = x[~np.isnan(x)]
    r_vector = robjects.FloatVector(x)
    description = r_functions.describe_hr_histogram(r_vector)
    return str(description[0]) if description else ""

def describe_succ_inc(x):
    """
    Python wrapper to call R's describe_succ_inc function
    
    Args:
        x (np.ndarray): Time series data
    
    Returns:
        str: Description of consecutive increases
    """
    x = x[~np.isnan(x)]
    r_vector = robjects.FloatVector(x)
    description = r_functions.describe_succ_inc(r_vector)
    return str(description[0]) if description else ""

def describe_succ_inc_summ(x):
    """
    Python wrapper to call R's describe_succ_inc_summ function
    
    Args:
        x (np.ndarray): Time series data
    
    Returns:
        str: Summary description of consecutive increases
    """
    x = x[~np.isnan(x)]
    r_vector = robjects.FloatVector(x)
    description = r_functions.describe_succ_inc_summ(r_vector)
    return str(description[0]) if description else ""


def describe_succ_unc_summ(x):
    """
    Python wrapper to call R's describe_succ_unc_summ function
    
    Args:
        x (np.ndarray): Time series data
    
    Returns:
        str: Summary description of consecutive unchanges
    """
    x = x[~np.isnan(x)]
    r_vector = robjects.FloatVector(x)
    description = r_functions.describe_succ_unc_summ(r_vector)
    return str(description[0]) if description else ""

