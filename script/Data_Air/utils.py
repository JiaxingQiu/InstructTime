#%pip install sktime==0.34.0   # brings the TSF loader
import pandas as pd
import numpy as np
from sktime.datasets import load_tsf_to_dataframe
import matplotlib.pyplot as plt


def expand_row(row):
    """Return a DataFrame with a DateTimeIndex and one column 'value'."""
    start = pd.to_datetime(row["start_timestamp"])
    ts    = pd.date_range(start=start,
                          periods=len(row["series_value"]),
                          freq="H",           # 1‑hour frequency
                          name="time")
    return pd.DataFrame(
        {
            "series_id":  row["series_name"],          # T1 … T270
            "city":       row["city"],
            "station":    row["station"],
            "pollutant":  row["air_quality_measurement"],
            "value":      row["series_value"],         # the numeric readings
        },
        index=ts
    )


# Process the DataFrame into weekly chunks
def process_weekly_chunks(df):
    # Sort by time to ensure chronological order
    df = df.sort_values(['city', 'station', 'time'])
    
    # Create helper columns for week identification
    df['hour_seq'] = df.groupby(['city', 'station']).cumcount()
    df['week_id'] = df['hour_seq'] // 168
    
    # Calculate weekstamp (start of each week)
    # First, get the first timestamp for each city
    first_timestamps = df.groupby(['city', 'station'])['time'].transform('first')
    # Then calculate the weekstamp by rounding down to the start of the week
    df['weekstamp'] = first_timestamps + pd.to_timedelta(df['week_id'] * 168, unit='h')
    
    # Keep only full weeks (exactly 168 hours)
    full_weeks = df.groupby(['city', 'station', 'week_id']).filter(lambda g: len(g) == 168)
    
    # Create hour column (1-168)
    full_weeks['hour_in_week'] = (full_weeks['hour_seq'] % 168) + 1
    
    # Pivot the data to get weekly chunks
    result = (full_weeks
             .pivot_table(index=['city', 'station', 'season', 'weekstamp'],
                         columns='hour_in_week',
                         values='value',
                         aggfunc='first')
             .reset_index())
    
    # Rename numeric columns to strings to preserve order
    result.columns = [str(c) if isinstance(c, (int, np.integer)) else c 
                     for c in result.columns]
    
    return result


def plot_ts(df, idx, len=168):
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
    plt.figure(figsize=(5, 3))
    plt.plot(ts, 'b-', linewidth=2)
    plt.title(f'ID: {idx}, caption: {df.loc[idx, "text"]}')
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Heart Rate')
    # plt.ylim(50, 200)
    plt.grid(True)
    plt.show()
    

# ------------------------------------------------------------
# helper: does this 168‑hour vector contain > 48 identical
#         readings *in a row*?   (NaN breaks the streak)
# ------------------------------------------------------------
def has_run_longer_than(arr, limit=12):
    longest = cur = 1
    for i in range(1, len(arr)):
        a, b = arr[i - 1], arr[i]

        if pd.isna(a) or pd.isna(b):           # NaN → break the streak
            cur = 1
        elif a == b:                           # identical reading
            cur += 1
            if cur > limit:
                return True                    # early exit
        else:                                  # value changed
            cur = 1
    return False

def process_weekly_df(pm25_df):

    # Apply the processing
    weekly_df = process_weekly_chunks(pm25_df)
    weekly_df = weekly_df.dropna()
    weekly_df.columns = weekly_df.columns.astype(str)
    weekly_df = weekly_df.reset_index(drop=True)

    weekly_df['weekstamp'] = pd.to_datetime(weekly_df['weekstamp'])
    weekly_df['year'] = weekly_df['weekstamp'].dt.year



    # ----------------------add text--------------------------------------
    weekly_df['city_str'] = "This is air quality in " + weekly_df['city'] + "."
    weekly_df['station_str'] = "It is measured by weather station " + weekly_df['station'] + "."
    weekly_df['year_str'] = "It is measured in " + weekly_df['year'].astype(str) + "."
    weekly_df['season_str'] = "The season is " + weekly_df['season'].str.lower() + "."
    weekly_df['text'] = weekly_df['city_str'] + " " + weekly_df['year_str'] + " " + weekly_df['season_str']
    #  + " " + weekly_df['station_str']
    print(weekly_df.text.value_counts())

    # ----------------------removers--------------------------------------
    # keep rows with ≥ 5 distinct values
    hour_cols = [str(i) for i in range(1, 169)]
    unique_counts = weekly_df[hour_cols].nunique(axis=1)
    weekly_df = weekly_df.loc[unique_counts >= 5].copy()
    # remove rows that have more than 12 continuous unchanged values
    hour_cols = [str(i) for i in range(1, 169)]
    mask_long_run = weekly_df[hour_cols].apply(
        lambda row: has_run_longer_than(row.values), axis=1
    )

    weekly_df = weekly_df.loc[~mask_long_run].copy()
    return weekly_df