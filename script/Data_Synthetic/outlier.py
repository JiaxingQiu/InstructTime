import numpy as np
from typing import Tuple, List, Optional
import random

def generate_no_outlier(
    length: int,
    base_value: float = 0.0,
    noise_std: float = 0.05
) -> np.ndarray:
    """
    Generate a time series with no outliers (control case).
    
    Args:
        length: Length of the time series
        base_value: Base value around which the series fluctuates
        noise_std: Standard deviation of Gaussian noise
        
    Returns:
        np.ndarray: Generated time series
    """
    return np.random.normal(base_value, noise_std, length)

def generate_spikes(
    length: int,
    base_value: float = 0.0,
    noise_std: float = 0.05,
    spike_magnitude_range: Tuple[float, float] = (2.0, 10.0),
    n_spikes_range: Tuple[int, int] = (1, 5),  # default to single spike
    min_spike_gap: int = 2  # minimum gap between spikes if multiple
) -> np.ndarray:
    """
    Generate a time series with one or more spikes.
    
    Args:
        length: Length of the time series
        base_value: Base value around which the series fluctuates
        noise_std: Standard deviation of Gaussian noise
        spike_magnitude_range: Range for the spike magnitude
        n_spikes_range: Range for number of spikes (min, max)
        min_spike_gap: Minimum number of time units between spikes
        
    Returns:
        np.ndarray: Generated time series
    """
    series = np.random.normal(base_value, noise_std, length)
    n_spikes = random.randint(*n_spikes_range)
    
    if n_spikes == 1:
        # Single spike
        spike_location = random.randint(0, length-1)
        spike_magnitude = random.uniform(*spike_magnitude_range)
        series[spike_location] += spike_magnitude
    else:
        # Multiple spikes with minimum gap
        # Calculate maximum possible start position for last spike
        max_start = length - (n_spikes * min_spike_gap)
        if max_start <= 0:
            # If not enough space, reduce number of spikes
            n_spikes = max(1, length // (min_spike_gap + 1))
            max_start = length - (n_spikes * min_spike_gap)
        
        # Generate spike locations with minimum gap
        spike_locations = []
        current_pos = 0
        
        for i in range(n_spikes):
            # Calculate available space for this spike
            remaining_spikes = n_spikes - i - 1
            max_pos = length - (remaining_spikes * min_spike_gap) - 1
            
            if i == 0:
                # First spike can be anywhere in the available space
                loc = random.randint(0, max_pos)
            else:
                # Subsequent spikes must maintain minimum gap
                loc = random.randint(current_pos + min_spike_gap, max_pos)
            
            spike_locations.append(loc)
            current_pos = loc
        
        # Add spikes at the calculated locations
        for loc in spike_locations:
            spike_magnitude = random.uniform(*spike_magnitude_range)
            series[loc] += spike_magnitude
    
    return series

def generate_step_spike(
    length: int,
    base_value: float = 0.0,
    noise_std: float = 0.05,
    sign: int = 1,
    step_magnitude_range: Tuple[float, float] = (15.0, 20.0),
    step_duration_range: Tuple[int, int] = (50, 100)
) -> np.ndarray:
    """
    Generate a time series with a step spike (persistent change).
    
    Args:
        length: Length of the time series
        base_value: Base value around which the series fluctuates
        noise_std: Standard deviation of Gaussian noise
        step_magnitude_range: Range for the step magnitude
        step_duration_range: Range for the step duration
        
    Returns:
        np.ndarray: Generated time series
    """
    series = np.random.normal(base_value, noise_std, length)
    step_duration = random.randint(*step_duration_range)
    step_start = random.randint(int(length*0.1), int(length*0.9)-step_duration)
    step_magnitude = random.uniform(*step_magnitude_range)
    
    # Randomly choose sign of the shift
    # sign = random.choice([-1, 1])
    step_magnitude *= sign
    
    # Ensure step doesn't exceed series length
    step_end = min(step_start + step_duration, length)
    series[step_start:step_end] += step_magnitude
    return series

def generate_level_shift(
    length: int,
    base_value: float = 0.0,
    noise_std: float = 0.05,
    sign: int = 1,
    shift_magnitude_range: Tuple[float, float] = (15.0, 20.0)
) -> np.ndarray:
    """
    Generate a time series with a level shift (persistent change).
    
    Args:
        length: Length of the time series
        base_value: Base value around which the series fluctuates
        noise_std: Standard deviation of Gaussian noise
        shift_magnitude_range: Range for the shift magnitude
        
    Returns:
        np.ndarray: Generated time series
    """
    series = np.random.normal(base_value, noise_std, length)
    shift_location = random.randint(int(length*0.1), int(length*0.9))
    shift_magnitude = random.uniform(*shift_magnitude_range)
    
    # Randomly choose sign of the shift
    # sign = random.choice([-1, 1])
    shift_magnitude *= sign
    
    series[shift_location:] += shift_magnitude
    return series

def generate_outlier_series(
    N: int,
    L: int,
    outlier_type: str = 'no'
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Generate N samples of length L with specified outlier pattern.
    
    Args: 
        N: Number of series to generate
        L: Length of each series
        outlier_type: Type of outlier ('no', 'spikes', 'step_spike', 'level_shift')
        
    Returns:
        Tuple[List[np.ndarray], List[str]]: Generated series and their descriptions
    """
    series_list = []
    description_list = []
    
    for _ in range(N):
        if outlier_type == 'no':
            series = generate_no_outlier(L)
            description = "No sharp shifts."
        elif outlier_type == 'spikes':
            series = generate_spikes(L)
            description = "Spike outliers."
        elif outlier_type == 'step_spike_up':
            series = generate_step_spike(L)
            description = "There is an upward step spike."
        elif outlier_type == 'step_spike_down':
            series = generate_step_spike(L, sign=-1)
            description = "There is a downward step spike."
        elif outlier_type == 'level_shift_up':
            series = generate_level_shift(L)
            description = "The mean of the time series shifts upwards."
        elif outlier_type == 'level_shift_down':
            series = generate_level_shift(L, sign=-1)
            description = "The mean of the time series shifts downwards."
        else:
            raise ValueError(f"Unknown outlier type: {outlier_type}")
        
        # center the series at zero 
        series = series - np.mean(series)
        series_list.append(series)
        description_list.append(description)
    
    return series_list, description_list
