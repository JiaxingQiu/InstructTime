# generation function for time series with varying trends 

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import random

def generate_linear_trend(
    length: int,
    slope_range: Tuple[float, float] = (0.05, 0.2),#(0.1, 1.0),
    direction: Optional[str] = None
) -> np.ndarray:
    """
    Generate a time series with a linear trend.
    
    Args:
        length: Length of the time series
        slope_range: Range for the slope parameter (a)
        intercept_range: Range for the intercept parameter (b)
        direction: 'up' or 'down' to specify trend direction. If None, randomly chosen.
    
    Returns:
        np.ndarray: Generated time series
    """
    # Randomly choose direction if not specified
    if direction is None:
        direction = random.choice(['up', 'down'])
    
    # Generate parameters
    a = random.uniform(*slope_range)
    
    # Adjust slope based on direction
    if direction == 'down':
        a = -a
    
    # Generate time points
    t = np.arange(length)
    
    # Generate series
    series = a * t
    
    return series

def generate_quadratic_trend(
    length: int,
    a_range: Tuple[float, float] = (0.0001, 0.0005),
    b_range: Tuple[float, float] = (0, 50),
    direction: Optional[str] = None,
    ensure_non_negative: bool = True
) -> np.ndarray:
    """
    Generate a time series with a quadratic trend.
    
    Args:
        length: Length of the time series
        a_range: Range for the quadratic coefficient (a)
        b_range: Range for the linear coefficient (b)
        c_range: Range for the constant term (c)
        direction: 'up' or 'down' to specify trend direction. If None, randomly chosen.
        ensure_non_negative: If True, adjusts c to ensure all values are non-negative
    
    Returns:
        np.ndarray: Generated time series
    """
    # Randomly choose direction if not specified
    if direction is None:
        direction = random.choice(['up', 'down'])
    
    # Generate parameters
    a = random.uniform(*a_range)
    b = random.uniform(*b_range)
    
    # Adjust quadratic coefficient based on direction
    if direction == 'down':
        a = -a
    
    # Generate time points
    t = np.arange(length)
    
    # Generate series
    series = a * (t+b)**2
    
    # Ensure non-negative values if requested
    if ensure_non_negative and np.min(series) < 0:
        series = series - np.min(series)
    
    return series

def generate_flat_trend(
    length: int,
    value_range: Tuple[float, float] = (-5, 5)
) -> np.ndarray:
    """
    Generate a time series with a flat trend (constant value).
    
    Args:
        length: Length of the time series
        value_range: Range for the constant value
        
    Returns:
        np.ndarray: Generated time series
    """
    # Generate constant value
    value = random.uniform(*value_range)
    
    # Generate series
    series = np.ones(length) * value
    
    return series

# a function to generate N samples of length L with a given trend
def generate_trend_series(N, L,
                           trend_type = ['linear', 'quadratic', 'flat'][0], 
                           direction=['up', 'down'][0],
                           mean_range = (-5, 5)#, std_range = (5, 10)
                           ):
    series_list = []
    description_list = []
    for _ in range(N):
        if trend_type == 'linear':
            series = generate_linear_trend(L, direction=direction)
        elif trend_type == 'quadratic':
            series = generate_quadratic_trend(L, direction=direction)
        elif trend_type == 'flat':
            series = generate_flat_trend(L)
        
        if trend_type != 'flat':
            # # rescale the series to be unit variance
            # series = (series - np.mean(series)) / np.std(series)
            # # randomly sample a mean between -1 and 1 uniform distribution, and a standard deviation between 0.1 and 0.5 uniform distribution
            # mean = random.uniform(*mean_range)
            # std = random.uniform(*std_range)
            # series = series * std + mean
            series = series - np.mean(series)
            series = series + random.uniform(*mean_range)
            description = f"The time series shows {direction}ward {trend_type} trend."
        else:
            description = "No trend."  
            
        series_list.append(series)
        description_list.append(description)
    return series_list, description_list


