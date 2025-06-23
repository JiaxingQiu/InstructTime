import numpy as np
from typing import Tuple, List
import random


def generate_no_seasonal(
    length: int,
    period_range: Tuple[int, int] = (150, 200),
    amplitude_range: Tuple[float, float] = (0, 0.1),
    phase_range: Tuple[float, float] = (0, 2*np.pi),
    noise_std: float = 0.05
) -> np.ndarray:
    """
    Generate a time series with no seasonal pattern.
    
    Args:
        length: Length of the time series
        period_range: Range for random period selection (inclusive)
        amplitude_range: Range for random amplitude selection
        phase_range: Range for random phase selection (in radians)
        noise_std: Standard deviation of Gaussian noise
        
    Returns:
        np.ndarray: Generated time series
    """
    t = np.arange(length)
    # Randomly select parameters within their ranges
    period = random.randint(*period_range)
    amplitude = random.uniform(*amplitude_range)
    phase = random.uniform(*phase_range)
    
    # Generate the seasonal pattern
    seasonal = amplitude * np.sin(2 * np.pi * t / period + phase)
    
    # Add noise
    noise = np.random.normal(0, noise_std, length)
    return seasonal + noise


def generate_single_seasonal(
    length: int,
    period_range: Tuple[int, int] = (20, 50),
    amplitude_range: Tuple[float, float] = (5, 8),
    phase_range: Tuple[float, float] = (0, 2*np.pi),
    noise_std: float = 0.05
) -> np.ndarray:
    """
    Generate a time series with seasonal pattern of varying period, amplitude, and phase.
    
    Args:
        length: Length of the time series
        period_range: Range for random period selection (inclusive)
        amplitude_range: Range for random amplitude selection
        phase_range: Range for random phase selection (in radians)
        noise_std: Standard deviation of Gaussian noise
        
    Returns:
        np.ndarray: Generated time series
    """
    t = np.arange(length)
    # Randomly select parameters within their ranges
    period = random.randint(*period_range)
    amplitude = random.uniform(*amplitude_range)
    phase = random.uniform(*phase_range)
    
    # Generate the seasonal pattern
    seasonal = amplitude * np.sin(2 * np.pi * t / period + phase)
    
    # Add noise
    noise = np.random.normal(0, noise_std, length)
    return seasonal + noise


def generate_multiple_seasonal(
    length: int,
    k_range: Tuple[int, int] = (2, 4),  # number of seasonal patterns
    period_range: Tuple[int, int] = (20, 50),  # random sample from 1 to 10
    amplitude_range: Tuple[float, float] = (2, 4),  # random sample from 0.1 to 1
    phase_range: Tuple[float, float] = (0, 2*np.pi),  # random sample from 0 to 2*pi
    noise_std: float = 0.05
) -> np.ndarray:
    """
    Generate a time series with multiple seasonal patterns.
    
    Args:
        length: Length of the time series
        k: Number of seasonal patterns to generate
        periods_range: Range for random period selection
        amplitudes_range: Range for random amplitude selection
        phases_range: Range for random phase selection
        noise_std: Standard deviation of Gaussian noise
        
    Returns:
        np.ndarray: Generated time series
    """
    k = random.randint(*k_range)
    # Generate random periods, amplitudes, and phases
    periods = [random.randint(*period_range) for _ in range(k)]
    amplitudes = [random.uniform(*amplitude_range) for _ in range(k)]
    phases = [random.uniform(*phase_range) for _ in range(k)]
    
    t = np.arange(length)
    seasonal = np.zeros(length)
    
    for period, amplitude, phase in zip(periods, amplitudes, phases):
        seasonal += amplitude * np.sin(2 * np.pi * t / period + phase)
    
    noise = np.random.normal(0, noise_std, length)
    return seasonal + noise


def generate_seasonal_series(
    N: int,
    L: int,
    seasonal_type: str = 'single',
    # std_range: Tuple[float, float] = (5, 10),
    # mean_range: Tuple[float, float] = (-1, 1)
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Generate N samples of length L with specified seasonal pattern.
    
    Args: 
        N: Number of series to generate
        L: Length of each series
        seasonal_type: Type of seasonal pattern ('single', 'multiple')
        
    Returns:
        Tuple[List[np.ndarray], List[str]]: Generated series and their descriptions
    """
    series_list = []
    description_list = []
    
    for _ in range(N):
        if seasonal_type == 'no':
            # Use default parameters from generate_fixed_seasonal
            series = generate_no_seasonal(length=L)  # daily-ish pattern
            description = "No seasonal pattern."
        elif seasonal_type == 'single':
            # Use default parameters from generate_single_seasonal
            series = generate_single_seasonal(length=L)
            description = "The time series exhibits a seasonal pattern."
        elif seasonal_type == 'multiple':
            # Use default parameters from generate_multiple_seasonal
            series = generate_multiple_seasonal(length=L)
            description = "The time series exhibits a seasonal pattern."
        else:
            raise ValueError(f"Unknown seasonal type: {seasonal_type}")
        
        # # rescale the series to be unit variance
        # series = (series - np.mean(series)) / np.std(series)
        # # randomly sample a mean between -1 and 1 uniform distribution, and a standard deviation between 0.1 and 0.5 uniform distribution
        # mean = random.uniform(*mean_range)
        # std = random.uniform(*std_range)
        # series = series * std + mean

        series_list.append(series)
        description_list.append(description)
    
    return series_list, description_list


