import numpy as np
from typing import Tuple, List
import random


def generate_noise(
    length: int,
    noise_std_range: Tuple[float, float] = (1, 5)
) -> np.ndarray:
    """
    Generate a time series with varying noise.
    
    Args:
        length: Length of the time series
        noise_std_range: Range for random noise standard deviation selection (inclusive)
        
    Returns:
        np.ndarray: Generated time series
    """
    # Randomly select parameters within their ranges
    noise_std = random.uniform(*noise_std_range)
    # Generate the noise
    noise = np.random.normal(0, noise_std, length)
    return noise


def generate_nonstationary_noise(
    length: int,
    noise_std_range: Tuple[float, float] = (0.01, 0.05),
    change_points: int = 3,
    smooth_transition: bool = True
) -> np.ndarray:
    """
    Generate a time series with non-stationary noise that changes over time.
    
    Args:
        length: Length of the time series
        noise_std_range: Range for random noise standard deviation selection (inclusive)
        change_points: Number of points where noise level changes
        smooth_transition: Whether to use smooth transitions between noise levels
        
    Returns:
        np.ndarray: Generated time series with non-stationary noise
    """
    # Generate change points
    change_indices = sorted(np.random.randint(0, length, size=change_points))
    change_indices = [0] + change_indices + [length]
    
    # Generate noise levels for each segment
    noise_levels = [random.uniform(*noise_std_range) for _ in range(len(change_indices) - 1)]
    
    # Generate the time series
    series = np.zeros(length)
    
    if smooth_transition:
        # Create smooth transitions between noise levels
        for i in range(len(change_indices) - 1):
            start_idx = change_indices[i]
            end_idx = change_indices[i + 1]
            segment_length = end_idx - start_idx
            
            # Create transition window
            transition_length = min(50, segment_length // 4)  # 50 points or 1/4 of segment length
            if i > 0:  # Not the first segment
                # Create smooth transition from previous noise level
                transition = np.linspace(noise_levels[i-1], noise_levels[i], transition_length)
                series[start_idx:start_idx + transition_length] = np.random.normal(0, transition, transition_length)
                start_idx += transition_length
            
            # Generate main segment with current noise level
            remaining_length = end_idx - start_idx
            if remaining_length > 0:
                series[start_idx:end_idx] = np.random.normal(0, noise_levels[i], remaining_length)
    else:
        # Generate segments with abrupt changes
        for i in range(len(change_indices) - 1):
            start_idx = change_indices[i]
            end_idx = change_indices[i + 1]
            segment_length = end_idx - start_idx
            series[start_idx:end_idx] = np.random.normal(0, noise_levels[i], segment_length)
    
    return series


def generate_noise_series(
    N: int,
    L: int,
    noise_type: str = 'stationary'
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Generate N samples of length L with specified noise pattern.
    
    Args: 
        N: Number of series to generate
        L: Length of each series
        noise_type: Type of noise pattern ('stationary', 'nonstationary')
        std_range: Range for standard deviation of noise
        mean_range: Range for mean of noise
        
    Returns:
        Tuple[List[np.ndarray], List[str]]: Generated series and their descriptions
    """
    series_list = []
    description_list = []
    
    for _ in range(N):
        stationary = random.choice([True, False])
        if noise_type == 'low variability':
            if stationary:
                # Generate stationary noise
                series = generate_noise(length=L, noise_std_range=(0.01, 0.1))
            else:
                # Generate non-stationary noise
                series = generate_nonstationary_noise(length=L, noise_std_range=(0.01, 0.1)) # 0.01, 0.1
            description = "The time series exhibits low variability."
        elif noise_type == 'high variability':
            if stationary:
                series = generate_noise(length=L, noise_std_range=(1.5, 2))
            else:
                series = generate_nonstationary_noise(length=L, noise_std_range=(1.5, 2)) # 1.5, 2
            description = "The time series exhibits high variability."
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        
        series_list.append(series)
        description_list.append(description)
    
    return series_list, description_list

