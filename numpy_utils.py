"""
Simple NumPy utility functions to replace numpy-minmax and numpy-rms packages.
These functions provide the same functionality using standard NumPy operations.
"""

import numpy as np
from typing import Union, Tuple


def minmax(arr: np.ndarray, axis: Union[int, Tuple[int, ...], None] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute minimum and maximum values along specified axis.
    
    Args:
        arr: Input array
        axis: Axis or axes along which to compute min/max. If None, compute over entire array.
    
    Returns:
        Tuple of (minimum_values, maximum_values)
    """
    return np.min(arr, axis=axis), np.max(arr, axis=axis)


def rms(arr: np.ndarray, axis: Union[int, Tuple[int, ...], None] = None) -> np.ndarray:
    """
    Compute root mean square (RMS) value along specified axis.
    
    Args:
        arr: Input array
        axis: Axis or axes along which to compute RMS. If None, compute over entire array.
    
    Returns:
        RMS values
    """
    return np.sqrt(np.mean(np.square(arr), axis=axis))


def normalize_minmax(arr: np.ndarray, feature_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
    """
    Normalize array to specified range using min-max scaling.
    
    Args:
        arr: Input array
        feature_range: Target range for normalization
    
    Returns:
        Normalized array
    """
    min_val, max_val = np.min(arr), np.max(arr)
    if max_val == min_val:
        return np.full_like(arr, feature_range[0])
    
    # Scale to [0, 1] then to feature_range
    normalized = (arr - min_val) / (max_val - min_val)
    return normalized * (feature_range[1] - feature_range[0]) + feature_range[0]


def normalize_rms(arr: np.ndarray, target_rms: float = 1.0) -> np.ndarray:
    """
    Normalize array to have specified RMS value.
    
    Args:
        arr: Input array
        target_rms: Target RMS value
    
    Returns:
        RMS-normalized array
    """
    current_rms = rms(arr)
    if current_rms == 0:
        return arr
    return arr * (target_rms / current_rms)


# Aliases for compatibility
def numpy_minmax(arr: np.ndarray, axis=None):
    """Alias for minmax function."""
    return minmax(arr, axis)


def numpy_rms(arr: np.ndarray, axis=None):
    """Alias for rms function.""" 
    return rms(arr, axis) 