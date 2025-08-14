"""
Peak detection utilities
"""

import numpy as np
from scipy.ndimage import maximum_filter


def peak_local_maxima(image, min_distance=1, threshold_abs=None, indices=False):
    """
    Simple peak detection fallback function
    
    Args:
        image: Input image
        min_distance: Minimum distance between peaks
        threshold_abs: Absolute threshold for peak detection
        indices: If True, return indices; if False, return boolean mask
    
    Returns:
        Peak locations as indices or boolean mask
    """
    local_max = (image == maximum_filter(image, size=min_distance*2+1))
    if threshold_abs is not None:
        local_max = local_max & (image > threshold_abs)
    if indices:
        return np.where(local_max)
    return local_max