"""
Image processing functions for calcium imaging segmentation
Pure functions without UI dependencies
"""

import numpy as np
import cv2
from skimage.morphology import white_tophat, disk
from typing import Tuple, Optional


def apply_white_tophat(image: np.ndarray, disk_size: int = 20) -> np.ndarray:
    """
    Apply white tophat filter to enhance bright objects and correct illumination
    
    Args:
        image: Input grayscale image
        disk_size: Size of the morphological disk (should be larger than cells)
    
    Returns:
        Tophat filtered image
    """
    structuring_element = disk(disk_size)
    tophat_image = white_tophat(image, structuring_element)
    
    # Enhance: original + tophat to preserve structures while correcting background
    enhanced = image.astype(np.float32) + tophat_image.astype(np.float32)
    
    # Normalize back to uint8
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    return enhanced


def apply_clahe(image: np.ndarray, clip_limit: float = 1.0, grid_size: int = 30) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization
    
    Args:
        image: Input grayscale image
        clip_limit: Clipping limit for CLAHE
        grid_size: Size of the grid for histogram equalization
    
    Returns:
        CLAHE processed image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    return clahe.apply(image)


def apply_contrast_adjustment(image: np.ndarray, alpha: float = 1.2, beta: float = 0.2) -> np.ndarray:
    """
    Apply contrast and brightness adjustment
    
    Args:
        image: Input image
        alpha: Contrast multiplier
        beta: Brightness offset
    
    Returns:
        Contrast adjusted image
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def apply_bilateral_filter(image: np.ndarray, d: int = 5, sigma_color: float = 1000, 
                          sigma_space: float = 500) -> np.ndarray:
    """
    Apply bilateral filtering for edge-preserving smoothing
    
    Args:
        image: Input image
        d: Diameter of each pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space
    
    Returns:
        Filtered image
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def apply_adaptive_threshold(image: np.ndarray, block_size: int = 21, 
                           c_constant: float = 2) -> np.ndarray:
    """
    Apply adaptive thresholding
    
    Args:
        image: Input grayscale image
        block_size: Size of neighborhood area for threshold calculation
        c_constant: Constant subtracted from the mean
    
    Returns:
        Binary thresholded image
    """
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, block_size, c_constant
    )


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale if needed
    
    Args:
        image: Input image (can be RGB or already grayscale)
    
    Returns:
        Grayscale image
    """
    if len(image.shape) == 3:
        return np.mean(image, axis=2).astype(np.uint8)
    return image.copy()


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to uint8 range [0, 255]
    
    Args:
        image: Input image of any dtype
    
    Returns:
        Image normalized to uint8
    """
    if image.dtype == np.uint8:
        return image
    
    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        normalized = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(image, dtype=np.uint8)
    
    return normalized


def apply_exclusion_mask(image: np.ndarray, exclusion_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply exclusion mask to image by setting excluded areas to mean value
    
    Args:
        image: Input image
        exclusion_mask: Boolean mask of areas to exclude
    
    Returns:
        Image with exclusion applied
    """
    if exclusion_mask is None or not np.any(exclusion_mask):
        return image
    
    masked_image = image.copy()
    mean_value = np.mean(masked_image[~exclusion_mask])
    masked_image[exclusion_mask] = mean_value
    
    return masked_image


class ImageProcessingPipeline:
    """
    Container for processing an image through the enhancement pipeline
    """
    
    def __init__(self):
        self.original = None
        self.grayscale = None
        self.clahe = None
        self.contrast = None
        self.filtered = None
        self.binary = None
    
    def process(self, image: np.ndarray, exclusion_mask: Optional[np.ndarray] = None,
                tophat_params: dict = None, clahe_params: dict = None, 
                contrast_params: dict = None, filter_params: dict = None, 
                threshold_params: dict = None) -> dict:
        """
        Process image through the complete enhancement pipeline
        
        Args:
            image: Input image
            exclusion_mask: Optional exclusion mask
            tophat_params: Tophat filter parameters
            clahe_params: CLAHE parameters
            contrast_params: Contrast adjustment parameters
            filter_params: Bilateral filter parameters
            threshold_params: Threshold parameters
        
        Returns:
            Dictionary containing all processing stages
        """
        # Set default parameters
        tophat_params = tophat_params or {'disk_size': 20}
        clahe_params = clahe_params or {'clip_limit': 1.0, 'grid_size': 30}
        contrast_params = contrast_params or {'alpha': 1.2, 'beta': 0.2}
        filter_params = filter_params or {'d': 5, 'sigma_color': 1000, 'sigma_space': 500}
        threshold_params = threshold_params or {'block_size': 21, 'c_constant': 2}
        
        # Store original and convert to grayscale
        self.original = normalize_to_uint8(image)
        self.grayscale = convert_to_grayscale(self.original)
        
        # Apply exclusion mask to input
        input_image = apply_exclusion_mask(self.grayscale, exclusion_mask)
        
        # Apply processing pipeline with tophat first
        self.tophat = apply_white_tophat(input_image, **tophat_params)
        self.clahe = apply_clahe(self.tophat, **clahe_params)
        self.contrast = apply_contrast_adjustment(self.clahe, **contrast_params)
        self.filtered = apply_bilateral_filter(self.contrast, **filter_params)
        self.binary = apply_adaptive_threshold(self.filtered, **threshold_params)
        
        # Apply exclusion mask to final binary result
        if exclusion_mask is not None and np.any(exclusion_mask):
            self.binary[exclusion_mask] = 0
        
        return {
            'original': self.original,
            'grayscale': self.grayscale,
            'tophat': self.tophat,
            'clahe': self.clahe,
            'contrast': self.contrast,
            'filtered': self.filtered,
            'binary': self.binary
        }