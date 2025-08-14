"""
Analysis functions for missing cells and large labels
"""

import numpy as np
from skimage import feature, measure, filters
from scipy import ndimage
from utils.peak_detection import peak_local_maxima


def find_missing_cells(labels, intensity_image, sensitivity=0.5, min_distance=20):
    """
    Find potential cells that were missed by automated segmentation
    
    Args:
        labels: Current labeled image
        intensity_image: Original intensity image for detection
        sensitivity: Detection sensitivity (0.1 to 0.9)
        min_distance: Minimum distance from existing cells
    
    Returns:
        List of candidate dictionaries
    """
    # Get existing cell centers
    existing_centers = []
    if labels is not None:
        props = measure.regionprops(labels)
        existing_centers = [prop.centroid for prop in props]
    
    # 1. Blob detection for cell-like structures
    min_sigma = 2
    max_sigma = 15
    threshold = 0.1 + (1 - sensitivity) * 0.2  # Higher threshold = fewer detections
    
    # Detect blobs using Laplacian of Gaussian
    blobs_log = feature.blob_log(intensity_image, 
                               min_sigma=min_sigma, 
                               max_sigma=max_sigma,
                               threshold=threshold,
                               overlap=0.5)
    
    # 2. Intensity-based detection
    smoothed = filters.gaussian(intensity_image, sigma=3)
    threshold_value = np.percentile(smoothed, 60 + sensitivity * 30)
    local_maxima_mask = peak_local_maxima(smoothed, 
                                        min_distance=10,
                                        threshold_abs=threshold_value,
                                        indices=False)
    
    # Get coordinates of local maxima
    local_maxima_coords = np.where(local_maxima_mask)
    
    # Combine detections
    candidates = []
    
    # Add blob centers
    for blob in blobs_log:
        y, x, sigma = blob
        candidates.append((int(y), int(x), sigma * 1.414, 'blob'))
    
    # Add intensity maxima
    for y, x in zip(local_maxima_coords[0], local_maxima_coords[1]):
        candidates.append((y, x, 5, 'intensity'))
    
    # 3. Filter candidates
    filtered_candidates = []
    for candidate in candidates:
        y, x, radius, method = candidate
        
        # Check distance to existing cells
        too_close = False
        if labels is not None:
            # Check if any existing labels are nearby
            region_size = int(min_distance)
            y_start = max(0, y - region_size)
            y_end = min(labels.shape[0], y + region_size)
            x_start = max(0, x - region_size)
            x_end = min(labels.shape[1], x + region_size)
            
            region_labels = labels[y_start:y_end, x_start:x_end]
            if np.any(region_labels > 0):
                for existing_y, existing_x in existing_centers:
                    distance = np.sqrt((y - existing_y)**2 + (x - existing_x)**2)
                    if distance < min_distance:
                        too_close = True
                        break
                
                # Check if inside existing label
                if (0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]):
                    if labels[y, x] > 0:
                        too_close = True
        
        if not too_close:
            # Quality checks
            region_size = int(radius * 2)
            y_start = max(0, y - region_size)
            y_end = min(intensity_image.shape[0], y + region_size)
            x_start = max(0, x - region_size)
            x_end = min(intensity_image.shape[1], x + region_size)
            
            region = intensity_image[y_start:y_end, x_start:x_end]
            
            if region.size > 0:
                mean_intensity = np.mean(region)
                intensity_std = np.std(region)
                
                # Cell-like criteria
                if (mean_intensity > np.percentile(intensity_image, 30) and 
                    intensity_std > 5):
                    
                    filtered_candidates.append({
                        'position': (y, x),
                        'radius': radius,
                        'intensity': mean_intensity,
                        'method': method,
                        'confidence': mean_intensity / 255.0
                    })
    
    return filtered_candidates


def analyze_large_labels(labels, intensity_image, size_threshold=200):
    """
    Analyze large labels that might contain multiple cells
    
    Args:
        labels: Labeled image
        intensity_image: Original intensity image
        size_threshold: Minimum size to consider as "large"
    
    Returns:
        List of subdivision candidate dictionaries
    """
    # Find large labels
    props = measure.regionprops(labels, intensity_image=intensity_image)
    large_labels = [prop for prop in props if prop.area > size_threshold]
    
    subdivision_candidates = []
    
    for prop in large_labels:
        label_id = prop.label
        
        # Get the label mask and bounding box
        minr, minc, maxr, maxc = prop.bbox
        label_mask = (labels[minr:maxr, minc:maxc] == label_id)
        label_intensity = intensity_image[minr:maxr, minc:maxc]
        
        # Apply mask to intensity
        masked_intensity = label_intensity.copy()
        masked_intensity[~label_mask] = 0
        
        # Method 1: Find local minima (potential cell boundaries)
        inverted = filters.gaussian(255 - masked_intensity, sigma=2)
        inverted[~label_mask] = 0
        
        local_minima = peak_local_maxima(inverted, 
                                       min_distance=8,
                                       threshold_abs=np.percentile(inverted[label_mask], 70),
                                       indices=False)
        
        # Method 2: Watershed on distance transform
        distance = ndimage.distance_transform_edt(label_mask)
        distance_maxima = peak_local_maxima(distance,
                                          min_distance=10,
                                          threshold_abs=np.percentile(distance[label_mask], 80),
                                          indices=False)
        
        # Method 3: Check for elongated shape
        aspect_ratio = prop.major_axis_length / prop.minor_axis_length if prop.minor_axis_length > 0 else 1
        is_elongated = aspect_ratio > 2.5
        
        # Count potential subdivision points
        minima_count = np.sum(local_minima)
        maxima_count = np.sum(distance_maxima)
        
        # Determine if this label likely contains multiple cells
        likely_multiple = False
        confidence = 0.0
        method_used = []
        
        if minima_count >= 2:
            likely_multiple = True
            confidence += 0.4
            method_used.append("intensity_valleys")
        
        if maxima_count >= 2:
            likely_multiple = True
            confidence += 0.4
            method_used.append("distance_peaks")
        
        if is_elongated and (minima_count >= 1 or maxima_count >= 1):
            likely_multiple = True
            confidence += 0.3
            method_used.append("elongated_shape")
        
        # Check intensity variation
        intensity_std = np.std(masked_intensity[label_mask])
        intensity_range = np.ptp(masked_intensity[label_mask])
        if intensity_range > 30 and intensity_std > 15:
            confidence += 0.2
            method_used.append("intensity_variation")
        
        if likely_multiple and confidence > 0.4:
            global_centroid = (prop.centroid[0], prop.centroid[1])
            
            subdivision_candidates.append({
                'label_id': label_id,
                'position': global_centroid,
                'area': prop.area,
                'confidence': min(confidence, 1.0),
                'aspect_ratio': aspect_ratio,
                'methods': method_used,
                'minima_count': minima_count,
                'maxima_count': maxima_count
            })
    
    return subdivision_candidates


class AnalysisPipeline:
    """
    Pipeline for analyzing segmentation results
    """
    
    def __init__(self):
        self.missing_candidates = []
        self.subdivision_candidates = []
    
    def analyze_segmentation(self, labels, intensity_image, missing_params=None, 
                           subdivision_params=None):
        """
        Analyze segmentation for missing cells and subdivisions
        
        Args:
            labels: Labeled image
            intensity_image: Original intensity image
            missing_params: Parameters for missing cell detection
            subdivision_params: Parameters for subdivision analysis
        
        Returns:
            Dictionary with analysis results
        """
        # Set default parameters
        missing_params = missing_params or {'sensitivity': 0.5, 'min_distance': 20}
        subdivision_params = subdivision_params or {'size_threshold': 200}
        
        # Find missing cells
        self.missing_candidates = find_missing_cells(
            labels, intensity_image, **missing_params
        )
        
        # Analyze large labels
        self.subdivision_candidates = analyze_large_labels(
            labels, intensity_image, **subdivision_params
        )
        
        return {
            'missing_candidates': self.missing_candidates,
            'subdivision_candidates': self.subdivision_candidates,
            'missing_count': len(self.missing_candidates),
            'subdivision_count': len(self.subdivision_candidates)
        }