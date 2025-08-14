"""
Segmentation processing functions with MLGOC enhancements
"""

import numpy as np
from skimage import measure, morphology, feature
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt, label, gaussian_filter

from utils.peak_detection import peak_local_maxima
from utils.morphology_utils import (
    apply_morphology_with_compensation, 
    fast_merge_touching_labels,
    remove_small_objects_from_labels,
    relabel_consecutively
)


def apply_binary_size_filter(binary_mask, min_size=50):
    """
    Apply size filtering to binary mask
    
    Args:
        binary_mask: Input binary mask
        min_size: Minimum object size in pixels
    
    Returns:
        Tuple of (filtered_binary, remaining_objects)
    """
    # Convert to binary and label connected components
    binary_image = binary_mask > 0
    labeled_image = measure.label(binary_image)
    
    # Remove small objects
    size_filtered = morphology.remove_small_objects(labeled_image, min_size=min_size)
    
    # Convert back to binary
    size_filtered_binary = (size_filtered > 0).astype(np.uint8) * 255
    
    # Count remaining objects
    remaining_objects = len(np.unique(size_filtered)) - 1
    
    return size_filtered_binary, remaining_objects


def create_initial_labels(morphology_result, exclusion_mask=None):
    """
    Create labels from morphology result - one label per white region
    
    Args:
        morphology_result: Binary morphology result
        exclusion_mask: Optional exclusion mask
    
    Returns:
        Labeled image
    """
    # Convert to binary
    binary_mask = morphology_result > 0
    
    # Label connected components
    labeled_regions = measure.label(binary_mask)
    
    # Remove very small regions (noise)
    for region in measure.regionprops(labeled_regions):
        if region.area < 20:
            labeled_regions[labeled_regions == region.label] = 0
    
    # Apply exclusions if they exist
    if exclusion_mask is not None and np.any(exclusion_mask):
        labeled_regions[exclusion_mask] = 0
    
    # Relabel consecutively
    return relabel_consecutively(labeled_regions)


def apply_watershed_segmentation(binary_mask, min_distance=15, exclusion_mask=None):
    """
    Apply watershed segmentation to binary mask
    
    Args:
        binary_mask: Input binary mask
        min_distance: Minimum distance between seeds
        exclusion_mask: Optional exclusion mask
    
    Returns:
        Dictionary with segmentation results
    """
    # Convert to binary
    binary_mask = binary_mask > 0
    
    # Distance transform and find seeds
    distance_transform = distance_transform_edt(binary_mask)
    local_maxima = peak_local_maxima(
        distance_transform,
        min_distance=min_distance,
        threshold_abs=min_distance//3,
        indices=False
    )
    
    # Watershed segmentation
    markers = label(local_maxima)[0]
    elevation = -distance_transform
    segmented = watershed(elevation, markers, mask=binary_mask)
    
    # Remove small segments
    for region in measure.regionprops(segmented):
        if region.area < 30:
            segmented[segmented == region.label] = 0
    
    # Fast merge of touching labels
    segmented = fast_merge_touching_labels(segmented)
    
    # Apply exclusions
    if exclusion_mask is not None and np.any(exclusion_mask):
        segmented[exclusion_mask] = 0
    
    return {
        'segmented': segmented,
        'distance_transform': distance_transform,
        'markers': markers
    }


def apply_watershed_subdivision(existing_labels, min_distance=15):
    """
    Apply watershed to subdivide existing labels
    
    Args:
        existing_labels: Existing labeled image
        min_distance: Minimum distance between seeds
    
    Returns:
        Subdivided labeled image
    """
    original_labels = existing_labels.copy()
    subdivided_labels = np.zeros_like(original_labels)
    
    current_max_label = 0
    
    # Process each existing label individually
    for region in measure.regionprops(original_labels):
        label_id = region.label
        
        # Get mask for this label
        label_mask = (original_labels == label_id)
        
        # Only subdivide if label is reasonably large
        if region.area > 50:
            # Distance transform within this label
            distance_transform = distance_transform_edt(label_mask)
            
            # Find seeds within this label
            local_maxima = peak_local_maxima(
                distance_transform,
                min_distance=min_distance,
                threshold_abs=min_distance//3,
                indices=False
            )
            
            # If we found multiple seeds, apply watershed
            markers = label(local_maxima)[0]
            if len(np.unique(markers)) > 2:  # More than background + 1 seed
                # Watershed within this label only
                elevation = -distance_transform
                subdivided = watershed(elevation, markers, mask=label_mask)
                
                # Relabel to avoid conflicts
                for sub_region in measure.regionprops(subdivided):
                    if sub_region.area >= 20:
                        current_max_label += 1
                        subdivided_labels[subdivided == sub_region.label] = current_max_label
            else:
                # No subdivision needed
                current_max_label += 1
                subdivided_labels[label_mask] = current_max_label
        else:
            # Too small to subdivide
            current_max_label += 1
            subdivided_labels[label_mask] = current_max_label
    
    return subdivided_labels


def multi_layer_watershed(binary_mask, intensity_image=None, num_layers=3, 
                         min_distance=15, preferred_radius=17):
    """
    Apply multi-layer watershed for overlapping cells (MLGOC approach)
    
    Args:
        binary_mask: Input binary mask
        intensity_image: Original intensity image
        num_layers: Number of segmentation layers
        min_distance: Minimum distance between seeds
        preferred_radius: Preferred cell radius for size selectivity
    
    Returns:
        Combined labeled image from all layers
    """
    if intensity_image is None:
        intensity_image = binary_mask.astype(np.float32)
    
    layers = []
    remaining_mask = binary_mask.copy()
    all_labels = np.zeros_like(binary_mask, dtype=np.int32)
    next_label_id = 1
    
    for layer_idx in range(num_layers):
        if np.sum(remaining_mask) < 100:  # Stop if too few pixels left
            break
        
        print(f"Processing layer {layer_idx + 1}/{num_layers}...")
        
        # Apply size-selective watershed to current layer
        layer_result = size_selective_watershed(
            remaining_mask, intensity_image, 
            preferred_radius=preferred_radius,
            min_distance=min_distance
        )
        
        if layer_result is not None and np.max(layer_result) > 0:
            # Relabel to avoid conflicts
            unique_labels = np.unique(layer_result)[1:]  # Skip background
            for old_label in unique_labels:
                mask = layer_result == old_label
                all_labels[mask] = next_label_id
                next_label_id += 1
            
            layers.append(layer_result)
            
            # Remove segmented areas from remaining mask
            remaining_mask[layer_result > 0] = 0
            
            # Erode remaining mask slightly to separate layers
            remaining_mask = morphology.binary_erosion(remaining_mask, morphology.disk(1))
        else:
            break
    
    print(f"Multi-layer watershed complete: {len(layers)} layers, {next_label_id-1} total labels")
    return all_labels


def size_selective_watershed(binary_mask, intensity_image, preferred_radius=17, 
                           tolerance=0.3, min_distance=15):
    """
    Size-selective watershed that favors objects of preferred size
    
    Args:
        binary_mask: Input binary mask
        intensity_image: Original intensity image
        preferred_radius: Preferred cell radius
        tolerance: Size tolerance (0.3 = 30% variation allowed)
        min_distance: Minimum distance between seeds
    
    Returns:
        Labeled image with size-selective segmentation
    """
    if not np.any(binary_mask):
        return None
    
    # Create elevation map using additive intensity model
    elevation = create_additive_elevation_map(
        intensity_image, binary_mask, preferred_radius
    )
    
    # Intelligent seed initialization
    seeds = intelligent_seed_initialization(
        intensity_image, binary_mask, preferred_radius, min_distance
    )
    
    if seeds is None or np.max(seeds) <= 1:
        return None
    
    # Apply watershed
    segmented = watershed(elevation, seeds, mask=binary_mask)
    
    # Size-based validation and filtering
    validated = validate_segmentation_sizes(
        segmented, preferred_radius, tolerance
    )
    
    return validated


def create_additive_elevation_map(intensity_image, binary_mask, preferred_radius):
    """
    Create elevation map using additive intensity model (MLGOC approach)
    
    Args:
        intensity_image: Original intensity image
        binary_mask: Binary mask of region
        preferred_radius: Expected cell radius
    
    Returns:
        Elevation map for watershed
    """
    # Calculate expected single-cell intensity
    distance_transform = distance_transform_edt(binary_mask)
    
    # Areas far from edges likely contain single cells
    single_cell_mask = distance_transform > (preferred_radius * 0.7)
    
    if np.any(single_cell_mask):
        expected_intensity = np.median(intensity_image[single_cell_mask])
    else:
        expected_intensity = np.median(intensity_image[binary_mask])
    
    # Normalize intensity to expected single-cell value
    normalized_intensity = intensity_image.astype(np.float32) / max(expected_intensity, 1)
    
    # Areas with higher intensity likely have overlapping cells
    overlap_factor = np.clip(normalized_intensity - 1.0, 0, 2)
    
    # Combine distance transform with intensity information
    # Higher overlap = deeper valleys (more subdivision)
    elevation = -(distance_transform + overlap_factor * preferred_radius * 0.5)
    
    return elevation


def intelligent_seed_initialization(intensity_image, binary_mask, preferred_radius, min_distance):
    """
    Intelligent seed initialization (MLGOC approach)
    
    Args:
        intensity_image: Original intensity image
        binary_mask: Binary mask
        preferred_radius: Expected cell radius
        min_distance: Minimum distance between seeds
    
    Returns:
        Seed markers for watershed
    """
    # Method 1: Blob detection at preferred scale
    try:
        blobs = feature.blob_log(
            intensity_image, 
            min_sigma=preferred_radius * 0.4,
            max_sigma=preferred_radius * 1.2,
            threshold=0.05,
            overlap=0.5
        )
        
        # Convert blobs to binary mask
        blob_mask = np.zeros_like(binary_mask, dtype=bool)
        for blob in blobs:
            y, x, sigma = blob
            y, x = int(y), int(x)
            if (0 <= y < blob_mask.shape[0] and 0 <= x < blob_mask.shape[1] and
                binary_mask[y, x]):
                blob_mask[y, x] = True
    except:
        blob_mask = np.zeros_like(binary_mask, dtype=bool)
    
    # Method 2: Distance transform peaks
    distance_transform = distance_transform_edt(binary_mask)
    smoothed_distance = gaussian_filter(distance_transform, sigma=preferred_radius * 0.2)
    
    distance_peaks = peak_local_maxima(
        smoothed_distance,
        min_distance=min_distance,
        threshold_abs=preferred_radius * 0.4,
        indices=False
    )
    
    # Combine both methods
    combined_seeds = blob_mask | distance_peaks
    combined_seeds = combined_seeds & binary_mask
    
    # Create labeled markers
    markers = label(combined_seeds)[0]
    
    return markers


def validate_segmentation_sizes(segmented, preferred_radius, tolerance):
    """
    Validate and filter segmentation based on size expectations
    
    Args:
        segmented: Segmented image
        preferred_radius: Expected cell radius
        tolerance: Size tolerance
    
    Returns:
        Validated segmentation
    """
    expected_area = np.pi * preferred_radius ** 2
    min_area = expected_area * (1 - tolerance)
    max_area = expected_area * (1 + tolerance * 2)  # More lenient for max
    
    validated = segmented.copy()
    removed_count = 0
    
    for region in measure.regionprops(segmented):
        if not (min_area <= region.area <= max_area):
            validated[validated == region.label] = 0
            removed_count += 1
    
    if removed_count > 0:
        print(f"Size validation: removed {removed_count} objects outside size range")
    
    return relabel_consecutively(validated)


def calculate_auto_size_threshold(labels, std_multiplier=3.0):
    """
    Calculate automatic size threshold using median - std_multiplier * std_dev
    
    Args:
        labels: Labeled image
        std_multiplier: Multiplier for standard deviation
    
    Returns:
        Tuple of (threshold, statistics_dict)
    """
    props = measure.regionprops(labels)
    if len(props) == 0:
        return 10, {}
    
    sizes = [region.area for region in props]
    
    median_size = np.median(sizes)
    std_size = np.std(sizes)
    
    # Calculate threshold
    threshold = median_size - (std_multiplier * std_size)
    threshold = max(threshold, 10)  # Minimum threshold
    
    statistics = {
        'median_size': median_size,
        'std_size': std_size,
        'std_multiplier': std_multiplier,
        'threshold': threshold,
        'total_labels': len(props)
    }
    
    return threshold, statistics


class SegmentationPipeline:
    """
    Complete segmentation pipeline
    """
    
    def __init__(self):
        self.binary_mask = None
        self.size_filtered_binary = None
        self.morphology_result = None
        self.distance_transform = None
        self.final_labels = None
    
    def process(self, binary_mask, exclusion_mask=None, morphology_params=None, 
                watershed_params=None, size_filter_params=None):
        """
        Process complete segmentation pipeline
        
        Args:
            binary_mask: Input binary mask
            exclusion_mask: Optional exclusion mask
            morphology_params: Morphology parameters
            watershed_params: Watershed parameters
            size_filter_params: Size filter parameters
        
        Returns:
            Dictionary with all segmentation results
        """
        # Set default parameters
        morphology_params = morphology_params or {'iterations': 3}
        watershed_params = watershed_params or {'min_distance': 15}
        size_filter_params = size_filter_params or {'min_size': 50}
        
        # Store input
        self.binary_mask = binary_mask
        
        # Apply size filtering to binary
        self.size_filtered_binary, remaining_objects = apply_binary_size_filter(
            binary_mask, **size_filter_params
        )
        
        # Apply morphology
        self.morphology_result, method_used, removed_objects = apply_morphology_with_compensation(
            self.size_filtered_binary, **morphology_params
        )
        
        # Create initial labels
        initial_labels = create_initial_labels(self.morphology_result, exclusion_mask)
        
        # Apply watershed
        watershed_results = apply_watershed_segmentation(
            self.morphology_result, exclusion_mask=exclusion_mask, **watershed_params
        )
        
        self.final_labels = watershed_results['segmented']
        self.distance_transform = watershed_results['distance_transform']
        
        return {
            'binary_mask': self.binary_mask,
            'size_filtered_binary': self.size_filtered_binary,
            'morphology_result': self.morphology_result,
            'initial_labels': initial_labels,
            'final_labels': self.final_labels,
            'distance_transform': self.distance_transform,
            'method_used': method_used,
            'removed_objects': removed_objects,
            'remaining_objects': remaining_objects
        }