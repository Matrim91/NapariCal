"""
Morphological operation utilities
"""

import numpy as np
import cv2
from skimage import measure, morphology


def relabel_consecutively(labels):
    """
    Relabel an image so labels are consecutive starting from 1
    
    Args:
        labels: Input labeled image
    
    Returns:
        Relabeled image with consecutive labels
    """
    unique_labels = np.unique(labels)[1:]  # Skip background (0)
    relabeled = np.zeros_like(labels)
    
    for new_label, old_label in enumerate(unique_labels, start=1):
        relabeled[labels == old_label] = new_label
    
    return relabeled


def remove_small_objects_from_labels(labels, min_size):
    """
    Remove labels smaller than minimum size
    
    Args:
        labels: Input labeled image
        min_size: Minimum size in pixels
    
    Returns:
        Filtered labeled image
    """
    filtered_labels = labels.copy()
    removed_count = 0
    
    for region in measure.regionprops(labels):
        if region.area < min_size:
            filtered_labels[filtered_labels == region.label] = 0
            removed_count += 1
    
    return relabel_consecutively(filtered_labels), removed_count


def apply_morphology_with_compensation(binary_image, iterations=3):
    """
    Apply morphological operations optimized for horizontal cells with drift compensation
    
    Args:
        binary_image: Input binary image
        iterations: Number of morphological iterations
    
    Returns:
        Tuple of (processed_image, method_used)
    """
    # Method 1: Anchor-corrected approach
    kernel1 = np.ones((2, 4), np.uint8)
    kernel2 = np.ones((2, 5), np.uint8)
    
    # Use top-anchored morphology to prevent downward drift
    anchor1 = (0, 2)  # Top of kernel, horizontally centered
    anchor2 = (0, 2)  # Top of kernel, horizontally centered
    
    try:
        opening1 = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel1, 
                                anchor=anchor1, iterations=iterations)
        anchor_result = cv2.morphologyEx(opening1, cv2.MORPH_OPEN, kernel2, 
                                    anchor=anchor2, iterations=iterations)
        method_used = "anchor-corrected"
    except:
        # Fallback: Standard morphology with compensation
        opening1 = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel1, iterations=iterations)
        anchor_result = cv2.morphologyEx(opening1, cv2.MORPH_OPEN, kernel2, iterations=iterations)
        
        # Apply upward shift compensation
        compensated = np.zeros_like(anchor_result)
        shift_up = 2  # Pixels to shift up
        compensated[:-shift_up, :] = anchor_result[shift_up:, :]
        anchor_result = compensated
        method_used = "shift-compensated"
    
    # Additional horizontal-preserving cleanup
    ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 3))
    ellipse_cleaned = cv2.morphologyEx(anchor_result, cv2.MORPH_OPEN, ellipse_kernel, iterations=1)
    
    # Shape filtering
    binary_result = ellipse_cleaned > 0
    labeled = measure.label(binary_result)
    
    final_binary = np.zeros_like(labeled, dtype=bool)
    removed_objects = 0
    
    for region in measure.regionprops(labeled):
        # Allow more elongation for horizontal cells
        if region.minor_axis_length > 0:
            aspect_ratio = region.major_axis_length / region.minor_axis_length
            shape_ok = aspect_ratio <= 8  # More permissive for horizontal cells
        else:
            shape_ok = True
            
        # Check if it's reasonably compact
        compactness = (4 * np.pi * region.area) / (region.perimeter ** 2)
        compactness_ok = compactness > 0.1  # Very permissive
        
        if shape_ok and compactness_ok:
            final_binary[labeled == region.label] = True
        else:
            removed_objects += 1
    
    result = (final_binary * 255).astype(np.uint8)
    
    return result, method_used, removed_objects


def fast_merge_touching_labels(segmented):
    """
    Fast merge of touching labels using morphological operations
    
    Args:
        segmented: Input segmented image
    
    Returns:
        Merged segmentation
    """
    if len(np.unique(segmented)) <= 2:  # Only background + 1 label
        return segmented
    
    # Create binary mask of all cells
    binary_mask = segmented > 0
    
    # Apply slight morphological closing to connect very close cells
    kernel = morphology.disk(2)
    closed_mask = morphology.binary_closing(binary_mask, kernel)
    
    # Find connected components in the closed mask
    from skimage.measure import label
    connected_components = label(closed_mask)
    
    # Create new segmentation by mapping watershed labels to connected components
    new_segmented = np.zeros_like(segmented)
    
    # For each connected component, combine all watershed labels within it
    for cc_label in np.unique(connected_components)[1:]:  # Skip background
        cc_mask = connected_components == cc_label
        
        # Find all watershed labels in this connected component
        watershed_labels_in_cc = np.unique(segmented[cc_mask])
        watershed_labels_in_cc = watershed_labels_in_cc[watershed_labels_in_cc > 0]
        
        # Assign all pixels from these watershed labels to the new component
        for ws_label in watershed_labels_in_cc:
            new_segmented[segmented == ws_label] = cc_label
    
    return new_segmented


def clean_fragmented_labels(labels):
    """
    Clean up fragmented labels and relabel consecutively
    
    Args:
        labels: Input labeled image
    
    Returns:
        Cleaned labeled image
    """
    # Remove small fragments (less than 20 pixels)
    for region in measure.regionprops(labels):
        if region.area < 20:
            labels[labels == region.label] = 0
    
    return relabel_consecutively(labels)