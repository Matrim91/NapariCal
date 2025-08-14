"""
Region growing algorithms for cell segmentation
"""

import numpy as np
from collections import deque
from skimage import measure
from utils.morphology_utils import relabel_consecutively


def region_grow(image, existing_labels, seed, seed_intensity, tolerance, max_radius):
    """
    Region growing algorithm from a seed point
    
    Args:
        image: Input intensity image
        existing_labels: Existing labeled regions to avoid
        seed: Seed point (y, x)
        seed_intensity: Intensity at seed point
        tolerance: Intensity tolerance for growing
        max_radius: Maximum radius for growing
    
    Returns:
        Boolean mask of grown region
    """
    height, width = image.shape
    seed_y, seed_x = seed
    
    # Initialize
    grown_region = np.zeros((height, width), dtype=bool)
    visited = np.zeros((height, width), dtype=bool)
    queue = deque([seed])
    
    # 8-connectivity neighbors
    neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    while queue:
        y, x = queue.popleft()
        
        # Skip if already visited
        if visited[y, x]:
            continue
        visited[y, x] = True
        
        # Check bounds
        if y < 0 or y >= height or x < 0 or x >= width:
            continue
        
        # Check if too far from seed
        distance = np.sqrt((y - seed_y)**2 + (x - seed_x)**2)
        if distance > max_radius:
            continue
        
        # Check if already labeled
        if existing_labels[y, x] > 0:
            continue
        
        # Check intensity similarity
        pixel_intensity = image[y, x]
        if abs(pixel_intensity - seed_intensity) > tolerance:
            continue
        
        # Add pixel to region
        grown_region[y, x] = True
        
        # Add neighbors to queue
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if (0 <= ny < height and 0 <= nx < width and 
                not visited[ny, nx]):
                queue.append((ny, nx))
    
    return grown_region


def grow_missing_cells(labels, intensity_image, candidates, intensity_tolerance=20, 
                      max_radius=15, min_size=20, max_size=500):
    """
    Grow new cell regions from missing cell candidates
    
    Args:
        labels: Existing labeled image
        intensity_image: Original intensity image
        candidates: List of candidate dictionaries
        intensity_tolerance: Intensity tolerance for growing
        max_radius: Maximum growth radius
        min_size: Minimum acceptable grown region size
        max_size: Maximum acceptable grown region size
    
    Returns:
        Tuple of (new_labels, grown_count, growth_info)
    """
    if not candidates:
        return labels, 0, []
    
    # Use filtered image for better region growing
    growth_image = intensity_image.astype(np.float32)
    
    # Start with existing labels
    new_labels = labels.copy()
    next_label_id = np.max(new_labels) + 1
    
    grown_count = 0
    growth_info = []
    
    for i, candidate in enumerate(candidates):
        seed_y, seed_x = candidate['position']
        seed_y, seed_x = int(seed_y), int(seed_x)
        
        # Skip if seed is already in a labeled region
        if new_labels[seed_y, seed_x] > 0:
            continue
        
        # Get seed intensity
        seed_intensity = growth_image[seed_y, seed_x]
        
        # Region growing using flood fill
        grown_region = region_grow(
            growth_image, 
            new_labels,
            (seed_y, seed_x), 
            seed_intensity, 
            intensity_tolerance, 
            max_radius
        )
        
        # Check if grown region is reasonable size
        region_size = np.sum(grown_region)
        if min_size <= region_size <= max_size:
            # Add to labels
            new_labels[grown_region] = next_label_id
            grown_count += 1
            
            growth_info.append({
                'seed_position': (seed_y, seed_x),
                'label_id': next_label_id,
                'size': region_size,
                'candidate_info': candidate
            })
            
            next_label_id += 1
        else:
            growth_info.append({
                'seed_position': (seed_y, seed_x),
                'label_id': None,
                'size': region_size,
                'rejected_reason': 'size_out_of_range',
                'candidate_info': candidate
            })
    
    return new_labels, grown_count, growth_info


def preview_region_growth(labels, intensity_image, candidates, intensity_tolerance=20,
                         max_radius=15, preview_limit=10):
    """
    Generate preview of what region growing would produce
    
    Args:
        labels: Existing labeled image
        intensity_image: Original intensity image
        candidates: List of candidate dictionaries
        intensity_tolerance: Intensity tolerance for growing
        max_radius: Maximum growth radius
        preview_limit: Maximum number of candidates to preview
    
    Returns:
        Preview mask with different intensities for each region
    """
    growth_image = intensity_image.astype(np.float32)
    preview_mask = np.zeros(growth_image.shape, dtype=np.uint8)
    
    for i, candidate in enumerate(candidates[:preview_limit]):
        seed_y, seed_x = candidate['position']
        seed_y, seed_x = int(seed_y), int(seed_x)
        
        # Skip if seed is already labeled
        if labels[seed_y, seed_x] > 0:
            continue
        
        seed_intensity = growth_image[seed_y, seed_x]
        
        # Grow region
        grown_region = region_grow(
            growth_image,
            labels,
            (seed_y, seed_x),
            seed_intensity,
            intensity_tolerance,
            max_radius
        )
        
        # Add to preview (different intensity for each region)
        region_size = np.sum(grown_region)
        if 20 <= region_size <= 500:  # Only preview reasonable sizes
            preview_mask[grown_region] = min(255, (i + 1) * 25)
    
    return preview_mask


class RegionGrowingPipeline:
    """
    Pipeline for region growing operations
    """
    
    def __init__(self):
        self.growth_history = []
    
    def grow_from_candidates(self, labels, intensity_image, candidates, 
                           growth_params=None):
        """
        Grow regions from candidate positions
        
        Args:
            labels: Existing labeled image
            intensity_image: Original intensity image
            candidates: List of candidate dictionaries
            growth_params: Growth parameters
        
        Returns:
            Dictionary with growth results
        """
        # Set default parameters
        growth_params = growth_params or {
            'intensity_tolerance': 20,
            'max_radius': 15,
            'min_size': 20,
            'max_size': 500
        }
        
        # Perform growth
        new_labels, grown_count, growth_info = grow_missing_cells(
            labels, intensity_image, candidates, **growth_params
        )
        
        # Store history
        self.growth_history.append({
            'original_labels': labels,
            'new_labels': new_labels,
            'candidates': candidates,
            'growth_info': growth_info,
            'parameters': growth_params
        })
        
        # Calculate statistics
        successful_growths = [info for info in growth_info if info.get('label_id') is not None]
        rejected_growths = [info for info in growth_info if info.get('label_id') is None]
        
        return {
            'new_labels': new_labels,
            'grown_count': grown_count,
            'total_candidates': len(candidates),
            'successful_growths': successful_growths,
            'rejected_growths': rejected_growths,
            'growth_info': growth_info
        }
    
    def generate_preview(self, labels, intensity_image, candidates, growth_params=None):
        """
        Generate preview of region growing
        
        Args:
            labels: Existing labeled image
            intensity_image: Original intensity image
            candidates: List of candidate dictionaries
            growth_params: Growth parameters
        
        Returns:
            Preview mask
        """
        growth_params = growth_params or {
            'intensity_tolerance': 20,
            'max_radius': 15
        }
        
        return preview_region_growth(
            labels, intensity_image, candidates,
            intensity_tolerance=growth_params['intensity_tolerance'],
            max_radius=growth_params['max_radius']
        )