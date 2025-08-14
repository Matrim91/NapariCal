# __init__.py files for all modules

# calcium_segmentation/__init__.py
"""
Calcium Imaging Segmentation Tool
A comprehensive tool for segmenting cells in calcium imaging data
"""

__version__ = "1.0.0"
__author__ = "Calcium Segmentation Team"

from .main import main

__all__ = ['main']

# widgets/__init__.py
"""
User interface widgets for calcium segmentation
"""

from .main_widget import CalciumSegmentationWidget
from .image_enhancement import ImageEnhancementWidget  
from .segmentation_controls import SegmentationControlsWidget
from .analysis_window import AnalysisWindow
from .exclusion_controls import ExclusionControlsWidget

__all__ = [
    'CalciumSegmentationWidget',
    'ImageEnhancementWidget',
    'SegmentationControlsWidget', 
    'AnalysisWindow',
    'ExclusionControlsWidget'
]

# processing/__init__.py
"""
Core image processing and segmentation algorithms
"""

from .image_processing import ImageProcessingPipeline
from .segmentation import SegmentationPipeline
from .analysis import AnalysisPipeline
from .region_growing import RegionGrowingPipeline

__all__ = [
    'ImageProcessingPipeline',
    'SegmentationPipeline',
    'AnalysisPipeline', 
    'RegionGrowingPipeline'
]

# io/__init__.py
"""
Input/output utilities for loading and saving data
"""

from .tiff_loader import TiffLoader
from .data_export import DataExporter

__all__ = [
    'TiffLoader',
    'DataExporter'
]

# utils/__init__.py
"""
Utility functions for image processing and analysis
"""

from .peak_detection import peak_local_maxima
from .morphology_utils import (
    relabel_consecutively,
    remove_small_objects_from_labels,
    apply_morphology_with_compensation,
    fast_merge_touching_labels,
    clean_fragmented_labels
)

__all__ = [
    'peak_local_maxima',
    'relabel_consecutively',
    'remove_small_objects_from_labels', 
    'apply_morphology_with_compensation',
    'fast_merge_touching_labels',
    'clean_fragmented_labels'
]