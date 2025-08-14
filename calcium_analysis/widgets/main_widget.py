"""
Main widget that coordinates all segmentation functionality
"""

import numpy as np
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout

from .image_enhancement import ImageEnhancementWidget
from .segmentation_controls import SegmentationControlsWidget
from .analysis_window import AnalysisWindow
from .exclusion_controls import ExclusionControlsWidget

from processing.image_processing import ImageProcessingPipeline
from processing.segmentation import SegmentationPipeline
from io_dir.tiff_loader import TiffLoader
from io_dir.data_export import DataExporter


class CalciumSegmentationWidget(QWidget):
    """
    Main widget that coordinates all calcium segmentation functionality
    """
    
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        
        # Initialize pipelines
        self.image_pipeline = ImageProcessingPipeline()
        self.segmentation_pipeline = SegmentationPipeline()
        self.tiff_loader = TiffLoader()
        self.data_exporter = DataExporter()
        
        # State management
        self.state = {
            'file_info': {
                'file_name': '20230602_01_01_03',
                'cwd': "C:/0000 - Grant_Network/01 - CalciumAnalysisSuite",
                'tiff_file_path': None
            },
            'images': {
                'original': None,
                'grayscale': None,
                'processed_stages': {}
            },
            'segmentation': {
                'binary_mask': None,
                'morphology_result': None,
                'final_labels': None
            },
            'exclusions': {
                'mask': None,
                'exclude_touching': False
            },
            'analysis': {
                'missing_candidates': [],
                'subdivision_candidates': [],
                'labels_ready': False
            }
        }
        
        # Initialize UI
        self.init_ui()
        self.setup_connections()
    
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        
        # Create main sections
        main_layout = QHBoxLayout()
        
        # Left Column - Image Enhancement
        self.image_enhancement = ImageEnhancementWidget(self)
        main_layout.addWidget(self.image_enhancement)
        
        # Right Column - Segmentation Controls
        self.segmentation_controls = SegmentationControlsWidget(self)
        main_layout.addWidget(self.segmentation_controls)
        
        layout.addLayout(main_layout)
        
        # Bottom section - Exclusion controls and export
        self.exclusion_controls = ExclusionControlsWidget(self)
        layout.addWidget(self.exclusion_controls)
        
        self.setLayout(layout)
        
        # Initialize analysis window (not shown by default)
        self.analysis_window = None
    
    def setup_connections(self):
        """Setup signal connections between widgets"""
        # Image enhancement signals
        self.image_enhancement.tiff_loaded.connect(self.on_tiff_loaded)
        self.image_enhancement.processing_updated.connect(self.on_processing_updated)
        
        # Segmentation signals
        self.segmentation_controls.segmentation_updated.connect(self.on_segmentation_updated)
        self.segmentation_controls.labels_created.connect(self.on_labels_created)
        
        # Exclusion signals
        self.exclusion_controls.exclusion_updated.connect(self.on_exclusion_updated)
    
    def get_tiff_file_path(self):
        """Get the full TIFF file path"""
        file_info = self.state['file_info']
        return f"{file_info['cwd']}/{file_info['file_name']}.tif"
    
    def load_tiff(self):
        """Load TIFF file and update state"""
        try:
            tiff_path = self.get_tiff_file_path()
            averaged_image = self.tiff_loader.load_and_average(tiff_path, max_frames=100)
            
            # Update state
            self.state['images']['original'] = averaged_image
            
            # Convert to grayscale
            if len(averaged_image.shape) == 3:
                self.state['images']['grayscale'] = np.mean(averaged_image, axis=2).astype(np.uint8)
            else:
                self.state['images']['grayscale'] = averaged_image.copy()
            
            # Update napari
            self.update_napari_layer('Original', averaged_image, 'image')
            
            # Enable controls safely
            try:
                self.image_enhancement.set_enabled(True)
            except AttributeError as e:
                print(f"Warning: Could not enable image enhancement controls: {e}")
                # Enable individual buttons that exist
                for button_name in ['clahe_button', 'contrast_button', 'filter_button', 'reset_button']:
                    if hasattr(self.image_enhancement, button_name):
                        getattr(self.image_enhancement, button_name).setEnabled(True)
                # Try to enable tophat button if it exists
                if hasattr(self.image_enhancement, 'tophat_button'):
                    self.image_enhancement.tophat_button.setEnabled(True)
            
            self.segmentation_controls.set_enabled(True)
            self.exclusion_controls.set_enabled(True)
            
            print(f"Loaded TIFF: {averaged_image.shape}, dtype: {averaged_image.dtype}")
            
        except Exception as e:
            print(f"Error loading TIFF: {e}")
            import traceback
            traceback.print_exc()
    
    def process_image_enhancement(self, stage=None):
        """Process image through enhancement pipeline"""
        if self.state['images']['original'] is None:
            return
        
        # Get current parameters from widgets
        params = self.image_enhancement.get_parameters()
        exclusion_mask = self.get_exclusion_mask()
        
        # Process through pipeline
        results = self.image_pipeline.process(
            self.state['images']['original'],
            exclusion_mask=exclusion_mask,
            **params
        )
        
        # Update state
        self.state['images']['processed_stages'] = results
        
        # Update napari layers based on what stage was requested
        if stage:
            layer_name = stage.title()
            self.update_napari_layer(layer_name, results[stage.lower()], 'image')
        else:
            # Update all layers
            for stage_name, image in results.items():
                if stage_name != 'original':  # Don't update original
                    layer_name = stage_name.title()
                    self.update_napari_layer(layer_name, image, 'image')
    
    def process_segmentation(self, stage=None):
        """Process segmentation pipeline"""
        processed_images = self.state['images']['processed_stages']
        if not processed_images or 'binary' not in processed_images:
            self.process_image_enhancement()
            processed_images = self.state['images']['processed_stages']
        
        # Get segmentation parameters
        params = self.segmentation_controls.get_parameters()
        exclusion_mask = self.get_exclusion_mask()
        
        # Process segmentation
        results = self.segmentation_pipeline.process(
            processed_images['binary'],
            exclusion_mask=exclusion_mask,
            **params
        )
        
        # Update state
        self.state['segmentation'].update(results)
        
        # Update napari
        if 'final_labels' in results and results['final_labels'] is not None:
            self.update_napari_layer('Cell Labels', results['final_labels'], 'labels')
            self.state['analysis']['labels_ready'] = True
    
    def get_exclusion_mask(self):
        """Get current exclusion mask"""
        return self.state['exclusions']['mask']
    
    def update_napari_layer(self, name, data, layer_type='image'):
        """Update or create napari layer"""
        if name in [layer.name for layer in self.viewer.layers]:
            self.viewer.layers[name].data = data
        else:
            if layer_type == 'image':
                self.viewer.add_image(data, name=name, colormap='gray')
            elif layer_type == 'labels':
                self.viewer.add_labels(data, name=name)
            elif layer_type == 'points':
                self.viewer.add_points(data, name=name)
    
    def open_analysis_window(self):
        """Open the detachable analysis window"""
        if self.analysis_window is None:
            self.analysis_window = AnalysisWindow(self)
        
        self.analysis_window.show()
        self.analysis_window.raise_()
        self.analysis_window.activateWindow()
    
    # Signal handlers
    def on_tiff_loaded(self):
        """Handle TIFF loaded signal"""
        self.load_tiff()
    
    def on_processing_updated(self, stage):
        """Handle image processing update"""
        self.process_image_enhancement(stage)
    
    def on_segmentation_updated(self, stage):
        """Handle segmentation update"""
        if stage == 'threshold':
            # Process binary threshold
            processed_images = self.state['images']['processed_stages']
            if 'filtered' in processed_images:
                from processing.image_processing import apply_adaptive_threshold
                binary = apply_adaptive_threshold(processed_images['filtered'])
                self.state['segmentation']['binary_mask'] = binary
                self.update_napari_layer('Threshold', binary, 'image')
        
        elif stage == 'size_filter':
            # Process binary size filter
            if self.state['segmentation']['binary_mask'] is not None:
                from processing.segmentation import apply_binary_size_filter
                params = self.segmentation_controls.get_parameters()
                min_size = params['size_filter_params']['min_size']
                
                filtered_binary, remaining = apply_binary_size_filter(
                    self.state['segmentation']['binary_mask'], min_size
                )
                self.update_napari_layer('Size Filtered', filtered_binary, 'image')
                print(f"Size filter: {remaining} objects remaining")
        
        elif stage == 'morphology':
            # Process morphology
            if self.state['segmentation']['binary_mask'] is not None:
                from utils.morphology_utils import apply_morphology_with_compensation
                binary_input = self.state['segmentation']['binary_mask']
                
                morphology_result, method, removed = apply_morphology_with_compensation(binary_input)
                self.state['segmentation']['morphology_result'] = morphology_result
                self.update_napari_layer('Morphology', morphology_result, 'image')
                print(f"Morphology complete using {method}, removed {removed} objects")
        
        elif stage == 'create_labels':
            # Create initial labels
            if self.state['segmentation']['morphology_result'] is not None:
                from processing.segmentation import create_initial_labels
                labels = create_initial_labels(
                    self.state['segmentation']['morphology_result'],
                    self.get_exclusion_mask()
                )
                self.state['segmentation']['final_labels'] = labels
                self.update_napari_layer('Cell Labels', labels, 'labels')
                num_labels = len(np.unique(labels)) - 1
                print(f"Created {num_labels} initial labels")
        
        elif stage == 'watershed':
            # Apply watershed
            if self.state['segmentation']['morphology_result'] is not None:
                from processing.segmentation import apply_watershed_segmentation
                params = self.segmentation_controls.get_parameters()
                min_distance = params['watershed_params']['min_distance']
                
                results = apply_watershed_segmentation(
                    self.state['segmentation']['morphology_result'],
                    min_distance=min_distance,
                    exclusion_mask=self.get_exclusion_mask()
                )
                
                self.state['segmentation']['final_labels'] = results['segmented']
                self.update_napari_layer('Cell Labels', results['segmented'], 'labels')
                num_labels = len(np.unique(results['segmented'])) - 1
                print(f"Watershed complete: {num_labels} cells detected")
        
        elif stage == 'mlgoc':
            # Apply MLGOC multi-layer watershed
            if self.state['segmentation']['morphology_result'] is not None:
                from processing.segmentation import multi_layer_watershed
                params = self.segmentation_controls.get_parameters()
                
                # Get intensity image for MLGOC
                intensity_image = None
                if 'filtered' in self.state['images']['processed_stages']:
                    intensity_image = self.state['images']['processed_stages']['filtered']
                elif self.state['images']['grayscale'] is not None:
                    intensity_image = self.state['images']['grayscale']
                
                mlgoc_labels = multi_layer_watershed(
                    self.state['segmentation']['morphology_result'] > 0,
                    intensity_image=intensity_image,
                    exclusion_mask=self.get_exclusion_mask(),
                    **params['mlgoc_params']
                )
                
                self.state['segmentation']['final_labels'] = mlgoc_labels
                self.update_napari_layer('Cell Labels', mlgoc_labels, 'labels')
                num_labels = len(np.unique(mlgoc_labels)) - 1
                print(f"MLGOC multi-layer watershed complete: {num_labels} cells detected")
        
        elif stage == 'analysis_view':
            # Setup clean analysis view
            self.setup_analysis_view()
    
    def on_labels_created(self):
        """Handle labels created signal"""
        self.state['analysis']['labels_ready'] = True
        if self.analysis_window:
            self.analysis_window.update_button_states()
    
    def on_exclusion_updated(self, exclusion_mask):
        """Handle exclusion mask update"""
        self.state['exclusions']['mask'] = exclusion_mask
    
    def setup_analysis_view(self):
        """Setup clean view for analysis"""
        # Show only original and labels
        for layer in self.viewer.layers:
            if layer.name in ['Original', 'Cell Labels']:
                layer.visible = True
                if layer.name == 'Cell Labels':
                    layer.opacity = 0.7
            else:
                layer.visible = False
        
        num_cells = 0
        if self.state['segmentation']['final_labels'] is not None:
            num_cells = len(np.unique(self.state['segmentation']['final_labels'])) - 1
        
        print(f"Analysis view ready - {num_cells} cells detected")
    
    def reset_to_original(self):
        """Reset view to show only original image"""
        if self.state['images']['original'] is not None:
            # Hide all layers except original
            for layer in self.viewer.layers:
                if layer.name != 'Original':
                    layer.visible = False
                else:
                    layer.visible = True