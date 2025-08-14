"""
Analysis Window Widget - Detachable analysis window
"""

import numpy as np
from qtpy.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                           QSlider, QLabel)
from qtpy.QtCore import Qt, Signal
from processing.analysis import AnalysisPipeline
from processing.region_growing import RegionGrowingPipeline


class AnalysisWindow(QDialog):
    """
    Detachable window for advanced analysis functionality
    """
    
    # Signals
    analysis_updated = Signal(str, object)  # analysis_type, results
    
    def __init__(self, main_widget):
        super().__init__(main_widget)
        self.main_widget = main_widget
        
        # Initialize pipelines
        self.analysis_pipeline = AnalysisPipeline()
        self.region_growing_pipeline = RegionGrowingPipeline()
        
        # State
        self.current_candidates = []
        self.subdivision_candidates = []
        
        self.init_ui()
        self.setup_window()
    
    def setup_window(self):
        """Setup window properties"""
        self.setWindowTitle("Cell Analysis & Validation")
        self.setModal(False)  # Non-modal so you can interact with main window
        self.resize(400, 300)
    
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        
        # First row: Main analysis buttons
        main_buttons = QHBoxLayout()
        
        self.find_missing_button = QPushButton("Find Missing Cells")
        self.find_missing_button.clicked.connect(self.on_find_missing_cells)
        self.find_missing_button.setEnabled(self.get_labels_ready())
        self.find_missing_button.setMinimumHeight(35)
        main_buttons.addWidget(self.find_missing_button)
        
        self.analyze_large_button = QPushButton("Find Oversized Labels")
        self.analyze_large_button.clicked.connect(self.on_analyze_large_labels)
        self.analyze_large_button.setEnabled(self.get_labels_ready())
        self.analyze_large_button.setMinimumHeight(35)
        main_buttons.addWidget(self.analyze_large_button)
        
        layout.addLayout(main_buttons)
        
        # Second row: Visibility controls
        visibility_buttons = QHBoxLayout()
        
        self.show_candidates_button = QPushButton("Toggle Missing (Red)")
        self.show_candidates_button.clicked.connect(self.on_toggle_candidates)
        self.show_candidates_button.setEnabled(False)
        self.show_candidates_button.setMaximumHeight(25)
        visibility_buttons.addWidget(self.show_candidates_button)
        
        self.show_subdivisions_button = QPushButton("Toggle Oversized (Orange)")
        self.show_subdivisions_button.clicked.connect(self.on_toggle_subdivisions)
        self.show_subdivisions_button.setEnabled(False)
        self.show_subdivisions_button.setMaximumHeight(25)
        visibility_buttons.addWidget(self.show_subdivisions_button)
        
        layout.addLayout(visibility_buttons)
        
        # Region Growing Controls
        growing_separator = QLabel("─" * 40)
        growing_separator.setStyleSheet("color: #ccc; font-size: 8px;")
        layout.addWidget(growing_separator)
        
        # Growing buttons
        growing_buttons = QHBoxLayout()
        
        self.grow_cells_button = QPushButton("Grow Missing Cells")
        self.grow_cells_button.clicked.connect(self.on_grow_missing_cells)
        self.grow_cells_button.setEnabled(False)
        self.grow_cells_button.setMinimumHeight(30)
        growing_buttons.addWidget(self.grow_cells_button)
        
        self.preview_growth_button = QPushButton("Preview Growth")
        self.preview_growth_button.clicked.connect(self.on_preview_growth)
        self.preview_growth_button.setEnabled(False)
        self.preview_growth_button.setMinimumHeight(30)
        growing_buttons.addWidget(self.preview_growth_button)
        
        layout.addLayout(growing_buttons)
        
        # Growing parameters
        growing_controls = QHBoxLayout()
        
        # Left: Intensity tolerance
        intensity_controls = QVBoxLayout()
        self.intensity_tolerance_label = QLabel("Intensity Tolerance: 20")
        intensity_controls.addWidget(self.intensity_tolerance_label)
        
        self.intensity_tolerance_slider = QSlider(Qt.Horizontal)
        self.intensity_tolerance_slider.setRange(5, 50)
        self.intensity_tolerance_slider.setValue(20)
        self.intensity_tolerance_slider.valueChanged.connect(self.update_intensity_tolerance)
        intensity_controls.addWidget(self.intensity_tolerance_slider)
        
        growing_controls.addLayout(intensity_controls)
        
        # Right: Max radius
        radius_controls = QVBoxLayout()
        self.max_radius_label = QLabel("Max Radius: 15px")
        radius_controls.addWidget(self.max_radius_label)
        
        self.max_radius_slider = QSlider(Qt.Horizontal)
        self.max_radius_slider.setRange(5, 30)
        self.max_radius_slider.setValue(15)
        self.max_radius_slider.valueChanged.connect(self.update_max_radius)
        radius_controls.addWidget(self.max_radius_slider)
        
        growing_controls.addLayout(radius_controls)
        layout.addLayout(growing_controls)
        
        # Analysis parameter controls
        controls_layout = QHBoxLayout()
        
        # Left side: Missing cell sensitivity
        missing_controls = QVBoxLayout()
        self.sensitivity_label = QLabel("Missing Sensitivity: 50%")
        missing_controls.addWidget(self.sensitivity_label)
        
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(10, 90)
        self.sensitivity_slider.setValue(50)
        self.sensitivity_slider.setEnabled(self.get_labels_ready())
        self.sensitivity_slider.valueChanged.connect(self.update_detection_sensitivity)
        missing_controls.addWidget(self.sensitivity_slider)
        
        controls_layout.addLayout(missing_controls)
        
        # Right side: Large label threshold (NOW PERCENTILE-BASED)
        large_controls = QVBoxLayout()
        self.size_threshold_label = QLabel("Oversized Threshold: 90th percentile")
        large_controls.addWidget(self.size_threshold_label)
        
        self.size_threshold_slider = QSlider(Qt.Horizontal)
        self.size_threshold_slider.setRange(50, 99)  # 50th to 99th percentile
        self.size_threshold_slider.setValue(90)      # Start at 90th percentile
        self.size_threshold_slider.setEnabled(False)
        self.size_threshold_slider.valueChanged.connect(self.update_size_threshold_percentile)
        large_controls.addWidget(self.size_threshold_slider)
        
        controls_layout.addLayout(large_controls)
        layout.addLayout(controls_layout)
        
        # Watershed Subdivision Controls (NEW)
        subdivision_separator = QLabel("─" * 40)
        subdivision_separator.setStyleSheet("color: #ccc; font-size: 8px;")
        layout.addWidget(subdivision_separator)
        
        # Subdivision buttons
        subdivision_buttons = QHBoxLayout()
        
        self.auto_subdivide_button = QPushButton("Auto Subdivide Large Labels")
        self.auto_subdivide_button.clicked.connect(self.on_auto_subdivide)
        self.auto_subdivide_button.setEnabled(False)
        self.auto_subdivide_button.setMinimumHeight(30)
        subdivision_buttons.addWidget(self.auto_subdivide_button)
        
        self.preview_subdivide_button = QPushButton("Preview Subdivisions")
        self.preview_subdivide_button.clicked.connect(self.on_preview_subdivisions)
        self.preview_subdivide_button.setEnabled(False)
        self.preview_subdivide_button.setMinimumHeight(30)
        subdivision_buttons.addWidget(self.preview_subdivide_button)
        
        layout.addLayout(subdivision_buttons)
        
        # Subdivision parameters
        subdivision_controls = QHBoxLayout()
        
        # Left: Seed distance
        distance_controls = QVBoxLayout()
        self.seed_distance_label = QLabel("Seed Distance: 15px")
        distance_controls.addWidget(self.seed_distance_label)
        
        self.seed_distance_slider = QSlider(Qt.Horizontal)
        self.seed_distance_slider.setRange(5, 30)
        self.seed_distance_slider.setValue(15)
        self.seed_distance_slider.valueChanged.connect(self.update_seed_distance)
        distance_controls.addWidget(self.seed_distance_slider)
        
        subdivision_controls.addLayout(distance_controls)
        
        # Right: Seed threshold
        threshold_controls = QVBoxLayout()
        self.seed_threshold_label = QLabel("Seed Threshold: 50%")
        threshold_controls.addWidget(self.seed_threshold_label)
        
        self.seed_threshold_slider = QSlider(Qt.Horizontal)
        self.seed_threshold_slider.setRange(10, 90)
        self.seed_threshold_slider.setValue(50)
        self.seed_threshold_slider.valueChanged.connect(self.update_seed_threshold)
        threshold_controls.addWidget(self.seed_threshold_slider)
        
        subdivision_controls.addLayout(threshold_controls)
        
        # Intensity weight control (NEW)
        weight_controls = QVBoxLayout()
        self.intensity_weight_label = QLabel("Intensity Weight: 60%")
        weight_controls.addWidget(self.intensity_weight_label)
        
        self.intensity_weight_slider = QSlider(Qt.Horizontal)
        self.intensity_weight_slider.setRange(0, 100)
        self.intensity_weight_slider.setValue(60)
        self.intensity_weight_slider.valueChanged.connect(self.update_intensity_weight)
        weight_controls.addWidget(self.intensity_weight_slider)
        
        subdivision_controls.addLayout(weight_controls)
        layout.addLayout(subdivision_controls)
        
        # Status info
        self.detection_status = QLabel("Run analysis to find issues with segmentation")
        self.detection_status.setStyleSheet("color: #666; font-size: 9px;")
        layout.addWidget(self.detection_status)
        
        # Preset Group (NEW)
        preset_separator = QLabel("─" * 40)
        preset_separator.setStyleSheet("color: #ccc; font-size: 8px;")
        layout.addWidget(preset_separator)
        
        preset_group_layout = QHBoxLayout()
        self.endothelial_preset_button = QPushButton("Endothelial Cell Preset")
        self.endothelial_preset_button.clicked.connect(self.apply_endothelial_preset)
        self.endothelial_preset_button.setEnabled(True)
        self.endothelial_preset_button.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white;")
        preset_group_layout.addWidget(self.endothelial_preset_button)
        
        layout.addLayout(preset_group_layout)
        
        self.setLayout(layout)
    
    def apply_endothelial_preset(self):
        """Apply optimized preset for endothelial cells"""
        # Missing cell detection
        self.sensitivity_slider.setValue(35)  # 30-40% conservative
        
        # Large label analysis  
        self.size_threshold_slider.setValue(80)  # 80th percentile
        
        # Subdivision parameters
        self.seed_distance_slider.setValue(8)    # 6-10px for small cells
        self.seed_threshold_slider.setValue(40)  # 30-50% moderate
        self.intensity_weight_slider.setValue(85) # 80-90% favor intensity
        
        # Update labels
        self.sensitivity_label.setText(f"Missing Sensitivity: {self.sensitivity_slider.value()}%")
        self.seed_distance_label.setText(f"Seed Distance: {self.seed_distance_slider.value()}px")
        self.seed_threshold_label.setText(f"Seed Threshold: {self.seed_threshold_slider.value()}%")
        self.intensity_weight_label.setText(f"Intensity Weight: {self.intensity_weight_slider.value()}%")
        
        # Update size threshold if labels are available
        if hasattr(self, 'update_size_threshold_percentile'):
            self.update_size_threshold_percentile(self.size_threshold_slider.value())
        
        print("Applied endothelial cell analysis preset:")
        print("  Missing Sensitivity: 35%")
        print("  Oversized Threshold: 80th percentile") 
        print("  Seed Distance: 8px")
        print("  Seed Threshold: 40%")
        print("  Intensity Weight: 85%")
    
    def get_labels_ready(self) -> bool:
        """Check if labels are ready for analysis"""
        if hasattr(self.main_widget, 'state'):
            return self.main_widget.state['analysis']['labels_ready']
        return False
    
    def get_current_labels(self):
        """Get current labels from main widget"""
        if hasattr(self.main_widget, 'state'):
            return self.main_widget.state['segmentation']['final_labels']
        return None
    
    def get_intensity_image(self):
        """Get intensity image for analysis"""
        if hasattr(self.main_widget, 'state'):
            # Try to get filtered image first, then original
            stages = self.main_widget.state['images']['processed_stages']
            if stages and 'filtered' in stages:
                return stages['filtered']
            return self.main_widget.state['images']['grayscale']
        return None
    
    def update_button_states(self):
        """Update button states based on current analysis state"""
        labels_ready = self.get_labels_ready()
        has_candidates = len(self.current_candidates) > 0
        has_subdivisions = len(self.subdivision_candidates) > 0
        
        self.find_missing_button.setEnabled(labels_ready)
        self.analyze_large_button.setEnabled(labels_ready)
        self.sensitivity_slider.setEnabled(labels_ready)
        
        self.grow_cells_button.setEnabled(has_candidates)
        self.preview_growth_button.setEnabled(has_candidates)
        
        # Update visibility button states
        self.show_candidates_button.setEnabled(has_candidates)
        self.show_subdivisions_button.setEnabled(has_subdivisions)
        
        # Update subdivision button states
        self.auto_subdivide_button.setEnabled(has_subdivisions)
        self.preview_subdivide_button.setEnabled(has_subdivisions)
    
    # Parameter update methods
    def update_intensity_tolerance(self, value):
        """Update intensity tolerance label"""
        self.intensity_tolerance_label.setText(f"Intensity Tolerance: {value}")
    
    def update_max_radius(self, value):
        """Update max radius label"""
        self.max_radius_label.setText(f"Max Radius: {value}px")
    
    def update_detection_sensitivity(self, sensitivity):
        """Update detection sensitivity label"""
        self.sensitivity_label.setText(f"Missing Sensitivity: {sensitivity}%")
    
    def update_size_threshold_percentile(self, percentile_value):
        """Update size threshold label and recalculate actual threshold"""
        labels = self.get_current_labels()
        if labels is not None:
            import numpy as np
            from skimage import measure
            
            props = measure.regionprops(labels)
            sizes = [region.area for region in props]
            
            if sizes:
                actual_threshold = int(np.percentile(sizes, percentile_value))
                num_large = sum(1 for size in sizes if size > actual_threshold)
                
                # Update label to show percentile, actual size, and count
                self.size_threshold_label.setText(
                    f"Oversized Threshold: {percentile_value}th percentile ({actual_threshold}px, {num_large} labels)"
                )
                
                # Store the actual threshold for use in analysis
                self.current_size_threshold = actual_threshold
            else:
                self.size_threshold_label.setText(f"Oversized Threshold: {percentile_value}th percentile")
                self.current_size_threshold = 200  # Fallback
        else:
            self.size_threshold_label.setText(f"Oversized Threshold: {percentile_value}th percentile")
            self.current_size_threshold = 200  # Fallback
    
    def update_seed_distance(self, distance):
        """Update seed distance label"""
        self.seed_distance_label.setText(f"Seed Distance: {distance}px")
    
    def update_seed_threshold(self, threshold):
        """Update seed threshold label"""
        self.seed_threshold_label.setText(f"Seed Threshold: {threshold}%")
    
    def update_intensity_weight(self, weight):
        """Update intensity weight label"""
        self.intensity_weight_label.setText(f"Intensity Weight: {weight}%")
    
    # Event handlers
    def on_find_missing_cells(self):
        """Handle find missing cells button click"""
        labels = self.get_current_labels()
        intensity_image = self.get_intensity_image()
        
        if labels is None or intensity_image is None:
            print("No labels or intensity image available")
            return
        
        # Get parameters
        sensitivity = self.sensitivity_slider.value() / 100.0
        
        # Run analysis
        results = self.analysis_pipeline.analyze_segmentation(
            labels, intensity_image,
            missing_params={'sensitivity': sensitivity, 'min_distance': 20}
        )
        
        self.current_candidates = results['missing_candidates']
        
        # Create napari points layer
        if self.current_candidates:
            import numpy as np
            positions = np.array([c['position'] for c in self.current_candidates])
            
            # Update or create points layer
            viewer = self.main_widget.viewer
            if 'Missing Cell Candidates' in [layer.name for layer in viewer.layers]:
                points_layer = viewer.layers['Missing Cell Candidates']
                points_layer.data = positions
            else:
                points_layer = viewer.add_points(
                    positions, 
                    name='Missing Cell Candidates',
                    size=8,
                    face_color='red',
                    border_color='white',
                    border_width=0.2,
                    border_width_is_relative=True
                )
            
            print(f"Found {len(self.current_candidates)} potential missing cells")
            self.detection_status.setText(f"Found {len(self.current_candidates)} locations for new cells")
        else:
            print("No missing cells detected")
            self.detection_status.setText("No missing cells detected")
        
        self.update_button_states()
    
    def on_analyze_large_labels(self):
        """Handle analyze large labels button click"""
        labels = self.get_current_labels()
        intensity_image = self.get_intensity_image()
        
        if labels is None or intensity_image is None:
            print("No labels or intensity image available")
            return
        
        # Get parameters
        percentile_value = self.size_threshold_slider.value()
        
        # Calculate actual size threshold from percentile
        if labels is not None:
            import numpy as np
            from skimage import measure
            props = measure.regionprops(labels)
            sizes = [region.area for region in props]
            
            if sizes:
                size_threshold = int(np.percentile(sizes, percentile_value))
                
                # Update the label display
                num_large = sum(1 for size in sizes if size > size_threshold)
                self.size_threshold_label.setText(
                    f"Oversized Threshold: {percentile_value}th percentile ({size_threshold}px, {num_large} labels)"
                )
            else:
                size_threshold = 200  # Fallback
        else:
            size_threshold = 200  # Fallback
        
        # Enable the slider after first analysis
        self.size_threshold_slider.setEnabled(True)
        
        # Run analysis
        results = self.analysis_pipeline.analyze_segmentation(
            labels, intensity_image,
            subdivision_params={'size_threshold': size_threshold}
        )
        
        self.subdivision_candidates = results['subdivision_candidates']
        
        # Create napari points layer
        if self.subdivision_candidates:
            import numpy as np
            positions = np.array([c['position'] for c in self.subdivision_candidates])
            
            # Update or create points layer
            viewer = self.main_widget.viewer
            if 'Large Label Subdivisions' in [layer.name for layer in viewer.layers]:
                points_layer = viewer.layers['Large Label Subdivisions']
                points_layer.data = positions
            else:
                points_layer = viewer.add_points(
                    positions,
                    name='Large Label Subdivisions',
                    size=12,
                    face_color='orange',
                    border_color='black',
                    border_width=0.3,
                    border_width_is_relative=True
                )
            
            print(f"Found {len(self.subdivision_candidates)} labels that likely contain multiple cells")
            self.detection_status.setText(f"Found {len(self.subdivision_candidates)} labels to split")
        else:
            print("No large labels found that likely contain multiple cells")
            self.detection_status.setText("No labels need splitting")
        
        self.size_threshold_slider.setEnabled(True)
        self.update_button_states()
    
    def on_toggle_candidates(self):
        """Toggle visibility of missing cell candidates"""
        viewer = self.main_widget.viewer
        if 'Missing Cell Candidates' in [layer.name for layer in viewer.layers]:
            layer = viewer.layers['Missing Cell Candidates']
            layer.visible = not layer.visible
            status = "visible" if layer.visible else "hidden"
            print(f"Missing cell candidates are now {status}")
    
    def on_toggle_subdivisions(self):
        """Toggle visibility of subdivision candidates"""
        viewer = self.main_widget.viewer
        if 'Large Label Subdivisions' in [layer.name for layer in viewer.layers]:
            layer = viewer.layers['Large Label Subdivisions']
            layer.visible = not layer.visible
            status = "visible" if layer.visible else "hidden"
            print(f"Large label subdivisions are now {status}")
    
    def on_grow_missing_cells(self):
        """Handle grow missing cells button click"""
        labels = self.get_current_labels()
        intensity_image = self.get_intensity_image()
        
        if not self.current_candidates:
            print("No missing cell candidates found. Run 'Find Missing Cells' first.")
            return
        
        if labels is None or intensity_image is None:
            print("No labels or intensity image available")
            return
        
        # Get growth parameters
        growth_params = {
            'intensity_tolerance': self.intensity_tolerance_slider.value(),
            'max_radius': self.max_radius_slider.value(),
            'min_size': 20,
            'max_size': 500
        }
        
        # Perform region growing
        results = self.region_growing_pipeline.grow_from_candidates(
            labels, intensity_image, self.current_candidates, growth_params
        )
        
        # Update main widget state
        self.main_widget.state['segmentation']['final_labels'] = results['new_labels']
        
        # Update napari
        self.main_widget.update_napari_layer('Cell Labels', results['new_labels'], 'labels')
        
        # Update status
        grown_count = results['grown_count']
        total_cells = len(np.unique(results['new_labels'])) - 1
        
        print(f"Region growing complete: {grown_count} new cells added")
        print(f"Total cells: {total_cells}")
        self.detection_status.setText(f"Added {grown_count} cells via region growing")
    
    def on_preview_growth(self):
        """Handle preview growth button click"""
        labels = self.get_current_labels()
        intensity_image = self.get_intensity_image()
        
        if not self.current_candidates:
            print("No missing cell candidates found. Run 'Find Missing Cells' first.")
            return
        
        if labels is None or intensity_image is None:
            print("No labels or intensity image available")
            return
        
        # Get growth parameters
        growth_params = {
            'intensity_tolerance': self.intensity_tolerance_slider.value(),
            'max_radius': self.max_radius_slider.value()
        }
        
        # Generate preview
        preview_mask = self.region_growing_pipeline.generate_preview(
            labels, intensity_image, self.current_candidates, growth_params
        )
        
        # Add preview layer to napari
        viewer = self.main_widget.viewer
        if 'Growth Preview' in [layer.name for layer in viewer.layers]:
            viewer.layers['Growth Preview'].data = preview_mask
        else:
            viewer.add_image(
                preview_mask, 
                name='Growth Preview', 
                colormap='viridis',
                opacity=0.5
            )
        
        print("Preview generated - check 'Growth Preview' layer")
        print("Use 'Grow Missing Cells' to apply the growth")
    
    def on_auto_subdivide(self):
        """Handle auto subdivide large labels button click"""
        labels = self.get_current_labels()
        
        if not self.subdivision_candidates:
            print("No large labels found. Run 'Find Oversized Labels' first.")
            return
        
        if labels is None:
            print("No labels available")
            return
        
        # Get subdivision parameters
        seed_distance = self.seed_distance_slider.value()
        seed_threshold_percent = self.seed_threshold_slider.value()
        intensity_weight = self.intensity_weight_slider.value() / 100.0
        
        print(f"Auto-subdividing {len(self.subdivision_candidates)} large labels...")
        print(f"Parameters: seed_distance={seed_distance}px, threshold={seed_threshold_percent}%, intensity_weight={intensity_weight:.1f}")
        
        # Apply watershed subdivision to all large labels
        new_labels = self.apply_watershed_to_large_labels(
            labels, self.subdivision_candidates, seed_distance, seed_threshold_percent, intensity_weight
        )
        
        # Update main widget state and napari
        self.main_widget.state['segmentation']['final_labels'] = new_labels
        self.main_widget.update_napari_layer('Cell Labels', new_labels, 'labels')
        
        # Update statistics
        original_count = len(np.unique(labels)) - 1
        final_count = len(np.unique(new_labels)) - 1
        added_labels = final_count - original_count
        
        print(f"Subdivision complete:")
        print(f"  Original labels: {original_count}")
        print(f"  Final labels: {final_count}")
        print(f"  Added labels: +{added_labels}")
        
        self.detection_status.setText(f"Subdivided {len(self.subdivision_candidates)} large labels → +{added_labels} labels")
    
    def on_preview_subdivisions(self):
        """Handle preview subdivisions button click"""
        labels = self.get_current_labels()
        
        if not self.subdivision_candidates:
            print("No large labels found. Run 'Find Oversized Labels' first.")
            return
        
        if labels is None:
            print("No labels available")
            return
        
        # Get subdivision parameters
        seed_distance = self.seed_distance_slider.value()
        seed_threshold_percent = self.seed_threshold_slider.value()
        intensity_weight = self.intensity_weight_slider.value() / 100.0
        
        print("Generating subdivision preview...")
        
        # Create preview showing where subdivisions would occur
        preview_mask = self.create_subdivision_preview(
            labels, self.subdivision_candidates, seed_distance, seed_threshold_percent, intensity_weight
        )
        
        # Add preview layer to napari
        viewer = self.main_widget.viewer
        if 'Subdivision Preview' in [layer.name for layer in viewer.layers]:
            viewer.layers['Subdivision Preview'].data = preview_mask
        else:
            viewer.add_image(
                preview_mask,
                name='Subdivision Preview',
                colormap='plasma',
                opacity=0.6
            )
        
        print("Preview generated - check 'Subdivision Preview' layer")
        print("Use 'Auto Subdivide Large Labels' to apply subdivisions")
    
    def apply_watershed_to_large_labels(self, labels, candidates, seed_distance, seed_threshold_percent, intensity_weight):
        """Apply hybrid watershed subdivision to large labels using distance + intensity"""
        from skimage.segmentation import watershed
        from scipy.ndimage import distance_transform_edt
        from utils.peak_detection import peak_local_maxima
        from skimage.measure import label
        import numpy as np
        
        # Get intensity image for hybrid approach
        intensity_image = self.get_intensity_image()
        if intensity_image is None:
            print("Warning: No intensity image available, falling back to distance-only")
            intensity_weight = 0.0
        
        new_labels = labels.copy()
        next_label_id = np.max(new_labels) + 1
        processed_count = 0
        subdivided_count = 0
        
        for candidate in candidates:
            label_id = candidate['label_id']
            
            # Get mask for this specific label
            label_mask = (labels == label_id)
            
            # Check if label still exists
            if not np.any(label_mask):
                print(f"Warning: Label {label_id} not found in current segmentation, skipping...")
                continue
            
            # Create hybrid elevation map
            elevation = self.create_hybrid_elevation(
                intensity_image, label_mask, intensity_weight
            )
            
            # Calculate distance transform for seed detection
            distance_transform = distance_transform_edt(label_mask)
            
            # Get distance values within the label for threshold calculation
            distance_values = distance_transform[label_mask]
            
            # Check if we have valid distance values
            if len(distance_values) == 0:
                print(f"Warning: No distance values for label {label_id}, skipping...")
                continue
            
            # Calculate threshold based on percentile of distance values
            try:
                threshold_abs = np.percentile(distance_values, seed_threshold_percent)
            except Exception as e:
                print(f"Warning: Could not calculate percentile for label {label_id}: {e}")
                continue
            
            # Skip if threshold is too low
            if threshold_abs <= 0:
                print(f"Warning: Threshold too low ({threshold_abs}) for label {label_id}, skipping...")
                continue
            
            # Find seeds using peak detection on distance transform
            # (Seeds are still based on distance transform, but watershed uses hybrid elevation)
            local_maxima = peak_local_maxima(
                distance_transform,
                min_distance=seed_distance,
                threshold_abs=threshold_abs,
                indices=False
            )
            
            # Create markers for watershed
            markers = label(local_maxima)
            
            # Only proceed if we found multiple seeds
            num_seeds = len(np.unique(markers)) - 1  # Subtract background
            if num_seeds > 1:
                try:
                    # Apply watershed using hybrid elevation map
                    subdivided = watershed(elevation, markers, mask=label_mask)
                    
                    # Remove the original label
                    new_labels[label_mask] = 0
                    
                    # Add subdivided regions with new label IDs
                    subdivisions_added = 0
                    for region_label in np.unique(subdivided)[1:]:  # Skip background
                        region_mask = subdivided == region_label
                        region_size = np.sum(region_mask)
                        
                        # Only keep reasonably sized subdivisions
                        if region_size >= 20:
                            new_labels[region_mask] = next_label_id
                            next_label_id += 1
                            subdivisions_added += 1
                    
                    if subdivisions_added > 0:
                        subdivided_count += 1
                        print(f"  Label {label_id}: subdivided into {subdivisions_added} parts (hybrid: {intensity_weight:.1f} intensity)")
                    else:
                        # Restore original label if no valid subdivisions
                        new_labels[label_mask] = label_id
                        print(f"  Label {label_id}: no valid subdivisions, kept original")
                
                except Exception as e:
                    print(f"Warning: Watershed failed for label {label_id}: {e}")
                    # Restore original label on error
                    new_labels[label_mask] = label_id
            else:
                print(f"  Label {label_id}: only {num_seeds} seed(s) found, no subdivision needed")
            
            processed_count += 1
        
        print(f"Processed {processed_count}/{len(candidates)} labels, subdivided {subdivided_count} labels")
        return new_labels
    
    def create_hybrid_elevation(self, intensity_image, label_mask, intensity_weight):
        """Create hybrid elevation map combining distance transform and intensity"""
        from scipy.ndimage import distance_transform_edt
        import numpy as np
        
        # Shape component (distance transform)
        distance_transform = distance_transform_edt(label_mask)
        
        # Intensity component
        if intensity_image is not None and intensity_weight > 0:
            # Mask intensity to label region
            masked_intensity = intensity_image.copy().astype(np.float32)
            masked_intensity[~label_mask] = 0
            
            # Normalize intensity within the label to [0,1] range
            label_intensity = masked_intensity[label_mask]
            if len(label_intensity) > 0 and np.max(label_intensity) > np.min(label_intensity):
                # Normalize to 0-1 range
                intensity_min = np.min(label_intensity)
                intensity_max = np.max(label_intensity)
                normalized_intensity = np.zeros_like(masked_intensity)
                normalized_intensity[label_mask] = (label_intensity - intensity_min) / (intensity_max - intensity_min)
            else:
                normalized_intensity = np.zeros_like(masked_intensity)
            
            # Normalize distance transform to [0,1] range
            if np.max(distance_transform) > 0:
                normalized_distance = distance_transform / np.max(distance_transform)
            else:
                normalized_distance = distance_transform
            
            # Combine with weights
            distance_weight = 1.0 - intensity_weight
            combined = (normalized_distance * distance_weight + normalized_intensity * intensity_weight)
            
            # Create elevation (negative for watershed - valleys attract water)
            elevation = -combined
        else:
            # Fall back to distance-only
            elevation = -distance_transform
        
        return elevation
    
    def create_subdivision_preview(self, labels, candidates, seed_distance, seed_threshold_percent, intensity_weight):
        """Create preview mask showing where subdivisions would occur"""
        from scipy.ndimage import distance_transform_edt
        from utils.peak_detection import peak_local_maxima
        import numpy as np
        
        preview_mask = np.zeros(labels.shape, dtype=np.uint8)
        
        # Get intensity image for hybrid approach
        intensity_image = self.get_intensity_image()
        if intensity_image is None:
            intensity_weight = 0.0
        
        for i, candidate in enumerate(candidates[:10]):  # Preview first 10
            label_id = candidate['label_id']
            
            # Get mask for this label
            label_mask = (labels == label_id)
            
            # Check if label exists
            if not np.any(label_mask):
                continue
            
            # Calculate distance transform for seed detection
            distance_transform = distance_transform_edt(label_mask)
            
            # Get distance values
            distance_values = distance_transform[label_mask]
            if len(distance_values) == 0:
                continue
            
            # Calculate threshold
            try:
                threshold_abs = np.percentile(distance_values, seed_threshold_percent)
            except:
                continue
            
            if threshold_abs <= 0:
                continue
            
            # Find subdivision seeds (still use distance transform for seed detection)
            local_maxima = peak_local_maxima(
                distance_transform,
                min_distance=seed_distance,
                threshold_abs=threshold_abs,
                indices=False
            )
            
            # Mark seeds in preview
            seeds = local_maxima & label_mask
            if np.sum(seeds) > 1:  # Only if multiple seeds found
                preview_mask[seeds] = min(255, (i + 1) * 25)
        
        return preview_mask