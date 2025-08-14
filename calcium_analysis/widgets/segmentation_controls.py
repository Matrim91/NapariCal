"""
Segmentation Controls Widget - Right column controls
"""

import numpy as np
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                           QSlider, QLabel, QGroupBox)
from qtpy.QtCore import Qt, Signal


class SegmentationControlsWidget(QWidget):
    """
    Widget for segmentation controls (right column)
    """
    
    # Signals
    segmentation_updated = Signal(str)  # stage name
    labels_created = Signal()
    
    def __init__(self, main_widget):
        super().__init__()
        self.main_widget = main_widget
        self.enabled = False
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("Cell Segmentation")
        header.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        layout.addWidget(header)
        
        # 5. Threshold Group
        threshold_group = QGroupBox("5. Threshold")
        threshold_group.setMaximumHeight(60)
        threshold_layout = QVBoxLayout()
        
        self.threshold_button = QPushButton("Apply Threshold")
        self.threshold_button.clicked.connect(self.on_apply_threshold)
        self.threshold_button.setEnabled(False)
        threshold_layout.addWidget(self.threshold_button)
        
        threshold_group.setLayout(threshold_layout)
        layout.addWidget(threshold_group)
        
        # 6. Size Filter Group
        size_filter_group = QGroupBox("6. Size Filter")
        size_filter_group.setMaximumHeight(100)
        size_filter_layout = QVBoxLayout()
        
        self.size_filter_button = QPushButton("Apply Size Filter")
        self.size_filter_button.clicked.connect(self.on_apply_size_filter)
        self.size_filter_button.setEnabled(False)
        size_filter_layout.addWidget(self.size_filter_button)
        
        # Size filter controls
        self.binary_size_label = QLabel("Min Size: 50 px")
        size_filter_layout.addWidget(self.binary_size_label)
        
        self.binary_size_slider = QSlider(Qt.Horizontal)
        self.binary_size_slider.setRange(10, 300)
        self.binary_size_slider.setValue(50)
        self.binary_size_slider.setEnabled(False)
        self.binary_size_slider.valueChanged.connect(self.update_size_filter_label)
        size_filter_layout.addWidget(self.binary_size_slider)
        
        size_filter_group.setLayout(size_filter_layout)
        layout.addWidget(size_filter_group)
        
        # 7. Morphology Group
        morphology_group = QGroupBox("7. Morphology")
        morphology_group.setMaximumHeight(60)
        morphology_layout = QVBoxLayout()
        
        self.morphology_button = QPushButton("Apply Morphology")
        self.morphology_button.clicked.connect(self.on_apply_morphology)
        self.morphology_button.setEnabled(False)
        morphology_layout.addWidget(self.morphology_button)
        
        morphology_group.setLayout(morphology_layout)
        layout.addWidget(morphology_group)
        
        # 8. Create Labels Group
        labels_group = QGroupBox("8. Create Labels")
        labels_group.setMaximumHeight(60)
        labels_layout = QVBoxLayout()
        
        self.labels_button = QPushButton("Create Labels")
        self.labels_button.clicked.connect(self.on_create_labels)
        self.labels_button.setEnabled(False)
        labels_layout.addWidget(self.labels_button)
        
        labels_group.setLayout(labels_layout)
        layout.addWidget(labels_group)
        
        # 9. Watershed Group (CLEANED UP)
        watershed_group = QGroupBox("9. Watershed Subdivision")
        watershed_group.setMaximumHeight(140)
        watershed_layout = QVBoxLayout()
        
        # Watershed buttons
        watershed_buttons = QHBoxLayout()
        
        self.watershed_button = QPushButton("Apply Watershed")
        self.watershed_button.clicked.connect(self.on_apply_watershed)
        self.watershed_button.setEnabled(False)
        watershed_buttons.addWidget(self.watershed_button)
        
        self.mlgoc_button = QPushButton("MLGOC Multi-Layer")
        self.mlgoc_button.clicked.connect(self.on_apply_mlgoc)
        self.mlgoc_button.setEnabled(False)
        self.mlgoc_button.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        watershed_buttons.addWidget(self.mlgoc_button)
        
        watershed_layout.addLayout(watershed_buttons)
        
        # Watershed controls in a compact layout
        watershed_controls = QHBoxLayout()
        
        # Distance control
        distance_layout = QVBoxLayout()
        self.distance_label = QLabel("Min Distance: 15")
        distance_layout.addWidget(self.distance_label)
        self.min_distance = QSlider(Qt.Horizontal)
        self.min_distance.setRange(1, 50)
        self.min_distance.setValue(15)
        self.min_distance.valueChanged.connect(lambda v: self.distance_label.setText(f"Min Distance: {v}"))
        distance_layout.addWidget(self.min_distance)
        watershed_controls.addLayout(distance_layout)
        
        # MLGOC layers control
        layers_layout = QVBoxLayout()
        self.layers_label = QLabel("Layers: 3")
        layers_layout.addWidget(self.layers_label)
        self.num_layers = QSlider(Qt.Horizontal)
        self.num_layers.setRange(1, 5)
        self.num_layers.setValue(3)
        self.num_layers.valueChanged.connect(lambda v: self.layers_label.setText(f"Layers: {v}"))
        layers_layout.addWidget(self.num_layers)
        watershed_controls.addLayout(layers_layout)
        
        # Preferred radius control  
        radius_layout = QVBoxLayout()
        self.preferred_radius_label = QLabel("Cell Radius: 17px")
        radius_layout.addWidget(self.preferred_radius_label)
        self.preferred_radius = QSlider(Qt.Horizontal)
        self.preferred_radius.setRange(5, 40)
        self.preferred_radius.setValue(17)
        self.preferred_radius.valueChanged.connect(lambda v: self.preferred_radius_label.setText(f"Cell Radius: {v}px"))
        radius_layout.addWidget(self.preferred_radius)
        watershed_controls.addLayout(radius_layout)
        
        watershed_layout.addLayout(watershed_controls)
        watershed_group.setLayout(watershed_layout)
        layout.addWidget(watershed_group)
        
        # 10. Analysis View Group
        analysis_group = QGroupBox("10. Analysis View")
        analysis_group.setMaximumHeight(60)
        analysis_layout = QVBoxLayout()
        
        self.analysis_button = QPushButton("Setup Analysis View")
        self.analysis_button.clicked.connect(self.on_setup_analysis)
        self.analysis_button.setEnabled(False)
        analysis_layout.addWidget(self.analysis_button)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
        # 11. Label Size Filter Group
        label_filter_group = QGroupBox("11. Label Size Filter")
        label_filter_group.setMaximumHeight(180)
        label_filter_layout = QVBoxLayout()
        
        # Manual size filter controls
        manual_filter_layout = QHBoxLayout()
        
        self.label_size_filter_button = QPushButton("Apply Label Filter")
        self.label_size_filter_button.clicked.connect(self.on_apply_label_filter)
        self.label_size_filter_button.setEnabled(False)
        manual_filter_layout.addWidget(self.label_size_filter_button)
        
        self.auto_filter_button = QPushButton("Auto Remove Small")
        self.auto_filter_button.clicked.connect(self.on_auto_filter)
        self.auto_filter_button.setEnabled(False)
        manual_filter_layout.addWidget(self.auto_filter_button)
        
        label_filter_layout.addLayout(manual_filter_layout)
        
        # Size filter slider
        self.label_size_label = QLabel("Min Label Size: 50 px")
        label_filter_layout.addWidget(self.label_size_label)
        
        self.label_size_slider = QSlider(Qt.Horizontal)
        self.label_size_slider.setRange(10, 500)
        self.label_size_slider.setValue(50)
        self.label_size_slider.setEnabled(False)
        self.label_size_slider.valueChanged.connect(self.update_label_size_filter)
        label_filter_layout.addWidget(self.label_size_slider)
        
        # Standard deviation multiplier controls
        std_controls = QHBoxLayout()
        
        std_label_layout = QVBoxLayout()
        self.std_multiplier_label = QLabel("Std Dev Multiplier: 3.0")
        std_label_layout.addWidget(self.std_multiplier_label)
        std_controls.addLayout(std_label_layout)
        
        slider_layout = QVBoxLayout()
        self.std_multiplier_slider = QSlider(Qt.Horizontal)
        self.std_multiplier_slider.setRange(10, 50)  # 1.0 to 5.0 in steps of 0.1
        self.std_multiplier_slider.setValue(30)  # Default 3.0
        self.std_multiplier_slider.setEnabled(False)
        self.std_multiplier_slider.valueChanged.connect(self.update_std_multiplier_label)
        slider_layout.addWidget(self.std_multiplier_slider)
        std_controls.addLayout(slider_layout)
        
        label_filter_layout.addLayout(std_controls)
        
        # Status label for auto filter
        self.auto_filter_status = QLabel("Auto filter: Not applied")
        self.auto_filter_status.setStyleSheet("color: #666; font-size: 9px;")
        label_filter_layout.addWidget(self.auto_filter_status)
        
        label_filter_group.setLayout(label_filter_layout)
        layout.addWidget(label_filter_group)
        
        # Manual Editing Group
        editing_group = QGroupBox("Manual Editing")
        editing_group.setMaximumHeight(80)
        editing_layout = QVBoxLayout()
        
        edit_buttons = QHBoxLayout()
        
        self.update_labels_button = QPushButton("Update Labels")
        self.update_labels_button.clicked.connect(self.on_update_labels)
        self.update_labels_button.setEnabled(False)
        edit_buttons.addWidget(self.update_labels_button)
        
        self.randomize_colors_button = QPushButton("Randomize Colors")
        self.randomize_colors_button.clicked.connect(self.on_randomize_colors)
        self.randomize_colors_button.setEnabled(False)
        edit_buttons.addWidget(self.randomize_colors_button)
        
        editing_layout.addLayout(edit_buttons)
        editing_group.setLayout(editing_layout)
        layout.addWidget(editing_group)
        
        # Analysis Window Button
        analysis_window_button = QPushButton("Open Cell Analysis Window")
        analysis_window_button.clicked.connect(self.on_open_analysis_window)
        layout.addWidget(analysis_window_button)
        
        # Preset Group
        preset_group = QGroupBox("Presets")
        preset_group.setMaximumHeight(60)
        preset_layout = QVBoxLayout()
        
        self.endothelial_preset_button = QPushButton("Endothelial Cell Preset")
        self.endothelial_preset_button.clicked.connect(self.apply_endothelial_preset)
        self.endothelial_preset_button.setEnabled(True)
        preset_layout.addWidget(self.endothelial_preset_button)
        
        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)
        
        self.setLayout(layout)
    
    def get_parameters(self) -> dict:
        """Get current segmentation parameters"""
        return {
            'size_filter_params': {
                'min_size': self.binary_size_slider.value()
            },
            'morphology_params': {
                'iterations': 3
            },
            'watershed_params': {
                'min_distance': self.min_distance.value()
            },
            'mlgoc_params': {
                'num_layers': self.num_layers.value(),
                'preferred_radius': self.preferred_radius.value(),
                'min_distance': self.min_distance.value()
            },
            'label_filter_params': {
                'min_size': self.label_size_slider.value(),
                'std_multiplier': self.std_multiplier_slider.value() / 10.0
            }
        }
    
    def set_parameters(self, params: dict):
        """Set parameter values from dictionary"""
        if 'size_filter_params' in params:
            size_params = params['size_filter_params']
            self.binary_size_slider.setValue(int(size_params.get('min_size', 50)))
        
        if 'watershed_params' in params:
            watershed_params = params['watershed_params']
            self.min_distance.setValue(int(watershed_params.get('min_distance', 15)))
        
        if 'mlgoc_params' in params:
            mlgoc_params = params['mlgoc_params']
            self.num_layers.setValue(int(mlgoc_params.get('num_layers', 3)))
            self.preferred_radius.setValue(int(mlgoc_params.get('preferred_radius', 17)))
        
        if 'label_filter_params' in params:
            label_params = params['label_filter_params']
            self.label_size_slider.setValue(int(label_params.get('min_size', 50)))
            std_mult = label_params.get('std_multiplier', 3.0)
            self.std_multiplier_slider.setValue(int(std_mult * 10))
    
    def set_enabled(self, enabled: bool):
        """Enable or disable all controls"""
        self.enabled = enabled
        self.threshold_button.setEnabled(enabled)
        self.size_filter_button.setEnabled(enabled)
        self.morphology_button.setEnabled(enabled)
        self.labels_button.setEnabled(enabled)
        self.watershed_button.setEnabled(enabled)
        self.mlgoc_button.setEnabled(enabled)
        self.analysis_button.setEnabled(enabled)
        self.update_labels_button.setEnabled(enabled)
        self.randomize_colors_button.setEnabled(enabled)
    
    def enable_label_controls(self, enabled: bool):
        """Enable label-specific controls after labels are created"""
        self.label_size_filter_button.setEnabled(enabled)
        self.auto_filter_button.setEnabled(enabled)
        self.label_size_slider.setEnabled(enabled)
        self.std_multiplier_slider.setEnabled(enabled)
    
    def update_size_filter_label(self, value):
        """Update size filter label"""
        self.binary_size_label.setText(f"Min Size: {value} px")
    
    def update_label_size_filter(self, value):
        """Update label size filter in real-time"""
        from processing.segmentation import calculate_auto_size_threshold
        from utils.morphology_utils import remove_small_objects_from_labels
        
        if hasattr(self.main_widget, 'state') and self.main_widget.state['segmentation']['final_labels'] is not None:
            current_labels = self.main_widget.state['segmentation']['final_labels']
            
            # Apply filter
            filtered_labels, removed_count = remove_small_objects_from_labels(current_labels, value)
            
            # Update display
            remaining_count = len(np.unique(filtered_labels)) - 1
            self.label_size_label.setText(f"Min Label Size: {value} px ({remaining_count} labels)")
            
            # Update napari
            self.main_widget.update_napari_layer('Cell Labels', filtered_labels, 'labels')
            
            # Update state
            self.main_widget.state['segmentation']['final_labels'] = filtered_labels
    
    def update_std_multiplier_label(self, value):
        """Update standard deviation multiplier label"""
        multiplier = value / 10.0
        self.std_multiplier_label.setText(f"Std Dev Multiplier: {multiplier:.1f}")
    
    # Event handlers
    def on_apply_threshold(self):
        """Handle apply threshold button click"""
        self.segmentation_updated.emit('threshold')
    
    def on_apply_size_filter(self):
        """Handle apply size filter button click"""
        self.binary_size_slider.setEnabled(True)
        self.segmentation_updated.emit('size_filter')
    
    def on_apply_morphology(self):
        """Handle apply morphology button click"""
        self.segmentation_updated.emit('morphology')
        self.labels_button.setEnabled(True)
    
    def on_create_labels(self):
        """Handle create labels button click"""
        self.segmentation_updated.emit('create_labels')
        self.watershed_button.setEnabled(True)
        self.mlgoc_button.setEnabled(True)
        self.enable_label_controls(True)
        self.labels_created.emit()
    
    def on_apply_watershed(self):
        """Handle apply watershed button click"""
        self.segmentation_updated.emit('watershed')
    
    def on_apply_mlgoc(self):
        """Handle apply MLGOC multi-layer watershed button click"""
        self.segmentation_updated.emit('mlgoc')
    
    def on_setup_analysis(self):
        """Handle setup analysis button click"""
        self.segmentation_updated.emit('analysis_view')
    
    def on_apply_label_filter(self):
        """Handle apply label filter button click"""
        value = self.label_size_slider.value()
        self.update_label_size_filter(value)
        print(f"Label size filter applied: minimum {value} pixels")
    
    def on_auto_filter(self):
        """Handle auto filter button click"""
        from processing.segmentation import calculate_auto_size_threshold
        from utils.morphology_utils import remove_small_objects_from_labels
        
        if hasattr(self.main_widget, 'state') and self.main_widget.state['segmentation']['final_labels'] is not None:
            current_labels = self.main_widget.state['segmentation']['final_labels']
            std_multiplier = self.std_multiplier_slider.value() / 10.0
            
            # Calculate threshold
            threshold, statistics = calculate_auto_size_threshold(current_labels, std_multiplier)
            
            # Apply filter
            filtered_labels, removed_count = remove_small_objects_from_labels(current_labels, threshold)
            
            # Update display
            remaining_count = len(np.unique(filtered_labels)) - 1
            
            # Update UI
            self.label_size_slider.setValue(int(threshold))
            self.label_size_label.setText(f"Min Label Size: {int(threshold)} px ({remaining_count} labels)")
            self.auto_filter_status.setText(f"Auto filter: {removed_count} removed (< {threshold:.0f}px)")
            
            # Update napari and state
            self.main_widget.update_napari_layer('Cell Labels', filtered_labels, 'labels')
            self.main_widget.state['segmentation']['final_labels'] = filtered_labels
            
            # Print statistics
            print(f"Auto filter statistics:")
            print(f"  Median size: {statistics['median_size']:.1f} pixels")
            print(f"  Standard deviation: {statistics['std_size']:.1f} pixels")
            print(f"  Multiplier: {statistics['std_multiplier']:.1f}")
            print(f"  Threshold: {statistics['median_size']:.1f} - {statistics['std_multiplier']:.1f}×{statistics['std_size']:.1f} = {threshold:.1f}")
            print(f"  Removed: {removed_count} labels, Remaining: {remaining_count} labels")
    
    def on_update_labels(self):
        """Handle update labels button click"""
        print("Update labels from manual edits")
    
    def on_randomize_colors(self):
        """Handle randomize colors button click"""
        if 'Cell Labels' in [layer.name for layer in self.main_widget.viewer.layers]:
            labels_layer = self.main_widget.viewer.layers['Cell Labels']
            
            # Force napari to regenerate colors
            current_opacity = labels_layer.opacity
            labels_layer.opacity = current_opacity - 0.01
            labels_layer.opacity = current_opacity
            
            print("Label colors randomized")
    
    def on_open_analysis_window(self):
        """Handle open analysis window button click"""
        self.main_widget.open_analysis_window()
    
    def apply_endothelial_preset(self):
        """Apply optimized preset for endothelial cells"""
        # Segmentation settings
        self.binary_size_slider.setValue(30)
        self.min_distance.setValue(8)
        
        # MLGOC settings
        self.num_layers.setValue(2)
        self.preferred_radius.setValue(12)
        
        # Label size filter settings
        self.label_size_slider.setValue(20)
        self.std_multiplier_slider.setValue(25)
        
        # Update labels
        self.binary_size_label.setText(f"Min Size: {self.binary_size_slider.value()} px")
        self.distance_label.setText(f"Min Distance: {self.min_distance.value()}")
        self.layers_label.setText(f"Layers: {self.num_layers.value()}")
        self.preferred_radius_label.setText(f"Cell Radius: {self.preferred_radius.value()}px")
        self.label_size_label.setText(f"Min Label Size: {self.label_size_slider.value()} px")
        self.std_multiplier_label.setText(f"Std Dev Multiplier: {self.std_multiplier_slider.value()/10.0:.1f}")
        
        print("Applied endothelial cell preset:")
        print("  Binary Size Filter: 30px")
        print("  Watershed Distance: 8px")
        print("  MLGOC Layers: 2")
        print("  Preferred Radius: 12px") 
        print("  Label Size Filter: 20px")
        print("  Std Dev Multiplier: 2.5")