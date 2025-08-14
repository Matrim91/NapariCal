"""
Exclusion Controls Widget - ROI exclusion and data export controls
"""

import numpy as np
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                           QGroupBox, QCheckBox, QFileDialog)
from qtpy.QtCore import Qt, Signal


class ExclusionControlsWidget(QWidget):
    """
    Widget for exclusion region controls and data export/import
    """
    
    # Signals
    exclusion_updated = Signal(object)  # exclusion_mask
    data_exported = Signal(str)  # file_path
    data_imported = Signal(dict)  # imported_data
    
    def __init__(self, main_widget):
        super().__init__()
        self.main_widget = main_widget
        self.enabled = False
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        
        # ROI Exclusion Group
        roi_group = QGroupBox("Exclusion Regions")
        roi_group.setMaximumHeight(120)
        roi_layout = QVBoxLayout()
        
        # Exclusion buttons
        roi_buttons = QHBoxLayout()
        
        self.create_exclusion_button = QPushButton("Create Exclusion Layer")
        self.create_exclusion_button.clicked.connect(self.on_create_exclusion_layer)
        self.create_exclusion_button.setEnabled(False)
        roi_buttons.addWidget(self.create_exclusion_button)
        
        self.show_exclusion_button = QPushButton("Show/Hide Exclusions")
        self.show_exclusion_button.clicked.connect(self.on_toggle_exclusion_visibility)
        self.show_exclusion_button.setEnabled(False)
        roi_buttons.addWidget(self.show_exclusion_button)
        
        self.apply_exclusion_button = QPushButton("Apply Exclusions")
        self.apply_exclusion_button.clicked.connect(self.on_apply_exclusions)
        self.apply_exclusion_button.setEnabled(False)
        roi_buttons.addWidget(self.apply_exclusion_button)
        
        roi_layout.addLayout(roi_buttons)
        
        # Save/Load buttons
        export_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Save Labels")
        self.save_button.clicked.connect(self.on_save_data)
        self.save_button.setEnabled(False)
        export_layout.addWidget(self.save_button)
        
        self.load_button = QPushButton("Load Labels")
        self.load_button.clicked.connect(self.on_load_data)
        self.load_button.setEnabled(True)
        export_layout.addWidget(self.load_button)
        
        roi_layout.addLayout(export_layout)
        
        # Checkbox for touching cells
        self.exclude_touching_checkbox = QCheckBox("Exclude cells touching exclusion regions")
        self.exclude_touching_checkbox.setChecked(False)
        self.exclude_touching_checkbox.stateChanged.connect(self.on_toggle_exclude_touching)
        roi_layout.addWidget(self.exclude_touching_checkbox)
        
        roi_group.setLayout(roi_layout)
        layout.addWidget(roi_group)
        
        self.setLayout(layout)
    
    def set_enabled(self, enabled: bool):
        """Enable or disable controls"""
        self.enabled = enabled
        self.create_exclusion_button.setEnabled(enabled)
        self.show_exclusion_button.setEnabled(enabled)
        self.save_button.setEnabled(enabled)
    
    def get_exclusion_mask(self):
        """Get current exclusion mask"""
        viewer = self.main_widget.viewer
        if 'Exclusion Regions' in [layer.name for layer in viewer.layers]:
            exclusion_layer = viewer.layers['Exclusion Regions']
            return exclusion_layer.data > 0
        return None
    
    def get_exclude_touching(self) -> bool:
        """Get exclude touching setting"""
        return self.exclude_touching_checkbox.isChecked()
    
    # Event handlers
    def on_create_exclusion_layer(self):
        """Handle create exclusion layer button click"""
        # Get image shape from main widget state
        if hasattr(self.main_widget, 'state') and self.main_widget.state['images']['original'] is not None:
            image_shape = self.main_widget.state['images']['original'].shape[:2]
        else:
            print("Load image first")
            return
        
        viewer = self.main_widget.viewer
        
        # Check if exclusion layer already exists
        if 'Exclusion Regions' in [layer.name for layer in viewer.layers]:
            # Layer exists - just make it visible and select it
            exclusion_layer = viewer.layers['Exclusion Regions']
            exclusion_layer.visible = True
            viewer.layers.selection.active = exclusion_layer
            print("Exclusion layer is now visible and selected for editing")
        else:
            # Create new exclusion layer
            empty_exclusion = np.zeros(image_shape, dtype=np.uint8)
            
            # Add as labels layer for easy painting
            exclusion_layer = viewer.add_labels(empty_exclusion, name='Exclusion Regions')
            
            # Set proper red color for label 1 (painted areas)
            exclusion_layer.color = {
                0: [0, 0, 0, 0],     # Transparent background
                1: [1, 0, 0, 1],     # Bright red for exclusions
            }
            
            # Make sure it's selected and ready for painting
            viewer.layers.selection.active = exclusion_layer
            exclusion_layer.mode = 'paint'
            exclusion_layer.selected_label = 1  # Paint with red
            
            print("Exclusion layer created in red. Paint areas to exclude.")
        
        self.apply_exclusion_button.setEnabled(True)
    
    def on_toggle_exclusion_visibility(self):
        """Handle toggle exclusion visibility button click"""
        viewer = self.main_widget.viewer
        if 'Exclusion Regions' in [layer.name for layer in viewer.layers]:
            exclusion_layer = viewer.layers['Exclusion Regions']
            exclusion_layer.visible = not exclusion_layer.visible
            status = "visible" if exclusion_layer.visible else "hidden"
            print(f"Exclusion regions are now {status}")
        else:
            print("No exclusion layer found - create one first")
    
    def on_toggle_exclude_touching(self, state):
        """Handle exclude touching checkbox change"""
        exclude_touching = state == 2  # 2 = checked
        self.main_widget.state['exclusions']['exclude_touching'] = exclude_touching
        exclude_mode = "cells touching exclusions" if exclude_touching else "cells within exclusions"
        print(f"Exclusion mode: {exclude_mode}")
    
    def on_apply_exclusions(self):
        """Handle apply exclusions button click"""
        from skimage import morphology
        from utils.morphology_utils import clean_fragmented_labels
        from scipy.ndimage import binary_dilation
        
        viewer = self.main_widget.viewer
        
        if 'Exclusion Regions' not in [layer.name for layer in viewer.layers]:
            print("No exclusion regions found. Create exclusion layer first.")
            return
        
        if 'Cell Labels' not in [layer.name for layer in viewer.layers]:
            print("No cell labels found. Run segmentation first.")
            return
        
        # Get exclusion mask
        exclusion_layer = viewer.layers['Exclusion Regions']
        exclusion_mask = exclusion_layer.data > 0
        
        if not np.any(exclusion_mask):
            print("No exclusion regions painted")
            return
        
        # Get cell labels
        labels_layer = viewer.layers['Cell Labels']
        cell_labels = labels_layer.data.copy()
        
        # Apply exclusions
        original_count = len(np.unique(cell_labels)) - 1
        exclude_touching = self.get_exclude_touching()
        
        if exclude_touching:
            # Remove cells that touch exclusion regions (more aggressive)
            expanded_exclusion = binary_dilation(exclusion_mask, structure=morphology.disk(2))
            
            # Find labels that overlap with expanded exclusion
            touching_labels = np.unique(cell_labels[expanded_exclusion])
            touching_labels = touching_labels[touching_labels > 0]  # Remove background
            
            # Remove these labels
            for label_id in touching_labels:
                cell_labels[cell_labels == label_id] = 0
            
            exclusion_type = "touching"
        else:
            # Only remove cells within exclusion regions (less aggressive)
            cell_labels[exclusion_mask] = 0
            exclusion_type = "within"
        
        # Clean up fragmented labels
        cleaned_labels = clean_fragmented_labels(cell_labels)
        
        # Update the labels layer and state
        labels_layer.data = cleaned_labels
        self.main_widget.state['segmentation']['final_labels'] = cleaned_labels
        
        # Update exclusion mask in state
        self.main_widget.state['exclusions']['mask'] = exclusion_mask
        self.exclusion_updated.emit(exclusion_mask)
        
        # Count remaining cells
        final_count = len(np.unique(cleaned_labels)) - 1
        excluded_count = original_count - final_count
        
        print(f"Exclusions applied: {excluded_count} cells removed ({exclusion_type} exclusions)")
        print(f"Remaining cells: {final_count}")
    
    def on_save_data(self):
        """Handle save data button click"""
        from io.data_export import DataExporter
        
        # Get file path
        file_info = self.main_widget.state['file_info']
        default_filename = f"{file_info['file_name']}_rois.h5"
        default_path = f"{file_info['cwd']}/{default_filename}"
        
        filename, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Labels and Exclusions", 
            default_path,
            "HDF5 files (*.h5 *.hdf5)"
        )
        
        if not filename:
            return
        
        # Prepare data for export
        data_dict = {
            'final_labels': self.main_widget.state['segmentation']['final_labels'],
            'original_image': self.main_widget.state['images']['original'],
            'exclusion_mask': self.get_exclusion_mask(),
            'parameters': self.get_all_parameters(),
            'file_info': self.main_widget.state['file_info']
        }
        
        # Export data
        try:
            exporter = DataExporter()
            success = exporter.export_segmentation_data(filename, data_dict)
            
            if success:
                print(f"Successfully saved to: {filename}")
                self.data_exported.emit(filename)
            
        except Exception as e:
            print(f"Save error: {str(e)}")
    
    def on_load_data(self):
        """Handle load data button click"""
        from io.data_export import DataExporter
        
        # Get file to load
        filename, _ = QFileDialog.getOpenFileName(
            self, 
            "Load Labels and Exclusions", 
            "",
            "HDF5 files (*.h5 *.hdf5)"
        )
        
        if not filename:
            return
        
        try:
            exporter = DataExporter()
            imported_data = exporter.import_segmentation_data(filename)
            
            # Update main widget state
            if 'final_labels' in imported_data:
                self.main_widget.state['segmentation']['final_labels'] = imported_data['final_labels']
                self.main_widget.update_napari_layer('Cell Labels', imported_data['final_labels'], 'labels')
                num_cells = len(np.unique(imported_data['final_labels'])) - 1
                print(f"Loaded cell labels: {num_cells} cells")
            
            if 'exclusion_mask' in imported_data:
                exclusion_data = imported_data['exclusion_mask']
                self.main_widget.state['exclusions']['mask'] = exclusion_data
                
                # Add to napari or update existing
                viewer = self.main_widget.viewer
                if 'Exclusion Regions' in [layer.name for layer in viewer.layers]:
                    viewer.layers['Exclusion Regions'].data = exclusion_data
                else:
                    exclusion_layer = viewer.add_labels(exclusion_data, name='Exclusion Regions')
                    exclusion_layer.color = {
                        0: [0, 0, 0, 0],     # Transparent background
                        1: [1, 0, 0, 1],     # Bright red for exclusions
                    }
                
                excluded_pixels = np.sum(exclusion_data > 0)
                if excluded_pixels > 0:
                    print(f"Loaded exclusion regions: {excluded_pixels} pixels excluded")
                    self.apply_exclusion_button.setEnabled(True)
                else:
                    print("No exclusion regions in file")
            
            if 'original_image' in imported_data:
                original_image = imported_data['original_image']
                self.main_widget.state['images']['original'] = original_image
                self.main_widget.update_napari_layer('Original', original_image, 'image')
                
                # Set grayscale version
                if len(original_image.shape) == 3:
                    self.main_widget.state['images']['grayscale'] = np.mean(original_image, axis=2).astype(np.uint8)
                else:
                    self.main_widget.state['images']['grayscale'] = original_image.copy()
            
            # Load parameters if available
            if 'parameters' in imported_data:
                self.set_all_parameters(imported_data['parameters'])
                print("Parameters restored from file")
            
            # Show metadata
            if 'metadata' in imported_data:
                metadata = imported_data['metadata']
                export_date = metadata.get('export_date', 'Unknown')
                source_file = metadata.get('source_file', 'Unknown')
                print(f"File info: Exported {export_date}, Source: {source_file}")
            
            # Enable controls
            self.main_widget.image_enhancement.set_enabled(True)
            self.main_widget.segmentation_controls.set_enabled(True)
            self.set_enabled(True)
            
            # Set analysis ready flag
            self.main_widget.state['analysis']['labels_ready'] = True
            
            print(f"Successfully loaded from: {filename}")
            self.data_imported.emit(imported_data)
            
        except Exception as e:
            print(f"Load error: {str(e)}")
    
    def get_all_parameters(self) -> dict:
        """Get all current parameters from widgets"""
        params = {}
        
        # Get image enhancement parameters
        if hasattr(self.main_widget, 'image_enhancement'):
            params.update(self.main_widget.image_enhancement.get_parameters())
        
        # Get segmentation parameters
        if hasattr(self.main_widget, 'segmentation_controls'):
            params.update(self.main_widget.segmentation_controls.get_parameters())
        
        # Get exclusion parameters
        params['exclusion_params'] = {
            'exclude_touching': self.get_exclude_touching()
        }
        
        return params
    
    def set_all_parameters(self, params: dict):
        """Set all parameters in widgets"""
        try:
            # Set image enhancement parameters
            if hasattr(self.main_widget, 'image_enhancement'):
                self.main_widget.image_enhancement.set_parameters(params)
            
            # Set segmentation parameters
            if hasattr(self.main_widget, 'segmentation_controls'):
                self.main_widget.segmentation_controls.set_parameters(params)
            
            # Set exclusion parameters
            if 'exclusion_params' in params:
                exclusion_params = params['exclusion_params']
                exclude_touching = exclusion_params.get('exclude_touching', False)
                self.exclude_touching_checkbox.setChecked(exclude_touching)
                self.main_widget.state['exclusions']['exclude_touching'] = exclude_touching
        
        except Exception as e:
            print(f"Warning: Could not restore some parameters: {e}")