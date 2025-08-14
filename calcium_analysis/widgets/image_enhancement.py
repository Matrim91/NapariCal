"""
Image Enhancement Widget - Left column controls
"""

from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                           QSlider, QLabel, QGroupBox)
from qtpy.QtCore import Qt, Signal


class ImageEnhancementWidget(QWidget):
    """
    Widget for image enhancement controls (left column)
    """
    
    # Signals
    tiff_loaded = Signal()
    processing_updated = Signal(str)  # stage name
    
    def __init__(self, main_widget):
        super().__init__()
        self.main_widget = main_widget
        self.enabled = False
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("Image Enhancement")
        header.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        layout.addWidget(header)
        
        # 1. Load TIFF Group
        load_group = QGroupBox("1. Load TIFF")
        load_group.setMaximumHeight(60)
        load_layout = QVBoxLayout()
        
        self.load_button = QPushButton("Load TIFF")
        self.load_button.clicked.connect(self.on_load_tiff)
        load_layout.addWidget(self.load_button)
        
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)
        
        # 2. CLAHE Group
        clahe_group = QGroupBox("2. CLAHE")
        clahe_group.setMaximumHeight(120)
        clahe_layout = QVBoxLayout()
        
        self.clahe_button = QPushButton("Apply CLAHE")
        self.clahe_button.clicked.connect(self.on_apply_clahe)
        self.clahe_button.setEnabled(False)
        clahe_layout.addWidget(self.clahe_button)
        
        # CLAHE Controls
        controls_layout = QHBoxLayout()
        
        clip_layout = QVBoxLayout()
        self.clip_label = QLabel("Clip: 1")
        clip_layout.addWidget(self.clip_label)
        self.clip_limit = QSlider(Qt.Horizontal)
        self.clip_limit.setRange(1, 10)
        self.clip_limit.setValue(1)
        self.clip_limit.valueChanged.connect(lambda v: self.clip_label.setText(f"Clip: {v}"))
        clip_layout.addWidget(self.clip_limit)
        controls_layout.addLayout(clip_layout)
        
        grid_layout = QVBoxLayout()
        self.grid_label = QLabel("Grid: 30")
        grid_layout.addWidget(self.grid_label)
        self.grid_size = QSlider(Qt.Horizontal)
        self.grid_size.setRange(8, 64)
        self.grid_size.setValue(30)
        self.grid_size.valueChanged.connect(lambda v: self.grid_label.setText(f"Grid: {v}"))
        grid_layout.addWidget(self.grid_size)
        controls_layout.addLayout(grid_layout)
        
        clahe_layout.addLayout(controls_layout)
        clahe_group.setLayout(clahe_layout)
        layout.addWidget(clahe_group)



        # 3. Top hat Group
        tophat__group = QGroupBox("3. Top hat")
        tophat__group.setMaximumHeight(120)
        tophat__layout = QVBoxLayout()
        
        self.tophat_button = QPushButton("Apply Tophat")
        self.tophat_button.clicked.connect(self.on_apply_tophat)
        self.tophat_button.setEnabled(False)
        tophat__layout.addWidget(self.tophat_button)
        
        # Tophat Controls
        tophat_controls = QHBoxLayout()
        tophat__layout_slider_layout = QVBoxLayout()
        self.tophat__layout_slider_label = QLabel("Disk: 25")
        tophat__layout_slider_layout.addWidget(self.tophat__layout_slider_label)
        self.tophat_size_slider = QSlider(Qt.Horizontal)
        self.tophat_size_slider.setRange(50, 300)
        self.tophat_size_slider.setValue(120)
        self.tophat_size_slider.valueChanged.connect(lambda v: self.tophat__layout_slider_label.setText(f"α: {v}%"))
        tophat__layout_slider_layout.addWidget(self.tophat_size_slider)
        tophat_controls.addLayout(tophat__layout_slider_layout)
        
        
        tophat__layout.addLayout(tophat_controls)
        tophat__group.setLayout(tophat__layout)
        layout.addWidget(tophat__group)







        
        
        # 3. Contrast Group
        contrast_group = QGroupBox("3. Contrast")
        contrast_group.setMaximumHeight(120)
        contrast_layout = QVBoxLayout()
        
        self.contrast_button = QPushButton("Apply Contrast")
        self.contrast_button.clicked.connect(self.on_apply_contrast)
        self.contrast_button.setEnabled(False)
        contrast_layout.addWidget(self.contrast_button)
        
        # Contrast Controls
        contrast_controls = QHBoxLayout()
        
        alpha_layout = QVBoxLayout()
        self.alpha_label = QLabel("α: 120%")
        alpha_layout.addWidget(self.alpha_label)
        self.alpha = QSlider(Qt.Horizontal)
        self.alpha.setRange(50, 300)
        self.alpha.setValue(120)
        self.alpha.valueChanged.connect(lambda v: self.alpha_label.setText(f"α: {v}%"))
        alpha_layout.addWidget(self.alpha)
        contrast_controls.addLayout(alpha_layout)
        
        beta_layout = QVBoxLayout()
        self.beta_label = QLabel("β: 2")
        beta_layout.addWidget(self.beta_label)
        self.beta = QSlider(Qt.Horizontal)
        self.beta.setRange(-50, 50)
        self.beta.setValue(2)
        self.beta.valueChanged.connect(lambda v: self.beta_label.setText(f"β: {v}"))
        beta_layout.addWidget(self.beta)
        contrast_controls.addLayout(beta_layout)
        
        contrast_layout.addLayout(contrast_controls)
        contrast_group.setLayout(contrast_layout)
        layout.addWidget(contrast_group)
        
        # 4. Filter Group
        filter_group = QGroupBox("4. Bilateral Filter")
        filter_group.setMaximumHeight(140)
        filter_layout = QVBoxLayout()
        
        self.filter_button = QPushButton("Apply Filter")
        self.filter_button.clicked.connect(self.on_apply_filter)
        self.filter_button.setEnabled(False)
        filter_layout.addWidget(self.filter_button)
        
        # Filter Controls
        self.filter_label = QLabel("Size: 5")
        filter_layout.addWidget(self.filter_label)
        self.filter_size = QSlider(Qt.Horizontal)
        self.filter_size.setRange(3, 15)
        self.filter_size.setValue(5)
        self.filter_size.valueChanged.connect(lambda v: self.filter_label.setText(f"Size: {v}"))
        filter_layout.addWidget(self.filter_size)
        
        sigma_controls = QHBoxLayout()
        
        sigma_color_layout = QVBoxLayout()
        self.sigma_color_label = QLabel("σ_c: 1000")
        sigma_color_layout.addWidget(self.sigma_color_label)
        self.sigma_color = QSlider(Qt.Horizontal)
        self.sigma_color.setRange(100, 2000)
        self.sigma_color.setValue(1000)
        self.sigma_color.valueChanged.connect(lambda v: self.sigma_color_label.setText(f"σ_c: {v}"))
        sigma_color_layout.addWidget(self.sigma_color)
        sigma_controls.addLayout(sigma_color_layout)
        
        sigma_space_layout = QVBoxLayout()
        self.sigma_space_label = QLabel("σ_s: 500")
        sigma_space_layout.addWidget(self.sigma_space_label)
        self.sigma_space = QSlider(Qt.Horizontal)
        self.sigma_space.setRange(50, 1000)
        self.sigma_space.setValue(500)
        self.sigma_space.valueChanged.connect(lambda v: self.sigma_space_label.setText(f"σ_s: {v}"))
        sigma_space_layout.addWidget(self.sigma_space)
        sigma_controls.addLayout(sigma_space_layout)
        
        filter_layout.addLayout(sigma_controls)
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        # Reset Button
        reset_group = QGroupBox("Reset")
        reset_group.setMaximumHeight(60)
        reset_layout = QVBoxLayout()
        
        self.reset_button = QPushButton("Reset to Original")
        self.reset_button.clicked.connect(self.on_reset)
        self.reset_button.setEnabled(False)
        reset_layout.addWidget(self.reset_button)
        
        reset_group.setLayout(reset_layout)
        layout.addWidget(reset_group)
        
        # Preset Group (NEW)
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
    
    def apply_endothelial_preset(self):
        """Apply optimized preset for endothelial cells"""
        # Tophat settings
        self.tophat_size_slider.setValue(25)
        
        # CLAHE settings (can be lower after tophat)
        self.clip_limit.setValue(2)
        self.grid_size.setValue(18)
        
        # Contrast settings  
        self.alpha.setValue(150)  # 150%
        self.beta.setValue(10)    # 1.0 (slider * 10)
        
        # Filter settings
        self.filter_size.setValue(3)
        self.sigma_color.setValue(700)
        self.sigma_space.setValue(300)
        
        # Update labels
        self.tophat_size_label.setText(f"Disk Size: {self.tophat_size_slider.value()}px")
        self.clip_label.setText(f"Clip: {self.clip_limit.value()}")
        self.grid_label.setText(f"Grid: {self.grid_size.value()}")
        self.alpha_label.setText(f"α: {self.alpha.value()}%")
        self.beta_label.setText(f"β: {self.beta.value()}")
        self.filter_label.setText(f"Size: {self.filter_size.value()}")
        self.sigma_color_label.setText(f"σ_c: {self.sigma_color.value()}")
        self.sigma_space_label.setText(f"σ_s: {self.sigma_space.value()}")
        
        print("Applied endothelial cell preset:")
        print("  Tophat: Disk=25px")
        print("  CLAHE: Clip=2, Grid=18")
        print("  Contrast: α=150%, β=1.0") 
        print("  Filter: Size=3, σ_c=700, σ_s=300")
    
    def get_parameters(self) -> dict:
        """
        Get current parameter values
        
        Returns:
            Dictionary with current parameters
        """
        return {
            'tophat_params': {
                'disk_size': self.tophat_size_slider.value()
            },
            'clahe_params': {
                'clip_limit': self.clip_limit.value(),
                'grid_size': self.grid_size.value()
            },
            'contrast_params': {
                'alpha': self.alpha.value() / 100.0,
                'beta': self.beta.value() / 10.0
            },
            'filter_params': {
                'd': self.filter_size.value(),
                'sigma_color': self.sigma_color.value(),
                'sigma_space': self.sigma_space.value()
            }
        }
    
    def set_parameters(self, params: dict):
        """
        Set parameter values from dictionary
        
        Args:
            params: Dictionary with parameters
        """
        if 'tophat_params' in params:
            tophat_params = params['tophat_params']
            self.tophat_size_slider.setValue(int(tophat_params.get('disk_size', 20)))
        
        if 'clahe_params' in params:
            clahe_params = params['clahe_params']
            self.clip_limit.setValue(int(clahe_params.get('clip_limit', 1)))
            self.grid_size.setValue(int(clahe_params.get('grid_size', 30)))
        
        if 'contrast_params' in params:
            contrast_params = params['contrast_params']
            self.alpha.setValue(int(contrast_params.get('alpha', 1.2) * 100))
            self.beta.setValue(int(contrast_params.get('beta', 0.2) * 10))
        
        if 'filter_params' in params:
            filter_params = params['filter_params']
            self.filter_size.setValue(int(filter_params.get('d', 5)))
            self.sigma_color.setValue(int(filter_params.get('sigma_color', 1000)))
            self.sigma_space.setValue(int(filter_params.get('sigma_space', 500)))
    
    def set_enabled(self, enabled: bool):
        """Enable or disable all controls"""
        self.enabled = enabled
        self.tophat_button.setEnabled(enabled)
        self.clahe_button.setEnabled(enabled)
        self.contrast_button.setEnabled(enabled)
        self.filter_button.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
    
    # Event handlers
    def on_load_tiff(self):
        """Handle load TIFF button click"""
        self.tiff_loaded.emit()
    
    def on_apply_tophat(self):
        """Handle apply tophat button click"""
        self.processing_updated.emit('tophat')
    
    def on_apply_clahe(self):
        """Handle apply CLAHE button click"""
        self.processing_updated.emit('clahe')
    
    def on_apply_contrast(self):
        """Handle apply contrast button click"""
        self.processing_updated.emit('contrast')
    
    def on_apply_filter(self):
        """Handle apply filter button click"""
        self.processing_updated.emit('filtered')
    
    def on_reset(self):
        """Handle reset button click"""
        self.main_widget.reset_to_original()