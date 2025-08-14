# Structure
"""
calcium_segmentation/
├── main.py
├── __init__.py
├── widgets/
│   ├── __init__.py
│   ├── main_widget.py
│   ├── image_enhancement.py
│   ├── segmentation_controls.py
│   ├── analysis_window.py
│   └── exclusion_controls.py
├── processing/
│   ├── __init__.py
│   ├── image_processing.py
│   ├── segmentation.py
│   ├── analysis.py
│   └── region_growing.py
├── io_dir/
│   ├── __init__.py
│   ├── tiff_loader.py
│   └── data_export.py
└── utils/
    ├── __init__.py
    ├── peak_detection.py
    └── morphology_utils.py
"""
import napari
from widgets.main_widget import CalciumSegmentationWidget


def main():
    """Main entry point for the calcium segmentation application"""
    # Create napari viewer
    viewer = napari.Viewer(title="Calcium Imaging Segmentation")
    
    # Create and add the main widget
    widget = CalciumSegmentationWidget(viewer)
    viewer.window.add_dock_widget(widget, area='right', name='Segmentation Controls')
    
    # Start napari
    napari.run()


if __name__ == "__main__":
    main()