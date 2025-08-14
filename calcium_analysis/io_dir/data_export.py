"""
Data export and import utilities for HDF5 format
"""

import numpy as np
import datetime
import os
from typing import Optional, Dict, Any


class DataExporter:
    """
    Handler for exporting and importing segmentation data
    """
    
    def __init__(self):
        self.last_export_info = {}
        self.last_import_info = {}
    
    def export_segmentation_data(self, file_path: str, data_dict: Dict[str, Any]) -> bool:
        """
        Export segmentation data to HDF5 file
        
        Args:
            file_path: Output file path
            data_dict: Dictionary containing all data to export
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import h5py
            
            with h5py.File(file_path, 'w') as f:
                # Create main groups
                labels_group = f.create_group('labels')
                exclusions_group = f.create_group('exclusions')
                metadata_group = f.create_group('metadata')
                parameters_group = f.create_group('parameters')
                
                # Save cell labels
                if 'final_labels' in data_dict and data_dict['final_labels'] is not None:
                    labels_data = data_dict['final_labels']
                    labels_group.create_dataset('cell_labels', data=labels_data, compression='gzip')
                    labels_group['cell_labels'].attrs['description'] = 'Final segmented cell labels'
                    labels_group['cell_labels'].attrs['num_cells'] = len(np.unique(labels_data)) - 1
                
                # Save exclusion regions
                exclusion_mask = data_dict.get('exclusion_mask')
                if exclusion_mask is not None:
                    exclusions_group.create_dataset('exclusion_mask', 
                                                   data=exclusion_mask.astype(np.uint8), 
                                                   compression='gzip')
                    exclusions_group['exclusion_mask'].attrs['description'] = 'Boolean mask of excluded regions'
                    exclusions_group['exclusion_mask'].attrs['excluded_pixels'] = np.sum(exclusion_mask)
                else:
                    # Create empty exclusion mask if original image is available
                    if 'original_image' in data_dict and data_dict['original_image'] is not None:
                        empty_mask = np.zeros(data_dict['original_image'].shape[:2], dtype=np.uint8)
                        exclusions_group.create_dataset('exclusion_mask', data=empty_mask, compression='gzip')
                        exclusions_group['exclusion_mask'].attrs['description'] = 'No exclusions defined'
                
                # Save original image for reference
                if 'original_image' in data_dict and data_dict['original_image'] is not None:
                    labels_group.create_dataset('original_image', 
                                              data=data_dict['original_image'], 
                                              compression='gzip')
                    labels_group['original_image'].attrs['description'] = 'Averaged original image'
                
                # Save processing parameters
                if 'parameters' in data_dict:
                    params = data_dict['parameters']
                    for key, value in params.items():
                        parameters_group.attrs[key] = value
                
                # Save file information
                if 'file_info' in data_dict:
                    file_info = data_dict['file_info']
                    for key, value in file_info.items():
                        if value is not None:
                            metadata_group.attrs[key] = str(value)
                
                # Save metadata
                metadata_group.attrs['export_date'] = datetime.datetime.now().isoformat()
                metadata_group.attrs['file_version'] = '1.0'
                metadata_group.attrs['software'] = 'Calcium Segmentation Tool'
                
                if 'original_image' in data_dict and data_dict['original_image'] is not None:
                    metadata_group.attrs['image_shape'] = data_dict['original_image'].shape
            
            # Store export info
            self.last_export_info = {
                'file_path': file_path,
                'export_time': datetime.datetime.now(),
                'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
                'groups_created': ['labels', 'exclusions', 'metadata', 'parameters']
            }
            
            return True
            
        except ImportError:
            raise RuntimeError("h5py not installed. Install with: pip install h5py")
        except Exception as e:
            raise RuntimeError(f"Export failed: {str(e)}")
    
    def import_segmentation_data(self, file_path: str) -> Dict[str, Any]:
        """
        Import segmentation data from HDF5 file
        
        Args:
            file_path: Input file path
        
        Returns:
            Dictionary containing all imported data
        """
        try:
            import h5py
            
            imported_data = {}
            
            with h5py.File(file_path, 'r') as f:
                # Load cell labels
                if 'labels/cell_labels' in f:
                    imported_data['final_labels'] = f['labels/cell_labels'][:]
                    imported_data['num_cells'] = len(np.unique(imported_data['final_labels'])) - 1
                
                # Load exclusion regions
                if 'exclusions/exclusion_mask' in f:
                    imported_data['exclusion_mask'] = f['exclusions/exclusion_mask'][:]
                
                # Load original image
                if 'labels/original_image' in f:
                    imported_data['original_image'] = f['labels/original_image'][:]
                
                # Load parameters
                if 'parameters' in f:
                    imported_data['parameters'] = {}
                    for key, value in f['parameters'].attrs.items():
                        imported_data['parameters'][key] = value
                
                # Load metadata
                if 'metadata' in f:
                    imported_data['metadata'] = {}
                    for key, value in f['metadata'].attrs.items():
                        imported_data['metadata'][key] = value
                
                # Load file info if available
                imported_data['file_info'] = {}
                if 'metadata' in f:
                    metadata_attrs = f['metadata'].attrs
                    file_info_keys = ['original_filename', 'source_file']
                    for key in file_info_keys:
                        if key in metadata_attrs:
                            imported_data['file_info'][key] = str(metadata_attrs[key])
            
            # Store import info
            self.last_import_info = {
                'file_path': file_path,
                'import_time': datetime.datetime.now(),
                'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
                'data_loaded': list(imported_data.keys())
            }
            
            return imported_data
            
        except ImportError:
            raise RuntimeError("h5py not installed. Install with: pip install h5py")
        except Exception as e:
            raise RuntimeError(f"Import failed: {str(e)}")
    
    def export_simple_format(self, file_path: str, labels: np.ndarray, 
                           original_image: Optional[np.ndarray] = None) -> bool:
        """
        Export in simple format (just labels and optionally original image)
        
        Args:
            file_path: Output file path
            labels: Labeled image
            original_image: Optional original image
        
        Returns:
            True if successful
        """
        data_dict = {
            'final_labels': labels,
            'original_image': original_image,
            'parameters': {},
            'file_info': {'export_type': 'simple'}
        }
        
        return self.export_segmentation_data(file_path, data_dict)
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about an HDF5 file without loading all data
        
        Args:
            file_path: Path to HDF5 file
        
        Returns:
            Dictionary with file information
        """
        try:
            import h5py
            
            info = {
                'file_path': file_path,
                'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
                'groups': [],
                'datasets': [],
                'metadata': {}
            }
            
            with h5py.File(file_path, 'r') as f:
                # Get groups
                def collect_info(name, obj):
                    if isinstance(obj, h5py.Group):
                        info['groups'].append(name)
                    elif isinstance(obj, h5py.Dataset):
                        info['datasets'].append({
                            'name': name,
                            'shape': obj.shape,
                            'dtype': str(obj.dtype)
                        })
                
                f.visititems(collect_info)
                
                # Get metadata
                if 'metadata' in f:
                    for key, value in f['metadata'].attrs.items():
                        info['metadata'][key] = str(value)
            
            return info
            
        except ImportError:
            raise RuntimeError("h5py not installed. Install with: pip install h5py")
        except Exception as e:
            raise RuntimeError(f"Failed to get file info: {str(e)}")
    
    def get_last_export_info(self) -> Dict[str, Any]:
        """Get information about the last export operation"""
        return self.last_export_info.copy()
    
    def get_last_import_info(self) -> Dict[str, Any]:
        """Get information about the last import operation"""
        return self.last_import_info.copy()


def create_backup(file_path: str, backup_suffix: str = "_backup") -> str:
    """
    Create a backup copy of a file
    
    Args:
        file_path: Original file path
        backup_suffix: Suffix to add to backup filename
    
    Returns:
        Path to backup file
    """
    import shutil
    
    base_name, ext = os.path.splitext(file_path)
    backup_path = f"{base_name}{backup_suffix}{ext}"
    
    try:
        shutil.copy2(file_path, backup_path)
        return backup_path
    except Exception as e:
        raise RuntimeError(f"Failed to create backup: {str(e)}")


def validate_hdf5_file(file_path: str) -> Dict[str, Any]:
    """
    Validate that an HDF5 file contains expected segmentation data
    
    Args:
        file_path: Path to HDF5 file
    
    Returns:
        Dictionary with validation results
    """
    try:
        import h5py
        
        validation = {
            'is_valid': True,
            'has_labels': False,
            'has_exclusions': False,
            'has_original': False,
            'has_metadata': False,
            'errors': [],
            'warnings': []
        }
        
        with h5py.File(file_path, 'r') as f:
            # Check for required groups
            required_groups = ['labels', 'exclusions', 'metadata']
            for group in required_groups:
                if group not in f:
                    validation['errors'].append(f"Missing required group: {group}")
                    validation['is_valid'] = False
            
            # Check for labels
            if 'labels/cell_labels' in f:
                validation['has_labels'] = True
                labels_data = f['labels/cell_labels']
                if labels_data.shape == (0,) or np.max(labels_data) == 0:
                    validation['warnings'].append("Labels dataset is empty")
            else:
                validation['errors'].append("Missing cell labels dataset")
                validation['is_valid'] = False
            
            # Check for exclusions
            if 'exclusions/exclusion_mask' in f:
                validation['has_exclusions'] = True
            
            # Check for original image
            if 'labels/original_image' in f:
                validation['has_original'] = True
            else:
                validation['warnings'].append("No original image found")
            
            # Check for metadata
            if 'metadata' in f and len(f['metadata'].attrs) > 0:
                validation['has_metadata'] = True
            else:
                validation['warnings'].append("No metadata found")
        
        return validation
        
    except ImportError:
        raise RuntimeError("h5py not installed. Install with: pip install h5py")
    except Exception as e:
        return {
            'is_valid': False,
            'errors': [f"Validation failed: {str(e)}"],
            'warnings': []
        }