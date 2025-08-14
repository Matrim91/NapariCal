"""
TIFF file loading utilities
"""

import numpy as np
from PIL import Image
from typing import Optional


class TiffLoader:
    """
    Handler for loading and processing TIFF files
    """
    
    def __init__(self):
        self.last_loaded_info = {}
    
    def load_and_average(self, file_path: str, max_frames: int = 100) -> np.ndarray:
        """
        Load TIFF file and calculate average of frames
        
        Args:
            file_path: Path to TIFF file
            max_frames: Maximum number of frames to process
        
        Returns:
            Averaged image as uint8 array
        """
        try:
            with Image.open(file_path) as img:
                frames = []
                frame_count = 0
                
                # Load frames
                while frame_count < max_frames:
                    try:
                        img.seek(frame_count)
                        frame_array = np.array(img)
                        frames.append(frame_array)
                        frame_count += 1
                    except EOFError:
                        break
                
                if not frames:
                    raise ValueError("No frames could be loaded from TIFF file")
                
                # Calculate average
                frames_array = np.array(frames)
                average_frame = np.mean(frames_array, axis=0)
                
                # Convert to uint8
                averaged_uint8 = self._normalize_to_uint8(average_frame)
                
                # Store loading info
                self.last_loaded_info = {
                    'file_path': file_path,
                    'total_frames_loaded': frame_count,
                    'max_frames_requested': max_frames,
                    'final_shape': averaged_uint8.shape,
                    'final_dtype': averaged_uint8.dtype,
                    'value_range': (averaged_uint8.min(), averaged_uint8.max())
                }
                
                return averaged_uint8
                
        except Exception as e:
            raise RuntimeError(f"Failed to load TIFF file {file_path}: {str(e)}")
    
    def load_single_frame(self, file_path: str, frame_index: int = 0) -> np.ndarray:
        """
        Load a single frame from TIFF file
        
        Args:
            file_path: Path to TIFF file
            frame_index: Index of frame to load
        
        Returns:
            Single frame as uint8 array
        """
        try:
            with Image.open(file_path) as img:
                img.seek(frame_index)
                frame_array = np.array(img)
                return self._normalize_to_uint8(frame_array)
        except Exception as e:
            raise RuntimeError(f"Failed to load frame {frame_index} from {file_path}: {str(e)}")
    
    def get_tiff_info(self, file_path: str) -> dict:
        """
        Get information about TIFF file without loading all data
        
        Args:
            file_path: Path to TIFF file
        
        Returns:
            Dictionary with TIFF file information
        """
        try:
            with Image.open(file_path) as img:
                # Count total frames
                frame_count = 0
                try:
                    while True:
                        img.seek(frame_count)
                        frame_count += 1
                except EOFError:
                    pass
                
                # Get first frame info
                img.seek(0)
                first_frame = np.array(img)
                
                return {
                    'total_frames': frame_count,
                    'frame_shape': first_frame.shape,
                    'frame_dtype': first_frame.dtype,
                    'file_size_mb': self._get_file_size_mb(file_path),
                    'estimated_memory_mb': self._estimate_memory_usage(first_frame.shape, frame_count)
                }
        except Exception as e:
            raise RuntimeError(f"Failed to get TIFF info for {file_path}: {str(e)}")
    
    def _normalize_to_uint8(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to uint8 range [0, 255]
        
        Args:
            image: Input image of any dtype
        
        Returns:
            Image normalized to uint8
        """
        if image.dtype == np.uint8:
            return image
        
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            normalized = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(image, dtype=np.uint8)
        
        return normalized
    
    def _get_file_size_mb(self, file_path: str) -> float:
        """Get file size in MB"""
        import os
        try:
            size_bytes = os.path.getsize(file_path)
            return size_bytes / (1024 * 1024)
        except:
            return 0.0
    
    def _estimate_memory_usage(self, frame_shape: tuple, frame_count: int) -> float:
        """Estimate memory usage in MB"""
        bytes_per_pixel = 4  # Assume float32 for processing
        total_pixels = np.prod(frame_shape) * frame_count
        return (total_pixels * bytes_per_pixel) / (1024 * 1024)
    
    def get_last_loaded_info(self) -> dict:
        """
        Get information about the last loaded file
        
        Returns:
            Dictionary with loading information
        """
        return self.last_loaded_info.copy()


def load_tiff_frames_generator(file_path: str, max_frames: Optional[int] = None):
    """
    Generator function to load TIFF frames one by one (memory efficient)
    
    Args:
        file_path: Path to TIFF file
        max_frames: Maximum number of frames to yield
    
    Yields:
        Individual frames as numpy arrays
    """
    try:
        with Image.open(file_path) as img:
            frame_count = 0
            
            while max_frames is None or frame_count < max_frames:
                try:
                    img.seek(frame_count)
                    frame_array = np.array(img)
                    yield frame_array, frame_count
                    frame_count += 1
                except EOFError:
                    break
    except Exception as e:
        raise RuntimeError(f"Failed to load TIFF frames from {file_path}: {str(e)}")


def calculate_frame_statistics(file_path: str, max_frames: int = 100):
    """
    Calculate statistics across frames without loading all into memory
    
    Args:
        file_path: Path to TIFF file
        max_frames: Maximum number of frames to analyze
    
    Returns:
        Dictionary with frame statistics
    """
    try:
        running_sum = None
        running_sum_squared = None
        frame_count = 0
        
        for frame, idx in load_tiff_frames_generator(file_path, max_frames):
            frame = frame.astype(np.float64)
            
            if running_sum is None:
                running_sum = frame.copy()
                running_sum_squared = frame ** 2
            else:
                running_sum += frame
                running_sum_squared += frame ** 2
            
            frame_count += 1
        
        if frame_count == 0:
            raise ValueError("No frames found in TIFF file")
        
        # Calculate statistics
        mean_frame = running_sum / frame_count
        variance_frame = (running_sum_squared / frame_count) - (mean_frame ** 2)
        std_frame = np.sqrt(variance_frame)
        
        return {
            'mean_frame': mean_frame,
            'std_frame': std_frame,
            'variance_frame': variance_frame,
            'frames_analyzed': frame_count,
            'global_mean': np.mean(mean_frame),
            'global_std': np.mean(std_frame)
        }
    
    except Exception as e:
        raise RuntimeError(f"Failed to calculate frame statistics for {file_path}: {str(e)}")