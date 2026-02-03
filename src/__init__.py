"""Brain Tumor Detector Package"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

from .image_processor import ImageProcessor
from .model import create_model
from .predict import predict_image, predict_batch

__all__ = ['ImageProcessor', 'create_model', 'predict_image', 'predict_batch']
