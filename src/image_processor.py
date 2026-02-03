"""
Image processing utilities for MRI scans
"""

import cv2
import numpy as np
from PIL import Image
import os
import sys

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import IMG_SIZE, IMG_CHANNELS


class ImageProcessor:
    """Class for processing and augmenting MRI images"""
    
    @staticmethod
    def load_image(image_path):
        """
        Load an image from file path
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Loaded image
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Read image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        return image
    
    @staticmethod
    def preprocess_image(image, target_size=(IMG_SIZE, IMG_SIZE)):
        """
        Preprocess image for model input
        
        Args:
            image (numpy.ndarray or str): Image array or path
            target_size (tuple): Target image size
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = ImageProcessor.load_image(image)
        
        # Resize image
        image_resized = cv2.resize(image, target_size)
        
        # Normalize to [0, 1]
        image_normalized = image_resized.astype('float32') / 255.0
        
        # Add channel dimension if needed
        if len(image_normalized.shape) == 2:
            image_normalized = np.expand_dims(image_normalized, axis=-1)
        
        return image_normalized
    
    @staticmethod
    def preprocess_batch(image_paths, target_size=(IMG_SIZE, IMG_SIZE)):
        """
        Preprocess a batch of images
        
        Args:
            image_paths (list): List of image paths
            target_size (tuple): Target image size
            
        Returns:
            numpy.ndarray: Batch of preprocessed images
        """
        batch = []
        for path in image_paths:
            img = ImageProcessor.preprocess_image(path, target_size)
            batch.append(img)
        
        return np.array(batch)
    
    @staticmethod
    def augment_image(image, rotation_range=20, width_shift_range=0.2,
                     height_shift_range=0.2, horizontal_flip=True, zoom_range=0.2):
        """
        Apply data augmentation to an image
        
        Args:
            image (numpy.ndarray): Input image
            rotation_range (int): Rotation angle in degrees
            width_shift_range (float): Width shift fraction
            height_shift_range (float): Height shift fraction
            horizontal_flip (bool): Whether to apply horizontal flip
            zoom_range (float): Zoom range
            
        Returns:
            numpy.ndarray: Augmented image
        """
        augmented = image.copy()
        
        # Random rotation
        angle = np.random.uniform(-rotation_range, rotation_range)
        h, w = augmented.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        augmented = cv2.warpAffine(augmented, M, (w, h))
        
        # Random width shift
        width_shift = int(w * np.random.uniform(-width_shift_range, width_shift_range))
        if width_shift > 0:
            augmented = np.pad(augmented, ((0, 0), (width_shift, 0)), mode='constant')[:, :w]
        elif width_shift < 0:
            augmented = np.pad(augmented, ((0, 0), (0, -width_shift)), mode='constant')[:, -w:]
        
        # Random height shift
        height_shift = int(h * np.random.uniform(-height_shift_range, height_shift_range))
        if height_shift > 0:
            augmented = np.pad(augmented, ((height_shift, 0), (0, 0)), mode='constant')[:h, :]
        elif height_shift < 0:
            augmented = np.pad(augmented, ((0, -height_shift), (0, 0)), mode='constant')[-h:, :]
        
        # Random horizontal flip
        if horizontal_flip and np.random.rand() > 0.5:
            augmented = cv2.flip(augmented, 1)
        
        # Random zoom
        zoom = np.random.uniform(1 - zoom_range, 1 + zoom_range)
        h_zoom = int(h * zoom)
        w_zoom = int(w * zoom)
        augmented = cv2.resize(augmented, (w_zoom, h_zoom))
        
        # Pad or crop to original size
        if zoom > 1:
            start_h = (h_zoom - h) // 2
            start_w = (w_zoom - w) // 2
            augmented = augmented[start_h:start_h+h, start_w:start_w+w]
        else:
            pad_h = (h - h_zoom) // 2
            pad_w = (w - w_zoom) // 2
            augmented = np.pad(augmented, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')[:h, :w]
        
        return augmented
    
    @staticmethod
    def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Apply Contrast Limited Adaptive Histogram Equalization
        
        Args:
            image (numpy.ndarray): Input image
            clip_limit (float): Contrast limit
            tile_grid_size (tuple): Tile grid size
            
        Returns:
            numpy.ndarray: Enhanced image
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(image)
        return enhanced
    
    @staticmethod
    def normalize_image(image):
        """
        Normalize image to [0, 1] range
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Normalized image
        """
        return image.astype('float32') / 255.0
    
    @staticmethod
    def save_image(image, output_path):
        """
        Save image to file
        
        Args:
            image (numpy.ndarray): Image to save
            output_path (str): Output file path
        """
        # Denormalize if needed
        if image.max() <= 1.0:
            image = (image * 255).astype('uint8')
        
        cv2.imwrite(output_path, image)
