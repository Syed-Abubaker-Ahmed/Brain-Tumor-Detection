"""
CNN Model for brain tumor detection
"""

import os
import sys

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from config import IMG_SIZE, IMG_CHANNELS, LEARNING_RATE


def create_model(input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS)):
    """
    Create a Convolutional Neural Network model for brain tumor detection
    
    Args:
        input_shape (tuple): Input shape (height, width, channels)
        
    Returns:
        keras.Model: Compiled model
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                     input_shape=input_shape, name='conv1_1'),
        layers.BatchNormalization(name='bn1_1'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'),
        layers.BatchNormalization(name='bn1_2'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        layers.Dropout(0.25, name='dropout1'),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'),
        layers.BatchNormalization(name='bn2_1'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'),
        layers.BatchNormalization(name='bn2_2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        layers.Dropout(0.25, name='dropout2'),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1'),
        layers.BatchNormalization(name='bn3_1'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2'),
        layers.BatchNormalization(name='bn3_2'),
        layers.MaxPooling2D((2, 2), name='pool3'),
        layers.Dropout(0.25, name='dropout3'),
        
        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_1'),
        layers.BatchNormalization(name='bn4_1'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_2'),
        layers.BatchNormalization(name='bn4_2'),
        layers.MaxPooling2D((2, 2), name='pool4'),
        layers.Dropout(0.25, name='dropout4'),
        
        # Global Average Pooling
        layers.GlobalAveragePooling2D(name='global_avg_pool'),
        
        # Dense layers
        layers.Dense(512, activation='relu', name='dense1'),
        layers.BatchNormalization(name='bn_dense1'),
        layers.Dropout(0.5, name='dropout_dense1'),
        
        layers.Dense(256, activation='relu', name='dense2'),
        layers.BatchNormalization(name='bn_dense2'),
        layers.Dropout(0.5, name='dropout_dense2'),
        
        # Output layer
        layers.Dense(1, activation='sigmoid', name='output')
    ])
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    return model


def create_mobilenet_model(input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS)):
    """
    Create a MobileNetV2 transfer learning model (lighter and faster)
    
    Args:
        input_shape (tuple): Input shape (height, width, channels)
        
    Returns:
        keras.Model: Compiled model
    """
    # Load pre-trained MobileNetV2
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom layers on top
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x) if x.shape[-1] == 1 else x),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    return model


def get_model_summary(model):
    """
    Print model architecture summary
    
    Args:
        model (keras.Model): Model to summarize
    """
    model.summary()
