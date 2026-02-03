"""
Configuration settings for Brain Tumor Detection model
"""

import os

# Image settings
IMG_SIZE = 224
IMG_CHANNELS = 1  # Grayscale
BATCH_SIZE = 32

# Model settings
LEARNING_RATE = 0.001
EPOCHS = 50
VALIDATION_SPLIT = 0.2

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'training_set')
TEST_DIR = os.path.join(DATA_DIR, 'testing_set')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'brain_tumor_model.h5')
MODEL_PATH_LATEST = os.path.join(MODELS_DIR, 'brain_tumor_model_latest.h5')

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Class labels
CLASS_LABELS = {0: 'No Tumor', 1: 'Tumor'}
CLASS_INDICES = {'no_tumor': 0, 'tumor': 1}

# Early stopping
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MONITOR = 'val_loss'

# Random seed for reproducibility
RANDOM_SEED = 42
