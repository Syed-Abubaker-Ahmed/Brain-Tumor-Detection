"""
Training script for brain tumor detection model
"""

import os
import sys

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

from .model import create_model
from .image_processor import ImageProcessor
from config import (TRAIN_DIR, MODEL_PATH, MODEL_PATH_LATEST, BATCH_SIZE, 
                    EPOCHS, EARLY_STOPPING_PATIENCE, IMG_SIZE, RANDOM_SEED)


def load_dataset(data_dir):
    """
    Load images and labels from directory structure
    
    Args:
        data_dir (str): Path to data directory with subdirectories 'no_tumor' and 'tumor'
        
    Returns:
        tuple: (image_paths, labels)
    """
    images = []
    labels = []
    
    # Load no tumor images
    no_tumor_dir = os.path.join(data_dir, 'no_tumor')
    if os.path.exists(no_tumor_dir):
        for img_file in os.listdir(no_tumor_dir):
            if img_file.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                images.append(os.path.join(no_tumor_dir, img_file))
                labels.append(0)
    
    # Load tumor images
    tumor_dir = os.path.join(data_dir, 'tumor')
    if os.path.exists(tumor_dir):
        for img_file in os.listdir(tumor_dir):
            if img_file.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                images.append(os.path.join(tumor_dir, img_file))
                labels.append(1)
    
    return np.array(images), np.array(labels)


def create_data_generators():
    """
    Create data augmentation generators for training
    
    Returns:
        tuple: (train_generator, validation_generator)
    """
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    return train_datagen, val_datagen


def train_model(train_dir=TRAIN_DIR):
    """
    Train the brain tumor detection model
    
    Args:
        train_dir (str): Path to training data directory
    """
    print("=" * 50)
    print("Brain Tumor Detection Model Training")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    # Load dataset
    print("\nLoading dataset...")
    image_paths, labels = load_dataset(train_dir)
    
    if len(image_paths) == 0:
        print("ERROR: No images found in training directory!")
        print(f"Please organize your data in: {train_dir}")
        print("Structure: no_tumor/ and tumor/ subdirectories")
        return
    
    print(f"Total images: {len(image_paths)}")
    print(f"No tumor images: {np.sum(labels == 0)}")
    print(f"Tumor images: {np.sum(labels == 1)}")
    
    # Preprocess images
    print("\nPreprocessing images...")
    X = ImageProcessor.preprocess_batch(image_paths)
    y = labels
    
    # Split data
    print("Splitting data (80/20 train/val)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model()
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH_LATEST,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model
    print(f"\nSaving model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    
    # Evaluate on validation set
    print("\nEvaluating model on validation set...")
    val_loss, val_acc, val_precision, val_recall = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")
    
    # Get predictions
    y_pred = model.predict(X_val)
    y_pred_binary = (y_pred > 0.5).astype(int).flatten()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_binary, target_names=['No Tumor', 'Tumor']))
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred_binary)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot results
    plot_training_results(history, y_val, y_pred)
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)


def plot_training_results(history, y_true, y_pred):
    """
    Plot training history and evaluation metrics
    
    Args:
        history: Training history from model.fit()
        y_true: True labels
        y_pred: Predicted probabilities
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Confusion Matrix
    y_pred_binary = (y_pred > 0.5).astype(int).flatten()
    cm = confusion_matrix(y_true, y_pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=['No Tumor', 'Tumor'],
                yticklabels=['No Tumor', 'Tumor'])
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    axes[1, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
    axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[1, 1].set_title('ROC Curve')
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=100, bbox_inches='tight')
    print("\nTraining results saved to training_results.png")
    plt.show()


if __name__ == "__main__":
    train_model()
