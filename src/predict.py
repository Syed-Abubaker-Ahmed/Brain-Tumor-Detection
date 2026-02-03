"""
Prediction script for brain tumor detection
"""

import os
import sys

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from .image_processor import ImageProcessor
from config import MODEL_PATH, CLASS_LABELS, IMG_SIZE


def load_trained_model(model_path=MODEL_PATH):
    """
    Load trained model
    
    Args:
        model_path (str): Path to model file
        
    Returns:
        keras.Model: Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = keras.models.load_model(model_path)
    return model


def predict_image(image_path, model=None, model_path=MODEL_PATH, visualize=True):
    """
    Predict if an MRI image contains a brain tumor
    
    Args:
        image_path (str): Path to the image
        model (keras.Model): Optional pre-loaded model
        model_path (str): Path to model file if model is None
        visualize (bool): Whether to display the image with prediction
        
    Returns:
        dict: Dictionary with prediction results
    """
    # Load model if not provided
    if model is None:
        model = load_trained_model(model_path)
    
    # Preprocess image
    image = ImageProcessor.preprocess_image(image_path)
    image_batch = np.expand_dims(image, axis=0)
    
    # Make prediction
    prediction = model.predict(image_batch, verbose=0)[0][0]
    
    # Determine class
    class_idx = 1 if prediction > 0.5 else 0
    confidence = prediction if class_idx == 1 else 1 - prediction
    
    result = {
        'image_path': image_path,
        'prediction': class_idx,
        'class_name': CLASS_LABELS[class_idx],
        'confidence': float(confidence),
        'raw_probability': float(prediction)
    }
    
    if visualize:
        visualize_prediction(image_path, result)
    
    return result


def predict_batch(image_paths, model=None, model_path=MODEL_PATH):
    """
    Predict on a batch of images
    
    Args:
        image_paths (list): List of image paths
        model (keras.Model): Optional pre-loaded model
        model_path (str): Path to model file if model is None
        
    Returns:
        list: List of prediction results
    """
    # Load model if not provided
    if model is None:
        model = load_trained_model(model_path)
    
    results = []
    for image_path in image_paths:
        result = predict_image(image_path, model=model, visualize=False)
        results.append(result)
    
    return results


def visualize_prediction(image_path, result):
    """
    Visualize image with prediction
    
    Args:
        image_path (str): Path to the image
        result (dict): Prediction result dictionary
    """
    # Load original image
    image = ImageProcessor.load_image(image_path)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Display image
    ax.imshow(image, cmap='gray')
    
    # Add title with prediction
    title = f"Prediction: {result['class_name']}\nConfidence: {result['confidence']:.2%}"
    color = 'red' if result['class_name'] == 'Tumor' else 'green'
    ax.set_title(title, fontsize=14, fontweight='bold', color=color)
    
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def evaluate_on_test_set(test_dir, model=None, model_path=MODEL_PATH):
    """
    Evaluate model on test set
    
    Args:
        test_dir (str): Path to test directory with 'no_tumor' and 'tumor' subdirectories
        model (keras.Model): Optional pre-loaded model
        model_path (str): Path to model file if model is None
        
    Returns:
        dict: Evaluation metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    # Load model if not provided
    if model is None:
        model = load_trained_model(model_path)
    
    print("Evaluating on test set...")
    
    # Load test images
    image_paths = []
    true_labels = []
    
    # No tumor images
    no_tumor_dir = os.path.join(test_dir, 'no_tumor')
    if os.path.exists(no_tumor_dir):
        for img_file in os.listdir(no_tumor_dir):
            if img_file.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                image_paths.append(os.path.join(no_tumor_dir, img_file))
                true_labels.append(0)
    
    # Tumor images
    tumor_dir = os.path.join(test_dir, 'tumor')
    if os.path.exists(tumor_dir):
        for img_file in os.listdir(tumor_dir):
            if img_file.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                image_paths.append(os.path.join(tumor_dir, img_file))
                true_labels.append(1)
    
    if len(image_paths) == 0:
        print("No test images found!")
        return None
    
    # Make predictions
    predictions = predict_batch(image_paths, model=model)
    pred_labels = [p['prediction'] for p in predictions]
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    cm = confusion_matrix(true_labels, pred_labels)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'total_images': len(image_paths)
    }
    
    # Print results
    print(f"\nTest Set Results:")
    print(f"Total images: {results['total_images']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Brain Tumor Detection - Prediction Module")
    print("Use this module to make predictions on new MRI images")
