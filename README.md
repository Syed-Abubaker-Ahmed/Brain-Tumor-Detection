# Brain Tumor Detection from MRI Images

A deep learning project for detecting brain tumors in MRI scan images using Convolutional Neural Networks (CNN).

## Features

- Image Preprocessing: Resize, normalize, and augment MRI images
- CNN Model: Deep learning model for tumor classification
- Training Pipeline: Train model with validation and checkpointing
- Batch Prediction: Predict on single images or batch of images
- Visualization: Display images with predictions and confidence scores

## Installation and Setup

1. Navigate to the project directory:
```bash
cd brain_tumor_detector
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

**Requirements:**
- TensorFlow >= 2.16.0
- OpenCV >= 4.8.0
- NumPy, Matplotlib, scikit-learn, Pandas, Seaborn, Pillow

4. (Optional) Prepare directory structure:
```bash
python3 setup.py
```

## Data Preparation

Organize your MRI images in the following structure:
```
data/
├── training_set/
│   ├── no_tumor/      # Place non-tumor MRI scans here
│   └── tumor/         # Place tumor MRI scans here
└── testing_set/
    ├── no_tumor/      # Test non-tumor images
    └── tumor/         # Test tumor images
```

Supported image formats: JPG, JPEG, PNG, TIFF

Recommended minimum: 200 images total (100 per class) for training

You can download brain MRI datasets from:
- Kaggle: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri
- BRATS: https://www.med.upenn.edu/cbica/brats2020/
- The Cancer Imaging Archive: https://www.cancerimagingarchive.net/

## Running the Program

Train the Model:
```bash
python3 main.py --mode train
```
Trains a CNN model on your data and saves the best model to models/brain_tumor_model.h5

Predict on a Single Image:
```bash
python3 main.py --mode predict --image_path /path/to/mri_image.jpg
```
Shows prediction (Tumor or No Tumor) with confidence percentage

Predict on Multiple Images:
```bash
python3 main.py --mode predict_batch --image_dir /path/to/images/
```
Predicts on all images in a directory with summary statistics

Evaluate on Test Set:
```bash
python3 main.py --mode evaluate
```
Tests model on all images in data/testing_set/ and shows accuracy metrics

## Python API Usage

```python
from src.predict import predict_image, predict_batch, evaluate_on_test_set
from src.train import train_model

# Single prediction
result = predict_image('mri_scan.jpg', visualize=True)
print(f"Prediction: {result['class_name']}")
print(f"Confidence: {result['confidence']:.2%}")

# Batch predictions
results = predict_batch(['image1.jpg', 'image2.jpg', 'image3.jpg'])
for r in results:
    print(f"{r['image_path']}: {r['class_name']} ({r['confidence']:.2%})")

# Train model
train_model()

# Evaluate
evaluate_on_test_set('data/testing_set/')
```

## Project Structure

```
brain_tumor_detector/
├── main.py                     # Entry point
├── config.py                   # Configuration settings
├── utils.py                    # Utility functions
├── setup.py                    # Setup script
├── requirements.txt            # Project dependencies
├── src/
│   ├── __init__.py
│   ├── image_processor.py      # Image processing utilities
│   ├── model.py                # Model architecture
│   ├── train.py                # Training script
│   └── predict.py              # Prediction script
├── models/                     # Trained model storage
├── data/                       # Dataset directory
│   ├── training_set/
│   ├── testing_set/
│   └── README.md
└── README.md                   # This file
```

## Model Details

- Architecture: Convolutional Neural Network with multiple layers
- Input Size: 224x224 pixels (grayscale)
- Output: Binary classification (Tumor / No Tumor)
- Optimizer: Adam
- Loss Function: Binary Crossentropy

The model will display:
- Training/validation accuracy and loss
- Confusion matrix on test set
- Prediction confidence for new images
- ROC curve analysis

## Notes

- Ensure images are in grayscale or will be converted automatically
- Images are resized to 224x224 pixels during preprocessing
- Data augmentation is applied during training for better generalization

## Disclaimer

This project is for educational purposes. For clinical use, validation with medical professionals and regulatory compliance is required.
