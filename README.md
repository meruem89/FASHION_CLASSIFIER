# Visual Taxonomy Classification Project

## Overview

This project implements a multi-label classification system for fashion items using computer vision and deep learning techniques. The system predicts multiple attributes for different categories of clothing items including Sarees, Men's T-shirts, Kurtis, Women's T-shirts, and Women's Tops & Tunics.

## Dataset

The project uses the Visual Taxonomy dataset from Kaggle, which contains:
- **Categories**: 5 clothing categories
- **Attributes**: Up to 10 different attributes per item (attr_1 to attr_10)
- **Images**: High-resolution product images for training and testing
- **Structure**: Structured CSV files with image IDs and corresponding attribute labels

### Data Distribution
- **Sarees**: 18,346 samples
- **Men T-shirts**: Training samples with 5 relevant attributes
- **Kurtis**: Training samples with 9 relevant attributes  
- **Women T-shirts**: Training samples with 8 relevant attributes
- **Women Tops & Tunics**: Training samples with all 10 attributes

## Methodology

### 1. Data Preprocessing
- **Missing Value Handling**: 
  - Removed rows with more than 5 null values to maintain data quality
  - Applied mode imputation for remaining missing values
  - Added dummy values for irrelevant attributes in specific categories

- **Data Cleaning**:
  - Category-wise data separation
  - Threshold-based row filtering using `dropna(thresh=...)`
  - Mode-based imputation for categorical attributes

### 2. Feature Engineering
- **Image Processing**:
  - Resized all images to 224×224 pixels (VGG16 input requirement)
  - Applied VGG16 preprocessing using `preprocess_input`
  - Batch processing with size 256 for memory efficiency

- **Feature Extraction**:
  - Used pre-trained VGG16 (ImageNet weights) as feature extractor
  - Generated 1000-dimensional feature vectors per image
  - Concatenated batch predictions for complete feature sets

- **Label Encoding**:
  - Applied one-hot encoding to categorical attributes using `pd.get_dummies()`
  - Created binary representation for multi-label classification

### 3. Model Architecture

#### Deep Neural Network (Primary Model)
```python
model_final = Sequential([
    Dense(512, input_shape=(1000,), activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(48, activation='sigmoid')  # Multi-label output
])
```

**Configuration**:
- **Input**: 1000-dimensional VGG16 features
- **Architecture**: 4 hidden layers with decreasing neurons
- **Output**: 48-dimensional multi-label predictions
- **Activation**: Sigmoid for multi-label classification
- **Loss Function**: Binary crossentropy
- **Optimizer**: Adam
- **Training**: 20 epochs, 80/20 train-test split

#### Alternative: Random Forest Classifier
- **Algorithm**: RandomForestClassifier
- **Trees**: 100 estimators
- **Purpose**: Comparison with neural network approach

## Technical Implementation

### Dependencies
```python
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
```

### Key Features
- **Memory Efficient**: Batch processing for large image datasets
- **Transfer Learning**: Leverages pre-trained VGG16 for feature extraction
- **Multi-label Support**: Handles multiple attribute predictions per item
- **Categorical Processing**: Category-wise model training and evaluation
- **Visualization**: Training history plots for model performance monitoring

## Setup and Installation

### Prerequisites
```bash
pip install pandas numpy opencv-python tensorflow scikit-learn matplotlib
```

### Kaggle API Setup
1. Install Kaggle API: `pip install kaggle`
2. Download API credentials from Kaggle Account settings
3. Place `kaggle.json` in `~/.kaggle/` directory
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Data Download
```bash
kaggle competitions download -c visual-taxonomy
```

## Usage

### 1. Data Loading and Preprocessing
```python
# Load and preprocess training data
train_data = pd.read_csv('train.csv')
# Category-wise separation and cleaning
# Missing value imputation
```

### 2. Image Processing and Feature Extraction
```python
# Load and resize images
# Apply VGG16 preprocessing
# Extract features using pre-trained VGG16
```

### 3. Model Training
```python
# Train neural network model
history = model_final.fit(x_train, y_train, 
                         validation_data=(x_test, y_test),
                         epochs=20, batch_size=256)
```

### 4. Prediction and Evaluation
```python
# Generate predictions on test set
predictions = model_final.predict(test_features)
```

## Model Performance

The model architecture includes:
- **Training Monitoring**: Real-time loss and accuracy tracking
- **Validation**: 20% holdout validation set
- **Visualization**: Training curves for loss and accuracy
- **Multi-label Metrics**: Appropriate for fashion attribute prediction

## Project Structure

```
visual-taxonomy/
├── README.md                 # Project documentation
├── train.csv                 # Training labels and metadata
├── test.csv                  # Test labels and metadata
├── train_images/             # Training images directory
├── test_images/              # Test images directory
├── data_preprocessing.py     # Data cleaning and preparation
├── feature_extraction.py    # VGG16 feature extraction
├── model_training.py        # Neural network training
├── prediction.py            # Model inference
└── requirements.txt          # Project dependencies
```

## Results and Output

The system generates:
- **Feature Vectors**: 1000-dimensional representations for each image
- **Multi-label Predictions**: Binary predictions for each attribute
- **Performance Metrics**: Training and validation accuracy/loss curves
- **Submission Files**: CSV files with predictions for test data

## Key Achievements

- **Multi-category Support**: Handles 5 different clothing categories
- **Robust Preprocessing**: Handles missing data effectively
- **Transfer Learning**: Utilizes pre-trained VGG16 for feature extraction
- **Scalable Architecture**: Batch processing for memory efficiency
- **Comprehensive Pipeline**: End-to-end solution from raw images to predictions

## Future Enhancements

- **Advanced Architectures**: Integration of ResNet, EfficientNet, or Vision Transformers
- **Ensemble Methods**: Combining multiple model predictions
- **Data Augmentation**: Image transformations for improved generalization
- **Hyperparameter Tuning**: Automated optimization of model parameters
- **Real-time Inference**: API development for production deployment

## Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest improvements.

## License

This project is available under the MIT License. See LICENSE file for more details.

## Acknowledgments

- Kaggle for providing the Visual Taxonomy dataset
- TensorFlow team for the pre-trained VGG16 model
- OpenCV community for image processing tools
