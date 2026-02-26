
# Aircraft Surface Defect Detection

## Overview
Deep learning project for automatic detection of structural surface defects on aircraft components using computer vision.

## Installation
pip install -r requirements.txt

## Dataset Structure
data/raw/
    cracked/
    intact/

## Workflow
1. Import dataset into data/raw/
2. Split into train/val/test
3. Train model:
   python src/training/train.py

## Outputs
Best model saved in checkpoints/best_model.pth

## Features
- Custom CNN model
- Transfer Learning with ResNet18
- Automatic dataset split (train/val/test)
- Evaluation with confusion matrix & classification report
- Grad-CAM interpretability
- Inference script
- Modular structure

## Future Improvements
- Transfer learning (ResNet/EfficientNet)
- Segmentation instead of classification
- Advanced data augmentation
- Deployment as web API

