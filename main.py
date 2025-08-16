#!/usr/bin/env python3
"""
Diabetic Retinopathy Diagnosis using CNN (PyTorch)
Main script to run the complete pipeline
"""

import torch
import torch.nn as nn
import argparse
import os
import sys
import json
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from models.cnn_model import create_model, count_parameters
from utils.data_preprocessing import (
    load_dataset_from_csv, split_dataset, create_data_loaders, 
    calculate_class_weights, ImagePreprocessor
)
from train import Trainer
from utils.evaluation import ModelEvaluator, evaluate_saved_model

def setup_device():
    """
    Setup computing device (GPU/CPU)
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device

def create_sample_dataset():
    """
    Create a sample dataset CSV for demonstration
    """
    print("Creating sample dataset structure...")
    
    # Create sample CSV file
    sample_data = {
        'image_name': [f'image_{i:04d}.jpg' for i in range(1000)],
        'diagnosis': [i % 5 for i in range(1000)]  # 0-4 for 5 classes
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('data/sample_dataset.csv', index=False)
    
    print("Sample dataset CSV created at 'data/sample_dataset.csv'")
    print("Please replace this with your actual dataset!")
    print("Expected format: image_name, diagnosis (0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative)")
    
    return 'data/sample_dataset.csv'

def load_config(config_path):
    """
    Load configuration from JSON file
    """
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Default configuration
        return {
            'model_type': 'resnet50',
            'num_classes': 5,
            'batch_size': 32,
            'num_epochs': 50,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'scheduler': 'reduce_on_plateau',
            'loss_function': 'weighted_ce',
            'weight_decay': 1e-4,
            'patience': 10,
            'grad_clip': 1.0,
            'image_size': [224, 224],
            'test_size': 0.2,
            'val_size': 0.2,
            'num_workers': 4
        }

def save_config(config, config_path):
    """
    Save configuration to JSON file
    """
    # Create a copy of config and convert tensors to lists
    config_copy = config.copy()
    for key, value in config_copy.items():
        if hasattr(value, 'tolist'):  # Check if it's a tensor or numpy array
            config_copy[key] = value.tolist()
    
    with open(config_path, 'w') as f:
        json.dump(config_copy, f, indent=4)
    print(f"Configuration saved to {config_path}")

def train_model(args):
    """
    Train the diabetic retinopathy classification model
    """
    print("\n" + "="*60)
    print("TRAINING DIABETIC RETINOPATHY CLASSIFICATION MODEL")
    print("="*60)
    
    # Setup device
    device = setup_device()
    
    # Load configuration
    config = load_config(args.config)
    print(f"\nLoaded configuration from {args.config}")
    
    # Check if dataset exists
    if not os.path.exists(args.dataset_csv):
        print(f"Dataset CSV not found at {args.dataset_csv}")
        if args.create_sample:
            args.dataset_csv = create_sample_dataset()
        else:
            print("Use --create-sample to create a sample dataset structure")
            return
    
    # Load dataset
    print(f"\nLoading dataset from {args.dataset_csv}...")
    try:
        image_paths, labels = load_dataset_from_csv(args.dataset_csv, args.image_dir)
        print(f"Loaded {len(image_paths)} images")
        
        if len(image_paths) == 0:
            print("No valid images found. Please check your dataset and image directory.")
            return
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Split dataset
    print("\nSplitting dataset...")
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = split_dataset(
        image_paths, labels, 
        test_size=config['test_size'], 
        val_size=config['val_size']
    )
    
    print(f"Train: {len(train_paths)} images")
    print(f"Validation: {len(val_paths)} images")
    print(f"Test: {len(test_paths)} images")
    
    # Calculate class weights for imbalanced dataset
    class_weights = calculate_class_weights(train_labels)
    config['class_weights'] = class_weights
    print(f"\nClass weights: {class_weights}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader = create_data_loaders(
        train_paths, train_labels, val_paths, val_labels,
        batch_size=config['batch_size'],
        image_size=tuple(config['image_size']),
        num_workers=config['num_workers']
    )
    
    # Create model
    print(f"\nCreating {config['model_type']} model...")
    model = create_model(
        model_type=config['model_type'],
        num_classes=config['num_classes'],
        pretrained=True
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, device, config)
    
    # Start training
    print("\nStarting training...")
    history = trainer.train()
    
    # Save final configuration
    save_config(config, args.config)
    
    print("\nTraining completed!")
    print(f"Best model saved in checkpoints/best_model.pth")
    
    return history

def evaluate_model(args):
    """
    Evaluate a trained model
    """
    print("\n" + "="*60)
    print("EVALUATING DIABETIC RETINOPATHY CLASSIFICATION MODEL")
    print("="*60)
    
    # Setup device
    device = setup_device()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Model not found at {args.model_path}")
        return
    
    # Load dataset for testing
    if not os.path.exists(args.dataset_csv):
        print(f"Dataset CSV not found at {args.dataset_csv}")
        return
    
    print(f"\nLoading test dataset from {args.dataset_csv}...")
    image_paths, labels = load_dataset_from_csv(args.dataset_csv, args.image_dir)
    
    # For evaluation, we'll use the test split
    config = load_config(args.config)
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = split_dataset(
        image_paths, labels, 
        test_size=config['test_size'], 
        val_size=config['val_size']
    )
    
    # Create test data loader
    from utils.data_preprocessing import DiabeticRetinopathyDataset, get_val_transforms
    from torch.utils.data import DataLoader
    
    test_dataset = DiabeticRetinopathyDataset(
        test_paths, test_labels,
        transform=get_val_transforms(tuple(config['image_size'])),
        image_size=tuple(config['image_size'])
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=config['num_workers']
    )
    
    print(f"Test set: {len(test_paths)} images")
    
    # Evaluate model
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    metrics, y_true, y_pred, y_prob = evaluate_saved_model(
        args.model_path, test_loader, device, class_names
    )
    
    print("\nEvaluation completed!")
    print("Results saved in results/ directory")

def predict_single_image(args):
    """
    Predict diabetic retinopathy for a single image
    """
    print("\n" + "="*60)
    print("SINGLE IMAGE PREDICTION")
    print("="*60)
    
    # Setup device
    device = setup_device()
    
    # Check if model and image exist
    if not os.path.exists(args.model_path):
        print(f"Model not found at {args.model_path}")
        return
    
    if not os.path.exists(args.image_path):
        print(f"Image not found at {args.image_path}")
        return
    
    # Load model
    from models.cnn_model import create_model
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint['config']
    
    model = create_model(config['model_type'], config['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load and preprocess image
    from PIL import Image
    from utils.data_preprocessing import get_val_transforms
    
    image = Image.open(args.image_path).convert('RGB')
    transform = get_val_transforms(tuple(config['image_size']))
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Display results
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    
    print(f"\nImage: {args.image_path}")
    print(f"Predicted Class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.4f}")
    
    print("\nAll Class Probabilities:")
    for i, (class_name, prob) in enumerate(zip(class_names, probabilities[0])):
        print(f"{class_name:15}: {prob:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Diabetic Retinopathy Diagnosis using CNN')
    parser.add_argument('mode', choices=['train', 'evaluate', 'predict'], 
                       help='Mode: train, evaluate, or predict')
    
    # Common arguments
    parser.add_argument('--config', default='config.json', 
                       help='Configuration file path')
    parser.add_argument('--dataset-csv', default='data/dataset.csv',
                       help='Dataset CSV file path')
    parser.add_argument('--image-dir', default='data/images',
                       help='Directory containing images')
    
    # Training arguments
    parser.add_argument('--create-sample', action='store_true',
                       help='Create sample dataset structure')
    
    # Evaluation arguments
    parser.add_argument('--model-path', default='checkpoints/best_model.pth',
                       help='Path to trained model')
    
    # Prediction arguments
    parser.add_argument('--image-path', help='Path to single image for prediction')
    
    args = parser.parse_args()
    
    # Create default config if it doesn't exist
    if not os.path.exists(args.config):
        config = load_config(args.config)
        save_config(config, args.config)
    
    # Run based on mode
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'evaluate':
        evaluate_model(args)
    elif args.mode == 'predict':
        if not args.image_path:
            print("Error: --image-path is required for prediction mode")
            return
        predict_single_image(args)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()