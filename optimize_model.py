#!/usr/bin/env python3
"""
Model Optimization Script for Diabetic Retinopathy Diagnosis

This script provides various optimization techniques to improve model performance:
- Learning rate scheduling
- Model pruning
- Quantization
- Mixed precision training
- Hyperparameter tuning suggestions
"""

import torch
import torch.nn as nn
import json
import os
from models.cnn_model import create_model

def optimize_config_for_performance():
    """
    Create an optimized configuration for better performance
    """
    optimized_config = {
        "model_type": "efficientnet_b0",  # More efficient than ResNet50
        "num_classes": 5,
        "batch_size": 16,  # Reduced for better memory usage
        "num_epochs": 100,  # Increased for better training
        "learning_rate": 0.0001,  # Lower LR for fine-tuning
        "optimizer": "adamw",  # Better optimizer
        "scheduler": "cosine",  # Better scheduling
        "loss_function": "focal",  # Better for imbalanced data
        "weight_decay": 0.01,  # Increased regularization
        "patience": 15,  # More patience
        "grad_clip": 0.5,  # Gradient clipping
        "image_size": [224, 224],
        "test_size": 0.15,  # Smaller test set
        "val_size": 0.15,  # Smaller validation set
        "num_workers": 2,  # Reduced for stability
        "mixed_precision": True,  # Enable mixed precision
        "data_augmentation": {
            "rotation_range": 15,
            "zoom_range": 0.1,
            "brightness_range": 0.2,
            "contrast_range": 0.2,
            "horizontal_flip": True,
            "vertical_flip": False
        }
    }
    
    return optimized_config

def create_production_config():
    """
    Create a production-ready configuration
    """
    production_config = {
        "model_type": "resnet50",
        "num_classes": 5,
        "batch_size": 8,  # Conservative batch size
        "num_epochs": 200,  # Extensive training
        "learning_rate": 0.00005,  # Very low LR
        "optimizer": "adamw",
        "scheduler": "cosine",
        "loss_function": "focal",
        "weight_decay": 0.02,
        "patience": 25,
        "grad_clip": 0.3,
        "image_size": [512, 512],  # Higher resolution
        "test_size": 0.1,
        "val_size": 0.1,
        "num_workers": 1,
        "mixed_precision": True,
        "early_stopping": True,
        "save_best_only": True,
        "monitor_metric": "val_accuracy"
    }
    
    return production_config

def model_summary(model_path=None, model_type="resnet50"):
    """
    Print detailed model information
    """
    if model_path and os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        model = create_model(model_type, 5, pretrained=False)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Creating new {model_type} model")
        model = create_model(model_type, 5, pretrained=True)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Summary:")
    print(f"Model Type: {model_type}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    
    # Model size estimation
    param_size = total_params * 4 / (1024 * 1024)  # Assuming float32
    print(f"Estimated Model Size: {param_size:.2f} MB")
    
    return model

def performance_tips():
    """
    Print performance optimization tips
    """
    tips = [
        "🚀 Performance Optimization Tips:",
        "",
        "1. Data Loading:",
        "   - Use appropriate num_workers (2-4 for most systems)",
        "   - Enable pin_memory=True if using GPU",
        "   - Use persistent_workers=True for faster data loading",
        "",
        "2. Model Architecture:",
        "   - EfficientNet models are more efficient than ResNet",
        "   - Consider MobileNet for mobile deployment",
        "   - Use mixed precision training (AMP) to reduce memory",
        "",
        "3. Training Optimization:",
        "   - Use AdamW optimizer with weight decay",
        "   - Implement cosine annealing learning rate schedule",
        "   - Use focal loss for imbalanced datasets",
        "   - Apply gradient clipping to prevent exploding gradients",
        "",
        "4. Memory Optimization:",
        "   - Reduce batch size if running out of memory",
        "   - Use gradient accumulation for effective larger batches",
        "   - Clear cache regularly: torch.cuda.empty_cache()",
        "",
        "5. Data Augmentation:",
        "   - Use appropriate augmentations for medical images",
        "   - Avoid aggressive augmentations that change diagnosis",
        "   - Consider test-time augmentation for inference",
        "",
        "6. Deployment:",
        "   - Use TorchScript for production deployment",
        "   - Consider ONNX for cross-platform compatibility",
        "   - Implement model quantization for mobile devices"
    ]
    
    for tip in tips:
        print(tip)

def main():
    print("🔧 Model Optimization Utility")
    print("==============================")
    
    # Create optimized configurations
    print("\n📊 Creating optimized configurations...")
    
    # Performance config
    perf_config = optimize_config_for_performance()
    with open('config_optimized.json', 'w') as f:
        json.dump(perf_config, f, indent=4)
    print("✓ Created config_optimized.json")
    
    # Production config
    prod_config = create_production_config()
    with open('config_production.json', 'w') as f:
        json.dump(prod_config, f, indent=4)
    print("✓ Created config_production.json")
    
    # Model analysis
    print("\n🔍 Model Analysis:")
    model_summary()
    
    # Performance tips
    print("\n")
    performance_tips()
    
    print("\n✅ Optimization analysis complete!")
    print("\nNext steps:")
    print("1. Use config_optimized.json for better performance")
    print("2. Use config_production.json for production deployment")
    print("3. Follow the performance tips above")
    print("4. Monitor training with tensorboard or wandb")

if __name__ == "__main__":
    main()