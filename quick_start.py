#!/usr/bin/env python3
"""
Quick Start Script for Diabetic Retinopathy Diagnosis

This script demonstrates how to use the diabetic retinopathy diagnosis pipeline
with minimal setup. It creates sample data, trains a model, and makes predictions.
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✓ Success!")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Show last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        if e.stderr:
            print("Error details:", e.stderr)
        return False

def main():
    print("🔬 Diabetic Retinopathy Diagnosis - Quick Start")
    print("This script will demonstrate the complete pipeline.")
    
    # Check if we're in the right directory
    if not os.path.exists('main.py'):
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Step 1: Create sample data
    if not run_command("python create_sample_data.py", 
                      "Creating sample dataset"):
        print("❌ Failed to create sample data")
        return
    
    # Step 2: Train the model (quick training with 2 epochs)
    if not run_command("python main.py train --dataset-csv data/train_labels.csv --image-dir data",
                      "Training the model (2 epochs)"):
        print("❌ Training failed")
        return
    
    # Step 3: Make a prediction
    test_image = "data/test/class_0/No_DR_test_000.jpg"
    if os.path.exists(test_image):
        if not run_command(f"python main.py predict --image-path {test_image} --model-path checkpoints/best_model.pth",
                          "Making a prediction on test image"):
            print("❌ Prediction failed")
            return
    else:
        print(f"⚠️  Test image not found: {test_image}")
    
    print("\n🎉 Quick start completed successfully!")
    print("\n📁 Generated files:")
    print("   - checkpoints/best_model.pth (trained model)")
    print("   - config.json (configuration)")
    print("   - logs/ (training logs)")
    print("   - results/ (training plots)")
    print("   - data/ (sample dataset)")
    
    print("\n🚀 Next steps:")
    print("   1. Replace sample data with real retinal images")
    print("   2. Adjust config.json for your dataset")
    print("   3. Train with more epochs for better performance")
    print("   4. Use evaluation mode to assess model performance")
    
    print("\n📖 For more details, see README.md")

if __name__ == "__main__":
    main()