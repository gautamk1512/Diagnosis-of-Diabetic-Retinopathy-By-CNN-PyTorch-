#!/usr/bin/env python3
"""
Sample Data Generator for Diabetic Retinopathy CNN Project

This script creates synthetic sample data to test the complete pipeline
when real diabetic retinopathy datasets are not available.

Usage:
    python create_sample_data.py
"""

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import random
from pathlib import Path

def create_synthetic_retinal_image(width=512, height=512, severity=0):
    """
    Create a synthetic retinal fundus image with varying characteristics
    based on diabetic retinopathy severity.
    
    Args:
        width (int): Image width
        height (int): Image height
        severity (int): DR severity level (0-4)
    
    Returns:
        PIL.Image: Synthetic retinal image
    """
    # Create base circular retinal image
    img = Image.new('RGB', (width, height), color='black')
    draw = ImageDraw.Draw(img)
    
    # Create circular fundus background
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 2 - 20
    
    # Base retinal color (reddish-orange)
    base_colors = [
        (180, 100, 60),   # Normal retina
        (190, 110, 70),   # Mild changes
        (200, 120, 80),   # Moderate changes
        (210, 130, 90),   # Severe changes
        (220, 140, 100)   # Proliferative changes
    ]
    
    base_color = base_colors[min(severity, 4)]
    
    # Draw circular fundus
    draw.ellipse(
        [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
        fill=base_color
    )
    
    # Add optic disc (bright circular area)
    disc_x = center_x + random.randint(-50, 50)
    disc_y = center_y + random.randint(-50, 50)
    disc_radius = random.randint(30, 50)
    
    draw.ellipse(
        [disc_x - disc_radius, disc_y - disc_radius, 
         disc_x + disc_radius, disc_y + disc_radius],
        fill=(255, 220, 180)
    )
    
    # Add blood vessels
    vessel_color = (120, 40, 40)
    for _ in range(random.randint(8, 15)):
        start_x = center_x + random.randint(-radius//2, radius//2)
        start_y = center_y + random.randint(-radius//2, radius//2)
        end_x = start_x + random.randint(-100, 100)
        end_y = start_y + random.randint(-100, 100)
        
        # Ensure vessels stay within fundus
        if (end_x - center_x)**2 + (end_y - center_y)**2 <= radius**2:
            draw.line([start_x, start_y, end_x, end_y], 
                     fill=vessel_color, width=random.randint(2, 5))
    
    # Add pathological features based on severity
    if severity >= 1:  # Mild DR - microaneurysms
        for _ in range(random.randint(3, 8)):
            x = center_x + random.randint(-radius//2, radius//2)
            y = center_y + random.randint(-radius//2, radius//2)
            if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                draw.ellipse([x-2, y-2, x+2, y+2], fill=(80, 20, 20))
    
    if severity >= 2:  # Moderate DR - hemorrhages
        for _ in range(random.randint(2, 6)):
            x = center_x + random.randint(-radius//2, radius//2)
            y = center_y + random.randint(-radius//2, radius//2)
            if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                size = random.randint(5, 12)
                draw.ellipse([x-size, y-size, x+size, y+size], fill=(60, 10, 10))
    
    if severity >= 3:  # Severe DR - cotton wool spots
        for _ in range(random.randint(2, 5)):
            x = center_x + random.randint(-radius//3, radius//3)
            y = center_y + random.randint(-radius//3, radius//3)
            if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                size = random.randint(8, 15)
                draw.ellipse([x-size, y-size, x+size, y+size], fill=(220, 200, 180))
    
    if severity >= 4:  # Proliferative DR - neovascularization
        for _ in range(random.randint(3, 7)):
            x = center_x + random.randint(-radius//3, radius//3)
            y = center_y + random.randint(-radius//3, radius//3)
            if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                # Draw irregular vessel patterns
                for _ in range(random.randint(3, 6)):
                    end_x = x + random.randint(-20, 20)
                    end_y = y + random.randint(-20, 20)
                    draw.line([x, y, end_x, end_y], fill=(100, 30, 30), width=2)
    
    # Add some noise and blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Add random noise
    img_array = np.array(img)
    noise = np.random.normal(0, 5, img_array.shape).astype(np.uint8)
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    return img

def create_sample_dataset(base_dir='./data', num_samples_per_class=50):
    """
    Create a complete sample dataset with train/val/test splits.
    
    Args:
        base_dir (str): Base directory for dataset
        num_samples_per_class (int): Number of samples per class
    """
    print("Creating sample diabetic retinopathy dataset...")
    
    # Create directory structure
    splits = ['train', 'val', 'test']
    classes = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4']
    class_names = ['No_DR', 'Mild_DR', 'Moderate_DR', 'Severe_DR', 'Proliferative_DR']
    
    for split in splits:
        for class_dir in classes:
            os.makedirs(os.path.join(base_dir, split, class_dir), exist_ok=True)
    
    # Distribution of samples across splits
    split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    
    # Create CSV files for each split
    csv_data = {'train': [], 'val': [], 'test': []}
    
    for class_idx, (class_dir, class_name) in enumerate(zip(classes, class_names)):
        print(f"Generating {class_name} samples...")
        
        for split in splits:
            num_samples = int(num_samples_per_class * split_ratios[split])
            
            for i in range(num_samples):
                # Create synthetic image
                img = create_synthetic_retinal_image(severity=class_idx)
                
                # Save image
                filename = f"{class_name}_{split}_{i:03d}.jpg"
                img_path = os.path.join(base_dir, split, class_dir, filename)
                img.save(img_path, 'JPEG', quality=85)
                
                # Add to CSV data
                relative_path = os.path.join(split, class_dir, filename)
                csv_data[split].append({
                    'image_path': relative_path,
                    'label': class_idx,
                    'class_name': class_name,
                    'severity': class_idx
                })
    
    # Save CSV files
    for split in splits:
        df = pd.DataFrame(csv_data[split])
        csv_path = os.path.join(base_dir, f"{split}_labels.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} samples to {csv_path}")
    
    # Create combined dataset info
    all_data = []
    for split in splits:
        for row in csv_data[split]:
            row['split'] = split
            all_data.append(row)
    
    df_all = pd.DataFrame(all_data)
    df_all.to_csv(os.path.join(base_dir, 'dataset_info.csv'), index=False)
    
    # Print dataset statistics
    print("\n" + "="*50)
    print("DATASET CREATION COMPLETE")
    print("="*50)
    print(f"Total samples: {len(df_all)}")
    print(f"Classes: {len(classes)}")
    print("\nClass distribution:")
    for class_idx, class_name in enumerate(class_names):
        count = len(df_all[df_all['label'] == class_idx])
        print(f"  {class_name}: {count} samples")
    
    print("\nSplit distribution:")
    for split in splits:
        count = len(df_all[df_all['split'] == split])
        percentage = (count / len(df_all)) * 100
        print(f"  {split}: {count} samples ({percentage:.1f}%)")
    
    print(f"\nDataset saved to: {os.path.abspath(base_dir)}")
    print("\nTo use this dataset:")
    print("  python main.py --mode train --data_dir ./data")
    print("  python main.py --mode evaluate --test_dir ./data/test")
    
    return base_dir

def create_single_test_image(output_path='./sample_image.jpg', severity=2):
    """
    Create a single test image for inference testing.
    
    Args:
        output_path (str): Path to save the test image
        severity (int): DR severity level (0-4)
    """
    print(f"Creating single test image with severity {severity}...")
    
    img = create_synthetic_retinal_image(width=512, height=512, severity=severity)
    img.save(output_path, 'JPEG', quality=90)
    
    print(f"Test image saved to: {os.path.abspath(output_path)}")
    print(f"To test inference:")
    print(f"  python main.py --mode predict --image_path {output_path}")
    
    return output_path

def main():
    """
    Main function to create sample dataset and test image.
    """
    print("Diabetic Retinopathy Sample Data Generator")
    print("==========================================\n")
    
    # Create sample dataset
    dataset_dir = create_sample_dataset(
        base_dir='./data',
        num_samples_per_class=20  # Reduced for quick testing
    )
    
    print("\n" + "-"*50)
    
    # Create single test image
    test_image = create_single_test_image(
        output_path='./sample_retinal_image.jpg',
        severity=2  # Moderate DR
    )
    
    print("\n" + "="*50)
    print("SETUP COMPLETE!")
    print("="*50)
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Train model: python main.py --mode train")
    print("3. Evaluate model: python main.py --mode evaluate")
    print("4. Test prediction: python main.py --mode predict --image_path ./sample_retinal_image.jpg")
    print("\nNote: This is synthetic data for testing purposes.")
    print("For real applications, use actual diabetic retinopathy datasets.")

if __name__ == "__main__":
    main()