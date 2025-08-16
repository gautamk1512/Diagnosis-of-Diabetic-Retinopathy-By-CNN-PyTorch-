import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

class DiabeticRetinopathyDataset(Dataset):
    """
    Custom Dataset for Diabetic Retinopathy Classification
    """
    def __init__(self, image_paths, labels, transform=None, image_size=(224, 224)):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.image_size = image_size
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize image
        image = image.resize(self.image_size)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

class ImagePreprocessor:
    """
    Image preprocessing utilities for retinal images
    """
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single retinal image
        """
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Resize image
        image = cv2.resize(image, self.image_size)
        
        return image
    
    def crop_black_borders(self, image, threshold=10):
        """
        Remove black borders from retinal images
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour (should be the eye)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Crop the image
            cropped = image[y:y+h, x:x+w]
            return cropped
        
        return image

def get_train_transforms(image_size=(224, 224)):
    """
    Data augmentation transforms for training
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms(image_size=(224, 224)):
    """
    Validation transforms (no augmentation)
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_data_loaders(train_paths, train_labels, val_paths, val_labels, 
                       batch_size=32, image_size=(224, 224), num_workers=4):
    """
    Create data loaders for training and validation
    """
    # Create datasets
    train_dataset = DiabeticRetinopathyDataset(
        train_paths, train_labels, 
        transform=get_train_transforms(image_size),
        image_size=image_size
    )
    
    val_dataset = DiabeticRetinopathyDataset(
        val_paths, val_labels,
        transform=get_val_transforms(image_size),
        image_size=image_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader

def load_dataset_from_csv(csv_path, image_dir):
    """
    Load dataset from CSV file
    Expected CSV format: image_name/image_path, diagnosis/label
    diagnosis: 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative DR
    """
    df = pd.read_csv(csv_path)
    
    image_paths = []
    labels = []
    
    # Handle different column names
    if 'image_name' in df.columns:
        image_col = 'image_name'
    elif 'image_path' in df.columns:
        image_col = 'image_path'
    else:
        raise ValueError("CSV must contain 'image_name' or 'image_path' column")
    
    if 'diagnosis' in df.columns:
        label_col = 'diagnosis'
    elif 'label' in df.columns:
        label_col = 'label'
    else:
        raise ValueError("CSV must contain 'diagnosis' or 'label' column")
    
    for _, row in df.iterrows():
        image_name = row[image_col]
        diagnosis = row[label_col]
        
        # Construct full image path
        image_path = os.path.join(image_dir, image_name)
        
        if os.path.exists(image_path):
            image_paths.append(image_path)
            labels.append(diagnosis)
    
    return image_paths, labels

def split_dataset(image_paths, labels, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split dataset into train, validation, and test sets
    """
    # First split: train+val vs test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=val_size_adjusted, 
        random_state=random_state, stratify=train_val_labels
    )
    
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)

def calculate_class_weights(labels):
    """
    Calculate class weights for imbalanced dataset
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=classes, y=labels)
    
    return torch.FloatTensor(class_weights)

if __name__ == "__main__":
    # Example usage
    print("Data preprocessing utilities loaded successfully!")
    print("Available functions:")
    print("- DiabeticRetinopathyDataset: Custom dataset class")
    print("- ImagePreprocessor: Image preprocessing utilities")
    print("- get_train_transforms: Training data augmentation")
    print("- get_val_transforms: Validation transforms")
    print("- create_data_loaders: Create PyTorch data loaders")
    print("- load_dataset_from_csv: Load dataset from CSV")
    print("- split_dataset: Split data into train/val/test")
    print("- calculate_class_weights: Handle class imbalance")